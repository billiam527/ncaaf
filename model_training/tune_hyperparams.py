"""Nested hyperparameter search for the preseason and in-season models.

Called by create_experiment.sh --search. Prints the winning config as a single
JSON line on stdout (everything else goes to stderr) so the shell can capture it
straight into the experiment file's `model_params:` field.

The search is nested on purpose. Configs are scored by walk-forward MAE over an
inner block of validation seasons taken from the END of the training range, and
each config trains only on seasons before the one it is scored on. Scoring a
config on the seasons used to choose it makes almost any config look good.

Two candidates are always in the pool, so neither can be lost to a search that
happened to sample badly: library defaults, and the incumbent config passed on
--incumbent. A search that cannot beat what is already shipping should return
what is already shipping.

THE IN-SEASON BUILDER USED TO TUNE ON A DIFFERENT PROBLEM

inseason_season joined season_summaries on (team_id, season) - the end-of-year
leak that was removed from the model itself in the preprocess rebuild. So the
tuner was choosing hyperparameters for a 12-feature model that could see its own
season's results, and those parameters were then applied to a 72-feature model
that cannot. Fitting on one problem and applying to another, again.

It now builds the frame through in_season_model/preprocess.edit_files, the same
code the model trains on: rolling pre-game form plus the as-of-week adjustment,
season openers dropped, adjustment withheld before week 5.

    python tune_hyperparams.py --model in_season \
        --features rush_success_rolling_avg,... \
        --train-start 2017 --train-end 2025 \
        --incumbent '{"n_estimators": 400, ...}'
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

_REPO = '/home/bill/ncaaf'
sys.path.insert(0, f'{_REPO}/batch_prediction')

RESULTS = f'{_REPO}/etl/summarize/results'
GAMES = f'{_REPO}/etl/summarize/temp/games.csv'
TEAMS = f'{_REPO}/etl/collect/collect_espn_teams/temp/teams.csv'
ROLLING = f'{RESULTS}/rolling_summaries.csv'

DEFAULTS = {}          # XGBRegressor() as shipped
N_VALIDATION_SEASONS = 4
MIN_TRAIN_SEASONS = 3

# How much better than the incumbent a search winner has to be before it is
# allowed to replace shipped parameters.
#
# Not an arbitrary round number. The paired standard error of a per-game MAE
# difference on this validation block is about 0.022 points, and the winner of a
# ~60-config search is the argmax of ~60 draws, which reaches two standard
# errors by chance routinely - the in-season search did exactly that, beating
# the incumbent by 0.043 at t = 1.97 while its top five configs spanned 0.034.
# Requiring roughly four standard errors makes the search prove something the
# selection process cannot manufacture. research/in_season_hyperparams.sh is the
# paired test this number comes from.
MIN_IMPROVEMENT = 0.10


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def sample_configs(rng, n):
    """Weighted toward regularisation: the failure mode here is overfitting."""
    out = [DEFAULTS]
    for _ in range(n):
        out.append({
            'n_estimators':     int(rng.choice([60, 100, 200, 400, 700])),
            'max_depth':        int(rng.choice([2, 2, 3, 3, 4, 5, 6])),
            'learning_rate':    float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.15, 0.3])),
            'min_child_weight': int(rng.choice([1, 5, 10, 25, 50])),
            'subsample':        float(rng.choice([0.6, 0.8, 1.0])),
            'colsample_bytree': float(rng.choice([0.4, 0.6, 0.8, 1.0])),
            'reg_lambda':       float(rng.choice([1, 5, 20, 100])),
        })
    return out


def load_games():
    g = pd.read_csv(GAMES, low_memory=False)
    t = pd.read_csv(TEAMS)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
    return g[g['home_score_differential'].notna()]


def preseason_season(season, ss, games, features):
    import predict as P
    ss_edit = P.edit_files(season_summary_df=ss, features=features,
                           start_year=(season - 1) - 3, end_year=season - 1)
    g = games[games.season == season]
    if g.empty:
        return None, None
    X, meta = P.merge_games_and_stats(g, ss_edit)
    y = meta[['id']].merge(g[['id', 'home_score_differential']], on='id')
    return X.reset_index(drop=True), y['home_score_differential'].reset_index(drop=True)


def inseason_season(season, rolling, games, features):
    """One season's in-season training frame, built the way the model builds it.

    This used to join season_summaries on (team_id, season), which is the
    end-of-year leak the model itself no longer has. Going through
    preprocess.edit_files keeps one implementation, so a change to the feature
    construction cannot silently stop applying to the search.

    Column order matters and is the same order preprocess uses: the rolling
    block interleaved home/away in feature order, then the suffixed blocks
    sorted by name. The scaler and estimator index by position, not by name.
    """
    import contextlib
    import io

    sys.path.insert(0, f'{_REPO}/model_training/in_season_model')
    import preprocess as ISP

    g = games[games.season == season]
    if g.empty:
        return None, None
    # edit_files narrates each join; useful in a training run, noise in a sweep
    with contextlib.redirect_stdout(io.StringIO()):
        df = ISP.edit_files(games_df=g, rolling_df=rolling, features=features,
                            start_year=season, end_year=season)
    if df is None or df.empty:
        return None, None

    order = []
    for f in features:
        order += [f + '_home', f + '_away']
    order += sorted(c for c in df.columns
                    if c.endswith('_pri') or c.endswith('_pos')
                    or c.endswith('_adj'))
    missing = [c for c in order if c not in df.columns]
    if missing:
        log(f"  {season}: missing {len(missing)} expected columns, skipping")
        return None, None

    df = df.dropna(subset=['home_score_differential'])
    return (df[order].reset_index(drop=True),
            df['home_score_differential'].reset_index(drop=True))


def score(config, cache, val_seasons, first):
    total_err, total_n = 0.0, 0
    for S in val_seasons:
        train = [s for s in cache if first <= s < S]
        if len(train) < MIN_TRAIN_SEASONS:
            continue
        Xtr = pd.concat([cache[s][0] for s in train], ignore_index=True)
        ytr = pd.concat([cache[s][1] for s in train], ignore_index=True)
        Xte, yte = cache[S]
        sc = StandardScaler().fit(Xtr)
        m = XGBRegressor(random_state=0, **config).fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))
        total_err += float(np.abs(pred - yte).sum())
        total_n += len(yte)
    return total_err / total_n if total_n else np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['preseason', 'in_season'])
    ap.add_argument('--features', required=True, help='comma-separated feature list')
    ap.add_argument('--train-start', type=int, required=True)
    ap.add_argument('--train-end', type=int, required=True)
    ap.add_argument('--n-configs', type=int, default=45)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--incumbent', default=None,
                    help='JSON config currently shipping; always scored, and '
                         'returned unless something beats it')
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(',') if f.strip()]
    incumbent = json.loads(args.incumbent) if args.incumbent else None

    if args.model == 'preseason':
        builder, source = preseason_season, pd.read_csv(
            f'{RESULTS}/season_summaries.csv')
    else:
        # the in-season model reads rolling pre-game form, not season totals
        builder, source = inseason_season, pd.read_csv(ROLLING, low_memory=False)
    games = load_games()

    cache = {}
    for s in range(args.train_start, args.train_end + 1):
        X, y = builder(s, source, games, features)
        if X is not None:
            cache[s] = (X, y)
    if len(cache) < MIN_TRAIN_SEASONS + 1:
        log(f"  too few usable seasons ({len(cache)}); keeping library defaults")
        print(json.dumps(DEFAULTS))
        return

    seasons = sorted(cache)
    val_seasons = seasons[-N_VALIDATION_SEASONS:]
    val_seasons = [s for s in val_seasons
                   if len([x for x in seasons if x < s]) >= MIN_TRAIN_SEASONS]
    if not val_seasons:
        log("  no season has enough history to validate on; keeping defaults")
        print(json.dumps(DEFAULTS))
        return

    log(f"  {args.model}: {sum(len(v[0]) for v in cache.values())} games, "
        f"{cache[seasons[-1]][0].shape[1]} features")
    log(f"  validating on {val_seasons}, training on earlier seasons only")

    rng = np.random.default_rng(args.seed)
    configs = sample_configs(rng, args.n_configs)
    if incumbent is not None and incumbent not in configs:
        configs.append(incumbent)
    scored = []
    for i, c in enumerate(configs):
        scored.append((score(c, cache, val_seasons, args.train_start), c))
        if (i + 1) % 15 == 0:
            log(f"    scored {i + 1}/{len(configs)}")
    scored.sort(key=lambda t: t[0])

    best_mae, best = scored[0]
    default_mae = next(s for s, c in scored if c == DEFAULTS)
    log(f"  defaults {default_mae:.3f} MAE")
    if incumbent is not None:
        inc_mae = next(s for s, c in scored if c == incumbent)
        log(f"  incumbent {inc_mae:.3f} MAE")
    log(f"  best {best_mae:.3f} MAE  ({default_mae - best_mae:+.3f} vs defaults)")

    log("  top 5:")
    for m, c in scored[:5]:
        tag = ''
        if c == DEFAULTS:
            tag = '  <- library defaults'
        elif incumbent is not None and c == incumbent:
            tag = '  <- incumbent'
        log(f"    {m:.3f}  {json.dumps(c)}{tag}")

    # The incumbent has to be beaten by a real margin, not merely edged out.
    # See MIN_IMPROVEMENT: the winner of a search this size clears two standard
    # errors by chance, so a narrow win is the selection process talking.
    if incumbent is not None and best_mae >= inc_mae - MIN_IMPROVEMENT:
        log(f"  incumbent not beaten by {MIN_IMPROVEMENT} MAE "
            f"({inc_mae - best_mae:+.3f}); keeping it")
        print(json.dumps(incumbent))
        return

    if best_mae >= default_mae:
        log("  defaults were not beaten; keeping them")
        print(json.dumps(DEFAULTS))
        return

    log(f"  winning config: {json.dumps(best)}")
    print(json.dumps(best))


if __name__ == '__main__':
    main()
