"""Nested hyperparameter search for the preseason and in-season models.

Called by create_experiment.sh --search. Prints the winning config as a single
JSON line on stdout (everything else goes to stderr) so the shell can capture it
straight into the experiment file's `model_params:` field.

The search is nested on purpose. Configs are scored by walk-forward MAE over an
inner block of validation seasons taken from the END of the training range, and
each config trains only on seasons before the one it is scored on. Scoring a
config on the seasons used to choose it makes almost any config look good.

Library defaults are always included as a candidate, so if they win, they win.

    python tune_hyperparams.py --model in_season \
        --features adjusted_rush_success_off,... \
        --train-start 2015 --train-end 2024
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

DEFAULTS = {}          # XGBRegressor() as shipped
N_VALIDATION_SEASONS = 4
MIN_TRAIN_SEASONS = 3


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


def inseason_season(season, ss, games, features):
    s = ss[ss.season == season]
    cols = ['season', 'team_id'] + [f for f in features if f in s.columns]
    s = s.reset_index(drop=True)[cols]
    g = games[games.season == season]
    if g.empty or s.empty:
        return None, None
    away = g.merge(s, left_on=['away_team_id', 'season'], right_on=['team_id', 'season'])
    both = away.merge(s, left_on=['home_team_id', 'season'], right_on=['team_id', 'season'],
                      suffixes=('_away', '_home'))
    order = []
    for f in features:
        order += [f + '_home', f + '_away']
    if any(c not in both.columns for c in order):
        return None, None
    both = both.dropna(subset=order + ['home_score_differential'])
    return (both[order].reset_index(drop=True),
            both['home_score_differential'].reset_index(drop=True))


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
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(',') if f.strip()]
    builder = preseason_season if args.model == 'preseason' else inseason_season

    ss = pd.read_csv(f'{RESULTS}/season_summaries.csv')
    games = load_games()

    cache = {}
    for s in range(args.train_start, args.train_end + 1):
        X, y = builder(s, ss, games, features)
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
    scored = []
    for i, c in enumerate(configs):
        scored.append((score(c, cache, val_seasons, args.train_start), c))
        if (i + 1) % 15 == 0:
            log(f"    scored {i + 1}/{len(configs)}")
    scored.sort(key=lambda t: t[0])

    best_mae, best = scored[0]
    default_mae = next(s for s, c in scored if c == DEFAULTS)
    log(f"  defaults {default_mae:.3f} MAE  ->  best {best_mae:.3f} MAE "
        f"({default_mae - best_mae:+.3f})")

    if best_mae >= default_mae:
        log("  defaults were not beaten; keeping them")
        print(json.dumps(DEFAULTS))
        return

    log(f"  winning config: {json.dumps(best)}")
    print(json.dumps(best))


if __name__ == '__main__':
    main()
