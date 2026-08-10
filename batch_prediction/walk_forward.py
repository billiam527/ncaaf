"""Walk-forward generation of preseason / in-season predictions for a season.

The opponent adjustment is refit for every week using only games played in
earlier weeks, so week 3 features are built from weeks 1-2 and nothing later.
This matters anywhere historical predictions are produced:

  * backtesting - full-season adjusted stats would leak the rest of the season
    into early-week predictions and flatter the model.
  * blender training - model_blender.py learns how much weight to give each
    model per week. Fitting those weights on leaked features biases them
    towards the in-season model, which looks artificially strong early on.

There is a second, subtler leak that `generate_expanding_predictions` exists to
prevent: scoring a historical season with TODAY's models. Those models were
trained on that season, so their predictions are near-perfect recall rather
than prediction - measured at 0.98 correlation with actual results, against
0.29-0.43 for genuinely unseen seasons. Blend weights fit on that data load up
on whichever model did the memorising. Every historical season must therefore
be predicted by models retrained on seasons before it.

Used as a library (see `walk_forward_season`, `generate_expanding_predictions`)
and as a CLI backtest:

    python walk_forward.py --season 2025
    python walk_forward.py --season 2025 --pre-dir <dir> --ins-dir <dir>
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'etl', 'summarize'))
sys.path.insert(0, _HERE)

from summarize_games import OpponentAdjuster, AnalyticsConfig  # noqa: E402
import predict as P  # noqa: E402

RESULTS = os.path.join(_REPO, 'etl', 'summarize', 'results')
GAMES_CSV = os.path.join(_REPO, 'etl', 'summarize', 'temp', 'games.csv')
TEAMS_CSV = os.path.join(_REPO, 'etl', 'collect', 'collect_espn_teams', 'temp', 'teams.csv')

STATS = ("play_success rush_success pass_success yards_per_play rush_yards_per_play "
         "pass_yards_per_play explosive_play_rate explosive_rush_rate explosive_pass_rate "
         "epa_per_play epa_per_rush epa_per_pass").split()

POSTSEASON_WEEK = 90  # sorts bowls after the numbered weeks


def week_to_int(w):
    """'Week 3' -> 3, 'bowl' -> POSTSEASON_WEEK, anything else -> NaN."""
    if not isinstance(w, str):
        return np.nan
    if w.strip().lower().startswith('week'):
        try:
            return int(w.split()[-1])
        except ValueError:
            return np.nan
    return POSTSEASON_WEEK


def load_model(directory, token):
    """Return (features, model, scaler) for a trained experiment directory."""
    matches = [f for f in glob.glob(directory + '/*') if token in f.split(directory)[1]]
    if not matches:
        raise FileNotFoundError(f"no {token} experiment file in {directory}")
    features = None
    for line in open(matches[0]):
        if 'data_features' in line:
            features = line.split(': ')[1].rstrip('\n').split(',')
    model = pickle.load(open(directory + '/model.pkl', 'rb'))
    scaler = pickle.load(open(directory + '/scaler.pkl', 'rb'))
    return features, model, scaler


def fbs_team_ids():
    teams = pd.read_csv(TEAMS_CSV)
    return set(teams.loc[teams['fbs_ind'] == 1.0, 'id'])


def load_games(season, fbs_only=True):
    """Completed games for a season, with a sortable numeric week."""
    games = pd.read_csv(GAMES_CSV, low_memory=False)
    games['week_num'] = games['week'].map(week_to_int)
    games = games[(games.season == season) & games.week_num.notna()]
    games = games[games['home_score_differential'].notna()]
    if fbs_only:
        # Training filters to FBS (see model_training/*/preprocess.sh); scoring
        # on FCS opponents would feed the model games unlike any it has seen.
        fbs = fbs_team_ids()
        games = games[games.home_team_id.isin(fbs) & games.away_team_id.isin(fbs)]
    return games


def _merge_in_season(games, stats_df, features):
    """Attach team-level stats as <feature>_home / <feature>_away."""
    cols = ['season', 'team_id'] + [f for f in features if f in stats_df.columns]
    s = stats_df.reset_index(drop=True)[cols]
    away = games.merge(s, left_on=['away_team_id', 'season'],
                       right_on=['team_id', 'season'])
    both = away.merge(s, left_on=['home_team_id', 'season'],
                      right_on=['team_id', 'season'], suffixes=('_away', '_home'))

    ordered = []
    for f in features:
        ordered += [f + '_home', f + '_away']
    if any(c not in both.columns for c in ordered):
        return None, None

    carry = [c for c in ('id', 'week', 'week_num', 'date', 'short_name', 'season',
                         'home_team_id', 'away_team_id', 'home_score_differential')
             if c in both.columns]
    return both[carry + ordered].dropna(), ordered


def walk_forward_season(season, pre_dir, ins_dir, fbs_only=True, verbose=True):
    """Preseason + in-season predictions for one season, refit week by week.

    Returns a DataFrame with one row per predicted game.
    """
    pre_feats, pre_model, pre_scaler = load_model(pre_dir, 'preseason')
    ins_feats, ins_model, ins_scaler = load_model(ins_dir, 'in_season')

    games = load_games(season, fbs_only=fbs_only)
    if games.empty:
        return pd.DataFrame()

    gbg = pd.read_csv(f'{RESULTS}/game_by_game_summaries.csv', low_memory=False)
    wk = games[['id', 'week_num']].rename(columns={'id': 'game_id'})
    gbg = gbg[gbg.season == season].merge(wk, on='game_id', how='inner')

    # Preseason predictions use only prior seasons, so they need no walk-forward.
    ss = pd.read_csv(f'{RESULTS}/season_summaries.csv')
    ss_edit = P.edit_files(season_summary_df=ss, features=pre_feats,
                           start_year=(season - 1) - 3, end_year=season - 1)
    pre_matrix, pre_meta = P.merge_games_and_stats(games, ss_edit)
    pre_meta = pre_meta.copy()
    pre_meta['preseason_model_preds'] = pre_model.predict(pre_scaler.transform(pre_matrix))
    pre_lookup = pre_meta[['id', 'preseason_model_preds']]

    adjuster = OpponentAdjuster(AnalyticsConfig())
    out = []
    for W in sorted(w for w in gbg.week_num.unique() if w < POSTSEASON_WEEK):
        prior = gbg[gbg.week_num < W]
        target = games[games.week_num == W]
        if prior.empty or target.empty:
            continue
        try:
            adj = adjuster.adjust_for_opponents(prior, STATS)
            if adj is None or adj.empty:
                raise ValueError('adjuster produced no rows')
        except Exception as exc:
            if verbose:
                print(f"  week {int(W):>2}: no adjustment yet ({str(exc)[:50]})")
            continue

        feats, order = _merge_in_season(target, adj, ins_feats)
        if feats is None or feats.empty:
            continue
        feats = feats.copy()
        feats['in_season_model_preds'] = ins_model.predict(ins_scaler.transform(feats[order]))
        keep = [c for c in ('id', 'week', 'week_num', 'date', 'short_name', 'season',
                            'home_team_id', 'away_team_id', 'home_score_differential',
                            'in_season_model_preds') if c in feats.columns]
        merged = feats[keep].merge(pre_lookup, on='id', how='inner')
        out.append(merged)
        if verbose:
            print(f"  week {int(W):>2}: {prior.game_id.nunique():>4} prior games "
                  f"-> {len(merged):>3} predicted")

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


ANALYSIS_DIR = os.path.join(_REPO, 'analysis')
EXPANDING_CACHE = os.path.join(ANALYSIS_DIR, 'backtest_expanding_preds.csv')
PYTHON = sys.executable


def read_experiment_field(directory, token, field):
    """Pull a single `field: value` line out of a model's experiment file."""
    matches = [f for f in glob.glob(directory + '/*') if token in f.split(directory)[1]]
    if not matches:
        return None
    for line in open(matches[0]):
        if line.startswith(field + ':'):
            return line.split(': ', 1)[1].rstrip('\n')
    return None


def _train_for_season(model_subdir, prefix, features, train_end, train_start=2010,
                      model_params=None):
    """Retrain one model on train_start..train_end. Returns the new model dir.

    Deliberately bypasses preprocess.sh: that pulls from S3, whereas here we
    must cap the games file at train_end so the season being predicted cannot
    reach the training set.

    model_params is carried through verbatim so the retrained models match the
    production estimator configuration. Omitting it would silently retrain at
    library defaults and understate a tuned model's performance.
    """
    d = os.path.join(_REPO, 'model_training', model_subdir)
    subprocess.run(['rm', '-rf', os.path.join(d, 'temp')], check=True)
    os.makedirs(os.path.join(d, 'temp'), exist_ok=True)

    stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    with open(os.path.join(d, 'temp', f'{prefix}_experiment_{stamp}.txt'), 'w') as fh:
        fh.write(f"{stamp}\n")
        fh.write(f"train_season_range: {train_start} - {train_end}\n")
        # a fraction keeps every season in training, unlike holding one out
        fh.write("test_size: 0.15\n")
        fh.write(f"data_features: {','.join(features)}\n")
        fh.write("fbs_only_ind: True\n")
        fh.write("training_algorithm: xgboost_reg\n")
        if model_params:
            fh.write(f"model_params: {model_params}\n")

    subprocess.run(['cp', os.path.join(RESULTS, 'season_summaries.csv'),
                    os.path.join(d, 'temp', 'season_summaries.csv')], check=True)

    games = pd.read_csv(GAMES_CSV, low_memory=False)
    teams = pd.read_csv(TEAMS_CSV)
    teams = teams.loc[teams['fbs_ind'] == 1.0][['id', 'fbs_ind']].rename(
        columns={'id': 'team_id'})
    m = games.merge(teams, left_on='home_team_id', right_on='team_id')
    m = m.merge(teams, left_on='away_team_id', right_on='team_id')
    m[m['season'] <= train_end].to_csv(os.path.join(d, 'temp', 'games.csv'))

    before = set(os.listdir(d))
    for script in ('preprocess.py', 'train_model.py'):
        r = subprocess.run([PYTHON, script], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{model_subdir}/{script} failed:\n"
                               f"{r.stdout[-600:]}\n{r.stderr[-600:]}")
    new = sorted(set(os.listdir(d)) - before)
    if not new:
        raise RuntimeError(f"{model_subdir}: training produced no model directory")
    return os.path.join(d, new[-1])


def generate_expanding_predictions(seasons, reference_model_dir=None,
                                   cache_path=EXPANDING_CACHE, force=False,
                                   train_start=2010, verbose=True):
    """Walk-forward predictions for `seasons`, each from models trained before it.

    Expensive (a retrain per season), so results are cached. Pass force=True to
    rebuild. `reference_model_dir` supplies the feature list, so the retrained
    models stay in step with whatever your production experiment uses.
    """
    seasons = sorted(seasons)

    if os.path.exists(cache_path) and not force:
        cached = pd.read_csv(cache_path)
        if set(seasons).issubset(set(cached['test_season'].unique())):
            if verbose:
                print(f"  using cached expanding predictions: {cache_path}")
                print(f"  ({len(cached)} games, seasons "
                      f"{int(cached.test_season.min())}-{int(cached.test_season.max())}; "
                      f"pass force=True to rebuild)")
            return cached[cached['test_season'].isin(seasons)].copy()
        if verbose:
            missing = sorted(set(seasons) - set(cached['test_season'].unique()))
            print(f"  cache missing seasons {missing}; regenerating")

    ref = reference_model_dir or _newest_model_dir('in_season_model')
    features, _, _ = load_model(ref, 'in_season')

    # Mirror the production estimator config for each model, so the retrained
    # per-season models are comparable to what actually ships.
    ins_params = read_experiment_field(ref, 'in_season', 'model_params')
    pre_ref = _newest_model_dir('preseason_model')
    pre_params = read_experiment_field(pre_ref, 'preseason', 'model_params')
    if verbose:
        print(f"  retraining per season (features from {os.path.basename(ref)})")
        print(f"    preseason params: {pre_params or 'library defaults'}")
        print(f"    in-season params: {ins_params or 'library defaults'}")

    created, frames = [], []
    for season in seasons:
        try:
            pre_dir = _train_for_season('preseason_model', 'preseason', features, season - 1,
                                        train_start, model_params=pre_params)
            ins_dir = _train_for_season('in_season_model', 'in_season', features, season - 1,
                                        train_start, model_params=ins_params)
            created += [pre_dir, ins_dir]
            df = walk_forward_season(season, pre_dir, ins_dir, verbose=False)
            if df.empty:
                if verbose:
                    print(f"    {season}: no predictions")
                continue
            df['test_season'] = season
            frames.append(df)
            if verbose:
                print(f"    {season}: {len(df)} games "
                      f"(trained {train_start}-{season - 1})", flush=True)
        except Exception as exc:
            print(f"    {season}: FAILED - {str(exc)[:180]}")

    for d in created:                       # throwaway models, not production
        subprocess.run(['rm', '-rf', d], check=False)

    if not frames:
        raise RuntimeError("expanding generation produced no predictions")

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    out.to_csv(cache_path, index=False)
    if verbose:
        print(f"  cached -> {cache_path}")
    return out


def _newest_model_dir(kind):
    dirs = sorted(glob.glob(os.path.join(_REPO, 'model_training', kind, 'xgboost_reg_*')))
    if not dirs:
        raise SystemExit(f"no trained models in model_training/{kind}")
    return dirs[-1]


def add_walk_forward_blend(df, min_history=30):
    """Blend column whose coefficients are fit on earlier weeks only."""
    df = df.copy()
    df['blended_model'] = np.nan
    for W in sorted(df.week_num.unique()):
        hist = df[df.week_num < W]
        cur = df.week_num == W
        if len(hist) < min_history:
            df.loc[cur, 'blended_model'] = df.loc[cur, 'preseason_model_preds']
        else:
            lr = LinearRegression().fit(
                hist[['preseason_model_preds', 'in_season_model_preds']],
                hist['home_score_differential'])
            df.loc[cur, 'blended_model'] = lr.predict(
                df.loc[cur, ['preseason_model_preds', 'in_season_model_preds']])
    return df


def report(df):
    def m(sub, col):
        e = sub[col] - sub['home_score_differential']
        return abs(e).mean(), np.sqrt((e ** 2).mean())

    cols = ['preseason_model_preds', 'in_season_model_preds', 'blended_model']
    print("\n" + "=" * 74)
    print(f"{'week':>5} {'n':>5} | {'preseason':>18} | {'in-season':>18} | {'blended':>18}")
    print(f"{'':>5} {'':>5} | {'MAE':>8}{'RMSE':>10} | {'MAE':>8}{'RMSE':>10} | {'MAE':>8}{'RMSE':>10}")
    print("-" * 74)
    for W in sorted(df.week_num.unique()):
        s = df[df.week_num == W]
        vals = [m(s, c) for c in cols]
        print(f"{int(W):>5} {len(s):>5} | " +
              " | ".join(f"{a:>8.2f}{b:>10.2f}" for a, b in vals))
    print("-" * 74)
    vals = [m(df, c) for c in cols]
    print(f"{'ALL':>5} {len(df):>5} | " + " | ".join(f"{a:>8.2f}{b:>10.2f}" for a, b in vals))
    print("=" * 74)
    print(f"\nbaseline (always predict 0): MAE {abs(df['home_score_differential']).mean():.2f}")
    for c in cols:
        acc = (np.sign(df[c]) == np.sign(df['home_score_differential'])).mean() * 100
        print(f"  {c:<26} correct side {acc:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--pre-dir', default=None, help='preseason experiment directory')
    ap.add_argument('--ins-dir', default=None, help='in-season experiment directory')
    ap.add_argument('--all-teams', action='store_true',
                    help='include non-FBS games (training is FBS-only)')
    ap.add_argument('--out', default=None, help='write per-game results here')
    args = ap.parse_args()

    def newest(kind):
        dirs = sorted(glob.glob(os.path.join(_REPO, 'model_training', kind, 'xgboost_reg_*')))
        if not dirs:
            raise SystemExit(f"no trained models in model_training/{kind}")
        return dirs[-1]

    pre_dir = args.pre_dir or newest('preseason_model')
    ins_dir = args.ins_dir or newest('in_season_model')
    print(f"preseason model: {os.path.basename(pre_dir)}")
    print(f"in-season model: {os.path.basename(ins_dir)}\n")

    df = walk_forward_season(args.season, pre_dir, ins_dir, fbs_only=not args.all_teams)
    if df.empty:
        raise SystemExit(f"no predictions produced for {args.season}")
    df = add_walk_forward_blend(df)
    report(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nper-game results -> {args.out}")


if __name__ == '__main__':
    main()
