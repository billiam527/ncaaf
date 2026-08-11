"""Do the three prior-season lags each earn their place in the preseason model?

The model currently uses 12 adjusted features from each of the last three
seasons (_FY, _FY-1, _FY-2) for both teams = 72 columns. This trains variants
on different lag subsets and scores each on held-out seasons.

Everything is walk-forward: for test season S the model is fit on seasons
before S only, and the features for S come from seasons before S by
construction. Feature columns are built once by predict.edit_files, so every
variant sees identically-constructed inputs and differs only in which columns
it is allowed to use.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/bill/ncaaf/batch_prediction')
import predict as P  # noqa: E402

RESULTS = '/home/bill/ncaaf/etl/summarize/results'
GAMES = '/home/bill/ncaaf/etl/summarize/temp/games.csv'
TEAMS = '/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv'

FEATURES = ("adjusted_rush_success_off,adjusted_rush_success_def,adjusted_pass_success_off,"
            "adjusted_pass_success_def,adjusted_explosive_rush_rate_off,"
            "adjusted_explosive_rush_rate_def,adjusted_explosive_pass_rate_off,"
            "adjusted_explosive_pass_rate_def,adjusted_epa_per_rush_off,"
            "adjusted_epa_per_rush_def,adjusted_epa_per_pass_off,"
            "adjusted_epa_per_pass_def").split(',')

TRAIN_START = 2012
TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

VARIANTS = {
    'FY only':            ['_FY_'],
    'FY-1 only':          ['_FY-1_'],
    'FY-2 only':          ['_FY-2_'],
    'FY + FY-1':          ['_FY_', '_FY-1_'],
    'FY + FY-2':          ['_FY_', '_FY-2_'],
    'all three (current)': ['_FY_', '_FY-1_', '_FY-2_'],
}


def load_games():
    g = pd.read_csv(GAMES, low_memory=False)
    t = pd.read_csv(TEAMS)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
    return g[g['home_score_differential'].notna()]


def build_season(season, ss, games):
    """Feature matrix + target for one season, built exactly as predict.py does."""
    ss_edit = P.edit_files(season_summary_df=ss, features=FEATURES,
                           start_year=(season - 1) - 3, end_year=season - 1)
    g = games[games.season == season]
    if g.empty:
        return None, None
    X, meta = P.merge_games_and_stats(g, ss_edit)
    y = meta[['id']].merge(g[['id', 'home_score_differential']], on='id')
    return X.reset_index(drop=True), y['home_score_differential'].reset_index(drop=True)


def main():
    ss = pd.read_csv(f'{RESULTS}/season_summaries.csv')
    games = load_games()

    cache = {}
    for s in range(TRAIN_START, max(TEST_SEASONS) + 1):
        X, y = build_season(s, ss, games)
        if X is not None:
            cache[s] = (X, y)
    print(f"built {len(cache)} seasons, {sum(len(v[0]) for v in cache.values())} games, "
          f"{cache[max(cache)][0].shape[1]} columns\n")

    all_cols = list(cache[max(cache)][0].columns)
    rows = []
    for S in TEST_SEASONS:
        train_seasons = [s for s in cache if s < S]
        if len(train_seasons) < 4:
            continue
        Xtr = pd.concat([cache[s][0] for s in train_seasons], ignore_index=True)
        ytr = pd.concat([cache[s][1] for s in train_seasons], ignore_index=True)
        Xte, yte = cache[S]

        for name, tags in VARIANTS.items():
            cols = [c for c in all_cols if any(t in c for t in tags)]
            sc = StandardScaler().fit(Xtr[cols])
            m = XGBRegressor().fit(sc.transform(Xtr[cols]), ytr)
            pred = m.predict(sc.transform(Xte[cols]))
            rows.append({'season': S, 'variant': name, 'k': len(cols), 'n': len(yte),
                         'mae': np.abs(pred - yte).mean(),
                         'rmse': float(np.sqrt(((pred - yte) ** 2).mean())),
                         'side': (np.sign(pred) == np.sign(yte)).mean() * 100})
        # naive reference: always predict the training mean margin
        base = ytr.mean()
        rows.append({'season': S, 'variant': 'train mean (baseline)', 'k': 0, 'n': len(yte),
                     'mae': np.abs(base - yte).mean(),
                     'rmse': float(np.sqrt(((base - yte) ** 2).mean())),
                     'side': (np.sign(base) == np.sign(yte)).mean() * 100})
        print(f"  {S} done", flush=True)

    res = pd.DataFrame(rows)
    order = list(VARIANTS) + ['train mean (baseline)']

    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE MAE BY TEST SEASON")
    print("=" * 78)
    piv = res.pivot(index='season', columns='variant', values='mae')[order]
    print(piv.round(2).to_string())

    print("\n" + "=" * 78)
    print("POOLED (games-weighted)")
    print("=" * 78)
    print(f"{'variant':<22} {'cols':>5} {'MAE':>8} {'RMSE':>8} {'side%':>7}  {'vs FY only':>11}")
    ref = None
    for name in order:
        g = res[res.variant == name]
        mae = (g['mae'] * g['n']).sum() / g['n'].sum()
        rmse = float(np.sqrt((g['rmse'] ** 2 * g['n']).sum() / g['n'].sum()))
        side = (g['side'] * g['n']).sum() / g['n'].sum()
        if name == 'FY only':
            ref = mae
        delta = '' if ref is None else f"{ref - mae:>+11.3f}"
        print(f"{name:<22} {g['k'].iloc[0]:>5} {mae:>8.3f} {rmse:>8.3f} {side:>7.1f}  {delta}")

    print("\nwins by season (lowest MAE):")
    print(f"  {piv.idxmin(axis=1).value_counts().to_dict()}")

    # which lags does the full model actually lean on?
    print("\n" + "=" * 78)
    print("XGBOOST GAIN BY LAG (full model, fit on all training seasons)")
    print("=" * 78)
    Xtr = pd.concat([cache[s][0] for s in cache if s < max(TEST_SEASONS)], ignore_index=True)
    ytr = pd.concat([cache[s][1] for s in cache if s < max(TEST_SEASONS)], ignore_index=True)
    sc = StandardScaler().fit(Xtr)
    m = XGBRegressor().fit(pd.DataFrame(sc.transform(Xtr), columns=all_cols), ytr)
    imp = pd.Series(m.feature_importances_, index=all_cols)
    for tag, label in (('_FY_', 'FY   (last season)'),
                       ('_FY-1_', 'FY-1 (two back)'),
                       ('_FY-2_', 'FY-2 (three back)')):
        sub = imp[[c for c in all_cols if tag in c]]
        print(f"  {label:<22} total {sub.sum():.3f}   mean {sub.mean():.4f}   n={len(sub)}")

    res.to_csv('/home/bill/ncaaf/preseason_lag_ablation.csv', index=False)
    print("\nper-season results -> /home/bill/ncaaf/preseason_lag_ablation.csv")


if __name__ == '__main__':
    main()
