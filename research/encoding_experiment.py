"""How should the preseason model's features be constructed?

Five encodings, walk-forward: every test season is predicted by a model trained
only on earlier seasons, and every variant sees identically-built inputs so they
differ only in the transform applied.

Scored on MAE and on ATS against the closing line. Those are not the same
objective and this project has only ever optimised the first - a change that
costs a little accuracy but stops the model manufacturing confident
disagreements from stale rosters could still be the one worth having.

  1  current                104 columns, home and away separately, three flat lags
  2  differentials           52, home minus away
  3  + decayed lags          28, the three lags of each stat collapsed 4:2:1
  4  + run-game matchup      54, backfield against the front seven it faces
  5  + both                  30

Production hyperparameters throughout, NOT library defaults: defaults cost 1.265
MAE here, which would swamp the differences being measured.
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# This file applies its OWN encodings, so predict.py has to hand back the raw
# per-team matrix. Since the differential encoding was adopted,
# merge_games_and_stats applies it internally by default - which left this
# script transforming a frame that had already been transformed, finding no
# home/away pairs, and silently building an empty one. Set before the import:
# the flag is read at module load.
import os  # noqa: E402
os.environ['DIFFERENTIAL_ENCODING'] = '0'

sys.path.insert(0, '/home/bill/ncaaf/batch_prediction')
import predict as P  # noqa: E402

assert not P.DIFFERENTIAL_ENCODING, \
    'predict.py must yield raw per-team columns for this experiment'

R = '/home/bill/ncaaf'
RESULTS = f'{R}/etl/summarize/results'
GAMES = f'{R}/etl/summarize/temp/games.csv'
TEAMS = f'{R}/etl/collect/collect_espn_teams/temp/teams.csv'
LINES = f'{R}/etl/collect/collect_cfbd_games/cfbd_spread_data.csv'

FEATURES = ("adjusted_rush_success_off,adjusted_rush_success_def,"
            "adjusted_pass_success_off,adjusted_pass_success_def,"
            "adjusted_explosive_rush_rate_off,adjusted_explosive_rush_rate_def,"
            "adjusted_explosive_pass_rate_off,adjusted_explosive_pass_rate_def,"
            "adjusted_epa_per_rush_off,adjusted_epa_per_rush_def,"
            "adjusted_epa_per_pass_off,adjusted_epa_per_pass_def").split(',')

PARAMS = {"n_estimators": 1800, "max_depth": 5, "learning_rate": 0.0045,
          "min_child_weight": 12, "subsample": 0.51, "colsample_bytree": 0.49,
          "reg_lambda": 12.37, "reg_alpha": 0.5, "random_state": 0}

TRAIN_START = 2012
TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Recent seasons weigh more, by construction rather than by whatever the trees
# happen to pick. 4:2:1 halves each step back.
DECAY = np.array([4.0, 2.0, 1.0])
DECAY = DECAY / DECAY.sum()


def load_games():
    g = pd.read_csv(GAMES, low_memory=False)
    t = pd.read_csv(TEAMS)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
    return g[g['home_score_differential'].notna()]


def build_season(season, ss, games):
    ss_edit = P.edit_files(season_summary_df=ss, features=FEATURES,
                           start_year=(season - 1) - 3, end_year=season - 1)
    ss_edit = P.add_returning_production(ss_edit)
    g = games[games.season == season]
    if g.empty:
        return None
    X, meta = P.merge_games_and_stats(g, ss_edit)
    y = meta[['id']].merge(g[['id', 'home_score_differential']], on='id')
    return (X.reset_index(drop=True),
            y['home_score_differential'].reset_index(drop=True),
            meta['id'].reset_index(drop=True))


def pair_up(cols):
    """Map every column to its (base, lag, side) so pairs can be differenced."""
    out = {}
    for c in cols:
        m = re.match(r'^(.*?)_(FY(?:-\d)?)_(home|away)$', c)
        if m:
            out[c] = (m.group(1), m.group(2), m.group(3))
            continue
        m = re.match(r'^(.*?)_(home|away)$', c)
        if m:
            out[c] = (m.group(1), '', m.group(2))
    return out


def transform(X, kind, cols):
    """Build one encoding from the raw 104-column matrix."""
    info = pair_up(cols)
    bases = {}
    for c, (b, lag, side) in info.items():
        bases.setdefault((b, lag), {})[side] = c
    full = {k: v for k, v in bases.items() if len(v) == 2}

    if kind == 'current':
        return X[cols]

    out = {}
    for (b, lag), sides in sorted(full.items()):
        d = X[sides['home']].to_numpy(float) - X[sides['away']].to_numpy(float)
        out[f'{b}{"_" + lag if lag else ""}_diff'] = d

    if 'decay' in kind:
        merged, drop = {}, set()
        for b in sorted({b for b, lag in full if lag}):
            names = [f'{b}_{l}_diff' for l in ('FY', 'FY-1', 'FY-2')]
            if all(n in out for n in names):
                merged[f'{b}_decayed'] = sum(
                    w * out[n] for w, n in zip(DECAY, names))
                drop.update(names)
        for n in drop:
            out.pop(n)
        out.update(merged)

    if 'matchup' in kind:
        # A run game's value depends on the front seven it faces, not on whether
        # the opponent also runs well. The only cross-unit pairing that beat its
        # like-for-like twin: R2 0.136 against 0.095.
        for a, b, name in (('pf_rb_home', 'pf_f7_away', 'run_vs_front_home'),
                           ('pf_rb_away', 'pf_f7_home', 'run_vs_front_away')):
            if a in X.columns and b in X.columns:
                out[name] = X[a].to_numpy(float) - X[b].to_numpy(float)

    return pd.DataFrame(out, index=X.index)


VARIANTS = ['current', 'diff', 'diff+decay', 'diff+matchup',
            'diff+decay+matchup']


def main():
    ss = pd.read_csv(f'{RESULTS}/season_summaries.csv')
    games = load_games()

    cache = {}
    for s in range(TRAIN_START, max(TEST_SEASONS) + 1):
        b = build_season(s, ss, games)
        if b is not None:
            cache[s] = b
    cols = list(cache[max(cache)][0].columns)
    print(f"built {len(cache)} seasons, "
          f"{sum(len(v[0]) for v in cache.values()):,} games, {len(cols)} raw columns")
    for v in VARIANTS:
        print(f"  {v:<22}{transform(cache[max(cache)][0], v, cols).shape[1]:>4} columns")

    lines = pd.read_csv(LINES, low_memory=False)
    lines['spread'] = pd.to_numeric(lines['spread'], errors='coerce')
    lines = lines.dropna(subset=['spread', 'game_id'])
    mk = lines.groupby('game_id')['spread'].median()
    mk = (-mk).to_dict()          # to "points the home team wins by"

    rows = []
    for S in TEST_SEASONS:
        tr = [s for s in cache if s < S]
        if len(tr) < 4:
            continue
        ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
        Xte_raw, yte, ids = cache[S]
        market = ids.map(mk)

        for v in VARIANTS:
            Xtr = pd.concat([transform(cache[s][0], v, cols) for s in tr],
                            ignore_index=True)
            Xte = transform(Xte_raw, v, cols)
            sc = StandardScaler().fit(Xtr)
            m = XGBRegressor(**PARAMS).fit(sc.transform(Xtr), ytr)
            pred = m.predict(sc.transform(Xte))

            ok = market.notna().to_numpy()
            covered = np.where(yte[ok] > market[ok], 1.0,
                               np.where(yte[ok] < market[ok], 0.0, np.nan))
            edge = pred[ok] - market[ok].to_numpy(float)
            won = np.where(edge > 0, covered, 1 - covered)
            big = np.abs(edge) >= 6

            rows.append({
                'season': S, 'variant': v, 'k': Xtr.shape[1], 'n': len(yte),
                'mae': float(np.abs(pred - yte).mean()),
                'n_bets': int(np.isfinite(won).sum()),
                'ats': float(np.nanmean(won)),
                'n_big': int(np.isfinite(won[big]).sum()),
                'ats_big': float(np.nanmean(won[big])) if big.any() else np.nan,
                'mean_gap': float(np.abs(edge).mean()),
            })
        print(f"  {S} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f'{R}/analysis/encoding_experiment.csv', index=False)

    print("\n" + "=" * 84)
    print("POOLED, games-weighted")
    print("=" * 84)
    print(f"  {'encoding':<22}{'cols':>6}{'MAE':>9}{'ATS all':>10}{'ATS >=6':>10}"
          f"{'bets >=6':>10}{'mean gap':>10}")
    for v in VARIANTS:
        g = res[res.variant == v]
        n, nb, nbg = g['n'].sum(), g['n_bets'].sum(), g['n_big'].sum()
        print(f"  {v:<22}{g['k'].iloc[0]:>6}"
              f"{(g['mae'] * g['n']).sum() / n:>9.3f}"
              f"{(g['ats'] * g['n_bets']).sum() / nb:>10.1%}"
              f"{(g['ats_big'] * g['n_big']).sum() / nbg:>10.1%}"
              f"{nbg:>10}"
              f"{(g['mean_gap'] * g['n']).sum() / n:>10.2f}")
    print("\n  break-even at -110 is 52.4%")

    print("\n" + "=" * 84)
    print("MAE BY SEASON")
    print("=" * 84)
    print(res.pivot(index='season', columns='variant', values='mae')[VARIANTS]
          .round(3).to_string())
    print("\n  wins by season:",
          res.pivot(index='season', columns='variant',
                    values='mae')[VARIANTS].idxmin(axis=1).value_counts().to_dict())


if __name__ == '__main__':
    main()
