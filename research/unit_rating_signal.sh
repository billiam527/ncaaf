#!/usr/bin/env bash
# pf_db flipped sign in 33% of games and had the weakest margin relationship of
# the eight unit ratings. Both were measured under the old 104-column per-team
# encoding, where a pair's net was the sum of two separately-attributed terms and
# the diagonal-staircase artefact was in play. There is one column per comparison
# now, so re-measure before chasing anything.
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf
python - <<'PY'
import numpy as np
import pandas as pd

R = '/home/bill/ncaaf'
d = pd.read_csv(f'{R}/batch_prediction/prediction_file/prediction_formula.csv')
d = d[d['pre_raw'].notna()].copy()

UNITS = ['pf_qb', 'pf_rb', 'pf_wr', 'pf_te', 'pf_ol', 'pf_f7', 'pf_db', 'pf_st']
print("=== under the differential encoding, on the 2026 slate ===")
print(f"  {'rating':<10}{'sign flips':>12}{'corr(diff,effect)':>20}"
      f"{'mean |effect|':>15}")
for u in UNITS:
    v, c = d[f'pre__{u}_diff'], d[f'pre__{u}_diff__contrib']
    ok = v.notna() & c.notna()
    flip = ((v[ok] > 0) != (c[ok] > 0)).mean()
    print(f"  {u:<10}{flip:>11.0%}{v[ok].corr(c[ok]):>20.3f}"
          f"{c[ok].abs().mean():>15.3f}")

print("\n  Under the old encoding pf_db flipped 33% of the time and pf_st 1%.")

print("\n=== and how much signal does each carry, against actual margin? ===")
g = pd.read_csv(f'{R}/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv(f'{R}/etl/collect/collect_espn_teams/temp/teams.csv')
p = pd.read_csv(f'{R}/etl/summarize/results/position_ratings.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
g = g.dropna(subset=['home_score_differential'])
s = p[['team_id', 'season'] + UNITS].dropna()
m = g.merge(s.add_suffix('_h').rename(columns={'team_id_h': 'home_team_id',
                                               'season_h': 'season'}),
            on=['home_team_id', 'season'])
m = m.merge(s.add_suffix('_a').rename(columns={'team_id_a': 'away_team_id',
                                               'season_a': 'season'}),
            on=['away_team_id', 'season'])
y = m['home_score_differential'].to_numpy(float)
print(f"  {len(m):,} games\n")
print(f"  {'rating':<10}{'R2 alone':>11}{'slope':>10}{'se':>8}{'t':>8}")
rows = []
for u in UNITS:
    x = (m[f'{u}_h'] - m[f'{u}_a']).to_numpy(float)
    X = np.c_[np.ones(len(y)), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    s2 = (r ** 2).sum() / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    rows.append((r2, u, b[1], se))
for r2, u, slope, se in sorted(rows, reverse=True):
    print(f"  {u:<10}{r2:>11.4f}{slope:>10.2f}{se:>8.2f}{slope / se:>8.1f}")

print("\n=== all eight together: does pf_db add anything on top of the rest? ===")
X_all = np.c_[np.ones(len(y)), np.column_stack(
    [(m[f'{u}_h'] - m[f'{u}_a']).to_numpy(float) for u in UNITS])]
b, *_ = np.linalg.lstsq(X_all, y, rcond=None)
r = y - X_all @ b
s2 = (r ** 2).sum() / (len(y) - X_all.shape[1])
cov = s2 * np.linalg.inv(X_all.T @ X_all)
full = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"  R2 with all eight: {full:.4f}")
print(f"  {'rating':<10}{'coefficient':>13}{'se':>8}{'t':>8}")
for i, u in enumerate(UNITS, start=1):
    print(f"  {u:<10}{b[i]:>13.2f}{np.sqrt(cov[i, i]):>8.2f}"
          f"{b[i] / np.sqrt(cov[i, i]):>8.1f}")
PY
