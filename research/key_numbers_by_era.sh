#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d

g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)].copy()
g['margin'] = g['home_score_differential'].round()
g = g.dropna(subset=['margin', 'season'])
g['margin'] = g['margin'].astype(int)

print("=== source of the key-number curve ===")
print(f"  {len(g)} FBS-vs-FBS college games, seasons "
      f"{int(g.season.min())}-{int(g.season.max())}")
print("  (built from your games.csv - college only, no NFL data anywhere)")

LO, HI = -70, 70
GRID = np.arange(LO, HI + 1)

def mult(m):
    c = np.array([(m == k).sum() for k in GRID], float)
    e = c / c.sum()
    s = gaussian_filter1d(e, sigma=2.5, mode='nearest')
    return np.clip(np.where(s > 1e-9, e / s, 1.0), .15, 4.), e

eras = [(2010, 2014), (2015, 2019), (2020, 2025)]
print("\n=== key-number multiplier by era ===")
print(f"{'margin':>7}" + "".join(f"{f'{a}-{b}':>12}" for a, b in eras) + f"{'trend':>10}")
print("-" * 56)
curves = {}
for a, b in eras:
    m = g[(g.season >= a) & (g.season <= b)]['margin'].abs()
    curves[(a, b)] = mult(m)[0]
for k in (1, 2, 3, 6, 7, 8, 9, 10, 11, 14, 16, 17, 21):
    i = k - LO
    vals = [curves[e][i] for e in eras]
    d = vals[-1] - vals[0]
    arrow = '  up' if d > 0.15 else ('  down' if d < -0.15 else '  flat')
    print(f"{k:>7}" + "".join(f"{v:>12.2f}" for v in vals) + f"{arrow:>10}")

print("\n=== share of games on each margin, by era (looking for 2-pt effects) ===")
print(f"{'margin':>7}" + "".join(f"{f'{a}-{b}':>12}" for a, b in eras))
for k in (1, 2, 3, 5, 7, 8, 10, 14):
    row = []
    for a, b in eras:
        m = g[(g.season >= a) & (g.season <= b)]['margin'].abs()
        row.append((m == k).mean() * 100)
    print(f"{k:>7}" + "".join(f"{v:>11.2f}%" for v in row))

print("\n=== overall shape drift ===")
for a, b in eras:
    s = g[(g.season >= a) & (g.season <= b)]
    m = s['margin']
    key = m.abs().isin([3, 7, 10, 14, 17, 21]).mean()
    print(f"  {a}-{b}: n={len(s):>5}  sd={m.std():5.2f}  "
          f"mean|margin|={m.abs().mean():5.2f}  on key {key:.1%}")

print("\n=== per-season, the numbers most affected by 2-pt / OT rules ===")
print(f"{'season':>7}{'n':>6}{'|m|=1':>8}{'|m|=2':>8}{'|m|=8':>8}{'|m|=3':>8}{'sd':>7}")
for s, sub in g.groupby('season'):
    m = sub['margin'].abs()
    print(f"{int(s):>7}{len(sub):>6}{(m==1).mean()*100:>7.2f}%{(m==2).mean()*100:>7.2f}%"
          f"{(m==8).mean()*100:>7.2f}%{(m==3).mean()*100:>7.2f}%{sub['margin'].std():>7.2f}")
PY
