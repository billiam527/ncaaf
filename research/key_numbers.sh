#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd, numpy as np

g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
m = g['home_score_differential'].dropna().astype(int)

print(f"{len(m)} FBS games, seasons {int(g.season.min())}-{int(g.season.max())}\n")
print("=== how often each ABSOLUTE margin occurs ===")
a = m.abs()
vc = a.value_counts().sort_index()
pct = vc / len(a) * 100
print(f"{'margin':>7}{'games':>8}{'pct':>8}   {'':<30}")
for k in range(0, 29):
    if k not in vc:
        continue
    bar = '#' * int(round(pct[k] * 6))
    key = ' <- key' if k in (3, 7, 10, 14, 17, 21) else ''
    print(f"{k:>7}{vc[k]:>8}{pct[k]:>7.2f}%  {bar}{key}")

print("\n=== key numbers vs their neighbours ===")
for k in (3, 7, 10, 14, 17, 21):
    lo, hi = pct.get(k-1, 0), pct.get(k+1, 0)
    nb = (lo + hi) / 2
    print(f"  margin {k:>2}: {pct.get(k,0):5.2f}%   neighbours avg {nb:4.2f}%   "
          f"ratio {pct.get(k,0)/nb if nb else float('nan'):.2f}x")

top = pct.sort_values(ascending=False).head(8)
print(f"\n  most common margins: {[(int(i), round(v,2)) for i, v in top.items()]}")
print(f"  share of games landing on 3/7/10/14/17/21: "
      f"{sum(pct.get(k,0) for k in (3,7,10,14,17,21)):.1f}%")
PY
