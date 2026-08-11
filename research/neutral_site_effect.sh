#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
g = g.dropna(subset=['home_score_differential'])

print("=== neutral_site column ===")
print(f"  values: {g['neutral_site'].value_counts(dropna=False).to_dict()}")

home = g[g.neutral_site == 0]['home_score_differential']
neut = g[g.neutral_site == 1]['home_score_differential']

def ci(x, label):
    se = x.std() / np.sqrt(len(x))
    print(f"  {label:<22} n={len(x):>5}  mean {x.mean():+6.2f}  "
          f"95% CI [{x.mean()-1.96*se:+.2f}, {x.mean()+1.96*se:+.2f}]")

print("\n=== raw home advantage in the data (all 2010-2025 FBS) ===")
ci(home, 'true home games')
ci(neut, 'neutral site games')
diff = home.mean() - neut.mean()
se = np.sqrt(home.var()/len(home) + neut.var()/len(neut))
print(f"\n  implied home-field advantage: {diff:+.2f} points")
print(f"  95% CI [{diff-1.96*se:+.2f}, {diff+1.96*se:+.2f}]   "
      f"{'SIGNIFICANT' if abs(diff) > 1.96*se else 'not significant'}")

print("\n=== but is it confounded? neutral games are often better matchups ===")
print("  (kickoff classics and bowls pair stronger, closer teams)")
for lab, s in (('true home', g[g.neutral_site == 0]), ('neutral', g[g.neutral_site == 1])):
    print(f"  {lab:<12} mean |margin| {s['home_score_differential'].abs().mean():5.2f}  "
          f"sd {s['home_score_differential'].std():5.2f}")

print("\n=== the clean test: model residuals on neutral games ===")
h = pd.read_csv('/home/bill/ncaaf/analysis/backtest_expanding_preds.csv')
h = h[h.week_num < 90].dropna(subset=['in_season_model_preds','home_score_differential'])
h = h.merge(g[['id','neutral_site','week']].drop_duplicates('id'), on='id', how='left')
h['resid'] = h['home_score_differential'] - h['in_season_model_preds']
for v, lab in [(0,'true home'), (1,'neutral')]:
    s = h[h.neutral_site == v]['resid']
    if len(s) > 20:
        ci(s, lab)
n = h[h.neutral_site == 1]['resid']
print(f"\n  neutral-site bias is {'REAL' if abs(n.mean()) > 1.96*n.std()/np.sqrt(len(n)) else 'NOT distinguishable from noise'}")

print("\n=== where do neutral games occur? ===")
sub = h[h.neutral_site == 1]
print(f"  by week: {sub['week_num'].value_counts().sort_index().head(8).to_dict()}")
print(f"  season 1 games (kickoff classics): {(sub.week_num <= 2).sum()}")
PY
