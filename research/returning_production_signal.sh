#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np

f = pd.read_csv('etl/summarize/results/returning_production.csv', low_memory=False)
print("=== does my rebuild match CFBD's published number, every year? ===")
c = f.dropna(subset=['ret_overall','cfbd_percentPPA'])
for yr, d in c.groupby('season'):
    print(f"  {int(yr)}: n={len(d):>3}  r={d['ret_overall'].corr(d['cfbd_percentPPA']):+.3f}"
          f"  mean mine {d['ret_overall'].mean():.3f} vs theirs {d['cfbd_percentPPA'].mean():.3f}")

# team-season performance: average scoring margin
g = pd.read_csv('etl/summarize/temp/games.csv', low_memory=False)
g = g.dropna(subset=['home_score_differential'])
t = pd.read_csv('etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]

home = g[['season','home_team_id','home_score_differential']].copy()
home.columns = ['season','team_id','margin']
away = g[['season','away_team_id','home_score_differential']].copy()
away.columns = ['season','team_id','margin']; away['margin'] *= -1
perf = pd.concat([home,away]).groupby(['season','team_id'])['margin'].mean().reset_index()
perf.columns = ['season','team_id','avg_margin']

d = f.merge(perf, on=['season','team_id'], how='inner')
prior = perf.copy(); prior['season'] += 1
prior = prior.rename(columns={'avg_margin':'prior_margin'})
d = d.merge(prior, on=['season','team_id'], how='left').dropna(subset=['prior_margin'])
d['delta'] = d['avg_margin'] - d['prior_margin']
print(f"\n=== signal check on {len(d)} FBS team-seasons ({int(d.season.min())}-{int(d.season.max())}) ===")
print("  correlation with CHANGE in average margin from last season")
print("  (the model already knows last season, so change is what is left to explain)")
print(f"\n  {'feature':<22}{'n':>6}{'r with delta':>14}{'r with margin':>15}")
print("  " + "-"*56)
cols = ([c for c in d.columns if c.startswith('cfbd_')] +
        sorted(c for c in d.columns if c.startswith('ret_')) +
        ['portal_in','portal_out','draft_departures'])
rows = []
for c in cols:
    s = d[[c,'delta','avg_margin']].dropna()
    if len(s) < 200: continue
    rows.append((c, len(s), s[c].corr(s['delta']), s[c].corr(s['avg_margin'])))
for c, n, rd, rm in sorted(rows, key=lambda x: -abs(x[2])):
    print(f"  {c:<22}{n:>6}{rd:>+14.3f}{rm:>+15.3f}")

print("\n=== how much of delta can they explain together? ===")
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
for label, use in (('CFBD team-level only', [c for c in d.columns if c.startswith('cfbd_')]),
                   ('QB only', ['ret_QB_starter','ret_QB']),
                   ('all position groups', [c for c in d.columns if c.startswith('ret_')]),
                   ('all + churn', [c for c in d.columns if c.startswith('ret_')]
                                    + ['portal_in','portal_out','draft_departures'])):
    s = d[use + ['delta']].dropna()
    if len(s) < 200: continue
    X, y = s[use], s['delta']
    r2 = cross_val_score(Ridge(alpha=10.0), X, y, cv=5, scoring='r2').mean()
    print(f"  {label:<24} n={len(s):>4}  {len(use):>2} features  CV R2 {r2:>+.4f}")
PY
