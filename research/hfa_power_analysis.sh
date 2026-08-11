#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

# Cross-check on the FULL game set (12k games, double the walk-forward sample).
# No model needed: diff-in-diff on raw scoring margin. Confounded by schedule,
# but over 16 seasons home/away schedules roughly balance, and the extra sample
# gives more power to detect a real venue effect.
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
g = g.dropna(subset=['home_score_differential'])
if 'neutral_site' in g.columns:
    g = g[g.neutral_site != 1]

home = g[['home_team_id','home_score_differential','season']].copy()
home.columns = ['team','perf','season']; home['at_home'] = True
away = g[['away_team_id','home_score_differential','season']].copy()
away.columns = ['team','perf','season']; away['perf'] *= -1; away['at_home'] = False
tf = pd.concat([home, away], ignore_index=True)

def hfa(frame, min_n=15):
    x = frame.groupby(['team','at_home'])['perf'].agg(['mean','count']).unstack().dropna()
    x.columns = ['away_mean','home_mean','away_n','home_n']
    x = x[(x.home_n >= min_n) & (x.away_n >= min_n)]
    x['hfa'] = x['home_mean'] - x['away_mean']
    return x

full = hfa(tf)
sigma = g['home_score_differential'].std()
noise = (sigma**2) * (1/full['home_n'] + 1/full['away_n'])
tau2 = full['hfa'].var() - noise.mean()

print(f"{len(g)} non-neutral FBS games, {len(full)} teams with 15+ home and away")
print(f"  league mean HFA:        {full['hfa'].mean():+.2f} points")
print(f"  raw spread across teams: sd {full['hfa'].std():.2f}")
print(f"  expected from noise:     sd {np.sqrt(noise.mean()):.2f}")
print(f"  implied TRUE spread:     sd {np.sqrt(max(tau2,0)):.2f}")

mid = tf.season.median()
a = hfa(tf[tf.season <= mid], 10)['hfa']
b = hfa(tf[tf.season > mid], 10)['hfa']
both = pd.concat([a,b], axis=1, keys=['early','late']).dropna()
print(f"\n  split-half: {len(both)} teams, correlation {both['early'].corr(both['late']):+.3f}")

print("\n=== what size effect COULD we detect? ===")
n = full['home_n'].median()
per_team_noise = sigma * np.sqrt(2/n)
print(f"  median games per side: {n:.0f}")
print(f"  noise on one team's HFA estimate: +/- {per_team_noise:.2f} points")
for tau in (1, 2, 3, 4):
    rel = tau**2 / (tau**2 + per_team_noise**2)
    print(f"    if true spread were {tau} pts -> reliability {rel:.2f}"
          f"{'  (detectable)' if rel > 0.3 else '  (invisible)'}")
PY
