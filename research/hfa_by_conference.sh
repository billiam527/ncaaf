#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

# Conference labels come from the CFBD spread file (2021+); map team -> conference
c = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_cfbd_games/cfbd_spread_data.csv',
                low_memory=False)
conf = pd.concat([
    c[['home_team','home_conference']].rename(columns={'home_team':'team','home_conference':'conf'}),
    c[['away_team','away_conference']].rename(columns={'away_team':'team','away_conference':'conf'}),
]).dropna().drop_duplicates('team')
name2conf = dict(zip(conf['team'], conf['conf']))

t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
id2name = dict(zip(t['id'], t['location']))

g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
g = g.dropna(subset=['home_score_differential'])
if 'neutral_site' in g.columns:
    g = g[g.neutral_site != 1]

home = g[['home_team_id','home_score_differential']].copy()
home.columns = ['team','perf']; home['at_home'] = True
away = g[['away_team_id','home_score_differential']].copy()
away.columns = ['team','perf']; away['perf'] *= -1; away['at_home'] = False
tf = pd.concat([home, away], ignore_index=True)
tf['conf'] = tf['team'].map(lambda i: name2conf.get(id2name.get(i)))
tf = tf.dropna(subset=['conf'])

sigma = g['home_score_differential'].std()

def hfa(frame, key, min_n):
    x = frame.groupby([key,'at_home'])['perf'].agg(['mean','count']).unstack().dropna()
    x.columns = ['away_mean','home_mean','away_n','home_n']
    x = x[(x.home_n >= min_n) & (x.away_n >= min_n)]
    x['hfa'] = (x['home_mean'] - x['away_mean']) / 2      # same units as the site
    x['se'] = (sigma/2) * np.sqrt(1/x['home_n'] + 1/x['away_n'])
    return x

print("=== CONFERENCE-level home field advantage (same estimator, pooled) ===")
cf = hfa(tf, 'conf', 200).sort_values('hfa', ascending=False)
print(f"{'conference':<18}{'HFA':>8}{'+/-':>7}{'home n':>9}{'road n':>8}")
print("-" * 52)
for k, r in cf.iterrows():
    print(f"{str(k)[:17]:<18}{r['hfa']:>8.2f}{r['se']:>7.2f}{int(r['home_n']):>9}{int(r['away_n']):>8}")

noise = (cf['se']**2).mean()
tau2 = cf['hfa'].var() - noise
print("-" * 52)
print(f"  league mean:            {cf['hfa'].mean():.2f}")
print(f"  spread across confs:    sd {cf['hfa'].std():.2f}")
print(f"  expected from noise:    sd {np.sqrt(noise):.2f}")
print(f"  implied TRUE spread:    sd {np.sqrt(max(tau2,0)):.2f}")
print(f"  -> {'REAL conference differences' if tau2 > 0.05 else 'still indistinguishable from noise'}")

print("\n=== split-half: does a conference's edge persist? ===")
g2 = g.copy()
mid = g2.season.median()
tf['season'] = pd.concat([g[['season']], g[['season']]], ignore_index=True)['season'].values
a = hfa(tf[tf.season <= mid], 'conf', 80)['hfa']
b = hfa(tf[tf.season > mid], 'conf', 80)['hfa']
both = pd.concat([a,b], axis=1, keys=['early','late']).dropna()
print(f"  {len(both)} conferences in both halves, correlation "
      f"{both['early'].corr(both['late']):+.3f}")
print(both.round(2).to_string())
PY
