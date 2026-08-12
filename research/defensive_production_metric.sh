#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
T='etl/collect/collect_cfbd_players/temp/'

stats  = pd.read_csv(T+'cfbd_stats.csv', low_memory=False)
roster = pd.read_csv(T+'cfbd_roster.csv', low_memory=False)
cf     = pd.read_csv(T+'cfbd_teams.csv', low_memory=False)
roster['id'] = roster['id'].astype(str)
stats['playerId'] = stats['playerId'].astype(str)

d = stats[stats.category=='defensive'].copy()
print("=== what defensive stats do we actually have? ===")
piv = d.pivot_table(index=['season','team','playerId','position'],
                    columns='statType', values='stat',
                    aggfunc='first').reset_index()
for c in piv.columns:
    if c not in ('season','team','playerId','position'):
        piv[c] = pd.to_numeric(piv[c], errors='coerce')
statcols = [c for c in piv.columns if c not in ('season','team','playerId','position')]
print(f"  statTypes: {statcols}")
print(f"  player-seasons: {len(piv)}")
for c in statcols:
    print(f"    {c:<8} non-null {piv[c].notna().mean():>6.1%}  "
          f"mean {piv[c].mean():>7.2f}  max {piv[c].max():>7.0f}")

GRP = {'DL':'DL','DE':'DL','DT':'DL','NT':'DL','EDGE':'DL',
       'LB':'LB','ILB':'LB','OLB':'LB','MLB':'LB',
       'DB':'DB','CB':'DB','S':'DB','FS':'DB','SS':'DB'}
piv['group'] = piv['position'].map(GRP)
piv = piv.dropna(subset=['group'])

# candidate production metrics
piv['havoc'] = (piv.get('TFL',0).fillna(0) + piv.get('SACKS',0).fillna(0)
                + piv.get('PD',0).fillna(0))
piv['pressure'] = (piv.get('SACKS',0).fillna(0) + piv.get('TFL',0).fillna(0)
                   + piv.get('QB HUR',0).fillna(0))
piv['tot_havoc'] = piv['TOT'].fillna(0) + 3.0*piv['havoc']

METRICS = ['TOT','SOLO','TFL','SACKS','PD','havoc','pressure','tot_havoc']
SLOTS = {'DL':4,'LB':3,'DB':4}

# outcome
g = pd.read_csv('etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind']==1.0,'id'])
g = g.dropna(subset=['home_score_differential'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
h = g[['season','home_team_id','home_score_differential']].copy(); h.columns=['season','team_id','m']
a = g[['season','away_team_id','home_score_differential']].copy(); a.columns=['season','team_id','m']; a['m']*=-1
perf = pd.concat([h,a]).groupby(['season','team_id'])['m'].mean().reset_index()
perf.columns=['season','team_id','avg_margin']
name2id = dict(zip(cf['school'], cf['id']))

def build(metric):
    p = piv[['season','team','playerId','group',metric]].copy()
    p = p.rename(columns={metric:'w'})
    p['w'] = pd.to_numeric(p['w'], errors='coerce')
    p = p[p['w'] > 0]
    p['rank'] = p.groupby(['season','team','group'])['w'].rank(method='first', ascending=False)
    p['starter'] = p['rank'] <= p['group'].map(SLOTS)
    rows=[]
    for season in range(2017, 2026):
        prior = p[p.season == season-1]
        if prior.empty: continue
        back = set(roster.loc[roster.season==season,'team'] + '|' +
                   roster.loc[roster.season==season,'id'])
        pr = prior.copy()
        pr['ret'] = (pr['team'] + '|' + pr['playerId']).isin(back)
        for (team, grp), gd in pr.groupby(['team','group']):
            st = gd[gd.starter]
            if st['w'].sum() <= 0: continue
            rows.append({'season':season,'team':team,'group':grp,
                         'val': st.loc[st.ret,'w'].sum()/st['w'].sum()})
    fr = pd.DataFrame(rows)
    if fr.empty: return None
    wide = fr.pivot_table(index=['season','team'], columns='group',
                          values='val').reset_index()
    wide['team_id'] = wide['team'].map(name2id)
    return wide

base = perf.copy()
pr = perf.copy(); pr['season'] += 1; pr = pr.rename(columns={'avg_margin':'prior'})
base = base.merge(pr, on=['season','team_id'], how='left').dropna(subset=['prior'])
lr = LinearRegression().fit(base[['prior']], base['avg_margin'])
base['resid'] = base['avg_margin'] - lr.predict(base[['prior']])

print("\n=== correlation with residualised margin, by production metric ===")
print(f"  {'metric':<12}{'DL':>9}{'LB':>9}{'DB':>9}{'mean':>9}{'n':>7}")
print("  " + "-"*56)
best=None
for m in METRICS:
    w = build(m)
    if w is None: continue
    j = base.merge(w, on=['season','team_id'], how='inner')
    rs=[]
    line=f"  {m:<12}"
    for grp in ('DL','LB','DB'):
        if grp in j.columns:
            s = j[[grp,'resid']].dropna()
            r = s[grp].corr(s['resid']) if len(s)>200 else np.nan
            rs.append(r); line += f"{r:>+9.3f}"
        else:
            line += f"{'-':>9}"
    mean = np.nanmean(rs)
    line += f"{mean:>+9.3f}{len(j):>7}"
    print(line)
    if best is None or mean > best[1]: best=(m, mean)
print(f"\n  best metric: {best[0]} (mean r {best[1]:+.3f})   "
      f"current build uses TOT")
PY
