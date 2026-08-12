#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
T='etl/collect/collect_cfbd_players/temp/'

stats=pd.read_csv(T+'cfbd_stats.csv',low_memory=False)
roster=pd.read_csv(T+'cfbd_roster.csv',low_memory=False)
cf=pd.read_csv(T+'cfbd_teams.csv',low_memory=False)
roster['id']=roster['id'].astype(str); stats['playerId']=stats['playerId'].astype(str)
d=stats[(stats.category=='defensive')&(stats.statType=='TOT')].copy()
d['w']=pd.to_numeric(d['stat'],errors='coerce'); d=d.dropna(subset=['w']); d=d[d.w>0]

FRONT={'DL','DE','DT','NT','EDGE','LB','ILB','OLB','MLB'}
BACK={'DB','CB','S','FS','SS'}
d['half']=np.where(d.position.isin(FRONT),'front7',
          np.where(d.position.isin(BACK),'secondary',None))

g=pd.read_csv('etl/summarize/temp/games.csv',low_memory=False)
t=pd.read_csv('etl/collect/collect_espn_teams/temp/teams.csv')
fbs=set(t.loc[t['fbs_ind']==1.0,'id'])
g=g.dropna(subset=['home_score_differential'])
g=g[g.home_team_id.isin(fbs)&g.away_team_id.isin(fbs)]
h=g[['season','home_team_id','home_score_differential']].copy(); h.columns=['season','team_id','m']
a=g[['season','away_team_id','home_score_differential']].copy(); a.columns=['season','team_id','m']; a['m']*=-1
perf=pd.concat([h,a]).groupby(['season','team_id'])['m'].mean().reset_index()
perf.columns=['season','team_id','avg_margin']
name2id=dict(zip(cf['school'],cf['id']))
pr=perf.copy(); pr['season']+=1; pr=pr.rename(columns={'avg_margin':'prior'})
base=perf.merge(pr,on=['season','team_id'],how='left').dropna(subset=['prior'])
lr=LinearRegression().fit(base[['prior']],base['avg_margin'])
base['resid']=base['avg_margin']-lr.predict(base[['prior']])

def feat(frame, slots, label):
    s=frame.copy()
    s['rank']=s.groupby(['season','team'])['w'].rank(method='first',ascending=False)
    s=s[s['rank']<=slots]
    rows=[]
    for season in range(2017,2026):
        prior=s[s.season==season-1]
        if prior.empty: continue
        back=set(roster.loc[roster.season==season,'team']+'|'+
                 roster.loc[roster.season==season,'id'])
        p=prior.copy(); p['ret']=(p['team']+'|'+p['playerId']).isin(back)
        for team,gd in p.groupby('team'):
            if gd['w'].sum()<=0: continue
            rows.append({'season':season,'team':team,
                         label: gd.loc[gd.ret,'w'].sum()/gd['w'].sum()})
    fr=pd.DataFrame(rows); fr['team_id']=fr['team'].map(name2id)
    return fr[['season','team_id',label]]

pooled = feat(d.dropna(subset=['half']), 11, 'def_all')
f7     = feat(d[d.half=='front7'], 7, 'def_front7')
sec    = feat(d[d.half=='secondary'], 4, 'def_secondary')

j = base.merge(pooled,on=['season','team_id'],how='inner')
j = j.merge(f7,on=['season','team_id'],how='left').merge(sec,on=['season','team_id'],how='left')
j = j.dropna(subset=['def_all','def_front7','def_secondary'])
print(f"=== single correlations (n={len(j)}) ===")
for c in ('def_all','def_front7','def_secondary'):
    print(f"  {c:<16}{j[c].corr(j['resid']):>+8.3f}")
print(f"  front7 vs secondary correlation: {j['def_front7'].corr(j['def_secondary']):+.3f}")

print("\n=== which encoding predicts best? (CV R2 on residual) ===")
sets=[('pooled top 11',['def_all']),
      ('front7 + secondary',['def_front7','def_secondary']),
      ('pooled + both halves',['def_all','def_front7','def_secondary'])]
for lab,cols in sets:
    r2=cross_val_score(Ridge(alpha=10.0),j[cols],j['resid'],cv=5,scoring='r2').mean()
    print(f"  {lab:<24}{len(cols):>3}f   {r2:>+.4f}")

# compare against the 3-group build currently in the file
f=pd.read_csv('etl/summarize/results/returning_production.csv',low_memory=False)
three=[c for c in ('ret_DL_starter','ret_LB_starter','ret_DB_starter') if c in f.columns]
k=j.merge(f[['season','team_id']+three],on=['season','team_id'],how='inner').dropna(subset=three)
print(f"\n=== head to head on the same {len(k)} rows ===")
for lab,cols in (('current 3-group split',three),
                 ('pooled top 11',['def_all']),
                 ('front7 + secondary',['def_front7','def_secondary'])):
    r2=cross_val_score(Ridge(alpha=10.0),k[cols],k['resid'],cv=5,scoring='r2').mean()
    print(f"  {lab:<24}{len(cols):>3}f   {r2:>+.4f}")
PY
