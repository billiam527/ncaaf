#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np, requests, os
from sklearn.linear_model import LinearRegression
T='etl/collect/collect_cfbd_players/temp/'

print("=== is there game-level player data for opponent adjustment? ===")
KEY=open(os.path.expanduser('~/.cfbd_api_key')).read().strip()
H={'Authorization':f'Bearer {KEY}'}
for path,params in (('/ppa/players/games',dict(year=2024,team='Ohio State')),
                    ('/games/players',dict(year=2024,week=5))):
    try:
        r=requests.get('https://api.collegefootballdata.com'+path,headers=H,
                       params=params,timeout=45)
        if r.status_code==200:
            d=r.json()
            print(f"  {path:<22} OK {len(d)} rows")
            if d: print(f"    keys: {list(d[0].keys())[:14]}")
        else:
            print(f"  {path:<22} HTTP {r.status_code}")
    except Exception as e:
        print(f"  {path:<22} {type(e).__name__}")

usage=pd.read_csv(T+'cfbd_usage.csv',low_memory=False)
ppa=pd.read_csv(T+'cfbd_ppa.csv',low_memory=False)
roster=pd.read_csv(T+'cfbd_roster.csv',low_memory=False)
cf=pd.read_csv(T+'cfbd_teams.csv',low_memory=False)
for d_ in (usage,ppa,roster): d_['id']=d_['id'].astype(str)

p=usage[['season','team','id','position','usage_overall']].merge(
    ppa[['season','team','id','averagePPA_all','totalPPA_all']],
    on=['season','team','id'],how='left')
p['avg']=pd.to_numeric(p['averagePPA_all'],errors='coerce')
p['tot']=pd.to_numeric(p['totalPPA_all'],errors='coerce')
p['use']=pd.to_numeric(p['usage_overall'],errors='coerce')
p=p.dropna(subset=['use','avg','tot'])
GRP={'QB':'QB','RB':'RB','FB':'RB','WR':'WR','TE':'TE'}
p['group']=p['position'].map(GRP); p=p.dropna(subset=['group'])
SLOTS={'QB':1,'RB':1,'WR':3,'TE':1}

print(f"\n=== how much does quality vary among equal-usage players? ===")
q=p[p.group=='QB']
print(f"  QB averagePPA: mean {q['avg'].mean():+.3f}  sd {q['avg'].std():.3f}  "
      f"p10 {q['avg'].quantile(.1):+.3f}  p90 {q['avg'].quantile(.9):+.3f}")
print(f"  correlation usage vs averagePPA: {q['use'].corr(q['avg']):+.3f}")
print("  -> a returning-usage feature treats these as identical")

# outcome frame
g=pd.read_csv('etl/summarize/temp/games.csv',low_memory=False)
t=pd.read_csv('etl/collect/collect_espn_teams/temp/teams.csv')
fbs=set(t.loc[t['fbs_ind']==1.0,'id'])
g=g.dropna(subset=['home_score_differential'])
g=g[g.home_team_id.isin(fbs)&g.away_team_id.isin(fbs)]
h=g[['season','home_team_id','home_score_differential']].copy(); h.columns=['season','team_id','m']
a=g[['season','away_team_id','home_score_differential']].copy(); a.columns=['season','team_id','m']; a['m']*=-1
perf=pd.concat([h,a]).groupby(['season','team_id'])['m'].mean().reset_index()
perf.columns=['season','team_id','avg_margin']
pr=perf.copy(); pr['season']+=1; pr=pr.rename(columns={'avg_margin':'prior'})
base=perf.merge(pr,on=['season','team_id'],how='left').dropna(subset=['prior'])
lr=LinearRegression().fit(base[['prior']],base['avg_margin'])
base['resid']=base['avg_margin']-lr.predict(base[['prior']])
name2id=dict(zip(cf['school'],cf['id']))

def build(weight, label, absolute=False):
    s=p.copy(); s['w']=s[weight]
    if not absolute: s=s[s['w']>0]
    s['rank']=s.groupby(['season','team','group'])['w'].rank(method='first',ascending=False)
    s=s[s['rank']<=s['group'].map(SLOTS)]
    rows=[]
    for season in range(2016,2026):
        prior=s[s.season==season-1]
        if prior.empty: continue
        back=set(roster.loc[roster.season==season,'team']+'|'+
                 roster.loc[roster.season==season,'id'])
        pp=prior.copy(); pp['ret']=(pp['team']+'|'+pp['id']).isin(back)
        for team,gd in pp.groupby('team'):
            if absolute:
                # level: total quality of the players who return
                val=(gd.loc[gd.ret,'use']*gd.loc[gd.ret,'avg']).sum()
            else:
                tot=gd['w'].sum()
                if tot<=0: continue
                val=gd.loc[gd.ret,'w'].sum()/tot
            rows.append({'season':season,'team':team,label:val})
    fr=pd.DataFrame(rows); fr['team_id']=fr['team'].map(name2id)
    return fr[['season','team_id',label]]

print("\n=== weighting schemes, offence skill positions ===")
p['q_use']=p['use']
p['q_tot']=p['tot'].clip(lower=0)
p['q_mix']=p['use']*p['avg'].clip(lower=-0.2)
tests=[('q_use','share of snaps returning (current build)',False),
       ('q_tot','share of TOTAL PPA returning',False),
       ('q_mix','share of usage x efficiency returning',False),
       ('q_use','ABSOLUTE quality returning (usage x PPA, a level)',True)]
frames=[]
print(f"  {'scheme':<46}{'r':>9}{'n':>7}")
print("  "+"-"*62)
for w,lab,absolute in tests:
    col='v_'+w+('_abs' if absolute else '')
    fr=build(w,col,absolute)
    frames.append(fr)
    j=base.merge(fr,on=['season','team_id'],how='inner').dropna(subset=[col])
    print(f"  {lab:<46}{j[col].corr(j['resid']):>+9.3f}{len(j):>7}")

print("\n=== do share and level carry different information? ===")
m=frames[0]
for fr in frames[1:]: m=m.merge(fr,on=['season','team_id'],how='outer')
j=base.merge(m,on=['season','team_id'],how='inner').dropna()
cols=[c for c in j.columns if c.startswith('v_')]
print(j[cols].corr().round(3).to_string())
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
print("\n=== combined CV R2 on residualised margin ===")
for lab,use in (('share only',['v_q_use']),
                ('level only',['v_q_use_abs']),
                ('share + level',['v_q_use','v_q_use_abs']),
                ('all four',cols)):
    r2=cross_val_score(Ridge(alpha=10.0),j[use],j['resid'],cv=5,scoring='r2').mean()
    print(f"  {lab:<18}{len(use):>3}f  {r2:>+.4f}")
PY
