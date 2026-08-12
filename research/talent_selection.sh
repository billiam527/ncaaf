#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
T='etl/collect/collect_cfbd_players/temp/'

usage=pd.read_csv(T+'cfbd_usage.csv',low_memory=False)
ppa=pd.read_csv(T+'cfbd_ppa.csv',low_memory=False)
roster=pd.read_csv(T+'cfbd_roster.csv',low_memory=False)
cf=pd.read_csv(T+'cfbd_teams.csv',low_memory=False)
for d_ in (usage,ppa,roster): d_['id']=d_['id'].astype(str)

p=usage[['season','team','id','position','usage_overall']].merge(
    ppa[['season','team','id','averagePPA_all','totalPPA_all']],
    on=['season','team','id'],how='inner')
p['use']=pd.to_numeric(p['usage_overall'],errors='coerce')
p['avg']=pd.to_numeric(p['averagePPA_all'],errors='coerce')
p['tot']=pd.to_numeric(p['totalPPA_all'],errors='coerce')
GRP={'QB':'QB','RB':'RB','FB':'RB','WR':'WR','TE':'TE'}
p['group']=p['position'].map(GRP)
p=p.dropna(subset=['use','avg','group'])
p=p[p['use']>=0.02]          # players with a real role

# quality is only comparable within a position and season
p['z']=p.groupby(['season','group'])['avg'].transform(
    lambda x: (x-x.mean())/x.std() if x.std()>0 else 0.0)
print(f"player-seasons with a role: {len(p)}")
print(f"  z by group: {p.groupby('group')['z'].std().round(2).to_dict()}")

name2id=dict(zip(cf['school'],cf['id']))

rows=[]
for season in range(2016,2027):
    prior=p[p.season==season-1]
    if prior.empty: continue
    back=set(roster.loc[roster.season==season,'team']+'|'+
             roster.loc[roster.season==season,'id'])
    d=prior.copy(); d['ret']=(d['team']+'|'+d['id']).isin(back)
    for team,g in d.groupby('team'):
        if len(g)<5: continue
        w=g['use']
        tot=w.sum()
        if tot<=0: continue
        r=g[g.ret]; l=g[~g.ret]
        row={'season':season,'team':team,
             'ret_share': r['use'].sum()/tot}
        # 1. quality gap: are the ones staying better than the ones leaving?
        if len(r) and len(l) and r['use'].sum()>0 and l['use'].sum()>0:
            row['q_stay']=np.average(r['z'],weights=r['use'])
            row['q_leave']=np.average(l['z'],weights=l['use'])
            row['q_gap']=row['q_stay']-row['q_leave']
        # 2. split the roster by quality, measure each half's return rate
        med=g['z'].median()
        hi=g[g.z>=med]; lo=g[g.z<med]
        if hi['use'].sum()>0:
            row['ret_good']=hi.loc[hi.ret,'use'].sum()/hi['use'].sum()
        if lo['use'].sum()>0:
            row['ret_bad']=lo.loc[lo.ret,'use'].sum()/lo['use'].sum()
        if 'ret_good' in row and 'ret_bad' in row:
            row['ret_tilt']=row['ret_good']-row['ret_bad']
        # 3. did the single best player return?
        best=g.nlargest(1,'z')
        if len(best): row['best_back']=float(best['ret'].iloc[0])
        # 4. the most-used player (the star by volume)
        top=g.nlargest(1,'use')
        if len(top): row['star_back']=float(top['ret'].iloc[0])
        rows.append(row)

F=pd.DataFrame(rows)
F['team_id']=F['team'].map(name2id)
print(f"team-seasons built: {len(F)}")

# outcome
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

j=base.merge(F,on=['season','team_id'],how='inner')
print(f"\n=== does WHO leaves matter, beyond HOW MUCH? (n={len(j)}) ===")
print(f"  {'feature':<40}{'r':>9}{'n':>7}")
print("  "+"-"*58)
labels={'ret_share':'share of usage returning (baseline)',
        'ret_good':'return rate, above-median quality',
        'ret_bad':'return rate, below-median quality',
        'ret_tilt':'tilt: good return rate - bad return rate',
        'q_gap':'quality gap, stayers minus leavers',
        'q_stay':'mean quality of those who stay',
        'q_leave':'mean quality of those who leave',
        'best_back':'best player (by efficiency) returned',
        'star_back':'most-used player returned'}
for c,lab in labels.items():
    s=j[[c,'resid']].dropna()
    if len(s)<200: continue
    print(f"  {lab:<40}{s[c].corr(s['resid']):>+9.3f}{len(s):>7}")

print("\n=== held against the plain share: does tilt add anything? ===")
k=j.dropna(subset=['ret_share','ret_good','ret_bad','ret_tilt','q_gap'])
for lab,cols in (('share only',['ret_share']),
                 ('share + tilt',['ret_share','ret_tilt']),
                 ('share + quality gap',['ret_share','q_gap']),
                 ('good and bad separately',['ret_good','ret_bad']),
                 ('everything',['ret_share','ret_good','ret_bad','ret_tilt','q_gap'])):
    r2=cross_val_score(Ridge(alpha=10.0),k[cols],k['resid'],cv=5,scoring='r2').mean()
    print(f"  {lab:<26}{len(cols):>3}f  {r2:>+.4f}")

print("\n=== concrete: split teams by tilt, holding share roughly fixed ===")
mid=k[(k.ret_share>k.ret_share.quantile(.3))&(k.ret_share<k.ret_share.quantile(.7))]
hi=mid[mid.ret_tilt>mid.ret_tilt.median()]
lo=mid[mid.ret_tilt<=mid.ret_tilt.median()]
print(f"  teams in the middle 40% of returning share: n={len(mid)}")
print(f"    kept the better players (high tilt): resid {hi['resid'].mean():+.2f}  n={len(hi)}")
print(f"    kept the worse players  (low tilt): resid {lo['resid'].mean():+.2f}  n={len(lo)}")
print(f"    gap: {hi['resid'].mean()-lo['resid'].mean():+.2f} points of margin")

F.to_csv('/tmp/talent.csv',index=False)
print("\nwrote /tmp/talent.csv")
PY
