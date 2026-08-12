#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
T='etl/collect/collect_cfbd_players/temp/'

Q=pd.read_csv('etl/summarize/results/player_quality.csv',low_memory=False)
roster=pd.read_csv(T+'cfbd_roster.csv',low_memory=False)
cf=pd.read_csv(T+'cfbd_teams.csv',low_memory=False)
Q['id']=Q['id'].astype(str); roster['id']=roster['id'].astype(str)
Q=Q.dropna(subset=['group','usage_overall','z_raw'])
name2id=dict(zip(cf['school'],cf['id']))

def build(mode, tag, thresh=0.0):
    """mode: 'team' splits at the team's own median, 'league' at z=0 (the
    within-position league mean), 'elite' keeps only clearly good players."""
    rows=[]
    for season in sorted(Q.season.unique()):
        prior=Q[Q.season==season-1]
        if prior.empty: continue
        back=set(roster.loc[roster.season==season,'team']+'|'+
                 roster.loc[roster.season==season,'id'])
        d=prior.copy(); d['ret']=(d['team']+'|'+d['id']).isin(back)
        for team,g in d.groupby('team'):
            if len(g)<5: continue
            w=g['usage_overall']; tot=w.sum()
            if tot<=0: continue
            if mode=='team':
                cut=g['z_raw'].median(); hi=g[g.z_raw>=cut]; lo=g[g.z_raw<cut]
            elif mode=='league':
                hi=g[g.z_raw>=thresh]; lo=g[g.z_raw<thresh]
            else:  # elite vs the rest
                hi=g[g.z_raw>=thresh]; lo=g[g.z_raw<thresh]
            row={'season':int(season),'team':team,
                 f'share_{tag}': g.loc[g.ret,'usage_overall'].sum()/tot,
                 f'n_hi_{tag}': len(hi)}
            if hi['usage_overall'].sum()>0:
                row[f'good_{tag}']=(hi.loc[hi.ret,'usage_overall'].sum()
                                    /hi['usage_overall'].sum())
                # how much of the team's total usage sits in the good half
                row[f'wt_{tag}']=hi['usage_overall'].sum()/tot
            if lo['usage_overall'].sum()>0:
                row[f'bad_{tag}']=(lo.loc[lo.ret,'usage_overall'].sum()
                                   /lo['usage_overall'].sum())
            rows.append(row)
    f=pd.DataFrame(rows); f['team_id']=f['team'].map(name2id)
    return f.drop(columns=['team'])

variants=[('team','tm',0.0),('league','lg',0.0),
          ('elite','el',0.5),('elite','e1',1.0)]
F=None
for mode,tag,th in variants:
    b=build(mode,tag,th)
    F=b if F is None else F.merge(b,on=['season','team_id'],how='outer')

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

print(f"=== coverage: does a league split leave teams with an empty half? (n={len(j)}) ===")
for tag,lab in (('tm','team median'),('lg','league mean z=0'),
                ('el','elite z>=0.5'),('e1','elite z>=1.0')):
    print(f"  {lab:<20} good {j[f'good_{tag}'].notna().mean():>6.1%}   "
          f"bad {j[f'bad_{tag}'].notna().mean():>6.1%}   "
          f"mean players in good half {j[f'n_hi_{tag}'].mean():>4.1f}")

print(f"\n=== correlation with residualised margin ===")
print(f"  {'feature':<34}{'r':>9}{'n':>7}")
print("  "+"-"*52)
for c in sorted(x for x in j.columns if x.startswith(('good_','bad_','share_','wt_'))):
    s=j[[c,'resid']].dropna()
    if len(s)<200: continue
    print(f"  {c:<34}{s[c].corr(s['resid']):>+9.3f}{len(s):>7}")

print(f"\n=== spread between good-half and bad-half correlation ===")
for tag,lab in (('tm','team median'),('lg','league mean'),
                ('el','elite z>=0.5'),('e1','elite z>=1.0')):
    s=j[[f'good_{tag}',f'bad_{tag}','resid']].dropna()
    if len(s)<200: continue
    sp=s[f'good_{tag}'].corr(s['resid'])-s[f'bad_{tag}'].corr(s['resid'])
    print(f"  {lab:<20}{sp:>+8.3f}   (n={len(s)})")

print(f"\n=== CV R2, common rows ===")
k=j.dropna(subset=[f'{p}_{t2}' for t2 in ('tm','lg') for p in ('good','bad')]
                  +['share_tm'])
print(f"  common rows: {len(k)}")
for lab,cols in (('share only',['share_tm']),
                 ('good/bad, TEAM median',['good_tm','bad_tm']),
                 ('good/bad, LEAGUE mean',['good_lg','bad_lg']),
                 ('league + weight of good half',['good_lg','bad_lg','wt_lg']),
                 ('share + league split',['share_tm','good_lg','bad_lg'])):
    cols=[c for c in cols if c in k.columns]
    r2=cross_val_score(Ridge(alpha=10.0),k[cols],k['resid'],cv=5,scoring='r2').mean()
    print(f"  {lab:<32}{len(cols):>3}f  {r2:>+.4f}")

F.to_csv('/tmp/league_split.csv',index=False)
print("\nwrote /tmp/league_split.csv")
PY
