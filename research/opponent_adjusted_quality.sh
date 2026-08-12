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
Q=Q.dropna(subset=['group','usage_overall'])
print(f"player-seasons with adjusted quality: {len(Q):,}  "
      f"seasons {int(Q.season.min())}-{int(Q.season.max())}")

name2id=dict(zip(cf['school'],cf['id']))

def build(zcol, tag):
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
            med=g[zcol].median()
            hi=g[g[zcol]>=med]; lo=g[g[zcol]<med]
            row={'season':int(season),'team':team,
                 f'share_{tag}': g.loc[g.ret,'usage_overall'].sum()/tot}
            if hi['usage_overall'].sum()>0:
                row[f'good_{tag}']=(hi.loc[hi.ret,'usage_overall'].sum()
                                    /hi['usage_overall'].sum())
            if lo['usage_overall'].sum()>0:
                row[f'bad_{tag}']=(lo.loc[lo.ret,'usage_overall'].sum()
                                   /lo['usage_overall'].sum())
            rows.append(row)
    f=pd.DataFrame(rows); f['team_id']=f['team'].map(name2id)
    return f.drop(columns=['team'])

raw=build('z_raw','raw')
adj=build('z_adj','adj')
F=raw.merge(adj,on=['season','team_id'],how='inner')
print(f"team-seasons: {len(F)}")

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

j=base.merge(F,on=['season','team_id'],how='inner').dropna(
    subset=['good_raw','bad_raw','good_adj','bad_adj'])
print(f"\n=== does the opponent adjustment sharpen the split? (n={len(j)}) ===")
print(f"  {'feature':<34}{'r':>9}")
print("  "+"-"*45)
for c in ('share_raw','good_raw','bad_raw','good_adj','bad_adj'):
    lab={'share_raw':'plain returning share',
         'good_raw':'good half returns (RAW quality)',
         'bad_raw':'bad half returns (RAW quality)',
         'good_adj':'good half returns (ADJUSTED)',
         'bad_adj':'bad half returns (ADJUSTED)'}[c]
    print(f"  {lab:<34}{j[c].corr(j['resid']):>+9.3f}")
print(f"\n  spread good-minus-bad, raw:      "
      f"{j['good_raw'].corr(j['resid'])-j['bad_raw'].corr(j['resid']):+.3f}")
print(f"  spread good-minus-bad, adjusted: "
      f"{j['good_adj'].corr(j['resid'])-j['bad_adj'].corr(j['resid']):+.3f}")

print("\n=== CV R2 ===")
for lab,cols in (('share only',['share_raw']),
                 ('good/bad, raw quality',['good_raw','bad_raw']),
                 ('good/bad, adjusted quality',['good_adj','bad_adj']),
                 ('share + adjusted split',['share_raw','good_adj','bad_adj'])):
    r2=cross_val_score(Ridge(alpha=10.0),j[cols],j['resid'],cv=5,scoring='r2').mean()
    print(f"  {lab:<30}{len(cols):>3}f  {r2:>+.4f}")

print("\n=== how often does adjustment reclassify a player? ===")
Q2=Q.dropna(subset=['z_raw','z_adj']).copy()
flip=[]
for (s,tm),gg in Q2.groupby(['season','team']):
    if len(gg)<5: continue
    r=gg['z_raw']>=gg['z_raw'].median()
    a2=gg['z_adj']>=gg['z_adj'].median()
    flip.append((r!=a2).mean())
print(f"  mean share of a roster that changes half: {np.mean(flip):.1%}")

F.to_csv('/tmp/adj_split.csv',index=False)
print("\nwrote /tmp/adj_split.csv")
PY
