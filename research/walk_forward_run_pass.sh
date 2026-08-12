#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np, pickle, re, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

S=pd.read_csv('../etl/summarize/results/season_summaries.csv',low_memory=False)
G=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
T=pd.read_csv('../etl/collect/collect_espn_teams/temp/teams.csv')
RP=pd.read_csv('../etl/summarize/results/run_pass_ratio.csv',low_memory=False)
R=pd.read_csv('../etl/summarize/results/returning_production.csv',low_memory=False)
fbs=set(T.loc[T['fbs_ind']==1.0,'id'])

# the PRODUCTION feature set: 12 adjusted stats, not all 24
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})
print(f"production adjusted stats: {len(used)}")

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()
feat=G.copy()
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    for lag in (1,2,3):
        l=S[['team_id','season']+used].copy(); l['season']+=lag
        suf='FY' if lag==1 else f'FY-{lag-1}'
        l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in used})
        feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                        how='left').drop(columns=['team_id'])
BASE=[c for c in feat.columns if 'adjusted' in c]
print(f"base features: {len(BASE)}")

RPC=['rush_rate_off','rush_rate_off_neutral','rush_rate_def','rush_rate_def_neutral']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    for lag in (1,2,3):
        l=RP[['team_id','season']+RPC].copy(); l['season']+=lag
        suf='FY' if lag==1 else f'FY-{lag-1}'
        l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in RPC})
        feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                        how='left').drop(columns=['team_id'])
RP_FY=[f'{s}_{c}_FY' for s in ('home','away') for c in RPC]
RP_ALL=[c for c in feat.columns if 'rush_rate' in c]

LEAN=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
      'ret_defense','ret_overall']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    l=R[['team_id','season']+LEAN].rename(columns={c:f'{side}_{c}' for c in LEAN})
    feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
LEAN_S=[f'{s}_{c}' for s in ('home','away') for c in LEAN]

SETS={'base (production 72)':BASE,
      'base + rush rate FY':BASE+RP_FY,
      'base + rush rate x3 lags':BASE+RP_ALL,
      'base + lean returning':BASE+LEAN_S,
      'base + rush rate + returning':BASE+RP_FY+LEAN_S}

need=sorted(set(BASE+RP_ALL+LEAN_S))
feat=feat.dropna(subset=need+['home_score_differential'])
print(f"common sample: {len(feat)} games "
      f"({int(feat.season.min())}-{int(feat.season.max())})")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)
res={}
print("\n=== walk-forward, identical games ===")
for label,cols in SETS.items():
    cols=[c for c in cols if c in feat.columns]
    P,A=[],[]
    for test in range(2019,2026):
        tr=feat[feat.season<test]; te=feat[feat.season==test]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
    P=np.array(P); A=np.array(A)
    res[label]=dict(p=P,a=A,mae=np.abs(P-A).mean(),side=((P>0)==(A>0)).mean())
    print(f"  {label:<32}{len(cols):>4}f  MAE {res[label]['mae']:.3f}  "
          f"side {res[label]['side']:.1%}")

b=res['base (production 72)']
print("\n=== vs base ===")
for label,r in res.items():
    if label.startswith('base (prod'): continue
    d=np.abs(r['p']-r['a'])-np.abs(b['p']-b['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {label:<32} MAE {r['mae']-b['mae']:+.3f}  (t={d.mean()/se:+.2f})  "
          f"side {r['side']-b['side']:+.1%}")

print("\n=== is the interaction real? rush rate x rush success ===")
j=RP.merge(S[['team_id','season','adjusted_rush_success_off']],
           on=['team_id','season'],how='inner')
gg=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
gg=gg.dropna(subset=['home_score_differential'])
gg=gg[gg.home_team_id.isin(fbs)&gg.away_team_id.isin(fbs)]
h=gg[['season','home_team_id','home_score_differential']].copy(); h.columns=['season','team_id','m']
aw=gg[['season','away_team_id','home_score_differential']].copy(); aw.columns=['season','team_id','m']; aw['m']*=-1
perf=pd.concat([h,aw]).groupby(['season','team_id'])['m'].mean().reset_index()
perf.columns=['season','team_id','margin']
j=j.merge(perf,on=['season','team_id'],how='inner').dropna()
j['runs_a_lot']=j['rush_rate_off_neutral']>j['rush_rate_off_neutral'].median()
j['good_at_it']=j['adjusted_rush_success_off']>j['adjusted_rush_success_off'].median()
print(f"  {'':<22}{'good at rushing':>18}{'bad at rushing':>17}")
for runs,lab in ((True,'runs a lot'),(False,'runs a little')):
    row=f"  {lab:<22}"
    for good in (True,False):
        s=j[(j.runs_a_lot==runs)&(j.good_at_it==good)]
        row+=f"{s['margin'].mean():>+13.2f} (n{len(s)})" if len(s) else f"{'-':>18}"
    print(row)
PY
