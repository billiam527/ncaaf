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

sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})
allc=sorted(c for c in S.columns if c.startswith('adjusted_'))
unused=[c for c in allc if c not in used]
print(f"used {len(used)}   unused {len(unused)}")
print(f"unused: {unused}")

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()
feat=G.copy()
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    for lag in (1,2,3):
        l=S[['team_id','season']+allc].copy(); l['season']+=lag
        suf='FY' if lag==1 else f'FY-{lag-1}'
        l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in allc})
        feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                        how='left').drop(columns=['team_id'])
BASE=[f'{s}_{c}_{sf}' for s in ('home','away') for c in used
      for sf in ('FY','FY-1','FY-2')]
EXTRA=[f'{s}_{c}_{sf}' for s in ('home','away') for c in unused
       for sf in ('FY','FY-1','FY-2')]

RPC=['rush_rate_off','rush_rate_off_neutral','rush_rate_def','rush_rate_def_neutral']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    l=RP[['team_id','season']+RPC].copy(); l['season']+=1
    l=l.rename(columns={c:f'{side}_{c}_FY' for c in RPC})
    feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
RP_FY=[f'{s}_{c}_FY' for s in ('home','away') for c in RPC]

LEAN=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
      'ret_defense','ret_overall']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    l=R[['team_id','season']+LEAN].rename(columns={c:f'{side}_{c}' for c in LEAN})
    feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
LEAN_S=[f'{s}_{c}' for s in ('home','away') for c in LEAN]

SETS={'base (production 72)':BASE,
      'base + 12 unused stats':BASE+EXTRA,
      'base + unused + rush rate':BASE+EXTRA+RP_FY,
      'base + lean returning':BASE+LEAN_S,
      'base + unused + returning':BASE+EXTRA+LEAN_S,
      'everything':BASE+EXTRA+RP_FY+LEAN_S}

need=sorted(set(BASE+EXTRA+RP_FY+LEAN_S))
feat=feat.dropna(subset=need+['home_score_differential'])
print(f"common sample: {len(feat)} games")

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
    print(f"  {label:<30}{len(cols):>4}f  MAE {res[label]['mae']:.3f}  "
          f"side {res[label]['side']:.1%}")

b=res['base (production 72)']
print("\n=== vs base ===")
for label,r in res.items():
    if label.startswith('base (prod'): continue
    d=np.abs(r['p']-r['a'])-np.abs(b['p']-b['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {label:<30} MAE {r['mae']-b['mae']:+.3f}  (t={d.mean()/se:+.2f})  "
          f"side {r['side']-b['side']:+.1%}")
PY
