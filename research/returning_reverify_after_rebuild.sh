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
R=pd.read_csv('../etl/summarize/results/returning_production.csv',low_memory=False)
fbs=set(T.loc[T['fbs_ind']==1.0,'id'])

# production feature set: the 12 adjusted stats the shipped model uses
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})
print(f"base stats: {len(used)}  (rebuilt, garbage-time included)")

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

LEAN=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
      'ret_defense','ret_overall']
GB=['ret_good','ret_bad']
ALL=sorted(c for c in R.columns if c.startswith('ret_')) + \
    ['portal_in','portal_out','draft_departures']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    l=R[['team_id','season']+ALL].rename(columns={c:f'{side}_{c}' for c in ALL})
    feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
sided=lambda cs:[f'{s}_{c}' for s in ('home','away') for c in cs if f'{s}_{c}' in feat.columns]

SETS={'base (rebuilt, 72f)':BASE,
      'base + lean':BASE+sided(LEAN),
      'base + lean + good/bad':BASE+sided(LEAN+GB),
      'base + lean, overall->good/bad':BASE+sided([c for c in LEAN if c!='ret_overall']+GB),
      'base + everything':BASE+sided(ALL)}

need=sorted(set(BASE+sided(ALL)))
feat=feat.dropna(subset=need+['home_score_differential'])
print(f"common sample: {len(feat)} games ({int(feat.season.min())}-{int(feat.season.max())})")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)
res={}
print("\n=== walk-forward 2019-2025 on the REBUILT stats, identical games ===")
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

b=res['base (rebuilt, 72f)']
print("\n=== vs base ===")
for label,r in res.items():
    if label.startswith('base (rebuilt'): continue
    d=np.abs(r['p']-r['a'])-np.abs(b['p']-b['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {label:<32} MAE {r['mae']-b['mae']:+.3f}  (t={d.mean()/se:+.2f})  "
          f"side {r['side']-b['side']:+.1%}")

ln=res['base + lean']
print("\n=== does good/bad add anything on top of lean? ===")
for label in ('base + lean + good/bad','base + lean, overall->good/bad'):
    r=res[label]
    d=np.abs(r['p']-r['a'])-np.abs(ln['p']-ln['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {label:<32} vs lean {r['mae']-ln['mae']:+.3f}  (t={d.mean()/se:+.2f})")
PY
