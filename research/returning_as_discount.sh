#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

S=pd.read_csv('../etl/summarize/results/season_summaries.csv',low_memory=False)
G=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
T=pd.read_csv('../etl/collect/collect_espn_teams/temp/teams.csv')
R=pd.read_csv('../etl/summarize/results/returning_production.csv',low_memory=False)
fbs=set(T.loc[T['fbs_ind']==1.0,'id'])
adj=sorted(c for c in S.columns if c.startswith('adjusted_'))

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()

feat=G.copy()
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    for lag in (1,2,3):
        l=S[['team_id','season']+adj].copy(); l['season']+=lag
        suf='FY' if lag==1 else f'FY-{lag-1}'
        l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in adj})
        feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                        how='left').drop(columns=['team_id'])
BASE=[c for c in feat.columns if 'adjusted' in c]

LEAN=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
      'ret_defense','ret_overall']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    r=R[['team_id','season']+LEAN].copy()
    r=r.rename(columns={c:f'{side}_{c}' for c in LEAN})
    feat=feat.merge(r,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
LEAN_S=[f'{s}_{c}' for s in ('home','away') for c in LEAN]

# --- the idea: use returning production to DISCOUNT last season's stats ---
# A team's FY stats describe players who may be gone. Shrink each FY feature
# toward the league mean in proportion to how little of that unit returns:
#   discounted = mean + (stat - mean) * returning_share
# Offensive FY features are scaled by offensive returning, defensive by
# defensive returning, so the discount matches the unit it applies to.
inter=[]
for side in ('home','away'):
    off_r=feat[f'{side}_ret_overall']
    def_r=feat[f'{side}_ret_defense'].fillna(feat[f'{side}_ret_overall'])
    for c in BASE:
        if not c.startswith(f'{side}_') or '_FY' not in c or c.endswith(('FY-1','FY-2')):
            continue
        share=def_r if '_def' in c else off_r
        mu=feat[c].mean()
        newc=f'{c}_disc'
        feat[newc]=mu+(feat[c]-mu)*share
        inter.append(newc)
print(f"discounted FY features created: {len(inter)}")

SETS={'base (model as-is)':BASE,
      'base + lean returning':BASE+LEAN_S,
      'base + discounted FY':BASE+inter,
      'base + lean + discounted':BASE+LEAN_S+inter}

need=sorted(set(BASE+LEAN_S+inter))
feat=feat.dropna(subset=need+['home_score_differential'])
print(f"common sample: {len(feat)} games")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)
seasons=list(range(2019,2026))
res={}
print("\n=== walk-forward, identical games ===")
for label,cols in SETS.items():
    cols=[c for c in cols if c in feat.columns]
    P,A=[],[]
    for test in seasons:
        tr=feat[feat.season<test]; te=feat[feat.season==test]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
    P=np.array(P); A=np.array(A)
    res[label]=dict(p=P,a=A,mae=np.abs(P-A).mean(),side=((P>0)==(A>0)).mean())
    print(f"  {label:<28}{len(cols):>4}f  MAE {res[label]['mae']:.3f}  "
          f"side {res[label]['side']:.1%}")

b=res['base (model as-is)']
print("\n=== against base ===")
for label,r in res.items():
    if label=='base (model as-is)': continue
    d=np.abs(r['p']-r['a'])-np.abs(b['p']-b['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {label:<28} MAE {r['mae']-b['mae']:+.3f}  (t={d.mean()/se:+.2f})  "
          f"side {r['side']-b['side']:+.1%}")
PY
