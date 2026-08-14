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
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})
used=[u for u in used if u in S.columns]

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

FULL=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
      'ret_defense','ret_good','ret_bad']
NODEF=[c for c in FULL if c!='ret_defense']       # available from 2015
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    l=R[['team_id','season']+FULL].rename(columns={c:f'{side}_{c}' for c in FULL})
    feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
sided=lambda cs:[f'{s}_{c}' for s in ('home','away') for c in cs]

# TEST set is common: 2019-2025 games complete for every candidate
test_ok=feat.dropna(subset=BASE+sided(FULL)+['home_score_differential'])['id']
test_ok=set(test_ok)
print(f"common test games: {len(test_ok)}")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)

def run(label, cols):
    """Train on every row that has these columns; predict the common test set."""
    P,A,N=[],[],[]
    for test in range(2019,2026):
        tr=feat[(feat.season<test)].dropna(subset=cols+['home_score_differential'])
        te=feat[(feat.season==test)&(feat['id'].isin(test_ok))]
        te=te.dropna(subset=cols+['home_score_differential'])
        if len(tr)<400 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
        N.append(len(tr))
    P=np.array(P); A=np.array(A)
    print(f"  {label:<40}{len(cols):>4}f  train~{int(np.mean(N)):>5}  "
          f"n={len(P):>5}  MAE {np.abs(P-A).mean():.3f}  "
          f"side {((P>0)==(A>0)).mean():.1%}")
    return P,A

print("\n=== production-realistic: each model trains on all rows IT can use ===")
res={}
res['base']=run('base only (trains back to 2013)',BASE)
res['full']=run('base + 7 returning (2017+ only)',BASE+sided(FULL))
res['nodef']=run('base + 6 returning, no defense (2015+)',BASE+sided(NODEF))

b=res['base']
print("\n=== vs base, on the common test games ===")
for k in ('full','nodef'):
    p,a=res[k]; pb,ab=b
    n=min(len(p),len(pb))
    d=np.abs(p[:n]-a[:n])-np.abs(pb[:n]-ab[:n])
    se=d.std()/np.sqrt(len(d))
    print(f"  {k:<8} MAE {np.abs(p-a).mean()-np.abs(pb-ab).mean():+.3f}  "
          f"(t={d.mean()/se:+.2f})")
PY
