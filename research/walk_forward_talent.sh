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
T2=pd.read_csv('/tmp/talent.csv',low_memory=False)
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
TAL=['ret_good','ret_bad','ret_tilt','q_gap','star_back','best_back']
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    r=R[['team_id','season']+LEAN].rename(
        columns={c:f'{side}_{c}' for c in LEAN})
    feat=feat.merge(r,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
    q=T2[['team_id','season']+TAL].rename(
        columns={c:f'{side}_{c}' for c in TAL})
    feat=feat.merge(q,left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])

L=[f'{s}_{c}' for s in ('home','away') for c in LEAN]
TALS=[f'{s}_{c}' for s in ('home','away') for c in TAL]
STAR=[f'{s}_star_back' for s in ('home','away')]
TILT=[f'{s}_ret_tilt' for s in ('home','away')]
GOODBAD=[f'{s}_{c}' for s in ('home','away') for c in ('ret_good','ret_bad')]

SETS={'base':BASE,
      'base + lean':BASE+L,
      'base + lean + star returned':BASE+L+STAR,
      'base + lean + tilt':BASE+L+TILT,
      'base + lean + good/bad split':BASE+L+GOODBAD,
      'base + lean + all talent':BASE+L+TALS}
need=sorted(set(BASE+L+TALS))
feat=feat.dropna(subset=need+['home_score_differential'])
print(f"common sample: {len(feat)} games, {int(feat.season.min())}-{int(feat.season.max())}")

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

b=res['base']; ln=res['base + lean']
print("\n=== vs base, and vs lean ===")
for label,r in res.items():
    if label=='base': continue
    d1=np.abs(r['p']-r['a'])-np.abs(b['p']-b['a'])
    d2=np.abs(r['p']-r['a'])-np.abs(ln['p']-ln['a'])
    print(f"  {label:<32} vs base {r['mae']-b['mae']:+.3f} (t={d1.mean()/(d1.std()/np.sqrt(len(d1))):+.2f})"
          f"   vs lean {r['mae']-ln['mae']:+.3f} (t={d2.mean()/(d2.std()/np.sqrt(len(d2))):+.2f})")
PY
