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
TL=pd.read_csv('../etl/summarize/results/team_talent.csv',low_memory=False)
L=pd.read_csv('../etl/collect/collect_cfbd_games/cfbd_spread_data.csv',low_memory=False)
fbs=set(T.loc[T['fbs_ind']==1.0,'id']); id2name=dict(zip(T['id'],T['location']))
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})
used=[u for u in used if u in S.columns]

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()
G['home']=G['home_team_id'].map(id2name); G['away']=G['away_team_id'].map(id2name)
Lx=L.dropna(subset=['spread']).copy()
Lx['market_margin']=-pd.to_numeric(Lx['spread'],errors='coerce')
Lg=(Lx.groupby(['season','home_team','away_team'])['market_margin'].median()
    .reset_index().rename(columns={'home_team':'home','away_team':'away'}))
G=G.merge(Lg,on=['season','home','away'],how='left')

feat=G.copy()
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    for lag in (1,2,3):
        l=S[['team_id','season']+used].copy(); l['season']+=lag
        suf='FY' if lag==1 else f'FY-{lag-1}'
        l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in used})
        feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                        how='left').drop(columns=['team_id'])
RET=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
     'ret_defense','ret_good','ret_bad']
Rf=R[['team_id','season']+RET].copy()
for c in RET: Rf[c]=Rf[c].fillna(Rf[c].median())
TAL=['talent_roll_pct','talent_pct']
Tf=TL.dropna(subset=['team_id'])[['team_id','season']+TAL].copy()
Tf['team_id']=Tf['team_id'].astype(float)
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    feat=feat.merge(Rf.rename(columns={c:f'{side}_{c}' for c in RET}),
                    left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
    feat=feat.merge(Tf.rename(columns={c:f'{side}_{c}' for c in TAL}),
                    left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])

BASE=[c for c in feat.columns if 'adjusted' in c]
RETS=[f'{s}_{c}' for s in ('home','away') for c in RET]
ROLL=[f'{s}_talent_roll_pct' for s in ('home','away')]
BOTH=ROLL+[f'{s}_talent_pct' for s in ('home','away')]
for c in RETS+BOTH: feat[c]=feat[c].fillna(feat[c].median())
feat=feat.dropna(subset=BASE+['home_score_differential'])
print(f"rows {len(feat)}   with a line {int(feat.market_margin.notna().sum())}")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)
def walk(cols,label):
    P,A,M=[],[],[]
    for test in range(2019,2026):
        tr=feat[feat.season<test]; te=feat[(feat.season==test)]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        p=m.predict(sc.transform(te[cols]))
        P.extend(p); A.extend(te['home_score_differential']); M.extend(te['market_margin'])
    P,A,M=np.array(P),np.array(A),np.array(M)
    ok=~np.isnan(M)
    print(f"  {label:<34}{len(cols):>4}f  MAE {np.abs(P-A).mean():.3f}  "
          f"side {((P>0)==(A>0)).mean():.1%}  sd {P.std():.1f}  |  "
          f"vs mkt: corr {np.corrcoef(P[ok],M[ok])[0,1]:+.3f}  "
          f"|P-M| {np.abs(P[ok]-M[ok]).mean():.2f}")
    return P,A,M

print("\n=== walk-forward 2019-2025 ===")
b=walk(BASE+RETS,'current model (86f)')
r=walk(BASE+RETS+ROLL,'+ rolling talent percentile')
t=walk(BASE+RETS+BOTH,'+ rolling + 247 composite')
o=walk(BASE+ROLL,'talent instead of returning')

d=np.abs(r[0]-r[1])-np.abs(b[0]-b[1]); se=d.std()/np.sqrt(len(d))
print(f"\n  rolling talent vs current: MAE {np.abs(r[0]-r[1]).mean()-np.abs(b[0]-b[1]).mean():+.3f}"
      f"  (t={d.mean()/se:+.2f})")
d2=np.abs(t[0]-t[1])-np.abs(b[0]-b[1]); se2=d2.std()/np.sqrt(len(d2))
print(f"  both talent   vs current: MAE {np.abs(t[0]-t[1]).mean()-np.abs(b[0]-b[1]).mean():+.3f}"
      f"  (t={d2.mean()/se2:+.2f})")
PY
