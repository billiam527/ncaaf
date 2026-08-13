#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np, pickle, re, os, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

G=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
T=pd.read_csv('../etl/collect/collect_espn_teams/temp/teams.csv')
R=pd.read_csv('../etl/summarize/results/returning_production.csv',low_memory=False)
TL=pd.read_csv('../etl/summarize/results/team_talent.csv',low_memory=False)
RT=pd.read_csv('../etl/summarize/results/roster_talent.csv',low_memory=False)
L=pd.read_csv('../etl/collect/collect_cfbd_games/cfbd_spread_data.csv',low_memory=False)
fbs=set(T.loc[T['fbs_ind']==1.0,'id']); id2name=dict(zip(T['id'],T['location']))
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
USED=sorted({m.group(1) for m in (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
             for n in sc0.feature_names_in_) if m})

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()
G['home']=G['home_team_id'].map(id2name); G['away']=G['away_team_id'].map(id2name)
Lx=L.dropna(subset=['spread']).copy()
Lx['market_margin']=-pd.to_numeric(Lx['spread'],errors='coerce')
Lg=(Lx.groupby(['season','home_team','away_team'])['market_margin'].median()
    .reset_index().rename(columns={'home_team':'home','away_team':'away'}))
G=G.merge(Lg,on=['season','home','away'],how='left')

RET=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
     'ret_defense','ret_good','ret_bad']
Rf=R[['team_id','season']+RET].copy()
for c in RET: Rf[c]=Rf[c].fillna(Rf[c].median())
Tf=TL.dropna(subset=['team_id'])[['team_id','season','talent_roll_pct']].copy()
Tf['team_id']=pd.to_numeric(Tf['team_id'],errors='coerce')
Nf=RT.dropna(subset=['team_id'])[['team_id','season','blue_chip_ratio_pct',
                                  'top22_rating_pct']].copy()
Nf['team_id']=pd.to_numeric(Nf['team_id'],errors='coerce')
EXTRA=RET+['talent_roll_pct','blue_chip_ratio_pct','top22_rating_pct']

PAR=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
         subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)

SRC=[('../etl/summarize/results/season_summaries_unadjusted.csv','unadjusted'),
     ('../etl/summarize/results_a0.1/season_summaries.csv','alpha 0.1'),
     ('../etl/summarize/results/season_summaries.csv','alpha 1 (current)'),
     ('../etl/summarize/results_a20/season_summaries.csv','alpha 20'),
     ('../etl/summarize/results_a100/season_summaries.csv','alpha 100'),
     ('../etl/summarize/results_a500/season_summaries.csv','alpha 500')]

print(f"{'version':<22}{'games':>7}{'MAE':>9}{'RMSE':>9}{'side':>8}{'mkt corr':>10}")
print("-"*66)
res={}
for path,lab in SRC:
    if not os.path.exists(path): continue
    S=pd.read_csv(path,low_memory=False)
    used=[u for u in USED if u in S.columns]
    if len(used)<12:
        print(f"{lab:<22}  only {len(used)} of 12 stats present - skipped"); continue
    feat=G.copy()
    for side,idcol in (('home','home_team_id'),('away','away_team_id')):
        for lag in (1,2,3):
            l=S[['team_id','season']+used].copy(); l['season']+=lag
            suf='FY' if lag==1 else f'FY-{lag-1}'
            l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in used})
            feat=feat.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                            how='left').drop(columns=['team_id'])
        for src,cols in ((Rf,RET),(Tf,['talent_roll_pct']),
                         (Nf,['blue_chip_ratio_pct','top22_rating_pct'])):
            feat=feat.merge(src.rename(columns={c:f'{side}_{c}' for c in cols}),
                            left_on=[idcol,'season'],right_on=['team_id','season'],
                            how='left').drop(columns=['team_id'])
    BASE=[c for c in feat.columns if 'adjusted' in c and '_FY' in c]
    EX=[f'{s}_{c}' for s in ('home','away') for c in EXTRA]
    for c in EX: feat[c]=feat[c].fillna(feat[c].median())
    d=feat.dropna(subset=BASE+['home_score_differential'])
    cols=BASE+EX
    P,A,M=[],[],[]
    for test in range(2019,2026):
        tr=d[d.season<test]; te=d[d.season==test]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PAR).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
        M.extend(te['market_margin'])
    P,A,M=np.array(P),np.array(A),np.array(M); ok=~np.isnan(M)
    res[lab]=(P,A)
    print(f"{lab:<22}{len(P):>7}{np.abs(P-A).mean():>9.3f}"
          f"{np.sqrt(((P-A)**2).mean()):>9.3f}{((P>0)==(A>0)).mean():>8.1%}"
          f"{np.corrcoef(P[ok],M[ok])[0,1]:>10.3f}")

if 'alpha 1 (current)' in res:
    b=res['alpha 1 (current)']
    print("\nvs the current alpha of 1:")
    for lab,(P,A) in res.items():
        if lab=='alpha 1 (current)': continue
        n=min(len(P),len(b[0]))
        dd=np.abs(P[:n]-A[:n])-np.abs(b[0][:n]-b[1][:n])
        se=dd.std()/np.sqrt(len(dd))
        print(f"  {lab:<22}{np.abs(P-A).mean()-np.abs(b[0]-b[1]).mean():>+8.3f}"
              f"  (t={dd.mean()/se:+.2f})")
PY
