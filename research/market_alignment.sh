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
L=pd.read_csv('../etl/collect/collect_cfbd_games/cfbd_spread_data.csv',low_memory=False)
fbs=set(T.loc[T['fbs_ind']==1.0,'id'])
id2name=dict(zip(T['id'],T['location']))

sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})
used=[u for u in used if u in S.columns]

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()
G['home']=G['home_team_id'].map(id2name); G['away']=G['away_team_id'].map(id2name)

# market target
Lx=L.dropna(subset=['spread']).copy()
Lx['market_margin']=-pd.to_numeric(Lx['spread'],errors='coerce')
Lg=(Lx.groupby(['season','home_team','away_team'])['market_margin'].median()
      .reset_index().rename(columns={'home_team':'home','away_team':'away'}))
G=G.merge(Lg,on=['season','home','away'],how='left')
print(f"games {len(G)}   with a line {int(G.market_margin.notna().sum())} "
      f"({G.market_margin.notna().mean():.0%})")

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
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    feat=feat.merge(Rf.rename(columns={c:f'{side}_{c}' for c in RET}),
                    left_on=[idcol,'season'],right_on=['team_id','season'],
                    how='left').drop(columns=['team_id'])
COLS=[c for c in feat.columns if 'adjusted' in c or 'ret_' in c]
for c in COLS: feat[c]=feat[c].fillna(feat[c].median())
feat=feat.dropna(subset=COLS+['home_score_differential'])
print(f"usable rows {len(feat)}   with a line {int(feat.market_margin.notna().sum())}")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)

def walk(target, label, need_target=True):
    P,A,M=[],[],[]
    for test in range(2019,2026):
        tr=feat[feat.season<test]
        if need_target: tr=tr.dropna(subset=[target])
        te=feat[(feat.season==test)&feat.market_margin.notna()]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[COLS])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[COLS]),tr[target])
        P.extend(m.predict(sc.transform(te[COLS])))
        A.extend(te['home_score_differential']); M.extend(te['market_margin'])
    P,A,M=np.array(P),np.array(A),np.array(M)
    slope=np.polyfit(M,P,1)[0]
    print(f"  {label:<26}n={len(P):>5}  corr w/ mkt {np.corrcoef(P,M)[0,1]:+.3f}  "
          f"slope {slope:.3f}  |P-M| {np.abs(P-M).mean():>5.2f}  "
          f"MAE vs actual {np.abs(P-A).mean():.2f}  sd {P.std():.1f}")
    return P,A,M

print("\n=== walk-forward 2019-2025, scored on games that have a line ===")
pm,a,m = walk('home_score_differential','target = actual margin')
ps,_,_  = walk('market_margin','target = market spread')
print(f"  {'the market itself':<26}n={len(m):>5}  corr w/ mkt {1.0:+.3f}  "
      f"slope {1.0:.3f}  |P-M| {0.0:>5.2f}  MAE vs actual {np.abs(m-a).mean():.2f}  "
      f"sd {m.std():.1f}")

print("\n=== does the spread-trained model predict OUTCOMES better too? ===")
d=np.abs(ps-a)-np.abs(pm-a)
se=d.std()/np.sqrt(len(d))
print(f"  spread-target vs margin-target: MAE {np.abs(ps-a).mean()-np.abs(pm-a).mean():+.3f}"
      f"  (t={d.mean()/se:+.2f})")
print(f"  correct side: margin {((pm>0)==(a>0)).mean():.1%}   "
      f"spread {((ps>0)==(a>0)).mean():.1%}   market {((m>0)==(a>0)).mean():.1%}")

print("\n=== blend of the two targets ===")
for w in (0.25,0.5,0.75):
    q=w*ps+(1-w)*pm
    print(f"  {w:.0%} spread + {1-w:.0%} margin:  corr w/ mkt "
          f"{np.corrcoef(q,m)[0,1]:+.3f}  |P-M| {np.abs(q-m).mean():5.2f}  "
          f"MAE vs actual {np.abs(q-a).mean():.2f}")
PY
