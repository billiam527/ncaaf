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
RT=pd.read_csv('../etl/summarize/results/roster_talent.csv',low_memory=False)
C=pd.read_csv('../etl/collect/collect_cfbd_players/temp/cfbd_classification.csv',
              low_memory=False)
L=pd.read_csv('../etl/collect/collect_cfbd_games/cfbd_spread_data.csv',low_memory=False)
static_fbs=set(T.loc[T.fbs_ind==1.0,'id']); id2name=dict(zip(T['id'],T['location']))
season_fbs=set(zip(C.loc[C.fbs==1,'season'],C.loc[C.fbs==1,'team_id']))
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
             for n in sc0.feature_names_in_) if m})
used=[u for u in used if u in S.columns]

G=G.dropna(subset=['home_score_differential']).copy()
G['season']=pd.to_numeric(G['season'],errors='coerce')
G['home']=G['home_team_id'].map(id2name); G['away']=G['away_team_id'].map(id2name)
Lx=L.dropna(subset=['spread']).copy()
Lx['market_margin']=-pd.to_numeric(Lx['spread'],errors='coerce')
Lg=(Lx.groupby(['season','home_team','away_team'])['market_margin'].median()
    .reset_index().rename(columns={'home_team':'home','away_team':'away'}))
G=G.merge(Lg,on=['season','home','away'],how='left')

G['static_ok']=G.home_team_id.isin(static_fbs)&G.away_team_id.isin(static_fbs)
G['season_ok']=[ (s,h) in season_fbs and (s,a) in season_fbs
                 for s,h,a in zip(G.season,G.home_team_id,G.away_team_id)]
print(f"games total {len(G):,}")
print(f"  static filter passes  {int(G.static_ok.sum()):,}")
print(f"  season filter passes  {int(G.season_ok.sum()):,}")
print(f"  in static but not season: {int((G.static_ok&~G.season_ok).sum()):,}")
print(f"  in season but not static: {int((~G.static_ok&G.season_ok).sum()):,}")

def build(mask,label):
    d=G[mask][['id','season','home_team_id','away_team_id',
               'home_score_differential','market_margin']].copy()
    for side,idcol in (('home','home_team_id'),('away','away_team_id')):
        for lag in (1,2,3):
            l=S[['team_id','season']+used].copy(); l['season']+=lag
            suf='FY' if lag==1 else f'FY-{lag-1}'
            l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in used})
            d=d.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                      how='left').drop(columns=['team_id'])
        RET=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
             'ret_defense','ret_good','ret_bad']
        Rf=R[['team_id','season']+RET].copy()
        for c in RET: Rf[c]=Rf[c].fillna(Rf[c].median())
        Tf=TL.dropna(subset=['team_id'])[['team_id','season','talent_roll_pct']].copy()
        Tf['team_id']=pd.to_numeric(Tf['team_id'],errors='coerce')
        Nf=RT.dropna(subset=['team_id'])[['team_id','season','blue_chip_ratio_pct',
                                          'top22_rating_pct']].copy()
        Nf['team_id']=pd.to_numeric(Nf['team_id'],errors='coerce')
        for src,cols in ((Rf,RET),(Tf,['talent_roll_pct']),
                         (Nf,['blue_chip_ratio_pct','top22_rating_pct'])):
            d=d.merge(src.rename(columns={c:f'{side}_{c}' for c in cols}),
                      left_on=[idcol,'season'],right_on=['team_id','season'],
                      how='left').drop(columns=['team_id'])
    return d

PAR=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
         subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)

# the TEST set is held identical - only the TRAINING set differs
common=build(G.static_ok&G.season_ok,'common')
BASE=[c for c in common.columns if 'adjusted' in c and '_FY' in c]
EX=[c for c in common.columns if c.startswith(('home_ret','away_ret','home_talent',
    'away_talent','home_blue','away_blue','home_top22','away_top22'))]
for c in EX: common[c]=common[c].fillna(common[c].median())
common=common.dropna(subset=BASE+['home_score_differential'])
test_ids=set(common['id'])
print(f"\ncommon test pool: {len(test_ids):,} games")

print(f"\n{'training filter':<26}{'train n':>9}{'MAE':>9}{'side':>8}{'mkt corr':>10}")
print("-"*62)
res={}
for lab,mask in (('static (current)',G.static_ok),('season-aware',G.season_ok)):
    d=build(mask,lab)
    for c in EX:
        if c in d.columns: d[c]=d[c].fillna(common[c].median())
    d=d.dropna(subset=BASE+['home_score_differential'])
    cols=BASE+[c for c in EX if c in d.columns]
    P,A,M,N=[],[],[],[]
    for test in range(2019,2026):
        tr=d[d.season<test]
        te=common[(common.season==test)]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PAR).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
        M.extend(te['market_margin']); N.append(len(tr))
    P,A,M=np.array(P),np.array(A),np.array(M); ok=~np.isnan(M)
    res[lab]=(P,A)
    print(f"  {lab:<24}{int(np.mean(N)):>9}{np.abs(P-A).mean():>9.3f}"
          f"{((P>0)==(A>0)).mean():>8.1%}{np.corrcoef(P[ok],M[ok])[0,1]:>10.3f}")

a,b=res['static (current)'],res['season-aware']
d=np.abs(b[0]-b[1])-np.abs(a[0]-a[1]); se=d.std()/np.sqrt(len(d))
print(f"\n  season-aware vs static: MAE {np.abs(b[0]-b[1]).mean()-np.abs(a[0]-a[1]).mean():+.3f}"
      f"  (t={d.mean()/se:+.2f})")
PY
