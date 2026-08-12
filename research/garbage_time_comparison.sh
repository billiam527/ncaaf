#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np, pickle, re, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

G=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
T=pd.read_csv('../etl/collect/collect_espn_teams/temp/teams.csv')
fbs=set(T.loc[T['fbs_ind']==1.0,'id'])
sc0=pickle.load(open('../model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in
             (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
              for n in sc0.feature_names_in_) if m})

G=G.dropna(subset=['home_score_differential'])
G=G[G.home_team_id.isin(fbs)&G.away_team_id.isin(fbs)]
G=G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()

SOURCES={
 'OLD  (pre-rebuild, backup)':'/home/bill/ncaaf_backup_20260812/etl/summarize/results/season_summaries.csv',
 'NEW  garbage REMOVED'      :'../etl/summarize/results/season_summaries.csv',
 'NEW  garbage INCLUDED'     :'../etl/summarize/results_with_garbage/season_summaries.csv',
}

def build(path):
    S=pd.read_csv(path,low_memory=False)
    f=G.copy()
    for side,idcol in (('home','home_team_id'),('away','away_team_id')):
        for lag in (1,2,3):
            l=S[['team_id','season']+used].copy(); l['season']+=lag
            suf='FY' if lag==1 else f'FY-{lag-1}'
            l=l.rename(columns={c:f'{side}_{c}_{suf}' for c in used})
            f=f.merge(l,left_on=[idcol,'season'],right_on=['team_id','season'],
                      how='left').drop(columns=['team_id'])
    cols=[c for c in f.columns if 'adjusted' in c]
    return f, cols

frames={}
for lab,p in SOURCES.items():
    f,cols=build(p)
    frames[lab]=(f,cols)
    print(f"  {lab:<28} {len(cols)} features, "
          f"{f.dropna(subset=cols).shape[0]} complete games")

# one common set of game ids across all three
common=None
for lab,(f,cols) in frames.items():
    ids=set(f.dropna(subset=cols+['home_score_differential'])['id'])
    common=ids if common is None else (common & ids)
print(f"\ncommon games across all three: {len(common)}")

PARAMS=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
            subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)
res={}
print("\n=== walk-forward 2019-2025, identical games ===")
for lab,(f,cols) in frames.items():
    d=f[f['id'].isin(common)]
    P,A=[],[]
    for test in range(2019,2026):
        tr=d[d.season<test]; te=d[d.season==test]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
    P=np.array(P); A=np.array(A)
    res[lab]=dict(p=P,a=A,mae=np.abs(P-A).mean(),
                  rmse=float(np.sqrt(((P-A)**2).mean())),
                  side=((P>0)==(A>0)).mean())
    print(f"  {lab:<28} n={len(P):>5}  MAE {res[lab]['mae']:.3f}  "
          f"RMSE {res[lab]['rmse']:.3f}  side {res[lab]['side']:.1%}")

base=res['OLD  (pre-rebuild, backup)']
print("\n=== against the pre-rebuild baseline ===")
for lab,r in res.items():
    if lab.startswith('OLD'): continue
    d=np.abs(r['p']-r['a'])-np.abs(base['p']-base['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {lab:<28} MAE {r['mae']-base['mae']:+.3f}  (t={d.mean()/se:+.2f})  "
          f"side {r['side']-base['side']:+.1%}")

a=res['NEW  garbage REMOVED']; b=res['NEW  garbage INCLUDED']
d=np.abs(b['p']-b['a'])-np.abs(a['p']-a['a'])
se=d.std()/np.sqrt(len(d))
print(f"\n=== the question: including garbage vs removing it ===")
print(f"  including - removing: MAE {b['mae']-a['mae']:+.3f}  "
      f"(SE {se:.3f}, t={d.mean()/se:+.2f})")
print(f"  -> {'REMOVING is better' if b['mae']>a['mae'] else 'INCLUDING is better'}"
      f"{' (significant)' if abs(d.mean()/se)>2 else ' (not significant)'}")

print("\n=== how much did the EPA features themselves move? ===")
old=pd.read_csv(SOURCES['OLD  (pre-rebuild, backup)'],low_memory=False)
new=pd.read_csv(SOURCES['NEW  garbage REMOVED'],low_memory=False)
j=old.merge(new,on=['team_id','season'],suffixes=('_old','_new'))
for c in ('adjusted_epa_per_rush_off','adjusted_epa_per_pass_off',
          'adjusted_rush_success_off','adjusted_pass_success_off'):
    if f'{c}_old' in j.columns:
        r=j[f'{c}_old'].corr(j[f'{c}_new'])
        md=(j[f'{c}_new']-j[f'{c}_old']).abs().mean()
        print(f"  {c:<32} r={r:+.4f}  mean |change| {md:.4f}")
PY
