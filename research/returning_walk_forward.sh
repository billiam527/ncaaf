#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

S = pd.read_csv('../etl/summarize/results/season_summaries.csv', low_memory=False)
G = pd.read_csv('../etl/summarize/temp/games.csv', low_memory=False)
T = pd.read_csv('../etl/collect/collect_espn_teams/temp/teams.csv')
R = pd.read_csv('../etl/summarize/results/returning_production.csv', low_memory=False)
fbs = set(T.loc[T['fbs_ind']==1.0,'id'])
adj = sorted(c for c in S.columns if c.startswith('adjusted_'))

G = G.dropna(subset=['home_score_differential'])
G = G[G.home_team_id.isin(fbs) & G.away_team_id.isin(fbs)]
G = G[['id','season','home_team_id','away_team_id','home_score_differential']].copy()

feat = G.copy()
for side, idcol in (('home','home_team_id'), ('away','away_team_id')):
    for lag in (1,2,3):
        l = S[['team_id','season']+adj].copy(); l['season'] += lag
        suf = 'FY' if lag==1 else f'FY-{lag-1}'
        l = l.rename(columns={c: f'{side}_{c}_{suf}' for c in adj})
        feat = feat.merge(l, left_on=[idcol,'season'], right_on=['team_id','season'],
                          how='left').drop(columns=['team_id'])
BASE = [c for c in feat.columns if 'adjusted' in c]

RET_ALL = sorted(c for c in R.columns if c.startswith('ret_')) + \
          ['portal_in','portal_out','draft_departures']
OFF_ST = [c for c in R.columns if c.endswith('_starter')
          and any(g in c for g in ('QB','RB','WR','TE'))]
LEAN = OFF_ST + ['ret_defense','ret_overall']

for side, idcol in (('home','home_team_id'), ('away','away_team_id')):
    r = R[['team_id','season']+[c for c in RET_ALL if c in R.columns]].copy()
    r = r.rename(columns={c: f'{side}_{c}' for c in r.columns
                          if c not in ('team_id','season')})
    feat = feat.merge(r, left_on=[idcol,'season'], right_on=['team_id','season'],
                      how='left').drop(columns=['team_id'])

sided = lambda cols: [f'{s}_{c}' for s in ('home','away') for c in cols
                      if f'{s}_{c}' in feat.columns]
SETS = {'base (model as-is)': BASE,
        'base + offense starters': BASE + sided(OFF_ST),
        'base + lean returning': BASE + sided(LEAN),
        'base + all returning': BASE + sided(RET_ALL)}

# ONE sample: rows complete for the widest set, used by every set
widest = SETS['base + all returning']
feat = feat.dropna(subset=[c for c in widest if c in feat.columns]
                          + ['home_score_differential'])
print(f"common sample: {len(feat)} games, seasons "
      f"{int(feat.season.min())}-{int(feat.season.max())}")

PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.01,
              min_child_weight=25, subsample=0.6, colsample_bytree=0.6,
              reg_lambda=1.0, random_state=0)
seasons = list(range(2019, 2026))

print("\n=== walk-forward, identical games for every feature set ===")
res={}
for label, cols in SETS.items():
    cols=[c for c in cols if c in feat.columns]
    P,A,SEA=[],[],[]
    for test in seasons:
        tr=feat[(feat.season<test)]; te=feat[feat.season==test]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PARAMS).fit(sc.transform(tr[cols]),
                                     tr['home_score_differential'])
        P.extend(m.predict(sc.transform(te[cols])))
        A.extend(te['home_score_differential']); SEA.extend([test]*len(te))
    P=np.array(P); A=np.array(A); SEA=np.array(SEA)
    res[label]=dict(p=P,a=A,s=SEA,mae=np.abs(P-A).mean(),
                    rmse=float(np.sqrt(((P-A)**2).mean())),
                    side=((P>0)==(A>0)).mean(),f=len(cols))
    print(f"  {label:<28}{len(cols):>4}f  n={len(P):>5}  MAE {res[label]['mae']:.3f}  "
          f"RMSE {res[label]['rmse']:.3f}  side {res[label]['side']:.1%}")

b=res['base (model as-is)']
print("\n=== against base, same games ===")
for label,r in res.items():
    if label=='base (model as-is)': continue
    d=np.abs(r['p']-r['a'])-np.abs(b['p']-b['a'])
    se=d.std()/np.sqrt(len(d))
    print(f"  {label:<28} MAE {r['mae']-b['mae']:+.3f}  (SE {se:.3f}, "
          f"t={d.mean()/se:+.2f})   side {r['side']-b['side']:+.1%}")

best=min(res.items(), key=lambda kv: kv[1]['mae'])[0]
print(f"\n=== per season: {best} vs base ===")
r=res[best]
print(f"  {'season':>7}{'n':>6}{'base':>9}{'new':>9}{'delta':>9}")
for s in seasons:
    mb=np.abs(b['p'][b['s']==s]-b['a'][b['s']==s])
    mn=np.abs(r['p'][r['s']==s]-r['a'][r['s']==s])
    if len(mb): print(f"  {s:>7}{len(mb):>6}{mb.mean():>9.3f}{mn.mean():>9.3f}"
                      f"{mn.mean()-mb.mean():>+9.3f}")
PY
