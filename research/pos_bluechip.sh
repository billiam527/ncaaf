#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np, ast, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import pickle, re
T='etl/collect/collect_cfbd_players/temp/'

ro=pd.read_csv(T+'cfbd_roster.csv',low_memory=False)
rc=pd.read_csv(T+'cfbd_recruits.csv',low_memory=False)
cf=pd.read_csv(T+'cfbd_teams.csv',low_memory=False)
def fid(x):
    try:
        v=ast.literal_eval(str(x))
        return str(v[0]) if isinstance(v,list) and v else None
    except Exception: return None
ro['rid']=ro['recruitIds'].map(fid)
rc['id']=rc['id'].astype(str)
rc['rating']=pd.to_numeric(rc['rating'],errors='coerce')
rc['stars']=pd.to_numeric(rc['stars'],errors='coerce')
j=ro.dropna(subset=['rid']).merge(rc[['id','stars','rating']],left_on='rid',
                                  right_on='id',how='inner').dropna(subset=['rating'])
GRP={'QB':'QB','RB':'RB','FB':'RB','WR':'WR','TE':'WR','ATH':'WR',
     'OL':'OL','C':'OL','G':'OL','OT':'OL','OG':'OL','T':'OL',
     'DL':'DL','DE':'DL','DT':'DL','NT':'DL','EDGE':'DL',
     'LB':'LB','ILB':'LB','OLB':'LB','MLB':'LB',
     'DB':'DB','CB':'DB','S':'DB','FS':'DB','SS':'DB'}
j['group']=j['position'].map(GRP)
j=j.dropna(subset=['group'])
j['blue']=j['stars']>=4
GROUPS=('QB','RB','WR','OL','DL','LB','DB')

rows=[]
for (team,season),g in j.groupby(['team','season']):
    if len(g)<20: continue
    row={'team':team,'season':int(season)}
    for grp in GROUPS:
        s=g[g.group==grp]
        row[f'bc_{grp}']=float(s['blue'].sum()) if len(s) else 0.0
        row[f'bcr_{grp}']=float(s['blue'].mean()) if len(s) else np.nan
        row[f'best_{grp}']=float(s['rating'].max()) if len(s) else np.nan
    rows.append(row)
P=pd.DataFrame(rows)
name2id=dict(zip(cf['school'],cf['id']))
P['team_id']=P['team'].map(name2id)
for c in [c for c in P.columns if c.startswith(('bc_','bcr_','best_'))]:
    P[f'{c}_pct']=P.groupby('season')[c].rank(pct=True)
print(f"built {len(P)} team-seasons")
print("\n=== best QB on roster, 2026 ===")
cur=P[P.season==2026].nlargest(6,'best_QB')
for _,r in cur.iterrows():
    print(f"  {str(r['team'])[:20]:<22}best QB {r['best_QB']:.4f}   "
          f"blue-chip QBs {int(r['bc_QB'])}")

# ---- walk-forward ----
S=pd.read_csv('etl/summarize/results/season_summaries.csv',low_memory=False)
G=pd.read_csv('etl/summarize/temp/games.csv',low_memory=False)
TT=pd.read_csv('etl/collect/collect_espn_teams/temp/teams.csv')
R=pd.read_csv('etl/summarize/results/returning_production.csv',low_memory=False)
TL=pd.read_csv('etl/summarize/results/team_talent.csv',low_memory=False)
RT=pd.read_csv('etl/summarize/results/roster_talent.csv',low_memory=False)
fbs=set(TT.loc[TT['fbs_ind']==1.0,'id'])
sc0=pickle.load(open('model_training/preseason_model/temp/scaler.pkl','rb'))
used=sorted({m.group(1) for m in (re.match(r'(adjusted_.+?)_(FY(?:-\d)?)_(home|away)$',n)
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
RET=['ret_QB_starter','ret_RB_starter','ret_WR_starter','ret_TE_starter',
     'ret_defense','ret_good','ret_bad']
Rf=R[['team_id','season']+RET].copy()
for c in RET: Rf[c]=Rf[c].fillna(Rf[c].median())
Tf=TL.dropna(subset=['team_id'])[['team_id','season','talent_roll_pct']].copy()
Tf['team_id']=pd.to_numeric(Tf['team_id'],errors='coerce')
Nf=RT.dropna(subset=['team_id'])[['team_id','season','blue_chip_ratio_pct',
                                  'top22_rating_pct']].copy()
Nf['team_id']=pd.to_numeric(Nf['team_id'],errors='coerce')
BEST=[f'best_{g}_pct' for g in GROUPS]
BCC=[f'bc_{g}_pct' for g in GROUPS]
Pf=P.dropna(subset=['team_id'])[['team_id','season']+BEST+BCC].copy()
Pf['team_id']=pd.to_numeric(Pf['team_id'],errors='coerce')
for side,idcol in (('home','home_team_id'),('away','away_team_id')):
    for src,cols in ((Rf,RET),(Tf,['talent_roll_pct']),
                     (Nf,['blue_chip_ratio_pct','top22_rating_pct']),
                     (Pf,BEST+BCC)):
        feat=feat.merge(src.rename(columns={c:f'{side}_{c}' for c in cols}),
                        left_on=[idcol,'season'],right_on=['team_id','season'],
                        how='left').drop(columns=['team_id'])
BASE=[c for c in feat.columns if 'adjusted' in c]
CUR=[f'{s}_{c}' for s in ('home','away') for c in
     RET+['talent_roll_pct','blue_chip_ratio_pct','top22_rating_pct']]
BSIDE=[f'{s}_{c}' for s in ('home','away') for c in BEST]
CSIDE=[f'{s}_{c}' for s in ('home','away') for c in BCC]
QBONLY=[f'{s}_best_QB_pct' for s in ('home','away')]
for c in CUR+BSIDE+CSIDE: feat[c]=feat[c].fillna(feat[c].median())
feat=feat.dropna(subset=BASE+['home_score_differential'])

PAR=dict(n_estimators=400,max_depth=3,learning_rate=0.01,min_child_weight=25,
         subsample=0.6,colsample_bytree=0.6,reg_lambda=1.0,random_state=0)
def walk(cols,label):
    Pr,A=[],[]
    for test in range(2019,2026):
        tr=feat[feat.season<test]; te=feat[feat.season==test]
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[cols])
        m=XGBRegressor(**PAR).fit(sc.transform(tr[cols]),tr['home_score_differential'])
        Pr.extend(m.predict(sc.transform(te[cols]))); A.extend(te['home_score_differential'])
    Pr,A=np.array(Pr),np.array(A)
    print(f"  {label:<38}{len(cols):>4}f  MAE {np.abs(Pr-A).mean():.3f}  "
          f"side {((Pr>0)==(A>0)).mean():.1%}")
    return Pr,A
print(f"\n=== walk-forward, {len(feat)} games ===")
b=walk(BASE+CUR,'current (talent x3)')
w1=walk(BASE+CUR+QBONLY,'+ best QB only')
w2=walk(BASE+CUR+BSIDE,'+ best player at each position')
w3=walk(BASE+CUR+CSIDE,'+ blue-chip counts by position')
for lab,r in (('best QB',w1),('best by position',w2),('bc counts',w3)):
    d=np.abs(r[0]-r[1])-np.abs(b[0]-b[1]); se=d.std()/np.sqrt(len(d))
    print(f"  {lab:<20} MAE {np.abs(r[0]-r[1]).mean()-np.abs(b[0]-b[1]).mean():+.3f}"
          f"  (t={d.mean()/se:+.2f})")
PY
