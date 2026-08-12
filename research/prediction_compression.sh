#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np
h=pd.read_csv('../analysis/backtest_expanding_preds.csv')
h=h.dropna(subset=['preseason_model_preds','home_score_differential'])
p=h['preseason_model_preds']; a=h['home_score_differential']

print("=== 1. is the model compressed against REALITY, not just the market? ===")
print(f"  walk-forward games: {len(h)}")
print(f"  prediction  sd {p.std():.2f}   range {p.min():+.1f} to {p.max():+.1f}")
print(f"  actual      sd {a.std():.2f}   range {a.min():+.1f} to {a.max():+.1f}")
sl=np.polyfit(p,a,1)
print(f"  actual = {sl[0]:.2f} x prediction {sl[1]:+.1f}   corr {p.corr(a):+.3f}")
print(f"  -> slope ABOVE 1 means the model under-states; below 1 means over-states")

print("\n=== 2. mean actual outcome by predicted bucket ===")
h=h.copy(); h['b']=pd.cut(p,[-99,-21,-14,-7,0,7,14,21,99])
g=h.groupby('b',observed=True).agg(n=('home_score_differential','size'),
    pred=('preseason_model_preds','mean'), act=('home_score_differential','mean'))
g['gap']=g['act']-g['pred']
print(f"  {'predicted':<16}{'n':>6}{'mean pred':>11}{'mean actual':>13}{'gap':>8}")
for k,r in g.iterrows():
    print(f"  {str(k):<16}{int(r['n']):>6}{r['pred']:>11.1f}{r['act']:>13.1f}{r['gap']:>+8.1f}")

print("\n=== 3. are the RATINGS themselves compressed? ===")
S=pd.read_csv('../etl/summarize/results/season_summaries.csv',low_memory=False)
T=pd.read_csv('../etl/collect/collect_espn_teams/temp/teams.csv')
fbs=set(T.loc[T['fbs_ind']==1.0,'id'])
s=S[S.team_id.isin(fbs)&(S.season==2025)]
for c in ('adjusted_epa_per_pass_off','adjusted_epa_per_rush_off',
          'adjusted_rush_success_def'):
    if c in s.columns:
        v=s[c].dropna()
        print(f"  {c:<32} sd {v.std():.4f}  p5 {v.quantile(.05):+.3f}  "
              f"p95 {v.quantile(.95):+.3f}  ratio {v.quantile(.95)/max(abs(v.quantile(.05)),1e-9):.2f}")

print("\n=== 4. how far apart ARE the best and worst teams, really? ===")
g2=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
g2=g2.dropna(subset=['home_score_differential'])
g2=g2[g2.home_team_id.isin(fbs)&g2.away_team_id.isin(fbs)]
hh=g2[['season','home_team_id','home_score_differential']].copy(); hh.columns=['season','tid','m']
aa=g2[['season','away_team_id','home_score_differential']].copy(); aa.columns=['season','tid','m']; aa['m']*=-1
perf=pd.concat([hh,aa]).groupby(['season','tid'])['m'].mean().reset_index()
r=perf[perf.season==2025]['m']
print(f"  2025 team scoring margin: sd {r.std():.2f}  best {r.max():+.1f}  worst {r.min():+.1f}")
print(f"  implied best-vs-worst neutral gap: {r.max()-r.min():.1f} points")
print(f"  model's widest 2026 prediction: +38.4")

print("\n=== 5. does the FBS-only training censor blowouts? ===")
allg=pd.read_csv('../etl/summarize/temp/games.csv',low_memory=False)
allg=allg.dropna(subset=['home_score_differential'])
mixed=allg[~(allg.home_team_id.isin(fbs)&allg.away_team_id.isin(fbs))]
fbsonly=allg[allg.home_team_id.isin(fbs)&allg.away_team_id.isin(fbs)]
print(f"  FBS v FBS games   n={len(fbsonly):>6}  |margin| mean {fbsonly.home_score_differential.abs().mean():.1f}  "
      f"p99 {fbsonly.home_score_differential.abs().quantile(.99):.0f}")
print(f"  games w/ an FCS   n={len(mixed):>6}  |margin| mean {mixed.home_score_differential.abs().mean():.1f}  "
      f"p99 {mixed.home_score_differential.abs().quantile(.99):.0f}")
print(f"  -> the model trains ONLY on the first row; the market prices both")
PY
