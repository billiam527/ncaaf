#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np
T='etl/collect/collect_cfbd_players/temp/'

print("=== 1. WHY is defensive coverage only 77%? ===")
stats = pd.read_csv(T+'cfbd_stats.csv', low_memory=False)
d = stats[(stats.category=='defensive') & (stats.statType=='TOT')]
per = d.groupby('season').agg(rows=('playerId','size'), teams=('team','nunique'))
print(per.to_string())

f = pd.read_csv('etl/summarize/results/returning_production.csv', low_memory=False)
cov = f.groupby('season')[['ret_DB_starter','ret_QB_starter']].apply(
    lambda x: pd.Series({'DB': x['ret_DB_starter'].notna().mean(),
                         'QB': x['ret_QB_starter'].notna().mean()}))
print("\n  coverage of starter-tier features by season:")
print((cov*100).round(1).to_string())

print("\n=== 2. is returning production REDUNDANT with what the model already knows? ===")
# the preseason model's inputs are 3 prior seasons of opponent-adjusted stats
s = pd.read_csv('etl/summarize/results/season_summaries.csv', low_memory=False)
print(f"  season_summaries: {len(s)} rows, {len(s.columns)} cols")
adj = [c for c in s.columns if c.startswith('adjusted_')]
print(f"  adjusted feature columns: {len(adj)}")

g = pd.read_csv('etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind']==1.0,'id'])
g = g.dropna(subset=['home_score_differential'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
h = g[['season','home_team_id','home_score_differential']].copy(); h.columns=['season','team_id','m']
a = g[['season','away_team_id','home_score_differential']].copy(); a.columns=['season','team_id','m']; a['m']*=-1
perf = pd.concat([h,a]).groupby(['season','team_id'])['m'].mean().reset_index()
perf.columns=['season','team_id','avg_margin']

# build the "what the model already has": prior 3 seasons of adjusted stats
key = 'team_id' if 'team_id' in s.columns else s.columns[0]
print(f"  joining season_summaries on: {key}")
lags = []
for lag in (1,2,3):
    l = s.copy()
    l['season'] = l['season'] + lag
    l = l[[key,'season'] + adj].rename(columns={c: f'{c}_L{lag}' for c in adj})
    lags.append(l)

d2 = perf.copy()
for l in lags:
    d2 = d2.merge(l, left_on=['team_id','season'], right_on=[key,'season'], how='left')
prior = perf.copy(); prior['season'] += 1
prior = prior.rename(columns={'avg_margin':'prior'})
d2 = d2.merge(prior, on=['season','team_id'], how='left')
d2 = d2.merge(f, on=['season','team_id'], how='inner')
lagcols = [c for c in d2.columns if '_L1' in c or '_L2' in c or '_L3' in c]
retcols = sorted(c for c in d2.columns if c.startswith('ret_')) + \
          ['portal_in','portal_out','draft_departures']
use = d2.dropna(subset=['avg_margin','prior'] + lagcols + retcols)
print(f"  usable rows: {len(use)}   lag features: {len(lagcols)}   returning: {len(retcols)}")

if len(use) > 200:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    y = use['avg_margin']
    sets = [('prior margin only', ['prior']),
            ('model lag features', lagcols),
            ('lags + prior', lagcols + ['prior']),
            ('lags + prior + RETURNING', lagcols + ['prior'] + retcols)]
    print(f"\n  {'feature set':<32}{'feat':>6}{'CV R2':>10}")
    print("  " + "-"*50)
    prev = None
    for lab, cols in sets:
        r2 = cross_val_score(Ridge(alpha=10.0), use[cols], y, cv=5, scoring='r2').mean()
        delta = f"  ({r2-prev:+.4f})" if prev is not None else ""
        print(f"  {lab:<32}{len(cols):>6}{r2:>10.4f}{delta}")
        prev = r2
    print("\n  -> the last line is the question: does returning production add anything")
    print("     ON TOP of what the model already sees?")
PY
