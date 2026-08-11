#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

h = pd.read_csv('/home/bill/ncaaf/analysis/backtest_expanding_preds.csv')
h = h[h.week_num < 90].dropna(subset=['in_season_model_preds','home_score_differential'])
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)

h = h.merge(g[['id','neutral_site','venue_id']].drop_duplicates('id'), on='id', how='left') \
     if 'neutral_site' in g.columns else h
h['resid'] = h['home_score_differential'] - h['in_season_model_preds']
print(f"{len(h)} games, residual sd {h.resid.std():.2f}\n")

print("=== 1. HOME FIELD: is average HFA already captured? ===")
print(f"  mean residual overall: {h.resid.mean():+.2f} points")
if 'neutral_site' in h.columns:
    for v, lab in [(0,'true home game'), (1,'neutral site')]:
        s = h[h.neutral_site == v]
        if len(s) > 40:
            print(f"  {lab:<18} n={len(s):>5}  mean residual {s.resid.mean():+.2f}")
    print("  -> a gap here means neutral-site games are mispriced")

print("\n=== 2. TEAM-SPECIFIC home advantage (is some HFA unexploited?) ===")
by = h.groupby('home_team_id')['resid'].agg(['mean','count'])
by = by[by['count'] >= 40].sort_values('mean')
print(f"  {len(by)} teams with 40+ home games")
print(f"  spread of team home residuals: sd {by['mean'].std():.2f} points")
print(f"  worst 3: {[round(x,1) for x in by['mean'].head(3)]}   "
      f"best 3: {[round(x,1) for x in by['mean'].tail(3)]}")
# split-half reliability: does a team's home edge persist?
h['half'] = (h.test_season >= h.test_season.median()).astype(int)
a = h[h.half==0].groupby('home_team_id')['resid'].mean()
b = h[h.half==1].groupby('home_team_id')['resid'].mean()
both = pd.concat([a,b],axis=1,keys=['early','late']).dropna()
both = both[both.index.isin(by.index)]
print(f"  early-vs-late correlation across teams: {both['early'].corr(both['late']):+.3f}")
print("  -> near 0 means team HFA differences are mostly noise; high means real & exploitable")

print("\n=== 3. REST: days since each team's previous game ===")
gg = g.dropna(subset=['home_score_differential']).copy()
gg['dt'] = pd.to_datetime(gg['date'], errors='coerce', utc=True)
long = pd.concat([
    gg[['id','season','dt','home_team_id']].rename(columns={'home_team_id':'team'}),
    gg[['id','season','dt','away_team_id']].rename(columns={'away_team_id':'team'})])
long = long.sort_values(['season','team','dt'])
long['rest'] = long.groupby(['season','team'])['dt'].diff().dt.days
rest_h = long.rename(columns={'team':'home_team_id','rest':'home_rest'})[['id','home_team_id','home_rest']]
rest_a = long.rename(columns={'team':'away_team_id','rest':'away_rest'})[['id','away_team_id','away_rest']]
h = h.merge(rest_h, on=['id','home_team_id'], how='left').merge(rest_a, on=['id','away_team_id'], how='left')
h['rest_diff'] = h['home_rest'] - h['away_rest']
s = h.dropna(subset=['rest_diff'])
print(f"  n with rest data: {len(s)}")
print(f"  corr(rest_diff, residual): {s['rest_diff'].corr(s['resid']):+.4f}")
for lo,hi,lab in [(-99,-4,'home on short rest'),(-4,4,'similar'),(4,99,'home extra rest')]:
    x = s[(s.rest_diff>=lo)&(s.rest_diff<hi)]
    if len(x)>60:
        print(f"    {lab:<20} n={len(x):>5}  mean residual {x.resid.mean():+.2f}")

print("\n=== 4. WEEK OF SEASON (proxy for roster settling / model staleness) ===")
print(f"  corr(week, residual): {h['week_num'].corr(h['resid']):+.4f}")
for lo,hi in [(2,5),(6,9),(10,14)]:
    x = h[(h.week_num>=lo)&(h.week_num<=hi)]
    print(f"    weeks {lo}-{hi}: n={len(x):>5}  mean residual {x.resid.mean():+.2f}  "
          f"sd {x.resid.std():.2f}")

print("\n=== what would each be worth? (residual sd is the ceiling) ===")
print(f"  current residual sd: {h.resid.std():.2f} points")
print("  a factor explaining X% of residual variance cuts sd by 1-sqrt(1-X)")
PY
