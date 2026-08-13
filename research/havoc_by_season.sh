#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/etl/summarize || exit 1
python - <<'PY'
import pandas as pd, numpy as np

h=pd.read_csv('results/havoc.csv',low_memory=False)
a=pd.read_csv('results/havoc_adjusted.csv',low_memory=False)
t=pd.read_csv('../collect/collect_espn_teams/temp/teams.csv')
fbs=set(t.loc[t['fbs_ind']==1.0,'id'])
name=dict(zip(t['id'],t['location']))

# raw season rates: weight each game by the snaps it contributed, so a
# team-season is the pooled rate rather than a mean of game rates
g=pd.read_csv('temp/games.csv',low_memory=False)[['id','season','home_team_id','away_team_id']].dropna()
for c in g.columns: g[c]=pd.to_numeric(g[c],errors='coerce')
g=g.dropna().rename(columns={'id':'game_id'}).drop_duplicates('game_id')

RATES=['tfl_rate','sack_rate','pass_defensed_rate','interception_rate',
       'fumble_rate','third_down_stop_rate','fourth_down_stop_rate',
       'redzone_stop_rate']
h=h[h.team_id.isin(fbs)]
raw=h.groupby(['team_id','season']).apply(
    lambda x: pd.Series({r: np.average(x[r].dropna(),
                          weights=x.loc[x[r].notna(),'def_plays'])
                         if x[r].notna().any() else np.nan for r in RATES}
                        | {'games': len(x), 'def_plays': x['def_plays'].sum()})
).reset_index()

adj_cols=[c for c in a.columns if c.startswith('adjusted_')]
a2=a[['team_id','season']+adj_cols].copy()
for c in ('team_id','season'):
    a2[c]=pd.to_numeric(a2[c],errors='coerce')
out=raw.merge(a2,on=['team_id','season'],how='left')
out.insert(0,'team',out['team_id'].map(name))
out=out.sort_values(['season','team']).reset_index(drop=True)

# tidier names on the adjusted side
out=out.rename(columns={c: c.replace('adjusted_','adj_').replace('_def','')
                        for c in adj_cols})
path='results/havoc_by_team_season.csv'
out.to_csv(path,index=False)
print(f"wrote {path}")
print(f"  {len(out)} team-seasons, {int(out.season.min())}-{int(out.season.max())}")
print(f"  columns: {list(out.columns)}")

print(f"\n=== 2025 havoc leaders (raw sack rate) ===")
c=out[out.season==2025].nlargest(10,'sack_rate')
print(f"  {'team':<22}{'sack':>7}{'tfl':>7}{'PD':>7}{'INT':>7}{'3rd stop':>10}{'RZ stop':>9}")
for _,r in c.iterrows():
    print(f"  {str(r['team'])[:20]:<22}{r['sack_rate']:>7.3f}{r['tfl_rate']:>7.3f}"
          f"{r['pass_defensed_rate']:>7.3f}{r['interception_rate']:>7.3f}"
          f"{r['third_down_stop_rate']:>10.3f}{r['redzone_stop_rate']:>9.3f}")

print(f"\n=== how much does the opponent adjustment move things, 2025? ===")
s=out[(out.season==2025)&out['adj_sack'].notna()]
for raw_c,adj_c in (('sack_rate','adj_sack'),('tfl_rate','adj_tfl'),
                    ('third_down_stop_rate','adj_third_down_stop')):
    d=(s[adj_c]-s[raw_c])
    print(f"  {raw_c:<24} r={s[raw_c].corr(s[adj_c]):+.3f}  "
          f"mean shift {d.mean():+.4f}  mean |shift| {d.abs().mean():.4f}")
PY
