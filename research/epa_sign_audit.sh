#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd, numpy as np
f = '/home/bill/ncaaf/etl/format/format_espn_pbp/temp/pbp_edit.csv'
d = pd.read_csv(f, nrows=400000, low_memory=False)
d['epa'] = pd.to_numeric(d['epa'], errors='coerce')

print("=== sanity check on OFFENSIVE plays (these feed the 12 model features) ===")
off = d[d.offensive_play == 1]

# touchdowns
td = off[off['play_type_text'].astype(str).str.contains('Touchdown', case=False, na=False)]
print(f"  offensive TD plays        n={len(td):>6}  mean EPA {td.epa.mean():>+7.3f}")

# turnovers
for pat in ('Interception', 'Fumble Recovery (Opponent)'):
    t = off[off['play_type_text'].astype(str).str.contains(pat, case=False, na=False)]
    if len(t):
        print(f"  {pat[:24]:<24}  n={len(t):>6}  mean EPA {t.epa.mean():>+7.3f}")

print("\n=== the modelled features: rush / pass EPA ===")
for col, lab in (('rushing_play','rush'), ('passing_play','pass')):
    if col in d.columns:
        s = d[(d[col] == 1)]
        print(f"  {lab:<6} n={len(s):>7}  mean EPA {s.epa.mean():>+7.4f}  sd {s.epa.std():.3f}")

print("\n=== do rush/pass overlap with special teams at all? ===")
for col in ('rushing_play','passing_play'):
    if col in d.columns:
        n = int(((d[col] == 1) & (d.special_teams_play == 1)).sum())
        print(f"  {col} & special_teams_play: {n}")

print("\n=== does a team's EPA correlate with winning, by play group? ===")
d['pt'] = np.where(d.special_teams_play == 1, 'st',
          np.where(d.offensive_play == 1, 'off', 'other'))
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
marg = {}
for _, r in g.dropna(subset=['home_score_differential']).iterrows():
    marg[(r['id'], r['home_team_id'])] = r['home_score_differential']
    marg[(r['id'], r['away_team_id'])] = -r['home_score_differential']

for grp in ('off','st'):
    sub = d[d.pt == grp]
    pg = sub.groupby(['game_id','team_id'])['epa'].sum().reset_index()
    pg['margin'] = [marg.get((a,b), np.nan) for a,b in zip(pg.game_id, pg.team_id)]
    pg = pg.dropna()
    if len(pg) > 100:
        print(f"  {grp:<4} EPA vs own margin: r={pg.epa.corr(pg.margin):>+.3f}  "
              f"(n={len(pg)}, sd of per-game EPA {pg.epa.std():.1f})")
PY
