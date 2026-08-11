#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd

print("=== game_by_game_summaries (per-game inputs for the adjuster) ===")
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/results/game_by_game_summaries.csv',
                low_memory=False, nrows=5)
print("  cols:", [c for c in g.columns])

print()
print("=== does it have week? and 2025 coverage ===")
g2 = pd.read_csv('/home/bill/ncaaf/etl/summarize/results/game_by_game_summaries.csv',
                 low_memory=False)
print("  has 'week':", 'week' in g2.columns)
s25 = g2[g2.season == 2025]
print(f"  2025 rows: {len(s25)}  unique games: {s25.game_id.nunique()}  teams: {s25.team_id.nunique()}")

print()
print("=== games.csv: week + actual scores (for truth) ===")
gm = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
score_cols = [c for c in gm.columns if 'score' in c.lower() or c == 'week']
print("  score/week cols:", score_cols)
g25 = gm[gm.season == 2025]
print(f"  2025 games: {len(g25)}  weeks: {sorted(g25.week.dropna().unique())[:20]}")
if 'home_score_differential' in gm.columns:
    print("  home_score_differential non-null 2025:", g25['home_score_differential'].notna().sum())
PY
