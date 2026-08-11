#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd, numpy as np

r = pd.read_csv('/home/bill/ncaaf/preseason_lag_ablation.csv')
g = r[r.variant == 'all three (current)']
model_mae = (g['mae'] * g['n']).sum() / g['n'].sum()

# Rebuild the constant baselines honestly: MAE is minimised by the median.
games = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
teams = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(teams.loc[teams['fbs_ind'] == 1.0, 'id'])
games = games[games.home_team_id.isin(fbs) & games.away_team_id.isin(fbs)]
games = games[games['home_score_differential'].notna()]

rows = []
for S in sorted(g['season'].unique()):
    tr = games[(games.season < S) & (games.season >= 2012)]['home_score_differential']
    te = games[games.season == S]['home_score_differential']
    rows.append({
        'season': S, 'n': len(te),
        'mean_const':   np.abs(tr.mean() - te).mean(),
        'median_const': np.abs(tr.median() - te).mean(),
        'zero':         np.abs(0 - te).mean(),
    })
b = pd.DataFrame(rows)
w = lambda c: (b[c] * b['n']).sum() / b['n'].sum()

print("Constant-prediction baselines vs the 72-feature preseason model\n")
print(f"  predict 0                {w('zero'):.3f}")
print(f"  predict training mean    {w('mean_const'):.3f}   (what the ablation used)")
print(f"  predict training median  {w('median_const'):.3f}   (the true MAE floor)")
print(f"  preseason model, 72 cols {model_mae:.3f}")
print(f"\n  model beats best constant by {min(w('zero'), w('mean_const'), w('median_const')) - model_mae:+.3f} MAE")

print("\n\nBut MAE is not the only lens. Side-picking:")
print(f"  always pick home  57.6%   (a constant can only ever pick one side)")
print(f"  preseason model   61.4%")
print(f"  in-season model   65.1%")
print(f"  blended           68.1%")
PY
