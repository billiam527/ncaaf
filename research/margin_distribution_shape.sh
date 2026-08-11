#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

# Real FBS margin distribution
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
real = g['home_score_differential'].dropna().round().astype(int).to_numpy()

p = pd.read_csv('/home/bill/ncaaf/batch_prediction/prediction_file/new_predictions.csv',
                index_col=0)
pred = p['blended_prediction'].dropna().to_numpy()

# Quantile mapping: a prediction at the q-th percentile of predictions becomes
# the q-th percentile of REAL margins. Ordering is preserved exactly; the
# resulting histogram matches reality, key numbers included.
ranks = pd.Series(pred).rank(pct=True, method='average').to_numpy()
mapped = np.quantile(real, np.clip(ranks, 0, 1)).round().astype(int)

key = {3, 7, 10, 14, 17, 21}
def report(name, vals):
    a = np.abs(np.round(vals).astype(int))
    onkey = np.isin(a, list(key)).mean()
    print(f"  {name:<22} sd {np.std(vals):5.1f}   on key numbers {onkey:5.1%}   "
          f"range {vals.min():+.0f}..{vals.max():+.0f}")

print("=== distribution shape ===")
report('real margins', real.astype(float))
report('raw predictions', pred)
report('quantile-mapped', mapped.astype(float))

print("\n=== what the mapping does to individual games ===")
idx = np.argsort(pred)
sel = [idx[1], idx[len(idx)//4], idx[len(idx)//2], idx[3*len(idx)//4], idx[-2]]
names = p['short_name'].dropna().to_numpy()
print(f"  {'matchup':<22}{'raw':>8}{'mapped':>8}")
for i in sel:
    print(f"  {str(names[i])[:21]:<22}{pred[i]:>8.1f}{mapped[i]:>8d}")

print("\n=== most common mapped values ===")
vc = pd.Series(mapped).value_counts().head(8).sort_index()
print("  " + ", ".join(f"{k:+d}:{v}" for k, v in vc.items()))

print("\n=== ordering preserved? ===")
print(f"  spearman(raw, mapped) = {pd.Series(pred).corr(pd.Series(mapped), method='spearman'):.4f}")
PY
