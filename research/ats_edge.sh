#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction
python - <<'PY'
import numpy as np, pandas as pd
d = pd.read_csv('prediction_file/predictions_with_lines.csv', index_col=0)
d = d[d.spread.notna()].copy()

print("Is the 'edge' real disagreement, or just model shrinkage?\n")
r = np.corrcoef(d.market_margin, d.edge)[0, 1]
print(f"  corr(market_margin, edge) = {r:+.3f}")
print("  near 0 -> genuine game-by-game disagreement")
print("  strongly negative -> the model simply under-predicts big favourites\n")

slope = np.polyfit(d.market_margin, d.blended_prediction, 1)[0]
print(f"  regression of model on market: slope = {slope:.3f}")
print(f"  a slope of 1.0 would mean matching the market's scale;")
print(f"  {slope:.2f} means the model compresses the range by {(1-slope)*100:.0f}%\n")

print(f"{'market range':<18}{'n':>4}{'mean model':>12}{'mean market':>13}{'mean edge':>11}")
print("-" * 60)
for lo, hi, lab in [(-99, 5, 'close (<5)'), (5, 15, 'moderate 5-15'),
                    (15, 25, 'large 15-25'), (25, 99, 'blowout 25+')]:
    g = d[(d.market_margin.abs() >= lo) & (d.market_margin.abs() < hi)]
    if len(g):
        print(f"  {lab:<16}{len(g):>4}{g.blended_prediction.mean():>12.1f}"
              f"{g.market_margin.mean():>13.1f}{g.edge.mean():>11.1f}")
print("-" * 60)
print(f"\n  model std {d.blended_prediction.std():.1f} vs market std {d.market_margin.std():.1f}")
PY
