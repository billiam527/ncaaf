#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

d = pd.read_csv('/home/bill/ncaaf/backtest_expanding_preds.csv')
d = d[d.week_num < 90].dropna(
    subset=['preseason_model_preds', 'in_season_model_preds', 'home_score_differential'])

b = pd.read_csv('/home/bill/ncaaf/model_training/model_blender/blended_model.csv', index_col=0)
b['wk'] = b['week'].str.extract(r'(\d+)').astype(int)
b = b.set_index('wk')

d = d[d.week_num.isin(b.index)].copy()
co = b.loc[d.week_num]
d['blend']    = (co.pre_szn_coefs.values * d.preseason_model_preds
                 + co.in_szn_coefs.values * d.in_season_model_preds
                 + co.intercepts.values)
d['no_icept'] = (co.pre_szn_coefs.values * d.preseason_model_preds
                 + co.in_szn_coefs.values * d.in_season_model_preds)

bins   = [-100, -14, -7, -3, 3, 7, 14, 100]
labels = ['road fav 14+', 'road fav 7-14', 'road fav 3-7', 'toss-up',
          'home fav 3-7', 'home fav 7-14', 'home fav 14+']
d['bucket'] = pd.cut(d['blend'], bins=bins, labels=labels)

print("Residual bias by predicted margin  (bias = predicted - actual;")
print("positive means the model over-favours the home team)\n")
print(f"{'bucket':<15} {'n':>5} {'mean pred':>10} {'mean actual':>12} "
      f"{'bias':>8} {'MAE w/':>8} {'MAE w/o':>9}")
print("-" * 74)
for lab in labels:
    g = d[d.bucket == lab]
    if len(g) < 20:
        continue
    bias = (g.blend - g.home_score_differential).mean()
    mae_w = (g.blend - g.home_score_differential).abs().mean()
    mae_wo = (g.no_icept - g.home_score_differential).abs().mean()
    print(f"{lab:<15} {len(g):>5} {g.blend.mean():>10.2f} "
          f"{g.home_score_differential.mean():>12.2f} {bias:>+8.2f} "
          f"{mae_w:>8.2f} {mae_wo:>9.2f}")

overall_w = (d.blend - d.home_score_differential).abs().mean()
overall_wo = (d.no_icept - d.home_score_differential).abs().mean()
print("-" * 74)
print(f"{'ALL':<15} {len(d):>5} {'':>10} {'':>12} "
      f"{(d.blend - d.home_score_differential).mean():>+8.2f} "
      f"{overall_w:>8.2f} {overall_wo:>9.2f}")

print("\n\nIs the bias flat across buckets? (a constant intercept assumes it is)")
sub = d.dropna(subset=['bucket'])
r = np.corrcoef(sub['blend'], sub['blend'] - sub['home_score_differential'])[0, 1]
print(f"  corr(prediction, bias) = {r:+.3f}")
print("  near 0 -> a constant shift is the right correction")
print("  far from 0 -> the bias scales with the prediction, so a slope is needed")

print("\n\nHeavy ROAD favourites specifically (predicted <= -14):")
g = d[d.blend <= -14]
print(f"  n={len(g)}  mean predicted {g.blend.mean():+.2f}  "
      f"mean actual {g.home_score_differential.mean():+.2f}")
print(f"  MAE with intercept {(g.blend-g.home_score_differential).abs().mean():.2f}  "
      f"vs without {(g.no_icept-g.home_score_differential).abs().mean():.2f}")
PY
