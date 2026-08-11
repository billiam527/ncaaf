#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction
python - <<'PY'
import numpy as np, pandas as pd
import margin_distribution as MD

h = MD.load_history()
raw = h['in_season_model_preds'].to_numpy(float)
act = h['home_score_differential'].to_numpy(float)
cal = MD.fit_calibrator()

print("=== step 1: what the model says vs what actually happens ===")
print(f"{'model says':>14}{'n':>7}{'actual mean':>14}{'shrunk by':>12}")
print("-" * 48)
for lo, hi in [(-99,-25),(-25,-15),(-15,-8),(-8,-3),(-3,3),(3,8),(8,15),(15,25),(25,99)]:
    s = (raw >= lo) & (raw < hi)
    if s.sum() < 40: continue
    lab = f"{lo:+d} to {hi:+d}" if abs(lo) < 90 else (f"< {hi:+d}" if lo < -90 else f"> {lo:+d}")
    print(f"{lab:>14}{s.sum():>7}{act[s].mean():>14.1f}"
          f"{raw[s].mean() - act[s].mean():>+12.1f}")

print("\n=== step 2: the fitted mapping (isotonic) ===")
print(f"{'raw':>8}{'calibrated':>13}{'change':>9}")
print("-" * 32)
for p in (-35,-25,-15,-10,-5,0,5,10,15,20,25,30,35):
    c = float(cal.predict([p])[0])
    print(f"{p:>+8d}{c:>13.1f}{c-p:>+9.1f}")

print("\n=== step 3: what it does and does not change ===")
c_all = cal.predict(raw)
print(f"  ordering (spearman raw vs calibrated): "
      f"{pd.Series(raw).corr(pd.Series(c_all), method='spearman'):.4f}")
print(f"  MAE            raw {np.abs(raw-act).mean():.2f}  ->  calibrated {np.abs(c_all-act).mean():.2f}")
print(f"  spread (sd)    raw {raw.std():.2f}  ->  calibrated {c_all.std():.2f}   (actual {act.std():.2f})")
print(f"  bias           raw {np.mean(raw-act):+.2f}  ->  calibrated {np.mean(c_all-act):+.2f}")

print("\n=== step 4: is it still honest after calibration? ===")
print(f"{'calibrated says':>17}{'n':>7}{'actual mean':>14}{'gap':>8}")
print("-" * 48)
for lo, hi in [(-99,-15),(-15,-8),(-8,-3),(-3,3),(3,8),(8,15),(15,99)]:
    s = (c_all >= lo) & (c_all < hi)
    if s.sum() < 40: continue
    lab = f"{lo:+d} to {hi:+d}" if abs(lo) < 90 else (f"< {hi:+d}" if lo < -90 else f"> {lo:+d}")
    print(f"{lab:>17}{s.sum():>7}{act[s].mean():>14.1f}{c_all[s].mean()-act[s].mean():>+8.1f}")
PY
