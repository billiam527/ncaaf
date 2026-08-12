#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from sklearn.isotonic import IsotonicRegression
import margin_distribution as MD

w = pd.read_csv('../model_training/model_blender/blended_model.csv')
w['wk'] = w['week'].str.extract(r'(\d+)').astype(int)
w = w.set_index('wk')

h = pd.read_csv('../analysis/backtest_expanding_preds.csv')
h = h.dropna(subset=['home_score_differential',
                     'preseason_model_preds','in_season_model_preds'])
h['blended'] = (h['week_num'].map(w['pre_szn_coefs']) * h['preseason_model_preds']
                + h['week_num'].map(w['in_szn_coefs']) * h['in_season_model_preds']
                + h['week_num'].map(w['intercepts']))

actual = h['home_score_differential'].to_numpy(float)
won = actual > 0
sw = MD.season_weights(h['test_season'].to_numpy(float), MD.HALF_LIFE)
games = MD.fbs_games()
weights = MD.key_number_weights(games, MD.HALF_LIFE)

def evaluate(fit_col, apply_col, label):
    iso = IsotonicRegression(out_of_bounds='clip').fit(
        h[fit_col], actual, sample_weight=sw)
    centre = iso.predict(h[apply_col])
    r = actual - centre
    mu = np.average(r, weights=sw)
    sigma = float(np.sqrt(np.average((r-mu)**2, weights=sw)))
    pw = np.array([MD.distribution(c, weights, sigma)[MD.GRID > 0].sum()
                   for c in centre])
    brier = np.mean((pw - won)**2)
    mae = np.abs(centre - actual).mean()
    print(f"  {label:<44} sigma {sigma:>5.2f}  MAE {mae:>5.2f}  Brier {brier:.4f}")
    return pw, brier, mae, sigma

print("=== calibrator fit on X, applied to Y ===")
cur = evaluate('in_season_model_preds', 'blended', 'CURRENT: fit in_season -> apply blended')
fix = evaluate('blended', 'blended', 'FIXED:   fit blended   -> apply blended')
ref = evaluate('in_season_model_preds', 'in_season_model_preds',
               'reference: fit in_season -> apply in_season')

print("\n=== P(home win) calibration bands ===")
for label, pw in (('CURRENT', cur[0]), ('FIXED', fix[0])):
    print(f"\n  {label}")
    print(f"    {'band':<14}{'n':>6}{'predicted':>11}{'actual':>9}{'gap':>8}")
    for lo, hi in zip([0,.2,.35,.5,.65,.8],[.2,.35,.5,.65,.8,1.01]):
        s = (pw >= lo) & (pw < hi)
        if s.sum() < 30: continue
        print(f"    {f'{lo:.0%}-{hi:.0%}':<14}{s.sum():>6}{pw[s].mean():>11.1%}"
              f"{won[s].mean():>9.1%}{won[s].mean()-pw[s].mean():>+8.1%}")
    worst = 0
    for lo, hi in zip([0,.2,.35,.5,.65,.8],[.2,.35,.5,.65,.8,1.01]):
        s = (pw >= lo) & (pw < hi)
        if s.sum() >= 30:
            worst = max(worst, abs(won[s].mean()-pw[s].mean()))
    print(f"    worst band error: {worst:.1%}")

print(f"\n=== summary ===")
print(f"  Brier  {cur[1]:.4f} -> {fix[1]:.4f}   ({fix[1]-cur[1]:+.4f})")
print(f"  MAE    {cur[2]:.2f} -> {fix[2]:.2f}   ({fix[2]-cur[2]:+.2f})")
print(f"  sigma  {cur[3]:.2f} -> {fix[3]:.2f}   ({fix[3]-cur[3]:+.2f})")
PY
