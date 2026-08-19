#!/usr/bin/env bash
# Does the isotonic calibrator earn its place?
#
# It was adopted on an in-sample reading of the leaked history (Brier 0.1960 ->
# 0.1918, MAE 13.48 -> 13.22) and has never been checked out of sample. Damping
# its correction toward the identity improves every measure monotonically, and
# the limit of that sweep is no calibrator at all:
#
#   centre                  MAE all  MAE mid  MAE tail   Brier  sigma
#   isotonic, unshrunk       12.730   12.685     13.42  0.1846  15.85
#   shrunk K=50              12.690   12.672     12.97  0.1844  15.90
#   shrunk K=200             12.672   12.659     12.87  0.1841  15.95
#   shrunk K=1000            12.661   12.648     12.85  0.1839  16.01
#   no calibrator at all     12.659   12.647     12.84  0.1839  16.04
#
# A monotone sweep across a smooth family is much stronger evidence than any
# two of those rows compared alone. The mechanism is plain: isotonic regression
# on ~4,200 points with no regularisation fits the noise, worst where the data
# is thinnest.
#
# Note the last column. Removing the calibrator also closes the sigma gap -
# 16.04 fitted against 16.07 realised. The "narrow sigma" was the calibrator's
# own in-sample overfit, not a property of the model.
#
# This pass confirms it season by season and separates the two changes.
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, '.')
sys.path.insert(0, '../model_training/model_blender')
import margin_distribution as MD
import model_blender as MB

h = pd.read_csv('../analysis/backtest_expanding_preds.csv')
h = h[h.week_num < 90]
h = h.dropna(subset=['home_score_differential', 'preseason_model_preds',
                     'in_season_model_preds'])
h['season'] = h['test_season']
kn = MD.key_number_weights(MD.fbs_games(), MD.HALF_LIFE)


def blend(frame, weights):
    w = weights.copy()
    w['w'] = w['week'].astype(str).str.extract(r'(\d+)').astype(float)
    w = w.dropna(subset=['w']).sort_values('w').set_index('w')
    k = frame['week_num'].astype(float).clip(upper=w.index.max())
    return (k.map(w['pre_szn_coefs']) * frame['preseason_model_preds']
            + k.map(w['in_szn_coefs']) * frame['in_season_model_preds']
            + k.map(w['intercepts'])).to_numpy(float)


def walk_forward(calibrate, use_offset):
    rows = []
    for s in sorted(h.season.unique())[3:]:
        tr, te = h[h.season < s], h[h.season == s]
        weights = MB.compare_results(tr)
        tb, eb = blend(tr, weights), blend(te, weights)
        ta = tr['home_score_differential'].to_numpy(float)
        ea = te['home_score_differential'].to_numpy(float)
        sw = MD.season_weights(tr['season'].to_numpy(float), MD.HALF_LIFE,
                               reference=s - 1)

        if calibrate:
            iso = IsotonicRegression(out_of_bounds='clip').fit(
                tb, ta, sample_weight=sw)
            tr_c, te_c = iso.predict(tb), iso.predict(eb)
        else:
            tr_c, te_c = tb, eb

        offset = 0.0
        if use_offset:
            m = tr.season.isin(
                sorted(tr.season.unique())[-MD.DRIFT_LOOKBACK:]).to_numpy()
            offset = float(np.mean(ta[m] - tr_c[m]))

        centre = te_c + offset
        r = ta - tr_c
        mu = np.average(r, weights=sw)
        sig = float(np.sqrt(np.average((r - mu) ** 2, weights=sw)))
        pw = np.array([MD.distribution(c, kn, sig)[MD.GRID > 0].sum()
                       for c in centre])
        won = ea > 0
        rows.append({'season': int(s), 'n': len(te),
                     'mae': float(np.abs(ea - centre).mean()),
                     'brier': float(np.mean((pw - won) ** 2)),
                     'bias': float(np.mean(ea - centre)),
                     'sigma': sig, 'realised': float((ea - centre).std()),
                     'pw': pw, 'won': won})
    return rows


VARIANTS = [
    ('isotonic + offset (current)', True, True),
    ('isotonic, no offset', True, False),
    ('no calibrator, no offset', False, False),
    ('no calibrator + offset', False, True),
]
store = {}
print(f"{'variant':<30}{'MAE':>8}{'Brier':>9}{'bias':>8}{'sigma':>8}"
      f"{'realised':>10}{'narrow by':>11}")
print("-" * 84)
for label, c, o in VARIANTS:
    r = walk_forward(c, o)
    store[label] = r
    d = pd.DataFrame([{k: v for k, v in x.items() if k not in ('pw', 'won')}
                      for x in r])
    n = d['n'].sum()
    wm = lambda col: (d[col] * d['n']).sum() / n
    print(f"{label:<30}{wm('mae'):>8.3f}{wm('brier'):>9.4f}{wm('bias'):>+8.2f}"
          f"{wm('sigma'):>8.2f}{wm('realised'):>10.2f}"
          f"{wm('realised') - wm('sigma'):>+11.2f}")

print("\n=== paired per season: does removing the calibrator win every time? ===")
a = store['isotonic + offset (current)']
b = store['no calibrator + offset']
print(f"  {'season':>7}{'MAE with':>10}{'MAE without':>13}{'diff':>8}"
      f"{'Brier with':>12}{'Brier without':>15}{'diff':>9}")
for x, y in zip(a, b):
    print(f"  {x['season']:>7}{x['mae']:>10.3f}{y['mae']:>13.3f}"
          f"{y['mae'] - x['mae']:>+8.3f}{x['brier']:>12.4f}"
          f"{y['brier']:>15.4f}{y['brier'] - x['brier']:>+9.4f}")
print("  (negative diff = removing the calibrator is better)")

print("\n=== P(home win) calibration, no calibrator + offset ===")
pw = np.concatenate([x['pw'] for x in b])
won = np.concatenate([x['won'] for x in b])
print(f"  {'band':<14}{'n':>6}{'predicted':>11}{'actual':>9}{'gap':>8}")
worst = 0.0
for lo, hi in zip([0, .2, .35, .5, .65, .8], [.2, .35, .5, .65, .8, 1.01]):
    m = (pw >= lo) & (pw < hi)
    if m.sum() < 30:
        continue
    gap = won[m].mean() - pw[m].mean()
    worst = max(worst, abs(gap))
    print(f"  {f'{lo:.0%}-{hi:.0%}':<14}{m.sum():>6}{pw[m].mean():>11.1%}"
          f"{won[m].mean():>9.1%}{gap:>+8.1%}")
print(f"  worst band {worst:.1%}   Brier {np.mean((pw - won) ** 2):.4f}")
PY
