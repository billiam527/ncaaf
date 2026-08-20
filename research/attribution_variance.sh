#!/usr/bin/env bash
# I have twice said the random-games holdout distorts SHAP attribution, on the
# evidence that the production fit's talent curve is badly non-monotone while a
# 2012-2024 fit's is clean. Those two models differ in more than the split rule:
# training window, whether 2025 is included, and which rows the random 10% took.
#
# And there is a simpler candidate I did not rule out. The model runs with
# subsample 0.51 and colsample_bytree 0.49, so two fits on IDENTICAL data with
# different seeds are different models. If the curve moves that much between
# seeds, attribution is just noisy and the claim about the split is unsupported.
#
# Train the same config on the same rows with five seeds and look.
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/research
python - <<'PY'
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

os.environ['DIFFERENTIAL_ENCODING'] = '0'
sys.path.insert(0, '/home/bill/ncaaf/research')
sys.path.insert(0, '/home/bill/ncaaf/batch_prediction')
import encoding_experiment as E

R = '/home/bill/ncaaf'
COL = 'talent_roll_pct_diff'
BANDS = [(-0.60, -0.30), (-0.30, -0.10), (-0.10, -0.02), (-0.02, 0.02),
         (0.02, 0.10), (0.10, 0.30), (0.30, 0.60)]

ss = pd.read_csv(f'{E.RESULTS}/season_summaries.csv')
games = E.load_games()
cache = {}
for s in range(2015, 2026):
    b = E.build_season(s, ss, games)
    if b is not None:
        cache[s] = b
cols = list(cache[max(cache)][0].columns)
TRAIN = [s for s in cache if s < 2025]

Xtr = pd.concat([E.transform(cache[s][0], 'diff+decay', cols) for s in TRAIN],
                ignore_index=True)
ytr = pd.concat([cache[s][1] for s in TRAIN], ignore_index=True)
Xte = E.transform(cache[2025][0], 'diff+decay', cols)
sc = StandardScaler().fit(Xtr)
names = list(Xte.columns)
i = names.index(COL)
Ste, Str = sc.transform(Xte), sc.transform(Xtr)
print(f"identical training rows every time: {len(Xtr):,} games, "
      f"scored on {len(Xte)}\n")


def curve(seed):
    p = dict(E.PARAMS)
    p['random_state'] = seed
    m = XGBRegressor(**p).fit(Str, ytr)
    c = m.get_booster().predict(
        xgb.DMatrix(Ste, feature_names=names), pred_contribs=True)[:, i]
    df = pd.DataFrame({'v': Xte[COL].to_numpy(), 'c': c})
    out = []
    for lo, hi in BANDS:
        x = df[(df.v >= lo) & (df.v < hi)]
        out.append(x['c'].mean() if len(x) >= 15 else np.nan)
    return np.array(out)


curves = np.array([curve(s) for s in (0, 1, 2, 3, 4)])
print(f"  {'band':<20}" + ''.join(f"{'seed ' + str(s):>10}" for s in range(5))
      + f"{'spread':>9}")
for j, (lo, hi) in enumerate(BANDS):
    row = curves[:, j]
    if np.isnan(row).all():
        continue
    print(f"  {f'{lo:+.2f} to {hi:+.2f}':<20}"
          + ''.join(f"{v:>10.2f}" for v in row)
          + f"{np.nanmax(row) - np.nanmin(row):>9.2f}")


def wrong_way(c):
    v = [x for x in c if not np.isnan(x)]
    return sum(1 for a, b in zip(v, v[1:]) if b < a), len(v) - 1


print(f"\n  {'seed':<8}{'wrong-way steps':>18}")
for s in range(5):
    bad, n = wrong_way(curves[s])
    print(f"  {s:<8}{f'{bad} of {n}':>18}")

print(f"\n  mean band-to-band spread across seeds: "
      f"{np.nanmean(np.nanmax(curves, axis=0) - np.nanmin(curves, axis=0)):.2f} points")
print("  If that is comparable to the differences between fits, the curve is")
print("  fitting variance and the split rule was never the explanation.")
PY
