#!/usr/bin/env bash
# Should the in-season hyperparameters be re-derived?
#
# The premise for asking: the shipped config was chosen when this model had 12
# features and could see its own season's final stats. It now has 72 and cannot.
# Parameters picked for one problem, applied to another.
#
# The premise held for the TUNER, which is the real finding here.
# tune_hyperparams.inseason_season still joined season_summaries on
# (team_id, season) - the same end-of-year leak preprocess.py was rebuilt to
# remove - so a --search run would have tuned the old leaking 12-feature problem
# and handed the answer to the new one. It now builds the frame through
# preprocess.edit_files, so there is one implementation.
#
# The premise did NOT hold for the parameters. Two searches, both scored
# walk-forward on 2022-2025 with training on earlier seasons only:
#
#   stock random grid, 61 configs
#     library defaults   14.805
#     incumbent          13.540
#     best draw          13.497   -0.043 against the incumbent
#
#   Paired on the same 2,941 games, that -0.043 carries a standard error of
#   0.022, so t = 1.97. But it is the argmax of 61 draws, and the argmax of 61
#   draws reaches two standard errors by chance routinely. The top five configs
#   spanned 0.034, less than the winning margin. Not evidence.
#
#   focused sweep of the slow region, 24 configs
#     Many trees, low learning rate, real regularisation - roughly where the
#     preseason model was hand-tuned to (1800 trees at 0.0045, reg_alpha 0.5),
#     and a region the stock grid never samples: it stops at 700 trees and 0.01,
#     and has no reg_alpha at all.
#
#     configs that beat the incumbent   15 of 24
#     mean difference                   -0.011 MAE
#     median difference                 -0.032 MAE
#     worst / best                      +0.123 / -0.081
#
#   The top eight all beat the incumbent, which looks like a regional effect
#   rather than a lucky draw - and a region being better is not subject to the
#   argmax bias. But the full sample says otherwise: 15 of 24 is close to a coin
#   flip and the mean is -0.011. Reading the top of a sorted list is the same
#   selection error the paired test was written to avoid.
#
#   A CENTRAL member of the region, chosen for being typical rather than for
#   winning, scored -0.078 at t = -2.19 - and then failed the season breakdown:
#
#       season   incumbent   central     diff
#         2022      13.642    13.655   +0.013
#         2023      13.529    13.461   -0.068
#         2024      13.463    13.448   -0.014
#         2025      13.532    13.298   -0.234
#
#   Three seasons tied, one season carried the whole result.
#
#   The best draw from that region does win all four seasons at -0.081, which
#   is the strongest showing anywhere in this investigation. It is still the
#   argmax of the sweep, and the sweep's own average says -0.011.
#
# CONCLUSION: keep the incumbent. Having tuned parameters is worth 1.265 MAE
# over library defaults at t = 9.87, so the tuning matters enormously; RE-tuning
# it gains nothing that survives being looked at properly. tune_hyperparams now
# requires MIN_IMPROVEMENT before it will replace a shipped config.
#
# There may be 0.05-0.08 MAE in the slow region. Settling it honestly needs a
# larger sweep scored on a season held out of the sweep, then a walk-forward
# blend run - because this model is judged on the blend, and a 0.08 change to
# one component at t = 2 will not show up there.
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/model_training
python - <<'PY'
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, '.')
import tune_hyperparams as T

FEATURES = ('rush_success_rolling_avg,rush_success_def_rolling_avg,'
            'pass_success_rolling_avg,pass_success_def_rolling_avg,'
            'explosive_rush_rate_rolling_avg,explosive_rush_rate_def_rolling_avg,'
            'explosive_pass_rate_rolling_avg,explosive_pass_rate_def_rolling_avg,'
            'epa_per_rush_rolling_avg,epa_per_rush_def_rolling_avg,'
            'epa_per_pass_rolling_avg,epa_per_pass_def_rolling_avg').split(',')

INCUMBENT = {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.01,
             "min_child_weight": 25, "subsample": 0.6, "colsample_bytree": 0.6,
             "reg_lambda": 1.0}
CANDIDATES = {
    'grid best': {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.01,
                  "min_child_weight": 25, "subsample": 0.6,
                  "colsample_bytree": 0.4, "reg_lambda": 20.0},
    'slow best': {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.008,
                  "min_child_weight": 12, "subsample": 0.5,
                  "colsample_bytree": 0.35, "reg_lambda": 25.0,
                  "reg_alpha": 0.0},
    'slow central': {"n_estimators": 1200, "max_depth": 4,
                     "learning_rate": 0.005, "min_child_weight": 25,
                     "subsample": 0.6, "colsample_bytree": 0.45,
                     "reg_lambda": 12.0, "reg_alpha": 0.0},
    'defaults': {},
}

rolling = pd.read_csv(T.ROLLING, low_memory=False)
games = T.load_games()
cache = {}
for s in range(2017, 2026):
    X, y = T.inseason_season(s, rolling, games, FEATURES)
    if X is not None:
        cache[s] = (X, y)
val = sorted(cache)[-4:]
pre = {}
for S in val:
    tr = [s for s in cache if s < S]
    Xtr = pd.concat([cache[s][0] for s in tr], ignore_index=True)
    ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
    sc = StandardScaler().fit(Xtr)
    pre[S] = (sc.transform(Xtr), ytr, sc.transform(cache[S][0]),
              cache[S][1].to_numpy())
print(f"{sum(len(v[0]) for v in cache.values())} games, "
      f"{cache[val[-1]][0].shape[1]} features; validating on {val}")
print("training on earlier seasons only, for every validation season\n")


def errs(config):
    out = []
    for S in val:
        Xtr, ytr, Xte, yte = pre[S]
        m = XGBRegressor(random_state=0, **config).fit(Xtr, ytr)
        out.append(np.abs(m.predict(Xte) - yte))
    return np.concatenate(out)


base = errs(INCUMBENT)
print(f"  incumbent MAE {base.mean():.3f} on {len(base)} games\n")
print(f"  {'config':<14}{'MAE':>9}{'vs inc':>9}{'SE':>8}{'t':>7}"
      f"{'seasons better':>16}")
for name, cfg in CANDIDATES.items():
    e = errs(cfg)
    d = e - base
    se = d.std(ddof=1) / np.sqrt(len(d))
    i, wins = 0, 0
    for S in val:
        n = len(cache[S][1])
        if e[i:i + n].mean() < base[i:i + n].mean():
            wins += 1
        i += n
    print(f"  {name:<14}{e.mean():>9.3f}{d.mean():>+9.3f}{se:>8.3f}"
          f"{d.mean() / se:>7.2f}{f'{wins} of {len(val)}':>16}")

print("\n  'slow best' wins all four seasons, the strongest showing here - but")
print("  it is the argmax of the 24-config sweep, and that sweep as a whole")
print("  beat the incumbent only 15 times in 24, mean -0.011. 'slow central',")
print("  from the same region but not chosen for winning, takes its entire")
print("  margin from 2025 alone.")
print("\n  So there may be 0.05-0.08 MAE in the slow region. It is not")
print("  established, and this model is judged on the blend rather than its own")
print("  holdout - a 0.08 change to one component at t = 2 will not show there.")
print("  Settling it needs a larger sweep scored on a season held out of the")
print("  sweep itself, then a walk-forward blend run.")
print("\n  Library defaults lose by 1.265 at t = 9.87, so tuned parameters are")
print("  worth having - they are just not worth re-deriving on this evidence.")
PY
