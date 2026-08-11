#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd

b = pd.read_csv('/home/bill/ncaaf/model_training/model_blender/blended_model.csv', index_col=0)
b['wk'] = b['week'].str.extract(r'(\d+)').astype(int)
b = b.sort_values('wk').set_index('week')

PRE, INS = 7.0, 3.0
print(f"Both models predict the home team by {PRE:.0f} and {INS:.0f}. What comes out?\n")
print(f"{'week':<9} {'coef sum':>9} {'blended':>9}   arithmetic")
print("-" * 74)
for wk in ['Week 2', 'Week 5', 'Week 8', 'Week 10', 'Week 14']:
    r = b.loc[wk]
    out = r.pre_szn_coefs * PRE + r.in_szn_coefs * INS + r.intercepts
    tot = r.pre_szn_coefs + r.in_szn_coefs
    print(f"{wk:<9} {tot:>9.2f} {out:>9.2f}   "
          f"{r.pre_szn_coefs:.3f}x{PRE:.0f} + {r.in_szn_coefs:.3f}x{INS:.0f} "
          f"{r.intercepts:+.2f}")

print("\n\nWhy the spread? Two separate jobs the coefficients are doing.\n")

print("1. MIXING - how much each model is trusted (their ratio)")
for wk in ['Week 2', 'Week 10', 'Week 14']:
    r = b.loc[wk]
    tot = r.pre_szn_coefs + r.in_szn_coefs
    print(f"   {wk:<8} preseason {r.pre_szn_coefs/tot:>5.0%} / in-season {r.in_szn_coefs/tot:>5.0%}")

print("\n2. SCALING - how confident to be overall (their sum)")
for wk in ['Week 2', 'Week 10', 'Week 14']:
    r = b.loc[wk]
    tot = r.pre_szn_coefs + r.in_szn_coefs
    verb = 'amplifies' if tot > 1 else 'shrinks toward 0'
    print(f"   {wk:<8} sum {tot:.2f}  -> {verb}")

print("\n\nDoes the intercept track anything real? Mean actual home margin by week:")
d = pd.read_csv('/home/bill/ncaaf/backtest_expanding_preds.csv')
d = d[d.week_num < 90]
print(f"   {'week':<9} {'actual mean':>12} {'intercept':>11}")
for wk in ['Week 2', 'Week 5', 'Week 8', 'Week 10', 'Week 14']:
    n = int(wk.split()[-1])
    g = d[d.week_num == n]
    print(f"   {wk:<9} {g.home_score_differential.mean():>12.2f} "
          f"{b.loc[wk, 'intercepts']:>11.2f}")

print("\n\nIs shrinking correct? Compare spread of predictions vs reality:")
print(f"   actual margins      std {d.home_score_differential.std():6.2f}")
print(f"   preseason preds     std {d.preseason_model_preds.std():6.2f}")
print(f"   in-season preds     std {d.in_season_model_preds.std():6.2f}")
print("   A model that only explains part of the variance SHOULD predict a")
print("   narrower range than reality. Coefficients summing under 1 enforce that.")
PY
