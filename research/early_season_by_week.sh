#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np

w = pd.read_csv('../model_training/model_blender/blended_model.csv')
w['wk'] = w['week'].str.extract(r'(\d+)').astype(int)
w = w.set_index('wk')

h = pd.read_csv('../analysis/backtest_expanding_preds.csv')
h = h.dropna(subset=['home_score_differential',
                     'preseason_model_preds', 'in_season_model_preds'])

h['blended'] = (h['week_num'].map(w['pre_szn_coefs']) * h['preseason_model_preds']
                + h['week_num'].map(w['in_szn_coefs']) * h['in_season_model_preds']
                + h['week_num'].map(w['intercepts']))
print(f"blended built for {h['blended'].notna().sum()} of {len(h)} rows")

print("\n=== residual bias / spread BY WEEK ===")
print(f"{'wk':>3}{'n':>6}{'pre bias':>10}{'in bias':>10}{'BLEND bias':>12}"
      f"{'blend sd':>10}{'blend MAE':>11}")
print("-"*62)
for wk, s in h.groupby('week_num'):
    rb = s['home_score_differential'] - s['blended']
    rp = s['home_score_differential'] - s['preseason_model_preds']
    ri = s['home_score_differential'] - s['in_season_model_preds']
    print(f"{int(wk):>3}{len(s):>6}{rp.mean():>+10.2f}{ri.mean():>+10.2f}"
          f"{rb.mean():>+12.2f}{rb.std():>10.2f}{rb.abs().mean():>11.2f}")

print("\n=== early (2-5) vs mid (6-13), on the BLEND ===")
e = h[h.week_num.between(2,5)]; m = h[h.week_num.between(6,13)]
for lab, sub in (('early 2-5', e), ('mid 6-13', m)):
    r = sub['home_score_differential'] - sub['blended']
    print(f"  {lab:<10} n={len(sub):>5}  bias {r.mean():>+6.2f} "
          f"(SE {r.std()/np.sqrt(len(sub)):.2f})  sd {r.std():>6.2f}  MAE {r.abs().mean():.2f}")

re_ = (e['home_score_differential'] - e['blended'])
rm  = (m['home_score_differential'] - m['blended'])
rng = np.random.default_rng(0)
a, b = re_.to_numpy(), rm.to_numpy()
ratios = [rng.choice(a,len(a),True).std()/rng.choice(b,len(b),True).std() for _ in range(4000)]
lo, hi = np.percentile(ratios,[2.5,97.5])
print(f"  sd ratio {a.std()/b.std():.3f}  CI [{lo:.3f}, {hi:.3f}]  "
      f"-> {'REAL' if (lo>1 or hi<1) else 'not distinguishable'}")

print("\n=== week 2 and 3 alone, on the BLEND ===")
for wk in (2,3,4,5):
    s = h[h.week_num == wk]
    r = s['home_score_differential'] - s['blended']
    se = r.std()/np.sqrt(len(s))
    print(f"  week {wk}: n={len(s):>4}  bias {r.mean():>+6.2f}  SE {se:.2f}  "
          f"t={r.mean()/se:>+5.2f}  sd {r.std():.2f}")
PY
