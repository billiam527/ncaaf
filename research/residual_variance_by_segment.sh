#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression

h = pd.read_csv('../analysis/backtest_expanding_preds.csv')
h = h.dropna(subset=['in_season_model_preds','home_score_differential'])
g = pd.read_csv('../etl/summarize/temp/games.csv', low_memory=False)
h = h.merge(g[['id','neutral_site']].drop_duplicates('id'), on='id', how='left')
h['neutral'] = h['neutral_site'].fillna(0).astype(bool)

iso = IsotonicRegression(out_of_bounds='clip').fit(
    h['in_season_model_preds'], h['home_score_differential'])
h['resid'] = h['home_score_differential'] - iso.predict(h['in_season_model_preds'])

def boot(a, b, n=4000):
    rng = np.random.default_rng(0)
    r = [rng.choice(a, len(a), True).std() / rng.choice(b, len(b), True).std()
         for _ in range(n)]
    return np.percentile(r, [2.5, 97.5])

print("=== residual bias and spread by part of season ===")
h['part'] = np.where(h.week_num >= 15, 'postseason (15-16)',
            np.where(h.week_num == 14, 'champ week (14)',
            np.where(h.week_num <= 5, 'early (2-5)', 'mid (6-13)')))
print(f"{'segment':<22}{'n':>6}{'bias':>9}{'resid sd':>10}")
print("-" * 47)
for k in ('early (2-5)', 'mid (6-13)', 'champ week (14)', 'postseason (15-16)'):
    s = h[h.part == k]['resid']
    print(f"{k:<22}{len(s):>6}{s.mean():>+9.2f}{s.std():>10.2f}")

base = h[h.part == 'mid (6-13)']['resid'].to_numpy()
print("\n=== each segment vs the mid-season baseline ===")
for k in ('early (2-5)', 'champ week (14)', 'postseason (15-16)'):
    a = h[h.part == k]['resid'].to_numpy()
    lo, hi = boot(a, base)
    sig = 'REAL' if (lo > 1.0 or hi < 1.0) else 'not distinguishable'
    print(f"  {k:<20} sd ratio {a.std()/base.std():.3f}  CI [{lo:.3f}, {hi:.3f}]  {sig}")

print("\n=== is the NEUTRAL effect just the postseason effect? ===")
reg = h[h.week_num < 14]
for lab, s in (('neutral', reg[reg.neutral]), ('normal', reg[~reg.neutral])):
    r = s['resid']
    print(f"  regular season only, {lab:<8} n={len(s):>5}  "
          f"bias {r.mean():>+6.2f}  sd {r.std():>6.2f}")
a = reg.loc[reg.neutral,'resid'].to_numpy(); b = reg.loc[~reg.neutral,'resid'].to_numpy()
lo, hi = boot(a, b)
print(f"  ratio {a.std()/b.std():.3f}  CI [{lo:.3f}, {hi:.3f}]  "
      f"-> {'REAL' if hi < 1.0 else 'not distinguishable'}")
PY
