#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction || exit 1
python - <<'PY'
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression

h = pd.read_csv('../analysis/backtest_expanding_preds.csv')
h = h[h.week_num < 90].dropna(subset=['in_season_model_preds','home_score_differential'])
g = pd.read_csv('../etl/summarize/temp/games.csv', low_memory=False)
h = h.merge(g[['id','neutral_site']].drop_duplicates('id'), on='id', how='left')
h['neutral'] = h['neutral_site'].fillna(0).astype(bool)

iso = IsotonicRegression(out_of_bounds='clip').fit(
    h['in_season_model_preds'], h['home_score_differential'])
h['centre'] = iso.predict(h['in_season_model_preds'])
h['resid'] = h['home_score_differential'] - h['centre']

print(f"walk-forward games: {len(h)}   neutral: {int(h.neutral.sum())}")

print("\n=== RAW margin spread (what the checklist quoted) ===")
for lab, s in (('neutral', h[h.neutral]), ('normal', h[~h.neutral])):
    d = s['home_score_differential']
    print(f"  {lab:<8} n={len(s):>5}  mean {d.mean():>+7.2f}  sd {d.std():>6.2f}")

print("\n=== RESIDUAL spread (what sigma actually needs) ===")
for lab, s in (('neutral', h[h.neutral]), ('normal', h[~h.neutral])):
    r = s['resid']
    # de-mean within group: the known -2.9 neutral bias is corrected elsewhere,
    # so it must not be allowed to inflate this group's spread
    print(f"  {lab:<8} n={len(s):>5}  mean {r.mean():>+7.2f}  "
          f"sd {r.std():>6.2f}  sd about own mean {np.sqrt(((r-r.mean())**2).mean()):>6.2f}")

a = h.loc[h.neutral, 'resid']; b = h.loc[~h.neutral, 'resid']
sa = np.sqrt(((a-a.mean())**2).mean()); sb = np.sqrt(((b-b.mean())**2).mean())
print(f"\n  ratio neutral/normal = {sa/sb:.3f}")

print("\n=== is that ratio distinguishable from 1? (bootstrap) ===")
rng = np.random.default_rng(0)
av, bv = a.to_numpy(), b.to_numpy()
ratios = []
for _ in range(4000):
    ra = rng.choice(av, len(av), replace=True)
    rb = rng.choice(bv, len(bv), replace=True)
    ratios.append(ra.std()/rb.std())
lo, hi = np.percentile(ratios, [2.5, 97.5])
print(f"  95% CI on the ratio: [{lo:.3f}, {hi:.3f}]")
print(f"  -> {'REAL difference' if hi < 1.0 else 'consistent with no difference'}")

print("\n=== how much would a separate sigma change P(home win)? ===")
from scipy.stats import norm
for pred in (-14, -7, -3, 0, 3, 7, 14):
    p1 = 1 - norm.cdf(0, pred, sb)
    p2 = 1 - norm.cdf(0, pred, sa)
    print(f"  centre {pred:>+4}:  sigma {sb:.1f} -> {p1:.1%}    "
          f"sigma {sa:.1f} -> {p2:.1%}    diff {p2-p1:>+6.1%}")

print("\n=== how many games does this touch? ===")
s = pd.read_csv('prediction_file/scheduled_games.csv', index_col=0)
print(f"  2026 regular-season slate: {int(s['neutral_site'].sum())} of {len(s)} neutral")
gg = g.dropna(subset=['home_score_differential']).copy()
gg['wk'] = pd.to_numeric(gg['week'], errors='coerce')
nn = gg[gg['neutral_site'] == 1]
print(f"  historical neutral games: {len(nn)} total, "
      f"{int((nn['wk'] < 15).sum())} in wk<15, {int((nn['wk'] >= 15).sum())} in wk>=15 (bowls)")
print(f"  share of all bowl-week games that are neutral: "
      f"{(gg[gg['wk'] >= 15]['neutral_site'] == 1).mean():.1%}")
PY
