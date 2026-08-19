#!/usr/bin/env bash
# Is the published distribution honest? Refits the ENTIRE chain per season on
# earlier seasons only - blend weights, isotonic calibrator, sigma, drift
# offset - and scores the held-out season with it.
#
# margin_distribution --validate cannot answer this. It fits on the whole
# history and checks against the same games, and isotonic regression is
# flexible enough to flatter itself that way: worst band 0.8% in sample
# against 4.0% out of it.
#
# Two questions, two answers.
#
#   Is sigma too narrow, as feared? Barely. 15.85 fitted against 16.07
#   realised, a gap of 0.22 points, and narrow is the expected direction for
#   any spread measured on the residuals it was fitted to. The sigma published
#   before the walk-forward history was rebuilt was 16.58, so the error then
#   ran the other way.
#
#   Then why was every band wrong? Because the fault is in the centre. All six
#   bands missed with the SAME sign - a shift, not a bad width. Home advantage
#   has risen from 51.0% of games in 2020 to 59.3% in 2025 and a calibrator
#   fitted on older seasons lags it, leaving +0.95 points of systematic lean
#   toward the away team. DRIFT_LOOKBACK adds that level back.
#
# Note the offset makes IN-SAMPLE calibration slightly worse. It corrects drift
# that has not happened yet within the training seasons, so in sample there is
# nothing for it to correct and it only adds bias. That is the signature of a
# forward correction, not evidence against it.
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
    """Weeks the training seasons never reached fall back to the last fitted
    week rather than to NaN. Test seasons run a week or two longer than the
    seasons before them, and dropping those games would quietly shrink the
    sample every comparison here is measured on."""
    w = weights.copy()
    w['w'] = w['week'].astype(str).str.extract(r'(\d+)').astype(float)
    w = w.dropna(subset=['w']).sort_values('w').set_index('w')
    k = frame['week_num'].astype(float).clip(upper=w.index.max())
    return (k.map(w['pre_szn_coefs']) * frame['preseason_model_preds']
            + k.map(w['in_szn_coefs']) * frame['in_season_model_preds']
            + k.map(w['intercepts'])).to_numpy(float)


def bands(pw, won, label):
    print(f"\n  {label}")
    print(f"    {'band':<14}{'n':>6}{'predicted':>11}{'actual':>9}{'gap':>8}")
    worst = 0.0
    for lo, hi in zip([0, .2, .35, .5, .65, .8], [.2, .35, .5, .65, .8, 1.01]):
        s = (pw >= lo) & (pw < hi)
        if s.sum() < 30:
            continue
        gap = won[s].mean() - pw[s].mean()
        worst = max(worst, abs(gap))
        print(f"    {f'{lo:.0%}-{hi:.0%}':<14}{s.sum():>6}{pw[s].mean():>11.1%}"
              f"{won[s].mean():>9.1%}{gap:>+8.1%}")
    print(f"    worst band {worst:.1%}   Brier {np.mean((pw - won) ** 2):.4f}")
    return worst


def walk_forward(lookback):
    rows, all_pw, all_won = [], [], []
    for s in sorted(h.season.unique())[3:]:
        tr, te = h[h.season < s], h[h.season == s]
        weights = MB.compare_results(tr)
        tb, eb = blend(tr, weights), blend(te, weights)
        ta = tr['home_score_differential'].to_numpy(float)
        ea = te['home_score_differential'].to_numpy(float)

        sw = MD.season_weights(tr['season'].to_numpy(float), MD.HALF_LIFE,
                               reference=s - 1)
        iso = IsotonicRegression(out_of_bounds='clip').fit(tb, ta,
                                                           sample_weight=sw)
        offset = 0.0
        if lookback:
            m = tr.season.isin(sorted(tr.season.unique())[-lookback:]).to_numpy()
            offset = float(np.mean(ta[m] - iso.predict(tb[m])))

        r = ta - iso.predict(tb)
        mu = np.average(r, weights=sw)
        sigma = float(np.sqrt(np.average((r - mu) ** 2, weights=sw)))

        centre = iso.predict(eb) + offset
        pw = np.array([MD.distribution(c, kn, sigma)[MD.GRID > 0].sum()
                       for c in centre])
        won = ea > 0
        all_pw.append(pw)
        all_won.append(won)
        rows.append({'season': int(s), 'n': len(te), 'offset': offset,
                     'sigma': sigma, 'realised': float((ea - centre).std()),
                     'bias': float(np.mean(ea - centre)),
                     'mae': float(np.abs(ea - centre).mean()),
                     'brier': float(np.mean((pw - won) ** 2))})
    return (pd.DataFrame(rows), np.concatenate(all_pw),
            np.concatenate(all_won))


print("=== per season, chain refitted on earlier seasons only ===")
print(f"  {'season':>7}{'n':>6}{'offset':>9}{'sigma':>8}{'realised':>10}"
      f"{'narrow by':>11}{'bias':>8}{'MAE':>8}{'Brier':>9}")
for lookback, tag in ((0, 'no offset'), (MD.DRIFT_LOOKBACK, 'with offset')):
    res, pw, won = walk_forward(lookback)
    print(f"  -- {tag}")
    for _, x in res.iterrows():
        print(f"  {x.season:>7.0f}{x.n:>6.0f}{x.offset:>+9.2f}{x.sigma:>8.2f}"
              f"{x.realised:>10.2f}{x.realised - x.sigma:>+11.2f}"
              f"{x.bias:>+8.2f}{x.mae:>8.2f}{x.brier:>9.4f}")
    n = res['n'].sum()
    wm = lambda c: (res[c] * res['n']).sum() / n
    print(f"  {'pooled':>7}{n:>6.0f}{wm('offset'):>+9.2f}{wm('sigma'):>8.2f}"
          f"{wm('realised'):>10.2f}{wm('realised') - wm('sigma'):>+11.2f}"
          f"{wm('bias'):>+8.2f}{wm('mae'):>8.2f}{wm('brier'):>9.4f}")
    globals()[f'keep_{lookback}'] = (res, pw, won)

_, pw0, won0 = keep_0
res1, pw1, won1 = globals()[f'keep_{MD.DRIFT_LOOKBACK}']
w0 = bands(pw0, won0, 'OUT OF SAMPLE, no offset')
w1 = bands(pw1, won1, f'OUT OF SAMPLE, offset from last {MD.DRIFT_LOOKBACK} seasons')

# In sample, for contrast: the whole chain fitted on everything.
w_all = MB.compare_results(h)
b_all = blend(h, w_all)
a_all = h['home_score_differential'].to_numpy(float)
sw_all = MD.season_weights(h['season'].to_numpy(float), MD.HALF_LIFE)
iso = IsotonicRegression(out_of_bounds='clip').fit(b_all, a_all,
                                                   sample_weight=sw_all)
r = a_all - iso.predict(b_all)
mu = np.average(r, weights=sw_all)
sig = float(np.sqrt(np.average((r - mu) ** 2, weights=sw_all)))
pwi = np.array([MD.distribution(c, kn, sig)[MD.GRID > 0].sum()
                for c in iso.predict(b_all)])
wi = bands(pwi, a_all > 0, 'IN SAMPLE, no offset (what --validate reports)')

print(f"\n=== summary ===")
print(f"  worst band   {wi:.1%} in sample, {w0:.1%} out of sample, "
      f"{w1:.1%} out of sample with the offset")
print(f"  sigma        {sig:.2f} published; out of sample it is narrow by "
      f"{(res1.realised * res1.n).sum() / res1.n.sum() - (res1.sigma * res1.n).sum() / res1.n.sum():.2f}")
PY
