"""Turn a point prediction into a football-shaped distribution over margins.

Real margins are lumpy: 3 and 7 occur roughly 3x as often as their neighbours,
9 is nearly a dead zone, and 34.5% of FBS games land on 3/7/10/14/17/21. A
point prediction should NOT be lumpy - snapping 8.2 to 7 only makes the
estimate worse - but the distribution around it certainly should be.

Method follows the approach nfelo describes for NFL spreads
(nfeloapp.com/analysis/margin-probabilities-from-nfl-spreads): a baseline
normal centred on the predicted margin, multiplied by a key-number weighting,
so the distribution stays centred where the model put it while carrying real
scoring structure.

    P(margin = m | prediction p)  proportional to  N(m; p, sigma) * w(m)

sigma is the residual spread of actual outcomes around the model's prediction.
w(m) is read off real games rather than assumed: the empirical frequency of
each margin divided by a smoothed version of itself, which isolates scoring
lumpiness from the overall shape. That yields 2.29x at 3, 2.13x at 7, and
0.39x at 9.

An earlier version of this file estimated the distribution by kernel-weighting
similar historical games. It produced a degenerate mode - 282 of 471 games came
out at +3, and a game predicted at +35 had its mode at +9 - because with a
sigma near 18 the single most likely integer is dominated by the global key
number effect rather than the game. Centring on the prediction fixes that.

WHAT --validate USED TO OVERSTATE

The calibrator and sigma were fitted on the whole walk-forward history and then
checked against that same history, which isotonic regression is flexible enough
to flatter. Refitting the entire chain per season on earlier seasons only -
blend weights, calibrator, sigma - moves the worst probability band from 0.8%
off to 4.0% off. research/calibration_walk_forward.sh does that and is the
number to trust; --validate is in-sample and says so.

That honest pass also answered the question it was run to answer. Sigma is not
dangerously narrow. It read 15.85 fitted against 16.07 realised, a gap of 0.22
points, and removing the calibrator (below) closes it to 16.04 against 15.98 -
marginally wide, which is the safe side. The sigma published before the
walk-forward history was rebuilt was 16.58, so the error then ran the other
way, and the overconfidence this was run to find was never there.

THE ISOTONIC CALIBRATOR IS OFF BY DEFAULT

It was adopted on an in-sample reading of the leaked history (Brier 0.1960 ->
0.1918, MAE 13.48 -> 13.22) and had never been checked out of sample. Damping
its correction toward the identity by local support improves every measure
monotonically, and the limit of that sweep is switching it off:

    centre                  MAE all  MAE mid  MAE tail   Brier  sigma
    isotonic, unshrunk       12.730   12.685     13.42  0.1846  15.85
    shrunk K=50              12.690   12.672     12.97  0.1844  15.90
    shrunk K=200             12.672   12.659     12.87  0.1841  15.95
    shrunk K=1000            12.661   12.648     12.85  0.1839  16.01
    no calibrator at all     12.659   12.647     12.84  0.1839  16.04

A monotone sweep across a smooth family is far stronger evidence than any two
of those rows compared alone, and the mechanism is plain: isotonic regression
on ~4,200 points with no regularisation fits noise, worst where the data is
thinnest. Its map across the top of the range, with the games behind each
point, shows how thin that gets:

    raw    calibrated   shift   support
  +30.0        +31.1    +1.1        86
  +35.0        +42.1    +7.1        51
  +42.5        +52.6   +10.1         5
  +50.0        +53.1    +3.1         0     <- clipped, nothing behind it

Ten of the 469 games on the 2026 slate sit beyond +35, including one at a raw
+50.6, past the largest prediction the history contains. Removing the
calibrator wins on MAE in all three held-out seasons.

--calibrate puts it back. research/calibrator_value.sh is the evidence.

What it was right about is that the model under-predicts blowouts: games
predicted +30 to +35 landed at +36.9 over 68 games. That signal is real and
still uncorrected - the top two probability bands remain 2-3% light. It is not
worth an unregularised monotone fit to chase, and a shrunk version of the same
fit does not beat switching it off.

THE DRIFT OFFSET

The real fault was in the centre, not the spread. Every out-of-sample band was
wrong with the same sign - home teams won 1.0 to 4.0 points more often than
predicted - which is a shift, not a miscalibrated width. Home advantage has
risen steadily and the blend lags it:

    season   home win%   mean margin   model centre   bias
      2020       51.0%          1.94           4.05  -2.11
      2021       57.4%          3.25           3.49  -0.25
      2022       57.8%          3.91           3.69  +0.22
      2023       58.7%          4.23           3.53  +0.70
      2024       59.0%          5.04           4.47  +0.57
      2025       59.3%          5.08           4.08  +1.00

A calibrator fitted on seasons with weaker home advantage cannot correct a
drift that postdates them, so the centre carries +0.95 points of systematic
lean toward the away team. That is a directional error on every game, which is
the kind that costs money rather than accuracy.

DRIFT_LOOKBACK adds back the mean residual of the most recent seasons (this
sweep predates switching the calibrator off, so the levels sit ~0.07 above the
shipped ones; the ordering is unchanged):

    offset from     offset   bias left   win% gap     MAE    Brier
    none             +0.00       +0.95      +2.5%   12.75   0.1850
    last 1 season    +0.67       +0.28      +1.2%   12.73   0.1845
    last 2 seasons   +0.53       +0.41      +1.4%   12.73   0.1846
    last 3 seasons   +0.21       +0.74      +2.1%   12.75   0.1849

Two seasons rather than one, though one scored marginally better: a single
season's mean residual carries a standard error near 0.58 points on an estimate
of 0.67, and the gap between them (0.0001 Brier over three test seasons) is far
inside that. Two halves the sampling error for the same result.

Extrapolating the bias linearly instead was tried and overshoots - +0.95
becomes -0.91 - because the fitted slope is dragged by 2020, played in empty
stadiums. Dropping 2020 outright halves the bias but costs MAE and Brier. The
offset is the smaller instrument and the only one that improved every measure.

THE GRID WAS TOO NARROW FOR LOPSIDED GAMES

distribution() renormalises over GRID, so mass falling past the edge was not
dropped, it was REDISTRIBUTED across the whole range - including below the
market line, where it counts as a loss. With the grid at +/-70 a centre of
+51.4 put 12% of its mass outside, and p_cover read 44.0% against a true 53.5%.
Nineteen of the 469 games on the 2026 slate were far enough out for the error
to exceed a point, and it grew with the centre, so it was worst exactly where
the model was most confident. The grid now runs +/-100, past anything football
does - the widest margin in 12,213 games is 78.

Produces per game:
    calibrated_margin the centre the distribution sits on: the blend plus the
                      drift offset, and the isotonic map only under --calibrate
    median_margin     the calibrated point estimate
    mode_margin       most likely single outcome, football-shaped
    p_home_win        probability the home team wins outright
    p_cover           probability of covering the market line, when known
    edge_calibrated   centre minus the market line. add_market_lines writes an
                      `edge` from the raw blend instead, which is a different
                      number and disagreed in sign with p_cover on 4 of the 37
                      priced games. Use this one with p_cover.
    margin_dist       the full P(margin = k), as JSON

    python margin_distribution.py
    python margin_distribution.py --game "UNC VS TCU" --show-dist
    python margin_distribution.py --validate
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

GAMES = os.path.join(_REPO, 'etl', 'summarize', 'temp', 'games.csv')
TEAMS = os.path.join(_REPO, 'etl', 'collect', 'collect_espn_teams', 'temp', 'teams.csv')
HISTORY = os.path.join(_REPO, 'analysis', 'backtest_expanding_preds.csv')
BLEND_WEIGHTS = os.path.join(_REPO, 'model_training', 'model_blender',
                             'blended_model.csv')
PREDICTIONS = os.path.join(_HERE, 'prediction_file', 'predictions_with_lines.csv')
FALLBACK = os.path.join(_HERE, 'prediction_file', 'new_predictions.csv')
OUT = os.path.join(_HERE, 'prediction_file', 'predictions_with_distribution.csv')

# The grid has to outrun the widest centre by several sigma. It ran -70 to +70,
# and because distribution() renormalises over it, mass past the edge was not
# dropped but REDISTRIBUTED across the whole range - including below the market
# line, where it counts as a loss. On the one 2026 game centred at +51.4 that
# published p_cover 44.0% against a true 53.5%; 19 of 469 games sat far enough
# out for the error to exceed a point. Real margins reach 78 in 12,213 games,
# so +/-100 is past anything football does and leaves under 0.3% outside even
# for the most lopsided centre on the slate.
LO, HI = -100, 100
GRID = np.arange(LO, HI + 1)
SMOOTHING = 2.5          # sigma used to build the "no lumpiness" reference
KEY_NUMBERS = (3, 7, 10, 14, 17, 21)

# Below this smoothed density there is no support for a lumpiness estimate, so
# the weight falls back to 1.0 (neutral). emp/smooth is a ratio of two small
# numbers once the history thins out: margin 65 drew weight 0.16 from ONE game
# and margin 70 drew 0.15 from none, suppressing those outcomes by 85% on no
# evidence at all. This cuts in beyond |margin| 57, where fewer than about five
# effective games back any estimate. Every key number is 21 or below, so none of
# them are touched at any threshold up to 1e-3.
MIN_SUPPORT = 2.5e-4

# Seasons are weighted 0.5 ** (age / HALF_LIFE). The scoring distribution
# drifts: two-point attempts have pushed margins of 2 from 2.16% to 3.26% and
# 8 from 2.26% to 2.80% since 2010, while 1-point and 14-point games have
# receded and overall spread has narrowed (sd 21.4 -> 20.2). Weighting keeps
# every season in the estimate while letting recent rules and play-calling
# dominate.
HALF_LIFE = 5.0

# Seasons of recent history the drift offset is measured on. See the docstring:
# home advantage is rising and the calibrator, fitted on older seasons, lags it.
DRIFT_LOOKBACK = 2


def season_weights(seasons, half_life=HALF_LIFE, reference=None):
    if half_life is None or half_life <= 0:
        return np.ones(len(seasons))
    ref = reference if reference is not None else seasons.max()
    return 0.5 ** ((ref - seasons) / half_life)


def games_by_regime():
    """The two regimes, kept apart, because they are not one population.

    Dropping crossover games entirely was the old behaviour, and it left the
    ~150 FBS-vs-FCS games a season being shaped by a curve fitted on games that
    look nothing like them:

                          FBS v FBS   crossover
      sd                      21.01       24.78
      |margin| > 28           18.5%       39.7%
      lands exactly on 3       9.75%       6.07%
      lands exactly on 7       8.44%       5.25%

    Both halves of the distribution differ, not only the width. A blowout does
    not end on a field-goal margin, so crossover games hit key numbers about a
    third less often - which means a shared lumpiness curve with a per-regime
    sigma would still be wrong.

    Returns {'fbs': df, 'cross': df}. The 'fbs' frame is exactly what
    fbs_games() always returned, so anything predicting only FBS games is
    unaffected.
    """
    g = pd.read_csv(GAMES, low_memory=False)
    t = pd.read_csv(TEAMS)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g.dropna(subset=['home_score_differential', 'season']).copy()
    g['margin'] = g['home_score_differential'].round().astype(int)
    h, a = g.home_team_id.isin(fbs), g.away_team_id.isin(fbs)
    return {'fbs': g[h & a][['season', 'margin']],
            'cross': g[h ^ a][['season', 'margin']]}


def regime_sigma_scale(half_life=HALF_LIFE):
    """How much wider the crossover regime is, as a multiplier on sigma.

    Sigma is fitted from model residuals, and the walk-forward history holds
    FBS games only, so there is no crossover residual to measure directly. The
    ratio of raw outcome spreads is the honest stand-in: it is what the two
    regimes' widths actually differ by, and it is applied rather than assumed
    to be 1.0, which is what dropping the games amounted to.
    """
    r = games_by_regime()
    a = r['fbs']['margin'].to_numpy(float)
    b = r['cross']['margin'].to_numpy(float)
    if len(b) < 200:
        return 1.0
    wa = season_weights(r['fbs']['season'].to_numpy(float), half_life)
    wb = season_weights(r['cross']['season'].to_numpy(float), half_life)
    va = np.average((a - np.average(a, weights=wa)) ** 2, weights=wa)
    vb = np.average((b - np.average(b, weights=wb)) ** 2, weights=wb)
    return float(np.sqrt(vb / va)) if va > 0 else 1.0


def fbs_games():
    """Kept so existing callers keep their exact behaviour."""
    return games_by_regime()['fbs']


def key_number_weights(games, half_life=HALF_LIFE):
    """Empirical / smoothed frequency, isolating scoring lumpiness.

    Recency-weighted so current play-calling drives the curve.
    """
    w = season_weights(games['season'].to_numpy(float), half_life)
    counts = np.zeros(len(GRID))
    idx = games['margin'].to_numpy() - LO
    ok = (idx >= 0) & (idx < len(GRID))
    np.add.at(counts, idx[ok], w[ok])
    emp = counts / counts.sum()
    smooth = gaussian_filter1d(emp, sigma=SMOOTHING, mode='nearest')
    # Past the widened grid's edges the smoothed density underflows to exactly
    # zero, so guard the division rather than letting np.where evaluate 0/0 for
    # every element and then discard the NaNs.
    ok = smooth > MIN_SUPPORT
    ratio = np.ones(len(GRID))
    np.divide(emp, smooth, out=ratio, where=ok)
    # Neutral wherever the history is too thin to say anything about lumpiness.
    return np.where(ok, np.clip(ratio, 0.15, 4.0), 1.0)


def add_blended(h):
    """Reconstruct the blended prediction on historical games.

    Inference centres the distribution on `blended_prediction`, so the
    calibrator and sigma have to be fitted against the same quantity. The
    walk-forward file stores only the two component models, so the per-week
    blend is rebuilt here using the weights that predict.py applies.
    """
    if not os.path.exists(BLEND_WEIGHTS):
        return h
    w = pd.read_csv(BLEND_WEIGHTS)
    if 'week' not in w.columns:
        return h
    w = w.copy()
    w['wk'] = w['week'].astype(str).str.extract(r'(\d+)').astype(float)
    w = w.dropna(subset=['wk']).set_index('wk')
    for col in ('pre_szn_coefs', 'in_szn_coefs', 'intercepts'):
        if col not in w.columns:
            return h
    wk = h['week_num'].astype(float)
    h = h.copy()
    h['blended'] = (wk.map(w['pre_szn_coefs']) * h['preseason_model_preds']
                    + wk.map(w['in_szn_coefs']) * h['in_season_model_preds']
                    + wk.map(w['intercepts']))
    return h


def load_history(fit_col='blended'):
    h = pd.read_csv(HISTORY)
    h = h[h.week_num < 90]
    needed = ['home_score_differential', 'in_season_model_preds']
    if fit_col == 'blended':
        h = add_blended(h)
        needed = ['home_score_differential', 'blended']
    elif fit_col in h.columns:
        needed = ['home_score_differential', fit_col]
    return h.dropna(subset=[c for c in needed if c in h.columns])


def resolve_fit_column(model_col):
    """History column corresponding to the column inference centres on.

    Fitting on one model and applying to another compresses everything toward
    50/50: a calibrator fitted on the in-season model and applied to the blend
    was off by up to 8.9 points in the 20-35% band, against 2.0 once matched.
    """
    if model_col in ('blended_prediction', 'blended_model'):
        return 'blended'
    return model_col


def fit_calibrator(half_life=HALF_LIFE, fit_col='blended'):
    """Map raw prediction -> conditional mean outcome, monotonically.

    The model is over-confident conditionally: games predicted at -17 land
    around -13 on average. Isotonic regression on out-of-sample walk-forward
    predictions corrects that without disturbing the ordering, and pulls the
    tails in - which is where P(home win) was worst calibrated.
    """
    h = load_history(fit_col)
    if fit_col not in h.columns:
        print(f"  '{fit_col}' could not be built from the walk-forward history; "
              f"calibrating on in_season_model_preds instead")
        fit_col = 'in_season_model_preds'
        h = load_history(fit_col)
    w = season_weights(h['test_season'].to_numpy(float), half_life)
    iso = IsotonicRegression(out_of_bounds='clip').fit(
        h[fit_col], h['home_score_differential'], sample_weight=w)
    return iso


def drift_offset(calibrator=None, fit_col='blended', lookback=DRIFT_LOOKBACK):
    """Mean residual over the most recent seasons, added back to the centre.

    Home advantage is drifting up and the model lags it, so the centre leans
    toward the away team by about a point. This is deliberately a constant: the
    bias is a shift, sigma is already right, and every attempt to fit its shape
    rather than its level overshot.
    """
    if not lookback:
        return 0.0
    h = load_history(fit_col)
    col = fit_col if fit_col in h.columns else 'in_season_model_preds'
    recent = sorted(h['test_season'].unique())[-lookback:]
    m = h['test_season'].isin(recent).to_numpy()
    if not m.any():
        return 0.0
    centre = (calibrator.predict(h[col]) if calibrator is not None
              else h[col].to_numpy(float))
    r = h['home_score_differential'].to_numpy(float)[m] - centre[m]
    return float(np.mean(r))


def residual_sigma(half_life=HALF_LIFE, calibrator=None, fit_col='blended'):
    """Recency-weighted spread of outcomes around the (optionally calibrated)
    prediction. Calibrating shifts the centre, so sigma must be measured
    against the same quantity the distribution is centred on.

    The drift offset is deliberately NOT applied here. It corrects a forward
    shift that has not happened yet within the training seasons, so adding it
    to historical residuals would widen sigma for a bias that is not there. The
    effect either way is under a tenth of a point.
    """
    h = load_history(fit_col)
    col = fit_col if fit_col in h.columns else 'in_season_model_preds'
    centre = (calibrator.predict(h[col]) if calibrator is not None
              else h[col].to_numpy(float))
    r = h['home_score_differential'].to_numpy(float) - centre
    w = season_weights(h['test_season'].to_numpy(float), half_life)
    mean = np.average(r, weights=w)
    return float(np.sqrt(np.average((r - mean) ** 2, weights=w)))


def distribution(prediction, weights, sigma):
    base = np.exp(-0.5 * ((GRID - prediction) / sigma) ** 2)
    p = base * weights
    total = p.sum()
    return p / total if total > 0 else p


def summarise(p, market_margin=np.nan):
    cum = np.cumsum(p)
    out = {
        'median_margin': int(GRID[int(np.searchsorted(cum, 0.5))]),
        'mode_margin': int(GRID[int(np.argmax(p))]),
        'mode_prob': float(p.max()),
        'p_home_win': float(p[GRID > 0].sum()),
        'p_tie': float(p[GRID == 0].sum()),
    }
    out['p_cover'] = (float(p[GRID > market_margin].sum())
                      if pd.notna(market_margin) else np.nan)
    return out


def validate(weights, sigma, calibrator=None, fit_col='blended', offset=0.0):
    """Are the probabilities honest? Compare predicted P(win) to realised.

    IN SAMPLE. The calibrator, sigma and offset below were all fitted on these
    same games, and isotonic regression flatters itself that way: the worst
    band here reads 0.8% against 4.0% when the chain is refitted per season on
    earlier seasons only. Use research/calibration_walk_forward.sh for the
    number that means something; this pass is for spotting gross breakage.
    """
    h = load_history(fit_col)
    col = fit_col if fit_col in h.columns else 'in_season_model_preds'
    raw = h[col].to_numpy(float)
    actual = h['home_score_differential'].to_numpy(float)
    preds = (calibrator.predict(raw) if calibrator is not None else raw) + offset

    pw = np.array([distribution(p, weights, sigma)[GRID > 0].sum() for p in preds])
    won = actual > 0

    print("=== calibration of P(home win) ===")
    print(f"{'predicted band':<18}{'n':>6}{'predicted':>11}{'actual':>9}{'gap':>8}")
    print("-" * 53)
    edges = [0, .2, .35, .5, .65, .8, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (pw >= lo) & (pw < hi)
        if s.sum() < 30:
            continue
        print(f"{f'{lo:.0%}-{hi:.0%}':<18}{s.sum():>6}{pw[s].mean():>11.1%}"
              f"{won[s].mean():>9.1%}{won[s].mean() - pw[s].mean():>+8.1%}")
    print("-" * 53)
    brier = np.mean((pw - won) ** 2)
    print(f"  Brier score {brier:.4f}   (0.25 = always guessing 50%)")

    print(f"  MAE of the centre: {np.abs(preds - actual).mean():.2f} points")
    med = np.array([summarise(distribution(p, weights, sigma))['median_margin']
                    for p in preds])
    print(f"  median tracks centre: mean |median - centre| = "
          f"{np.abs(med - preds).mean():.2f} points")
    kn = np.array([distribution(p, weights, sigma)[np.isin(np.abs(GRID), KEY_NUMBERS)].sum()
                   for p in preds])
    print(f"  mass on key numbers: {kn.mean():.1%}   (real games: "
          f"{np.isin(np.abs(actual).round(), KEY_NUMBERS).mean():.1%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sigma', type=float, default=None, help='override residual sd')
    ap.add_argument('--half-life', type=float, default=HALF_LIFE,
                    help='seasons for recency weight to halve; 0 disables weighting')
    ap.add_argument('--game', default=None, help='substring of short_name to inspect')
    ap.add_argument('--show-dist', action='store_true')
    ap.add_argument('--validate', action='store_true', help='check probability calibration')
    ap.add_argument('--calibrate', action='store_true',
                    help='apply the isotonic calibrator; off by default because '
                         'it costs MAE and Brier out of sample (see docstring)')
    ap.add_argument('--raw-centre', action='store_true',
                    help='no-op, kept for older scripts: raw is now the default')
    ap.add_argument('--drift-lookback', type=int, default=DRIFT_LOOKBACK,
                    help='seasons the home-advantage drift offset is measured '
                         'on; 0 disables it')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    hl = args.half_life if args.half_life and args.half_life > 0 else None
    games = fbs_games()
    weights = key_number_weights(games, hl)

    # Resolve the column inference will centre on before fitting anything, so
    # the calibrator and sigma describe that same quantity.
    src = PREDICTIONS if os.path.exists(PREDICTIONS) else FALLBACK
    preds = pd.read_csv(src, index_col=0) if os.path.exists(src) else None
    model_col = next((c for c in ('blended_prediction', 'blended_model',
                                  'preseason_model_preds')
                      if preds is not None and c in preds.columns), None)
    fit_col = resolve_fit_column(model_col) if model_col else 'blended'

    calibrator = fit_calibrator(hl, fit_col) if args.calibrate else None
    sigma = args.sigma or residual_sigma(hl, calibrator, fit_col)
    offset = drift_offset(calibrator, fit_col, args.drift_lookback)
    eff = season_weights(games['season'].to_numpy(float), hl).sum()
    print(f"sigma = {sigma:.2f} points; key-number weights from {len(games)} FBS games"
          f" ({eff:.0f} effective, half-life "
          f"{'off' if hl is None else f'{hl:g} seasons'}); "
          f"centre = {'raw prediction' if calibrator is None else 'isotonic-calibrated'}"
          f", fitted on {fit_col}")
    print(f"drift offset = {offset:+.2f} points "
          f"({'off' if not args.drift_lookback else f'last {args.drift_lookback} seasons'})\n")

    if args.validate:
        validate(weights, sigma, calibrator, fit_col, offset)
        print("\n=== effect of recency weighting on the key numbers ===")
        unw = key_number_weights(games, None)
        print(f"{'margin':>7}{'unweighted':>12}{'weighted':>11}{'change':>10}")
        for k in (1, 2, 3, 7, 8, 10, 14, 17, 21):
            i = k - LO
            print(f"{k:>7}{unw[i]:>12.2f}{weights[i]:>11.2f}"
                  f"{weights[i] - unw[i]:>+10.2f}")
        print(f"\n  sigma unweighted {residual_sigma(None):.2f}  ->  weighted {sigma:.2f}")
        return

    if preds is None:
        raise SystemExit(f"no predictions file at {PREDICTIONS} or {FALLBACK}")
    if model_col is None:
        raise SystemExit(f"{os.path.basename(src)} has no usable prediction column")
    print(f"predictions: {len(preds)} games from {os.path.basename(src)} ({model_col})")

    rows = []
    for _, r in preds.iterrows():
        if pd.isna(r[model_col]):
            rows.append({})
            continue
        centre = (float(calibrator.predict([r[model_col]])[0])
                  if calibrator is not None else float(r[model_col])) + offset
        p = distribution(centre, weights, sigma)
        s = {'calibrated_margin': round(centre, 2)}
        s.update(summarise(p, r.get('market_margin', np.nan)))
        # add_market_lines computes `edge` from the raw blend, before the drift
        # offset, so its sign can disagree with p_cover. Publish the edge that
        # matches the distribution alongside it rather than silently differing.
        if pd.notna(r.get('market_margin', np.nan)):
            s['edge_calibrated'] = round(centre - float(r['market_margin']), 2)
        keep = p > 0.002
        s['margin_dist'] = json.dumps({int(k): round(float(v), 4)
                                       for k, v in zip(GRID[keep], p[keep])})
        rows.append(s)

    out = pd.concat([preds.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out.to_csv(args.out)
    print(f"wrote {args.out}")

    done = out.dropna(subset=['median_margin'])
    on_key = done['mode_margin'].abs().isin(KEY_NUMBERS).mean()
    print(f"\nmode lands on a key number in {on_key:.0%} of games")
    print(f"median vs raw prediction: mean gap "
          f"{(done['median_margin'] - done[model_col]).abs().mean():.2f} points")

    if args.game:
        for _, r in done[done['short_name'].astype(str)
                         .str.contains(args.game, case=False)].iterrows():
            print(f"\n=== {r['short_name']} ===")
            print(f"  model prediction   {r[model_col]:+.1f}")
            print(f"  median margin      {int(r['median_margin']):+d}")
            print(f"  most likely margin {int(r['mode_margin']):+d} "
                  f"(p={r['mode_prob']:.3f})")
            print(f"  P(home win)        {r['p_home_win']:.1%}")
            if pd.notna(r.get('market_margin')):
                print(f"  market line        {r['market_margin']:+.1f}")
                print(f"  P(home covers)     {r['p_cover']:.1%}")
            if args.show_dist:
                d = {int(k): v for k, v in json.loads(r['margin_dist']).items()}
                top = sorted(d.items(), key=lambda kv: -kv[1])[:12]
                for k, v in sorted(top):
                    star = ' *' if abs(k) in KEY_NUMBERS else ''
                    print(f"    {k:+4d}  {v:6.2%}  {'#' * int(v * 300)}{star}")


if __name__ == '__main__':
    main()
