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

Produces per game:
    median_margin     the calibrated point estimate
    mode_margin       most likely single outcome, football-shaped
    p_home_win        probability the home team wins outright
    p_cover           probability of covering the market line, when known
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

LO, HI = -70, 70
GRID = np.arange(LO, HI + 1)
SMOOTHING = 2.5          # sigma used to build the "no lumpiness" reference
KEY_NUMBERS = (3, 7, 10, 14, 17, 21)

# Seasons are weighted 0.5 ** (age / HALF_LIFE). The scoring distribution
# drifts: two-point attempts have pushed margins of 2 from 2.16% to 3.26% and
# 8 from 2.26% to 2.80% since 2010, while 1-point and 14-point games have
# receded and overall spread has narrowed (sd 21.4 -> 20.2). Weighting keeps
# every season in the estimate while letting recent rules and play-calling
# dominate.
HALF_LIFE = 5.0


def season_weights(seasons, half_life=HALF_LIFE, reference=None):
    if half_life is None or half_life <= 0:
        return np.ones(len(seasons))
    ref = reference if reference is not None else seasons.max()
    return 0.5 ** ((ref - seasons) / half_life)


def fbs_games():
    g = pd.read_csv(GAMES, low_memory=False)
    t = pd.read_csv(TEAMS)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
    g = g.dropna(subset=['home_score_differential', 'season']).copy()
    g['margin'] = g['home_score_differential'].round().astype(int)
    return g[['season', 'margin']]


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
    return np.clip(np.where(smooth > 1e-9, emp / smooth, 1.0), 0.15, 4.0)


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


def residual_sigma(half_life=HALF_LIFE, calibrator=None, fit_col='blended'):
    """Recency-weighted spread of outcomes around the (optionally calibrated)
    prediction. Calibrating shifts the centre, so sigma must be measured
    against the same quantity the distribution is centred on."""
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


def validate(weights, sigma, calibrator=None, fit_col='blended'):
    """Are the probabilities honest? Compare predicted P(win) to realised."""
    h = load_history(fit_col)
    col = fit_col if fit_col in h.columns else 'in_season_model_preds'
    raw = h[col].to_numpy(float)
    actual = h['home_score_differential'].to_numpy(float)
    preds = calibrator.predict(raw) if calibrator is not None else raw

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
    ap.add_argument('--raw-centre', action='store_true',
                    help='centre on the raw model output instead of the calibrated one')
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

    calibrator = None if args.raw_centre else fit_calibrator(hl, fit_col)
    sigma = args.sigma or residual_sigma(hl, calibrator, fit_col)
    eff = season_weights(games['season'].to_numpy(float), hl).sum()
    print(f"sigma = {sigma:.2f} points; key-number weights from {len(games)} FBS games"
          f" ({eff:.0f} effective, half-life "
          f"{'off' if hl is None else f'{hl:g} seasons'}); "
          f"centre = {'raw prediction' if calibrator is None else 'calibrated'}"
          f", fitted on {fit_col}\n")

    if args.validate:
        validate(weights, sigma, calibrator, fit_col)
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
                  if calibrator is not None else float(r[model_col]))
        p = distribution(centre, weights, sigma)
        s = {'calibrated_margin': round(centre, 2)}
        s.update(summarise(p, r.get('market_margin', np.nan)))
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
