"""Attach market spreads to the model's predictions and compute the edge.

Uses the CollegeFootballData lines already collected by
etl/collect/collect_cfbd_games (a documented, key-authenticated API), rather
than scraping a sportsbook. FanDuel in particular refuses automated access at
three layers - plain requests get 403, the JSON API rejects unauthenticated
calls, and a headless browser is served a captcha - so it is not a source that
can be read programmatically without deliberately defeating bot detection.
CFBD carries DraftKings, Bovada, ESPN Bet, Caesars and a consensus line, which
is equivalent for judging whether the model disagrees with the market.

Sign convention, which is easy to get backwards:
    CFBD `spread`   negative means the HOME team is favoured (TCU -7.5)
    market_margin   = -spread, so positive means home favoured, matching the
                      model's home_score_differential target
    edge            = model prediction - market_margin
                      positive => model likes the HOME side relative to market

    python add_market_lines.py
    python add_market_lines.py --provider DraftKings --min-edge 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

PREDICTIONS = os.path.join(_HERE, 'prediction_file', 'new_predictions.csv')
FALLBACK_PREDICTIONS = os.path.join(_HERE, 'prediction_file', 'predictions.csv')
TEAMS = os.path.join(_REPO, 'etl', 'collect', 'collect_espn_teams', 'temp', 'teams.csv')
LINES = os.path.join(_REPO, 'etl', 'collect', 'collect_cfbd_games', 'cfbd_spread_data.csv')
OUT = os.path.join(_HERE, 'prediction_file', 'predictions_with_lines.csv')

MODEL_COLS = ['blended_prediction', 'blended_model',
              'in_season_model_preds', 'preseason_model_preds']


def week_to_int(w):
    if pd.isna(w):
        return np.nan
    text = str(w).strip()
    if text.lower().startswith('week'):
        try:
            return int(text.split()[-1])
        except ValueError:
            return np.nan
    try:
        return int(float(text))
    except ValueError:
        return np.nan


def season_from_date(d):
    """College football seasons span a calendar year boundary."""
    ts = pd.to_datetime(d, errors='coerce', utc=True)
    if pd.isna(ts):
        return np.nan
    return ts.year if ts.month >= 8 else ts.year - 1


def espn_id_to_name():
    """ESPN team id -> the name CFBD uses. `location` matches CFBD directly."""
    t = pd.read_csv(TEAMS)
    for col in ('location', 'display_name', 'short_display_name', 'name'):
        if col in t.columns:
            return dict(zip(t['id'], t[col])), t
    raise SystemExit(f"no usable name column in {TEAMS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', default=None,
                    help="single book (e.g. DraftKings, Bovada, consensus). "
                         "Default: median across all books per game.")
    ap.add_argument('--min-edge', type=float, default=0.0,
                    help='only list games where |edge| is at least this many points')
    ap.add_argument('--week', type=int, default=None, help='limit to one week')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    src = PREDICTIONS if os.path.exists(PREDICTIONS) else FALLBACK_PREDICTIONS
    if not os.path.exists(src):
        raise SystemExit(f"no predictions found. Run predict.sh first.\n  looked for {PREDICTIONS}")
    preds = pd.read_csv(src, index_col=0)
    print(f"predictions: {len(preds)} games from {os.path.basename(src)}")

    model_col = next((c for c in MODEL_COLS if c in preds.columns), None)
    if model_col is None:
        raise SystemExit(f"no model prediction column found; looked for {MODEL_COLS}")
    print(f"  using model column: {model_col}")

    id_to_name, _ = espn_id_to_name()
    preds['home_team'] = preds['home_team_id'].map(id_to_name)
    preds['away_team'] = preds['away_team_id'].map(id_to_name)
    preds['week_num'] = preds['week'].map(week_to_int)
    preds['season'] = preds['date'].map(season_from_date)

    unmapped = preds['home_team'].isna().sum() + preds['away_team'].isna().sum()
    if unmapped:
        print(f"  WARNING: {unmapped} team id(s) had no name mapping")

    lines = pd.read_csv(LINES, low_memory=False)
    lines = lines[lines['spread'].notna()]
    if args.provider:
        before = len(lines)
        lines = lines[lines['provider'].str.lower() == args.provider.lower()]
        print(f"  provider filter '{args.provider}': {before} -> {len(lines)} line rows")
        if lines.empty:
            raise SystemExit("no rows for that provider")

    # One row per game: median spread across books is more robust than any
    # single book, and books rarely differ by more than a point.
    agg = {'spread': 'median', 'provider': lambda s: ', '.join(sorted(set(s)))}
    if 'over_under' in lines.columns:
        agg['over_under'] = 'median'
    grouped = (lines.groupby(['season', 'week', 'home_team', 'away_team'], as_index=False)
                    .agg(agg)
                    .rename(columns={'week': 'week_num', 'provider': 'books'}))

    merged = preds.merge(grouped, on=['season', 'week_num', 'home_team', 'away_team'],
                         how='left')

    merged['market_margin'] = -merged['spread']
    merged['edge'] = merged[model_col] - merged['market_margin']

    matched = merged['spread'].notna().sum()
    print(f"  matched a market line for {matched}/{len(merged)} games")
    if matched == 0:
        print("  (lines may not be posted yet for these weeks)")

    merged.to_csv(args.out)
    print(f"  wrote {args.out}")

    view = merged[merged['spread'].notna()].copy()
    if args.week is not None:
        view = view[view['week_num'] == args.week]
    view = view[view['edge'].abs() >= args.min_edge]
    if view.empty:
        print("\nno games meet the filter")
        return

    view = view.reindex(view['edge'].abs().sort_values(ascending=False).index)
    print(f"\n{'matchup':<24}{'wk':>4}{'model':>8}{'market':>8}{'edge':>8}  books")
    print("-" * 78)
    for _, r in view.iterrows():
        side = 'HOME' if r['edge'] > 0 else 'AWAY'
        print(f"{str(r['short_name'])[:23]:<24}{int(r['week_num']):>4}"
              f"{r[model_col]:>8.1f}{r['market_margin']:>8.1f}{r['edge']:>+8.1f}"
              f"  {side}  {str(r['books'])[:28]}")
    print("-" * 78)
    print(f"  {len(view)} games | mean |edge| {view['edge'].abs().mean():.2f} | "
          f"model leans home in {(view['edge'] > 0).mean():.0%}")


if __name__ == '__main__':
    main()
