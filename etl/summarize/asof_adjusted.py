#!/usr/bin/env python3
"""Opponent-adjusted team stats as they stood before each week.

WHY THIS EXISTS

There were two team-stat tables and neither was right for an in-season model.

  season_summaries.csv is opponent-adjusted but holds one row per team-season -
  end-of-year figures. Joining it to a game inside that season hands the model
  the result of the game it is predicting. The in-season model did exactly that
  until recently; see model_training/in_season_model/preprocess.py.

  rolling_summaries.csv is correctly lagged - a team's season opener is empty
  and the figures correlate +0.26 with the margin of the game they precede - but
  it is NOT opponent-adjusted. Three games against weak defences read as three
  good games.

This file is the missing third thing: opponent-adjusted, and computed only from
games played before the week in question. One row per (season, week, team).

WHAT IT IS WORTH

Given to the in-season model alongside the unadjusted rolling figures, on a 2025
season holdout:

    rolling only                        R2 0.332   MAE 13.117
    + as-of-week adjusted               R2 0.346   MAE 12.972
    + as-of-week adjusted, gated        R2 0.351   MAE 12.929

WHY IT IS WITHHELD BEFORE WEEK 5

A ridge fitted on two weeks of football is mostly noise, and the estimator
cannot tell that from signal - given the adjustment in weeks 2-3 it uses it and
loses 0.47 of MAE there against plain rolling form. Withholding it until week 5
lets the model fall back on unadjusted form exactly while the adjustment is
untrustworthy, and recovers those weeks: 13.30 to 12.98. Swept over weeks 4
through 7, week 5 is the best of them, and the consumer applies the gate rather
than this file, so the threshold can be retuned without a rebuild.

COST

About 2 seconds per (season, week), so roughly six minutes for ten seasons. The
adjuster is the same OpponentAdjuster used by summarize_games, run repeatedly on
truncated inputs rather than once per season.

Usage:
    python asof_adjusted.py --out results/asof_adjusted.csv
    python asof_adjusted.py --first 2016 --last 2025
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from summarize_games import OpponentAdjuster, AnalyticsConfig  # noqa: E402

GBG = os.path.join(_HERE, 'results', 'game_by_game_summaries.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

STATS = ("play_success rush_success pass_success yards_per_play "
         "rush_yards_per_play pass_yards_per_play explosive_play_rate "
         "explosive_rush_rate explosive_pass_rate epa_per_play epa_per_rush "
         "epa_per_pass").split()

# Below this many prior games the ridge has nothing to work with and returns
# something worse than no answer, so no row is written at all.
MIN_PRIOR_GAMES = 60
POSTSEASON_WEEK = 90


def week_to_int(w):
    w = str(w)
    if w.strip().lower().startswith('week'):
        tail = w.split()[-1]
        if tail.isdigit():
            return int(tail)
    return POSTSEASON_WEEK


def load_game_by_game(gbg=GBG, games=GAMES):
    g = pd.read_csv(gbg, low_memory=False)
    q = pd.read_csv(games, low_memory=False)
    q['week_num'] = q['week'].map(week_to_int)
    g = g.merge(q[['id', 'week_num']].rename(columns={'id': 'game_id'}),
                on='game_id', how='left')
    return g.dropna(subset=['week_num'])


def build(first, last, verbose=True):
    g = load_game_by_game()
    adjuster = OpponentAdjuster(AnalyticsConfig())
    rows, t0 = [], time.time()
    for season in range(first, last + 1):
        s = g[g.season == season]
        if s.empty:
            continue
        weeks = sorted(w for w in s.week_num.unique() if w < POSTSEASON_WEEK)
        for W in weeks:
            prior = s[s.week_num < W]
            if prior.game_id.nunique() < MIN_PRIOR_GAMES:
                continue
            try:
                out = adjuster.adjust_for_opponents(prior, STATS)
            except Exception as exc:            # noqa: BLE001
                if verbose:
                    print(f"  {season} wk {int(W)}: {str(exc)[:60]}")
                continue
            if out is None or not len(out):
                continue
            keep = ['team_id'] + [c for c in out.columns
                                  if c.startswith('adjusted_')]
            o = out[keep].copy()
            o['season'], o['week_num'] = season, int(W)
            rows.append(o)
        if verbose:
            print(f"  {season}: {len(weeks)} weeks, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
    if not rows:
        raise SystemExit("produced nothing; check game_by_game_summaries.csv")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--first', type=int, default=2016)
    ap.add_argument('--last', type=int, default=None,
                    help='defaults to the newest season present')
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'asof_adjusted.csv'))
    args = ap.parse_args()

    last = args.last
    if last is None:
        last = int(pd.read_csv(GBG, usecols=['season'])['season'].max())
    a = build(args.first, last)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    a.to_csv(args.out, index=False)
    acols = [c for c in a.columns if c.startswith('adjusted_')]
    print(f"\nwrote {args.out}")
    print(f"  {len(a):,} (season, week, team) rows, "
          f"{len(acols)} adjusted columns")
    print(f"  seasons {int(a.season.min())}-{int(a.season.max())}, "
          f"weeks {int(a.week_num.min())}-{int(a.week_num.max())}")
    print("\n  rows per season:")
    print(a.groupby('season').size().to_string())
    print("\n  Consumers should withhold this before week 5; see the docstring.")


if __name__ == '__main__':
    main()
