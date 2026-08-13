#!/usr/bin/env python3
"""Collect recruiting and roster-talent data from CFBD.

The model's features describe how teams have played. The market also prices how
much talent they have, which is why it stretches to spreads the model cannot
reach: a 50-point line on Ball State at Ohio State is a statement about the
two-deep, not about last season's efficiency against MAC opponents.

Two sources, because neither alone covers what is needed:

  /talent            the 247 roster talent composite, one number per team-season.
                     Exactly the right measure, but it stops at 2025 - there is
                     no figure for a season that has not started.
  /recruiting/teams  class ranking and points per year, available for 2026 and
                     back to 2010. A rolling weighted sum of the last four
                     classes approximates the roster composite for any year,
                     including one still to be played.

Usage:
    python collect_cfbd_recruiting.py --start-year 2010 --end-year 2026
"""

import argparse
import os
import sys
import time

import pandas as pd

from collect_cfbd_players import fetch, load_cfbd_key, KEY_FILE


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start-year', type=int, default=2010)
    ap.add_argument('--end-year', type=int, default=2026)
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'temp'))
    ap.add_argument('--pause', type=float, default=0.25)
    args = ap.parse_args()

    key = load_cfbd_key()
    if not key:
        print(f"ERROR: no CFBD API key. Set CFBD_API_KEY or write {KEY_FILE}")
        sys.exit(1)
    headers = {'Authorization': f'Bearer {key}'}
    os.makedirs(args.out_dir, exist_ok=True)

    talent, classes, players = [], [], []
    for year in range(args.start_year, args.end_year + 1):
        t = fetch('/talent', headers, {'year': year})
        c = fetch('/recruiting/teams', headers, {'year': year})
        # Per-recruit rows are what let roster.recruitIds be resolved into an
        # actual star rating, which is the difference between measuring the
        # talent a team signed and the talent it currently has.
        p = fetch('/recruiting/players', headers, {'year': year})
        if t:
            talent.append(pd.DataFrame(t))
        if c:
            classes.append(pd.DataFrame(c))
        if p:
            players.append(pd.DataFrame(p))
        print(f"  {year}: talent {len(t) if t else 0:>4}   "
              f"class {len(c) if c else 0:>4}   "
              f"recruits {len(p) if p else 0:>5}")
        time.sleep(args.pause)

    if talent:
        df = pd.concat(talent, ignore_index=True)
        df.to_csv(os.path.join(args.out_dir, 'cfbd_talent.csv'), index=False)
        print(f"\nwrote cfbd_talent.csv     {len(df):>6} rows, "
              f"{int(df.year.min())}-{int(df.year.max())}")
    if classes:
        df = pd.concat(classes, ignore_index=True)
        df.to_csv(os.path.join(args.out_dir, 'cfbd_recruiting.csv'), index=False)
        print(f"wrote cfbd_recruiting.csv {len(df):>6} rows, "
              f"{int(df.year.min())}-{int(df.year.max())}")
    if players:
        df = pd.concat(players, ignore_index=True)
        df = df.drop(columns=[c for c in ('hometownInfo',) if c in df.columns])
        df.to_csv(os.path.join(args.out_dir, 'cfbd_recruits.csv'), index=False)
        print(f"wrote cfbd_recruits.csv    {len(df):>6} rows, "
              f"{int(df.year.min())}-{int(df.year.max())}")


if __name__ == '__main__':
    main()
