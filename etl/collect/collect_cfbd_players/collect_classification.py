#!/usr/bin/env python3
"""Division classification per team per season.

teams.csv carries a single fbs_ind, set by scraping an ESPN page and marking
every team slug found on it. That is wrong in two ways. It does not check what
division a team plays in, so FCS programmes linked from the page arrive flagged
as FBS. And it is a single value applied to every season, while classification
changes: 42 teams move divisions between 2010 and 2026, and the FBS field grows
from 120 teams to 138.

Applied to the whole window, the current flag puts 231 games into the model as
FBS-vs-FBS that were not - James Madison before 2022, Sam Houston before 2024,
Liberty before 2019, Georgia State before 2013, and Idaho after it dropped back
to FCS in 2018. They average 20.8 points of margin against 16.9 overall, which
is what a mismatch looks like.

CFBD publishes classification per team per year and its team ids are ESPN team
ids, so this is an exact join.

Usage:
    python collect_classification.py --start-year 2010 --end-year 2026
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
    ap.add_argument('--pause', type=float, default=0.2)
    args = ap.parse_args()

    key = load_cfbd_key()
    if not key:
        print(f"ERROR: no CFBD API key. Set CFBD_API_KEY or write {KEY_FILE}")
        sys.exit(1)
    headers = {'Authorization': f'Bearer {key}'}
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for year in range(args.start_year, args.end_year + 1):
        teams = fetch('/teams', headers, {'year': year})
        for t in teams or []:
            rows.append({'season': year, 'team_id': t.get('id'),
                         'team': t.get('school'),
                         'conference': t.get('conference'),
                         'classification': t.get('classification')})
        fbs = sum(1 for t in (teams or []) if t.get('classification') == 'fbs')
        print(f"  {year}: {len(teams or []):>4} teams, {fbs:>3} FBS")
        time.sleep(args.pause)

    df = pd.DataFrame(rows).dropna(subset=['team_id'])
    df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
    df = df.dropna(subset=['team_id'])
    df['fbs'] = (df['classification'] == 'fbs').astype(int)

    path = os.path.join(args.out_dir, 'cfbd_classification.csv')
    df.to_csv(path, index=False)
    print(f"\nwrote {os.path.basename(path)}  ({len(df)} team-seasons)")
    moved = df.groupby('team')['classification'].nunique()
    print(f"  teams that change division in this window: {int((moved > 1).sum())}")


if __name__ == '__main__':
    main()
