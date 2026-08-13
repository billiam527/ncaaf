#!/usr/bin/env python3
"""Talent of the players actually on the roster, not the talent signed.

team_talent.py measures recruiting class points summed over four years. That
conflates quantity with quality - a large class of three-stars outscores a small
one with four blue-chips - and it counts recruits who transferred out years ago
while missing a five-star who arrived through the portal.

This joins the roster to per-recruit ratings through roster.recruitIds, so what
gets measured is the talent present this season. About 40% of roster rows carry
a resolvable recruit id; the rest are walk-ons and unrated signings, so ratios
here use the linked players as the denominator, which is what published
blue-chip figures do. The result lines up with them: Georgia 76.7%, Ohio State
74.4%, Alabama 74.0% for 2025.

Emitted per team-season:

  blue_chip_ratio   share of linked roster rated four or five stars
  mean_rating       mean recruit rating, independent of class size
  top22_rating      mean rating of the 22 best - a two-deep proxy
  <group>_rating    the same by position group, where the market's premium on
                    quarterback talent can show up separately

Usage:
    python roster_talent.py --out results/roster_talent.csv
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ETL = os.path.dirname(_HERE)
PLAYER_DIR = os.path.join(_ETL, 'collect', 'collect_cfbd_players', 'temp')

POSITION_GROUPS = {
    'QB': 'QB',
    'RB': 'SKILL', 'FB': 'SKILL', 'WR': 'SKILL', 'TE': 'SKILL', 'ATH': 'SKILL',
    'OL': 'OL', 'C': 'OL', 'G': 'OL', 'OT': 'OL', 'OG': 'OL', 'T': 'OL',
    'DL': 'DL', 'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'EDGE': 'DL',
    'LB': 'LB', 'ILB': 'LB', 'OLB': 'LB', 'MLB': 'LB',
    'DB': 'DB', 'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB',
}
GROUPS = ('QB', 'SKILL', 'OL', 'DL', 'LB', 'DB')
TWO_DEEP = 22
MIN_LINKED = 20            # below this a team-season ratio is not meaningful


def load(name):
    path = os.path.join(PLAYER_DIR, f'cfbd_{name}.csv')
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run collect_cfbd_recruiting.py first")
    return pd.read_csv(path, low_memory=False)


def first_recruit_id(value):
    """roster.recruitIds arrives as the string form of a list."""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'roster_talent.csv'))
    args = ap.parse_args()

    roster = load('roster')
    recruits = load('recruits')
    teams = load('teams')
    name_to_id = dict(zip(teams['school'], teams['id']))

    roster['rid'] = roster['recruitIds'].map(first_recruit_id)
    linked_share = roster['rid'].notna().mean()
    recruits['id'] = recruits['id'].astype(str)
    recruits['rating'] = pd.to_numeric(recruits['rating'], errors='coerce')
    recruits['stars'] = pd.to_numeric(recruits['stars'], errors='coerce')

    j = roster.dropna(subset=['rid']).merge(
        recruits[['id', 'stars', 'rating']], left_on='rid', right_on='id',
        how='inner', suffixes=('', '_rec'))
    j = j.dropna(subset=['rating'])
    print(f"roster rows {len(roster):,}, {linked_share:.1%} carry a recruit id")
    print(f"joined and rated: {len(j):,} player-seasons")

    j['group'] = j['position'].map(POSITION_GROUPS)
    j['blue'] = j['stars'] >= 4

    rows = []
    for (team, season), g in j.groupby(['team', 'season']):
        if len(g) < MIN_LINKED:
            continue
        row = {'team': team, 'season': int(season),
               'linked': len(g),
               'blue_chip_ratio': float(g['blue'].mean()),
               'mean_rating': float(g['rating'].mean()),
               'top22_rating': float(g['rating'].nlargest(TWO_DEEP).mean()),
               'five_stars': int((g['stars'] >= 5).sum())}
        for grp in GROUPS:
            sub = g[g['group'] == grp]
            row[f'{grp}_rating'] = float(sub['rating'].mean()) if len(sub) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)

    # Percentiles within season: the ratings scale drifts as the services
    # re-rate, so a level is not comparable across years.
    pct_cols = (['blue_chip_ratio', 'mean_rating', 'top22_rating']
                + [f'{g}_rating' for g in GROUPS])
    for col in pct_cols:
        out[f'{col}_pct'] = out.groupby('season')[col].rank(pct=True)

    out['team_id'] = out['team'].map(name_to_id)
    print(f"team id match: {out['team_id'].notna().mean():.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} team-seasons, "
          f"{int(out.season.min())}-{int(out.season.max())})")

    print("\n  coverage by season:")
    for s, n in out.groupby('season').size().items():
        print(f"    {int(s)}: {n:>4} teams")

    cur = out[out.season == out.season.max()]
    print(f"\n  blue-chip ratio, {int(out.season.max())}:")
    for _, r in cur.nlargest(6, 'blue_chip_ratio').iterrows():
        print(f"    {str(r['team'])[:22]:<24}{r['blue_chip_ratio']:>6.1%}"
              f"   QB {r['QB_rating']:.3f}   top22 {r['top22_rating']:.3f}")


if __name__ == '__main__':
    main()
