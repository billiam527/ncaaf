#!/usr/bin/env python3
"""One trustworthy class year per player-season.

THE PROBLEM

cfbd_roster's `year` column holds two different quantities. Most rows carry a
class year, 1 through 5. About a quarter carry the calendar season instead:

    1      26,184        4     100,359
    2      29,389        5         258
    3      41,335        2014-2025  63,599

Valentino Espinoza reads year=2018 on the 2018 roster and year=2019 on the 2019
roster. Nothing downstream noticed, because 2018 is a number and every consumer
treated it as one.

WHAT THAT BROKE

Any code doing `pd.to_numeric(year).clip(1, 4)` turned 2018 into 4, so a quarter
of players were silently labelled seniors. That is how the class column on the
line and defence pages was built, and it is how the offensive-line age-curve
test was run - the one that concluded every development curve measured worse
than flat. That conclusion rested on class data that was a quarter wrong and
should be re-derived, not trusted.

THE FIX

Class year is reconstructed from the recruiting record instead:

    class year = season - recruiting class + 1

A player who signed in the 2023 class is in year 1 in 2023 and year 3 in 2025.
This does not depend on the roster field at all, it is internally consistent
across seasons by construction, and it uses the recruit link that
talent_by_position, ol_projection, front_seven and defensive_backs already join
for ratings.

Where there is no recruiting record the roster field is used, but only when it
is plausibly a class year - 1 to 6. A calendar season is treated as missing,
because a wrong class year is worse than none: it moves a player to a specific
wrong place in the depth chart rather than leaving him unranked.

Redshirts are not modelled. A redshirt sophomore reads as year 3 here, which is
his year on campus rather than his year of eligibility. That is the right
quantity for a development curve - what matters is how long he has been in the
programme - but it is not what a roster page would call him.

Usage:
    from class_year import class_years
    cy = class_years()            # pid, season, class_yr, class_src
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')

MAX_CLASS = 6           # above this the value is a calendar year, not a class


def first_recruit_id(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return str(parsed[0]) if isinstance(parsed, list) and parsed else None


def class_years(verbose=False):
    """One row per (pid, season) with class_yr and where it came from."""
    roster = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_roster.csv'),
                         low_memory=False)
    recruits = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_recruits.csv'),
                           low_memory=False)[['id', 'year']]
    recruits['id'] = recruits['id'].astype(str)
    recruits['recruit_class'] = pd.to_numeric(recruits['year'],
                                              errors='coerce')

    r = roster.copy()
    r['pid'] = r['id'].astype(str)
    r['season'] = pd.to_numeric(r['season'], errors='coerce')
    r['rid'] = r['recruitIds'].map(first_recruit_id)
    r = r.merge(recruits[['id', 'recruit_class']].rename(columns={'id': 'rid'}),
                on='rid', how='left')

    # primary: seasons since signing
    derived = r['season'] - r['recruit_class'] + 1
    derived = derived.where((derived >= 1) & (derived <= MAX_CLASS))

    # fallback: the roster field, but only when it is plausibly a class year
    raw = pd.to_numeric(r['year'], errors='coerce')
    fallback = raw.where((raw >= 1) & (raw <= MAX_CLASS))

    r['class_yr'] = derived.fillna(fallback)
    r['class_src'] = np.where(derived.notna(), 'recruit',
                       np.where(fallback.notna(), 'roster', 'none'))

    out = (r.dropna(subset=['season'])
             .drop_duplicates(['pid', 'season'])
             [['pid', 'season', 'class_yr', 'class_src']])
    out['season'] = out['season'].astype(int)

    if verbose:
        n = len(out)
        print(f"  {n:,} player-seasons")
        for src, g in out.groupby('class_src'):
            print(f"    {src:<9}{len(g):>9,}  ({len(g) / n:.0%})")
        bad = int(((raw > MAX_CLASS) & derived.isna()).sum())
        print(f"  roster field was a calendar year and nothing else was "
              f"available: {bad:,}")
        ok = out.dropna(subset=['class_yr'])
        print(f"  class year distribution:")
        for y, c in ok['class_yr'].value_counts().sort_index().items():
            print(f"    year {int(y)}   {c:>9,}")
    return out


def check():
    """Does the derived class actually increment for a returning player?"""
    cy = class_years(verbose=True)
    cy = cy.dropna(subset=['class_yr'])
    nxt = cy.copy()
    nxt['season'] -= 1
    nxt = nxt.rename(columns={'class_yr': 'next_yr'})
    pair = cy.merge(nxt[['pid', 'season', 'next_yr']], on=['pid', 'season'],
                    how='inner')
    step = pair['next_yr'] - pair['class_yr']
    print(f"\n  consecutive seasons for the same player: {len(pair):,}")
    print(f"  the class advances by exactly one in {(step == 1).mean():.0%}")
    print("  distribution of the step:")
    for v, c in step.value_counts().sort_index().head(6).items():
        print(f"    {v:+.0f}   {c:>8,}")
    print("\n  under the old roster field, for comparison:")
    roster = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_roster.csv'),
                         low_memory=False)
    roster['pid'] = roster['id'].astype(str)
    roster['season'] = pd.to_numeric(roster['season'], errors='coerce')
    roster['old'] = pd.to_numeric(roster['year'], errors='coerce').clip(1, 4)
    o = roster.dropna(subset=['season', 'old']).drop_duplicates(
        ['pid', 'season'])[['pid', 'season', 'old']]
    on = o.copy()
    on['season'] -= 1
    on = on.rename(columns={'old': 'old_next'})
    op = o.merge(on, on=['pid', 'season'], how='inner')
    ostep = op['old_next'] - op['old']
    print(f"    advances by exactly one in {(ostep == 1).mean():.0%} "
          f"of {len(op):,} pairs")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'class_year.csv'))
    args = ap.parse_args()
    if args.check:
        check()
        return
    cy = class_years(verbose=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cy.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(cy):,} player-seasons)")


if __name__ == '__main__':
    main()
