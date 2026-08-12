#!/usr/bin/env python3
"""Team talent per season, from recruiting classes and the 247 composite.

The efficiency features describe how a team has played; these describe what it
has to play with. That distinction is most of why the model cannot reach the
spreads the market posts on mismatches - a MAC team's opponent-adjusted numbers
look respectable against MAC opponents, and nothing in the feature set says its
two-deep is three stars lower.

Two measures, because they cover different years:

  talent          the 247 roster composite straight from CFBD. The right
                  measure, but only 2015-2025 - there is none for a season not
                  yet played, so it is unusable for the thing we most want to
                  predict.
  talent_roll     a rolling weighted sum of recruiting class points over the
                  previous CLASS_WINDOW years, which exists for any season
                  including 2026. Classes are weighted by how much of a roster
                  they still occupy: a fourth-year class contributes less than a
                  second-year one because much of it has left.

Both are also emitted as a within-season percentile, which is what actually
matters - the raw points scale drifts as the industry re-rates.

Usage:
    python team_talent.py --out results/team_talent.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ETL = os.path.dirname(_HERE)
PLAYER_DIR = os.path.join(_ETL, 'collect', 'collect_cfbd_players', 'temp')

# A roster is roughly four recruiting classes. Weights taper because the oldest
# class has been thinned by departures and the newest is mostly redshirting.
CLASS_WINDOW = 4
CLASS_WEIGHTS = {1: 0.9, 2: 1.0, 3: 1.0, 4: 0.7}


def load(name):
    path = os.path.join(PLAYER_DIR, f'cfbd_{name}.csv')
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run collect_cfbd_recruiting.py first")
    return pd.read_csv(path, low_memory=False)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'team_talent.csv'))
    args = ap.parse_args()

    rec = load('recruiting')
    tal = load('talent')
    teams = load('teams')
    name_to_id = dict(zip(teams['school'], teams['id']))

    rec['points'] = pd.to_numeric(rec['points'], errors='coerce')
    rec = rec.dropna(subset=['points', 'team', 'year'])

    # rolling class composite: for season Y, sum classes Y-1 back to Y-CLASS_WINDOW
    rows = []
    seasons = range(int(rec.year.min()) + CLASS_WINDOW, int(rec.year.max()) + 2)
    by_team_year = {(r.team, int(r.year)): r.points for r in rec.itertuples()}
    all_teams = sorted(rec['team'].unique())
    for season in seasons:
        for team in all_teams:
            total, weight = 0.0, 0.0
            for lag in range(1, CLASS_WINDOW + 1):
                pts = by_team_year.get((team, season - lag))
                if pts is None:
                    continue
                w = CLASS_WEIGHTS.get(lag, 0.7)
                total += w * pts
                weight += w
            if weight >= 2.0:          # need most of a roster to be represented
                rows.append({'season': season, 'team': team,
                             'talent_roll': total / weight})
    roll = pd.DataFrame(rows)

    tal = tal.rename(columns={'year': 'season'})
    out = roll.merge(tal[['season', 'team', 'talent']], on=['season', 'team'],
                     how='outer')

    # Percentile within season: the raw scales drift year to year as the
    # services re-rate, so a level is not comparable across seasons.
    for col in ('talent', 'talent_roll'):
        out[f'{col}_pct'] = out.groupby('season')[col].rank(pct=True)

    out['team_id'] = out['team'].map(name_to_id)
    matched = out['team_id'].notna().mean()
    print(f"team id match: {matched:.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} team-seasons)")
    print(f"\n  coverage by column:")
    for c in ('talent', 'talent_roll', 'talent_pct', 'talent_roll_pct'):
        sub = out[out[c].notna()]
        if len(sub):
            print(f"    {c:<18} {out[c].notna().mean():>6.1%}  "
                  f"{int(sub.season.min())}-{int(sub.season.max())}")

    both = out.dropna(subset=['talent_pct', 'talent_roll_pct'])
    if len(both) > 100:
        print(f"\n  the two measures agree at r="
              f"{both['talent_pct'].corr(both['talent_roll_pct']):+.3f} "
              f"on {len(both)} overlapping team-seasons")

    cur = out[out.season == out.season.max()].nlargest(8, 'talent_roll')
    print(f"\n  most talent for {int(out.season.max())}:")
    for _, r in cur.iterrows():
        print(f"    {str(r['team'])[:24]:<26} rolling {r['talent_roll']:>7.1f}"
              f"   pct {r['talent_roll_pct']:.0%}")


if __name__ == '__main__':
    main()
