#!/usr/bin/env python3
"""Which teams are power-conference, in a given season.

WHY THIS IS NOT A CONSTANT

Every module that needed this used to carry its own frozen set,

    P4 = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12'}

and that set is wrong in three separate ways, each of which produced a wrong
answer in practice:

  Notre Dame      its conference is 'FBS Independents', so the biggest
                  independent in the sport counted as a group-of-five team.
  Pac-12 after    twelve teams through 2023, then two orphans in 2024-25, then
  the collapse    a rebuilt eight-team league in 2026 whose members are Boise
                  State, Colorado State, Fresno State, Oregon State, San Diego
                  State, Texas State, Utah State and Washington State. Reading
                  the 2026 Pac-12 as power promotes eight G5 teams.
  other           BYU was independent through 2022, and Army, Navy, UMass,
  independents    New Mexico State, Liberty and UConn have all been at times.
                  None of them are Notre Dame.

The defensive ratings score within tier specifically so recruiting is not paid
for flagging which tier a team is in, so getting the tier wrong corrupts the
thing the split exists to protect.

Membership read off the data, not from memory - see the docstring table in
conference history. FCS conferences (Big Sky, MVFC, SWAC and the rest) appear
in the frames as opponents and are simply not power.

Usage:
    from tiers import is_power, power_series
    d['power'] = power_series(d)          # needs conference, season, team
"""

import numpy as np
import pandas as pd

POWER_CONF = frozenset({'SEC', 'Big Ten', 'Big 12', 'ACC'})

# power while it was a twelve-team league; the 2024-25 remnant and the 2026
# rebuild are not the same competition wearing the same name
PAC12_LAST_POWER_SEASON = 2023

POWER_INDEPENDENTS = frozenset({'Notre Dame'})


def is_power(conference, season=None, team=None):
    """True if this team-season sits in a power conference."""
    if conference in POWER_CONF:
        return True
    if conference == 'Pac-12':
        return season is not None and season <= PAC12_LAST_POWER_SEASON
    if conference == 'FBS Independents':
        return team in POWER_INDEPENDENTS
    return False


def power_series(d, conference='conference', season='season', team='team'):
    """Boolean Series over a frame carrying conference, season and team.

    Missing conference gives False rather than an error - an unmatched team is
    not evidence of power-conference membership.
    """
    conf = d[conference] if conference in d else pd.Series(index=d.index,
                                                           dtype=object)
    sea = d[season] if season in d else pd.Series(index=d.index, dtype=float)
    tm = d[team] if team in d else pd.Series(index=d.index, dtype=object)
    return pd.Series(
        [is_power(c, s, t) for c, s, t in zip(conf, sea, tm)],
        index=d.index)


def tier_series(d, **kw):
    """'P4' / 'G5' labels, for the many places that want a string."""
    return np.where(power_series(d, **kw), 'P4', 'G5')


def main():
    """Print what this classifies, as a check."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    T = pd.read_csv(os.path.join(here, 'results', 'talent_by_position.csv'),
                    low_memory=False)[['team_id', 'season', 'conference']]
    T = T.drop_duplicates(['team_id', 'season'])
    tm = pd.read_csv(os.path.join(here, '..', 'collect', 'collect_espn_teams',
                                  'temp', 'teams.csv'))
    T['team'] = T['team_id'].map(dict(zip(tm['id'], tm['location'])))
    T['power'] = power_series(T)
    print(f"{'season':>8}{'power':>8}{'total':>8}")
    for s in sorted(T['season'].unique()):
        if s < 2014:
            continue
        g = T[T.season == s]
        print(f"{int(s):>8}{int(g.power.sum()):>8}{len(g):>8}")
    print("\nspot checks:")
    for team, season in (('Notre Dame', 2026), ('UConn', 2026),
                         ('Boise State', 2026), ('Oregon', 2023),
                         ('Oregon State', 2026), ('BYU', 2019),
                         ('BYU', 2026)):
        g = T[(T.team == team) & (T.season == season)]
        if len(g):
            x = g.iloc[0]
            print(f"  {team:<16}{season}  {str(x.conference):<20}"
                  f"{'POWER' if x.power else 'G5'}")


if __name__ == '__main__':
    main()
