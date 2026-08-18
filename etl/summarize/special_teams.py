#!/usr/bin/env python3
"""Special teams, valued in points above the average kick from the same spot.

This replaces an earlier version of this file that concluded special teams was
worthless. That conclusion was wrong, and it was wrong for three separate
reasons, all of them fixed here. The history is kept in the docstring because
each mistake is easy to repeat.

WHAT WAS WRONG BEFORE

  Attribution. team_id on a kick is not the kicking team. edit_pbp_file.py
  reassigns it to the next play's team on every kickoff, and on punts only when
  the punt was returned. Checked against the rosters of the men actually named
  in the play text: on kickoffs team_id is the RECEIVING team 27,167 times
  against 529, and on punts it is mixed, 19,396 to 7,198. Field goals are clean,
  7,724 to 0. So the old table's "kickoff touchback rate" was the rate at which
  a team's OPPONENTS kicked touchbacks, which repeats at 0.42 mostly because
  schedules repeat. Attribution here comes from the drive instead: a drive has
  one offense, which is the punting and field-goal team, and its opponent is the
  team that kicked off to it.

  Valuation. The old table used raw rates - net punting, touchback rate, field
  goal percentage. Rates ignore the situation. Fixed below by valuing every kick
  in expected points and then subtracting what an average kick from the same
  state is worth.

  Shape. Five collinear rate columns were offered to the margin model at once.
  One aggregate is the right object, and it is the one that works.

THE EPA COLUMN IN THE PBP IS UNUSABLE ON KICKS, AND HERE IS WHY

Two bugs upstream, in edit_pbp_file.py:

  play_type_id 59 is "Field Goal Good". The scoring dictionary assigns it 3
  points and then a later mask, believing 59 means Safety, overwrites it with
  -2. Real safeties are id 20 and that branch never touches them. So every made
  field goal in the file carries actual_points = -2 and epa = -2 - EP = -4.41.

  There is no sign flip across a change of possession. A punt is scored as
  next_EP - current_EP, but the next play belongs to the other team, so its EP
  has to be negated first. Uncorrected, a punt reads +2.39.

The expected_points column itself is sound - it falls smoothly from about +4.8
inside the ten to -0.28 backed up against a team's own goal line - so what is
rebuilt here is the application to kicks, not the model.

HOW A KICK IS VALUED

Signed to the kicking team:

    EPA = points_scored + (-EP_after) - EP_before

EP_after is the expected points of the state the other team inherits, negated
because it is theirs. A kickoff is a free kick rather than a down, so it has no
"before" state belonging to the kicker and is valued purely by what the
receiving team inherits.

That is correct EPA and it is NOT a unit rating. Raw EPA on a field goal climbs
with distance - -0.63 from inside 25 yards against +0.19 from 50-plus - because
a short attempt came from a position already worth four and a half points and
settling for three is a disappointment. That measures an offence stalling. So
each kick is compared with the league's average from the same state:

    field goals   by distance, in five-yard bands
    punts         by line of scrimmage, in ten-yard bands
    kickoffs      by season, because the touchback rules moved twice

all within season. What is left is the part the unit controls.

WHAT IT IS WORTH

Teams spread about 16.8 points a season above and below average, which is 1.4
points a game, and the aggregate repeats year over year at 0.248. So roughly a
third of a point per game is knowable in advance. Small, and real: added to the
margin model as one column it improves out-of-sample R2 by +0.0023 and MAE by
0.026 points on a 2017-22 / 2023-25 split, is positive on four of five rolling
splits, and beats 100% of 400 one-column noise placebos and 100% of 400
shuffled-feature placebos.

The pieces behave as football says they should. Kickoffs repeat best at 0.259
and punting at 0.191; field-goal accuracy repeats at 0.030 and is worth nothing,
which is the long-standing finding in every sport with a kicker in it. Offered
separately, the field-goal and punting columns each make the model worse and
only the aggregate helps, so the aggregate is what ships.

Usage:
    python special_teams.py --out results/special_teams.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

# a team-season needs this many kicks before its total means anything
MIN_KICKS = 60


def load_kicks(pbp=PBP, games=GAMES, chunksize=1_000_000):
    """Every punt, kickoff and field goal, valued in expected points."""
    g = pd.read_csv(games, low_memory=False)[
        ['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    season_of = dict(zip(g['id'].astype(int), g['season'].astype(int)))
    pair = g.set_index('id')[['home_team_id', 'away_team_id']].astype(int)

    cols = ['game_id', 'drive_id', 'team_id', 'play_type_text', 'down',
            'distance', 'yards_to_goal', 'expected_points',
            'special_teams_play', 'offensive_play']
    out = []
    for chunk in pd.read_csv(pbp, usecols=cols, chunksize=chunksize,
                             low_memory=False):
        chunk['season'] = chunk['game_id'].map(season_of)
        chunk['team_id'] = pd.to_numeric(chunk['team_id'], errors='coerce')
        chunk = chunk.dropna(subset=['season', 'team_id', 'game_id']).copy()
        chunk['team_id'] = chunk['team_id'].astype(int)
        chunk['season'] = chunk['season'].astype(int)

        # the drive's offence, from scrimmage plays, which do carry the right
        # team_id - this is what replaces team_id on kicks
        scr = chunk[(chunk.offensive_play == 1)
                    & (chunk.special_teams_play != 1)]
        off = (scr.groupby('drive_id')['team_id']
               .agg(lambda s: s.value_counts().idxmax()).rename('drive_off'))
        chunk = chunk.merge(off, left_on='drive_id', right_index=True,
                            how='left')
        chunk = chunk.join(pair, on='game_id')
        chunk['opp'] = np.where(chunk['drive_off'] == chunk['home_team_id'],
                                chunk['away_team_id'], chunk['home_team_id'])

        chunk['nxt_ep'] = chunk['expected_points'].shift(-1)
        chunk['nxt_off'] = chunk['drive_off'].shift(-1)
        broke = chunk['game_id'] != chunk['game_id'].shift(-1)
        chunk.loc[broke, ['nxt_ep', 'nxt_off']] = np.nan

        pt = chunk['play_type_text'].astype(str)
        punt, ko = pt.eq('Punt'), pt.str.startswith('Kickoff')
        fg = (pt.str.contains('Field Goal', case=False)
              & ~pt.str.contains('Blocked', case=False))
        c = chunk[punt | ko | fg].copy()
        p2 = c['play_type_text'].astype(str)
        c['kind'] = np.where(p2.eq('Punt'), 'punt',
                             np.where(p2.str.startswith('Kickoff'),
                                      'kickoff', 'fg'))
        c['made'] = p2.str.contains('Good').astype(int)
        c['kicking_team'] = np.where(c['kind'] == 'kickoff', c['opp'],
                                     c['drive_off'])
        c['pts'] = np.where((c['kind'] == 'fg') & (c['made'] == 1), 3.0, 0.0)
        for t, v in (('Punt Return Touchdown', -7.0),
                     ('Kickoff Return Touchdown', -7.0),
                     ('Blocked Punt Touchdown', -7.0)):
            c.loc[p2 == t, 'pts'] = v
        ours = c['nxt_off'] == c['kicking_team']
        c['ep_after'] = np.where(ours, c['nxt_ep'], -c['nxt_ep'])
        c['st_epa'] = c['pts'] + c['ep_after'] - c['expected_points']
        isko = c['kind'] == 'kickoff'
        c.loc[isko, 'st_epa'] = c.loc[isko, 'ep_after']
        c['dist'] = c['yards_to_goal'] + 17
        out.append(c[['season', 'kicking_team', 'kind', 'made', 'st_epa',
                      'dist', 'yards_to_goal']])
    k = pd.concat(out, ignore_index=True).dropna(
        subset=['kicking_team', 'st_epa'])
    k['kicking_team'] = k['kicking_team'].astype(int)
    return k


def above_expectation(k):
    """Subtract the league's average kick from the same state, same season."""
    k = k.copy()
    band = np.full(len(k), -1.0)
    fg = (k['kind'] == 'fg').to_numpy()
    pu = (k['kind'] == 'punt').to_numpy()
    band[fg] = (k.loc[fg, 'dist'] // 5) * 5
    band[pu] = (k.loc[pu, 'yards_to_goal'] // 10) * 10
    k['band'] = np.nan_to_num(band, nan=-1.0)
    k['st_epaa'] = k['st_epa'] - k.groupby(
        ['season', 'kind', 'band'])['st_epa'].transform('mean')
    return k


def by_team(k):
    a = k.groupby(['season', 'kicking_team']).agg(
        kicks=('st_epa', 'size'), st_epa=('st_epa', 'sum'),
        st_epaa=('st_epaa', 'sum'))
    for kind in ('fg', 'punt', 'kickoff'):
        s = k[k['kind'] == kind].groupby(['season', 'kicking_team']).agg(
            **{f'{kind}_n': ('st_epaa', 'size'),
               f'{kind}_epaa': ('st_epaa', 'sum')})
        a = a.join(s, how='left')
    a = a.reset_index().rename(columns={'kicking_team': 'team_id'})
    for kind in ('fg', 'punt', 'kickoff'):
        a[f'{kind}_epaa_per'] = (a[f'{kind}_epaa']
                                 / a[f'{kind}_n'].replace(0, np.nan))
    a['st_epaa_per_kick'] = a['st_epaa'] / a['kicks']
    a.loc[a['kicks'] < MIN_KICKS, ['st_epaa', 'st_epaa_per_kick']] = np.nan
    g = a.groupby('season')['st_epaa']
    a['z_st_epaa'] = (a['st_epaa'] - g.transform('mean')) / g.transform('std')
    return a


def project(a):
    """Carry a season forward, so the column stands BEFORE the season it names.

    No shrinkage. The feature is standardized and the model fits its own
    coefficient, so shrinking here would only rescale what the model rescales
    anyway.
    """
    p = a[['season', 'team_id', 'z_st_epaa']].copy()
    p['season'] += 1
    p = p.rename(columns={'z_st_epaa': 'proj_st_epaa'})
    return a.merge(p, on=['season', 'team_id'], how='outer')


def stability(a, col, vol, minv):
    x = a[['season', 'team_id', col, vol]].dropna(subset=[col])
    x = x[x[vol] >= minv]
    g = x.groupby('season')[col]
    x = x.assign(z=(x[col] - g.transform('mean')) / g.transform('std'))
    y = x[['season', 'team_id', 'z']].copy()
    y['season'] -= 1
    m = x.merge(y, on=['season', 'team_id'], suffixes=('', '_n'))
    return m['z'].corr(m['z_n']), len(m)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'special_teams.csv'))
    args = ap.parse_args()

    k = above_expectation(load_kicks())
    print(f"### {len(k):,} kicks, {int(k.season.min())}-"
          f"{int(k.season.max())} ###")
    print(k.groupby('kind')['st_epa'].agg(['size', 'mean']).round(3)
          .to_string())

    print("\n### the field-goal check: raw EPA is the situation, "
          "the residual is the kicker ###")
    fg = k[k['kind'] == 'fg']
    t = fg.groupby(pd.cut(fg['dist'], [0, 25, 30, 35, 40, 45, 50, 60]),
                   observed=True).agg(n=('st_epa', 'size'),
                                      make=('made', 'mean'),
                                      raw=('st_epa', 'mean'),
                                      above=('st_epaa', 'mean'))
    print(t.round(3).to_string())
    print("  make rate must fall with distance, raw EPA must RISE with it,")
    print("  and the residual must sit near zero in every band.")

    a = project(by_team(k))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    a.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(a):,} team-seasons)")
    latest = int(a['season'].max())
    print(f"  {latest} rows with a carried projection: "
          f"{int(a.loc[a.season == latest, 'proj_st_epaa'].notna().sum())}")
    print(f"  spread of a season, points above average: "
          f"sd {a['st_epaa'].std():.1f}  "
          f"({a['st_epaa'].std() / 12:.2f} a game)")

    print("\n### does it repeat? ###")
    print("  adjusted line yards 0.510; adjusted sack rate allowed 0.270\n")
    print(f"  {'measure':<20}{'r':>8}{'n':>8}")
    for c, v, mn in (('st_epaa', 'kicks', MIN_KICKS),
                     ('kickoff_epaa_per', 'kickoff_n', 30),
                     ('punt_epaa_per', 'punt_n', 30),
                     ('fg_epaa_per', 'fg_n', 10)):
        r, n = stability(a, c, v, mn)
        print(f"  {c:<20}{r:>8.3f}{n:>8,}")
    print("\n  Only the aggregate is used downstream. Field goals repeat at")
    print("  0.03 and hurt the model on their own; see the docstring.")


if __name__ == '__main__':
    main()
