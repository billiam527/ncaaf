#!/usr/bin/env python3
"""Field position and finishing drives, per team-season.

Two of Bill Connelly's five factors, and the two the model has no version of.
Efficiency and explosiveness describe what happens on a play; these describe
what happens to a possession, and two teams with identical per-play numbers can
differ by a touchdown a game because one starts drives near midfield and cashes
its red-zone trips while the other kicks field goals from the twenty.

  field position     mean yards-to-goal at the start of a possession, and the
                     same figure for possessions the defense hands over
  finishing drives   points per trip inside the opponent's forty

Points are read from play_type_text on scoring plays rather than from the score
columns or from actual_points. actual_points is zero on all but safeties, and
the running score is attached post-play but reverts on the following kickoff
row, so differencing it across a drive gives the wrong answer about a third of
the time. Only the offense's own scoring counts - a pick-six on a drive is not
that offense finishing well.

Usage:
    python drive_factors.py --out results/drive_factors.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

USECOLS = ['game_id', 'drive_id', 'team_id', 'yards_to_goal', 'offensive_play',
           'scoring_play', 'play_type_text', 'garbage_time_ind', 'down']
CHUNK = 500_000

# See the note in accumulate(). This is the one producer that drops garbage
# time on purpose; season_summaries and havoc both keep it.
KEEP_GARBAGE_TIME = False

# Points credited to the offense, by the text ESPN uses on a scoring play.
#
# The vocabulary changes partway through the data. Through about 2011 a
# touchdown is "Rush" or "Pass Completion" and the conversion appears
# separately as "Extra Point Good"; from about 2016 it is "Rushing Touchdown"
# or "Passing Touchdown" and the extra point is not recorded as a scoring play
# at all. Both spellings are matched, and a touchdown is scored flat at 7 with
# conversion rows ignored, so the two eras are treated identically. The error
# is small either way - extra points convert about 97% of the time and
# two-point tries are rare.
TOUCHDOWN = ('rushing touchdown', 'passing touchdown', 'receiving touchdown',
             'rush', 'pass completion', 'pass reception')
TOUCHDOWN_POINTS = 7
FIELD_GOAL = 'field goal good'
FIELD_GOAL_POINTS = 3

# Scoring plays that belong to the defense, or to no offense at all. Checked
# before the touchdown match, since "Interception Return Touchdown" would
# otherwise be read as a passing touchdown.
NOT_OFFENSE = ('interception', 'fumble', 'punt return', 'kickoff return',
               'blocked', 'safety', 'sack')

INSIDE_40 = 40


def drive_points(texts):
    """Points the offense scored on one drive."""
    total = 0
    for t in texts:
        s = str(t).lower().strip()
        if any(k in s for k in NOT_OFFENSE):
            continue
        if s.startswith(FIELD_GOAL):
            total += FIELD_GOAL_POINTS
        elif any(s == k or s.startswith(k) for k in TOUCHDOWN):
            total += TOUCHDOWN_POINTS
    return total


def accumulate(chunk, acc):
    c = chunk.copy()
    for col in ('game_id', 'drive_id', 'team_id', 'yards_to_goal',
                'offensive_play', 'scoring_play', 'garbage_time_ind'):
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors='coerce')
    c = c.dropna(subset=['drive_id', 'team_id', 'game_id'])

    # Deliberately the opposite of season_summaries and havoc, both of which
    # keep garbage time. Field position is the one measure where the blowout
    # snaps are not merely noisier but actively misleading: a drive starting on
    # the opponent's 30 because the game is over says nothing about field
    # position earned, whereas a garbage-time sack is still a sack. Flip this
    # only with a walk-forward behind it.
    if not KEEP_GARBAGE_TIME and 'garbage_time_ind' in c.columns:
        c = c[c['garbage_time_ind'] != 1]

    off = c[c['offensive_play'] == 1]
    if off.empty:
        return

    # A scoring opportunity is a FIRST DOWN inside the forty, not merely any
    # snap taken there. Counting every snap sweeps in drives that reach the 39
    # on third down and fail, which pushed the scoreless rate to 36% against a
    # real 15-20% and dragged points per trip down accordingly.
    off = off.copy()
    off['first_inside'] = (off['down'] == 1) & (off['yards_to_goal'] <= INSIDE_40)

    g = off.groupby('drive_id')
    drives = pd.DataFrame({
        'game_id': g['game_id'].first(),
        'team_id': g['team_id'].first(),
        'start_ytg': g['yards_to_goal'].first(),
        'reached': g['first_inside'].any(),
    })
    scoring = c[c['scoring_play'] == 1]
    pts = (scoring.groupby('drive_id')['play_type_text']
           .apply(drive_points).rename('points'))
    drives = drives.join(pts).fillna({'points': 0})
    drives = drives.dropna(subset=['start_ytg'])
    acc.append(drives.reset_index())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pbp', default=PBP)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'drive_factors.csv'))
    args = ap.parse_args()

    if not os.path.exists(args.pbp):
        raise SystemExit(f"missing {args.pbp}")

    games = pd.read_csv(GAMES, low_memory=False)
    games = games[['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    for col in games.columns:
        games[col] = pd.to_numeric(games[col], errors='coerce')
    games = games.dropna().rename(columns={'id': 'game_id'})
    games = games.drop_duplicates('game_id')

    acc, rows = [], 0
    for chunk in pd.read_csv(args.pbp, usecols=USECOLS, chunksize=CHUNK,
                             low_memory=False):
        accumulate(chunk, acc)
        rows += len(chunk)
        print(f"  {rows:,} plays", end='\r')
    print()

    drives = pd.concat(acc, ignore_index=True)
    drives = drives.merge(games, on='game_id', how='inner')
    print(f"drives: {len(drives):,}")

    # the defense's view: whichever side did not have the ball
    drives['opp_id'] = np.where(drives['team_id'] == drives['home_team_id'],
                                drives['away_team_id'], drives['home_team_id'])
    drives['trip'] = drives['reached'].fillna(False).astype(bool)

    out = []
    for side, key in (('off', 'team_id'), ('def', 'opp_id')):
        g = drives.groupby([key, 'season'])
        part = pd.DataFrame({
            f'start_ytg_{side}': g['start_ytg'].mean(),
            f'trips_{side}': g['trip'].sum(),
            f'trip_points_{side}': g.apply(lambda x: x.loc[x['trip'], 'points'].sum()),
            f'drives_{side}': g.size(),
        })
        part.index.names = ['team_id', 'season']
        out.append(part)

    res = out[0].join(out[1], how='outer').reset_index()
    for side in ('off', 'def'):
        res[f'points_per_trip_{side}'] = np.where(
            res[f'trips_{side}'] > 0,
            res[f'trip_points_{side}'] / res[f'trips_{side}'], np.nan)
        res[f'trip_rate_{side}'] = np.where(
            res[f'drives_{side}'] > 0,
            res[f'trips_{side}'] / res[f'drives_{side}'], np.nan)

    res = res[res['drives_off'] >= 80]
    keep = ['team_id', 'season', 'start_ytg_off', 'start_ytg_def',
            'points_per_trip_off', 'points_per_trip_def',
            'trip_rate_off', 'trip_rate_def', 'drives_off']
    res = res[keep]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(res)} team-seasons)")
    print(res[[c for c in keep if c not in ('team_id', 'season')]]
          .describe().round(3).to_string())


if __name__ == '__main__':
    main()
