#!/usr/bin/env python3
"""Havoc and high-leverage defensive stops, per team-game.

Efficiency and explosiveness describe what happens on an average play. These
describe disruption - how often a defence blows a play up before it starts, and
how often it holds when holding matters. Unlike field position or finishing
drives, neither is downstream of the success rates the model already has: a
defence that generates chaos is doing something the per-play averages do not
capture.

Output is per team-game so it can go through the same ridge opponent adjustment
as every other statistic. That rules out splitting front seven from secondary,
which needs player positions and those only exist in season-level stats.

Rates are over defensive snaps, and tackles for loss are kept separate from
sacks rather than summed, since a sack is already a tackle behind the line and
adding them double counts the same event.

  tfl_rate        rushing plays stopped behind the line
  sack_rate       over pass plays faced
  pass_defensed_rate
  interception_rate, fumble_rate
  third_down_stop_rate, fourth_down_stop_rate
  redzone_stop_rate   trips inside the twenty held without a touchdown

Tackles for loss are read from negative rushing yardage rather than play text,
which carries no "tackle for loss" string. Sacks, breakups, interceptions and
fumbles come from play_text, which does.

Usage:
    python havoc.py --out results/havoc.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

USECOLS = ['game_id', 'drive_id', 'team_id', 'offensive_play', 'rushing_play',
           'passing_play', 'offensive_yards', 'down', 'distance',
           'yards_to_goal', 'play_text', 'garbage_time_ind', 'scoring_play',
           'play_type_text']
CHUNK = 400_000
REDZONE = 20


def accumulate(chunk, acc):
    c = chunk.copy()
    for col in ('game_id', 'drive_id', 'team_id', 'offensive_play',
                'rushing_play', 'passing_play', 'offensive_yards', 'down',
                'yards_to_goal', 'garbage_time_ind', 'scoring_play'):
        c[col] = pd.to_numeric(c[col], errors='coerce')
    c = c[c['garbage_time_ind'] != 1]
    c = c.dropna(subset=['game_id', 'team_id'])
    off = c[c['offensive_play'] == 1].copy()
    if off.empty:
        return

    txt = off['play_text'].astype(str).str.lower()
    off['sack'] = txt.str.contains('sack', na=False)
    off['pbu'] = txt.str.contains('broken up', na=False)
    off['intc'] = txt.str.contains('intercept', na=False)
    off['fum'] = txt.str.contains('fumbl', na=False)
    # a sack is already behind the line; counting it as a TFL too would double
    # count the same event, so TFL is rushing plays only
    off['tfl'] = (off['rushing_play'] == 1) & (off['offensive_yards'] < 0)

    # a third down is converted when the same drive's next snap is a first down
    off = off.sort_values(['drive_id'], kind='stable')
    off['next_down'] = off.groupby('drive_id')['down'].shift(-1)
    off['converted'] = off['next_down'] == 1

    off['is3'] = off['down'] == 3
    off['is4'] = off['down'] == 4
    off['stop3'] = off['is3'] & ~off['converted']
    off['stop4'] = off['is4'] & ~off['converted']
    off['inrz'] = off['yards_to_goal'] <= REDZONE

    td = off['play_type_text'].astype(str).str.lower()
    off['rz_td'] = off['inrz'] & (off['scoring_play'] == 1) & (
        td.str.contains('touchdown', na=False)
        | td.isin(['rush', 'pass completion', 'pass reception']))

    g = off.groupby(['game_id', 'team_id'])
    part = pd.DataFrame({
        'plays': g.size(),
        'pass_plays': g['passing_play'].sum(),
        'rush_plays': g['rushing_play'].sum(),
        'tfl': g['tfl'].sum(),
        'sack': g['sack'].sum(),
        'pbu': g['pbu'].sum(),
        'intc': g['intc'].sum(),
        'fum': g['fum'].sum(),
        'third': g['is3'].sum(),
        'third_stop': g['stop3'].sum(),
        'fourth': g['is4'].sum(),
        'fourth_stop': g['stop4'].sum(),
        'rz_drives': g.apply(lambda x: x.loc[x['inrz'], 'drive_id'].nunique()),
        'rz_td': g.apply(lambda x: x.loc[x['rz_td'], 'drive_id'].nunique()),
    })
    acc.append(part.reset_index())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pbp', default=PBP)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results', 'havoc.csv'))
    args = ap.parse_args()

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

    o = pd.concat(acc, ignore_index=True)
    o = o.groupby(['game_id', 'team_id'], as_index=False).sum()
    o = o.merge(games, on='game_id', how='inner')

    # everything above is from the OFFENCE's point of view; credit it to the
    # defence that faced it
    o['def_id'] = np.where(o['team_id'] == o['home_team_id'],
                           o['away_team_id'], o['home_team_id'])

    d = o.rename(columns={'def_id': 'defense_id'})
    out = pd.DataFrame({
        'game_id': d['game_id'], 'team_id': d['defense_id'],
        'season': d['season'],
        'tfl_rate': d['tfl'] / d['plays'].replace(0, np.nan),
        'sack_rate': d['sack'] / d['pass_plays'].replace(0, np.nan),
        'pass_defensed_rate': d['pbu'] / d['pass_plays'].replace(0, np.nan),
        'interception_rate': d['intc'] / d['pass_plays'].replace(0, np.nan),
        'fumble_rate': d['fum'] / d['plays'].replace(0, np.nan),
        'third_down_stop_rate': d['third_stop'] / d['third'].replace(0, np.nan),
        'fourth_down_stop_rate': d['fourth_stop'] / d['fourth'].replace(0, np.nan),
        'redzone_stop_rate': 1 - d['rz_td'] / d['rz_drives'].replace(0, np.nan),
        'def_plays': d['plays'],
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} team-games, "
          f"{int(out.season.min())}-{int(out.season.max())})")
    cols = [c for c in out.columns if c.endswith('_rate')]
    print(out[cols].describe().round(4).to_string())


if __name__ == '__main__':
    main()
