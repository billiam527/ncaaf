#!/usr/bin/env python3
"""Run/pass balance per team-season, offense and defense.

The model already knows how well a team runs and passes - rush_success,
pass_success, epa_per_rush, epa_per_pass are all in there - but not how often it
chooses to do each. That leaves an interaction on the table: a team that runs on
two thirds of its snaps and is bad at it is in worse shape than the success rate
alone suggests, because the weakness is on most of its plays rather than a
third of them. Handing the trees the rate alongside the efficiency lets them
find that themselves.

Raw rush rate is partly an outcome rather than a trait: teams protecting a lead
run out the clock and teams chasing one throw, so a team's rate reflects how
often it was ahead. A neutral-situation rate is therefore computed alongside the
raw one - early downs, within one score, before the fourth quarter - which is
much closer to play-calling identity.

Defensive rates are what opponents chose to do against this team, which carries
its own signal: opponents run at defenses that cannot stop the run.

Garbage time is kept, matching season_summaries and havoc. The note that used
to sit here said garbage_time_ind was zero on every row and so unusable - that
was true when this was written, before the flag was fixed in the play-by-play
formatter, and is not true now. The neutral-situation rate is a stricter filter
than the garbage-time flag anyway: it already excludes late blowout snaps along
with every other non-neutral situation.

Usage:
    python run_pass_ratio.py --out results/run_pass_ratio.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from pbp_cache import read_pbp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ETL = os.path.dirname(_HERE)
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

USECOLS = ['game_id', 'team_id', 'rushing_play', 'passing_play',
           'down', 'period', 'home_score', 'away_score']

# "neutral" = early down, still a one-score game, before the fourth quarter
ONE_SCORE = 8
CHUNK = 500_000


def accumulate(chunk, games, acc):
    c = chunk.dropna(subset=['game_id', 'team_id']).copy()
    for col in ('rushing_play', 'passing_play', 'down', 'period',
                'home_score', 'away_score'):
        c[col] = pd.to_numeric(c[col], errors='coerce')
    # team ids arrive as a mix of ints and strings across chunks. Left alone,
    # the offensive and defensive accumulators index on different types and the
    # join at the end silently drops most of the defensive side.
    for col in ('game_id', 'team_id'):
        c[col] = pd.to_numeric(c[col], errors='coerce')
    c = c.dropna(subset=['game_id', 'team_id'])
    c = c[(c['rushing_play'] == 1) | (c['passing_play'] == 1)]
    if c.empty:
        return

    c = c.merge(games, on='game_id', how='inner')
    if c.empty:
        return

    # score from the perspective of whoever has the ball
    is_home = c['team_id'] == c['home_team_id']
    c['lead'] = np.where(is_home,
                         c['home_score'] - c['away_score'],
                         c['away_score'] - c['home_score'])
    c['neutral'] = (c['down'].isin([1, 2])
                    & (c['lead'].abs() <= ONE_SCORE)
                    & (c['period'] <= 3))

    # offense: the team with the ball. defense: their opponent that game.
    c['opp_id'] = np.where(is_home, c['away_team_id'], c['home_team_id'])

    for side, idcol in (('off', 'team_id'), ('def', 'opp_id')):
        g = c.groupby([idcol, 'season'])
        part = pd.DataFrame({
            f'rush_{side}': g['rushing_play'].sum(),
            f'pass_{side}': g['passing_play'].sum(),
            f'rush_{side}_neutral': g.apply(
                lambda x: x.loc[x['neutral'], 'rushing_play'].sum()),
            f'pass_{side}_neutral': g.apply(
                lambda x: x.loc[x['neutral'], 'passing_play'].sum()),
        })
        part.index.names = ['team_id', 'season']
        acc[side] = part if side not in acc else acc[side].add(part, fill_value=0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pbp', default=PBP)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'run_pass_ratio.csv'))
    args = ap.parse_args()

    if not os.path.exists(args.pbp):
        raise SystemExit(f"missing {args.pbp}")

    g = pd.read_csv(GAMES, low_memory=False)
    games = g[['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    games = games.rename(columns={'id': 'game_id'}).drop_duplicates('game_id')
    for col in ('game_id', 'home_team_id', 'away_team_id', 'season'):
        games[col] = pd.to_numeric(games[col], errors='coerce')
    games = games.dropna()
    print(f"games: {len(games)}")

    acc = {}
    rows = 0
    for chunk in read_pbp(args.pbp, usecols=USECOLS, chunksize=CHUNK,
                             low_memory=False):
        accumulate(chunk, games, acc)
        rows += len(chunk)
        print(f"  {rows:,} plays read", end='\r')
    print()

    if not acc:
        raise SystemExit("no plays accumulated")

    out = acc['off'].join(acc['def'], how='outer').fillna(0).reset_index()
    for side in ('off', 'def'):
        tot = out[f'rush_{side}'] + out[f'pass_{side}']
        out[f'rush_rate_{side}'] = np.where(tot > 0, out[f'rush_{side}'] / tot,
                                            np.nan)
        ntot = out[f'rush_{side}_neutral'] + out[f'pass_{side}_neutral']
        out[f'rush_rate_{side}_neutral'] = np.where(
            ntot > 100, out[f'rush_{side}_neutral'] / ntot, np.nan)
        out[f'plays_{side}'] = tot

    # a team that runs a lot AND throws a lot is simply fast; pace is worth
    # separating from balance
    out['plays_per_game_off'] = out['plays_off']

    keep = ['team_id', 'season', 'rush_rate_off', 'rush_rate_off_neutral',
            'rush_rate_def', 'rush_rate_def_neutral', 'plays_off', 'plays_def']
    out = out[keep]
    out = out[out['plays_off'] >= 200]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} team-seasons)")
    print(out[['rush_rate_off', 'rush_rate_off_neutral', 'rush_rate_def',
               'rush_rate_def_neutral']].describe().round(3).to_string())


if __name__ == '__main__':
    main()
