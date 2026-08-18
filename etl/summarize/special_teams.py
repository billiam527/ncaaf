#!/usr/bin/env python3
"""Special teams, measured from the play text. Deliberately NOT a model feature.

This module builds the unit and then argues against using it. Both halves are
the point: the numbers are here because they are hard to reproduce and someone
will want them, and the conclusion is here so nobody spends another day
rediscovering it.

WHY IT IS NOT WIRED INTO position_ratings.py

Five carried special-teams columns were tested against game margin alongside
prior-season EPA and the seven existing position features. In sample they added
+0.0029 R2, which looks like something until you notice that five free columns
on 5,533 rows buy about 0.0009 by construction.

Out of sample - fitted on 2017-22, scored on 2023-25 - every one of them makes
the model worse:

    baseline                      R2 0.2472   MAE 13.891
    + carried leg strength           0.2441       13.922   -0.0032
    + everything combined            0.2408       13.972   -0.0064
    + field goal percentage          0.2439       13.931   -0.0033
    + net punting                    0.2441       13.926   -0.0031
    + kickoff touchback rate         0.2462       13.904   -0.0010
    + all five together              0.2440       13.955   -0.0032

Against a placebo of five pure-noise columns over 200 draws, which average
-0.0012 out of sample, special teams manages -0.0032 and beats 12% of the noise
draws. It is not merely useless, it is worse than random columns of the same
shape. For scale, dropping the offensive line from the same model costs 0.0057
and the front seven 0.0054, each on ONE column.

WHAT REPEATS AND WHAT DOES NOT

Year-over-year correlation, standardized within season, at the volume floors
set below. Adjusted line yards repeats at 0.510 and adjusted sack rate allowed
at 0.270, for scale:

    kickoff touchback rate    0.421      leg
    kickoff distance          0.345      leg
    punt coverage rate        0.333
    punt average              0.315      leg
    net punting               0.240
    field goal distance       0.197
    kick return average       0.160
    kickoff coverage          0.110
    punt return average       0.080
    FIELD GOAL PERCENTAGE     0.045      accuracy

Legs repeat, accuracy does not. A team's field-goal percentage this season tells
you almost nothing about next season's, which is the long-standing finding in
every sport with a kicker in it, and it holds here. Raising the floor does not
rescue it: at 15 attempts rather than 10 it reaches 0.085, and restricted to
kicks from 40-49 yards it is 0.028.

IT FOLLOWS THE SPECIALIST, WHICH STILL DOES NOT HELP

Splitting consecutive team pairs on whether the same man punted, the way the
quarterback test splits sack rate: punt average repeats at 0.385 when he stays
and 0.231 when he does not, a gap of +0.154. The control is clean - splitting
KICKOFF touchback rate on whether the PUNTER stayed gives -0.018, the nothing it
should. So the effect is real and it is the man, not the scheme.

It is still not worth a feature, because the thing being carried is small. A
touchback instead of a return is a few yards of field position; over a season of
kickoffs that is a couple of points, spread thin.

THE EPA COLUMN IS UNUSABLE HERE

pbp_edit.csv carries an epa value on kicking plays and it is incoherent: a made
field goal averages -4.29 and a MISSED one -2.50, and punts average +2.32. EPA
in that file is built for scrimmage plays and the kicking plays were carried
along unmodelled. Everything below is built from the text and field position.

THREE TEXT FORMATS, AS EVERYWHERE ELSE IN THIS FILE SET

    FG      "Drew Basil 53 yard field goal"          through ~2013
            "Daniel Sullivan 22 yd FG GOOD"          after
    PUNT    "punt for 45 yards, returned by NAME for 18 yards"
            "punt for 43 yds , NAME returns for 4 yds"
            "punt for 28 yds, fair catch by NAME at the Mont 14"
            "punt for 53 yds for a touchback"
    KO      "kickoff for 64 yards returned by NAME for 25 yards"
            "kickoff for 65 yds , NAME return for 13 yds"

A return pattern that does not allow the "returned BY <name> FOR n" form misses
most returns, which puts punt return average at 0.01 yards and makes net punting
identical to gross. Coverage is checked and printed on every run for that
reason; it should sit above 95%.

Usage:
    python special_teams.py --out results/special_teams.csv
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

FG_D = re.compile(r"(\d{1,2})\s*(?:yd|yds|yard|yards)\s*(?:fg|field goal)",
                  re.I)
PUNT_Y = re.compile(r"punt\s+for\s+(\d{1,3})\s*(?:yd|yds|yard|yards)", re.I)
KICK_Y = re.compile(r"kickoff\s+for\s+(\d{1,3})\s*(?:yd|yds|yard|yards)", re.I)
RET_Y = re.compile(r"return(?:s|ed)?(?:\s+by\s+[A-Za-z.'\-\s]{1,32}?)?"
                   r"\s+for\s+(no gain|\d{1,3})", re.I)

# a kick has to be attempted this often before the rate means anything
MIN_KICKS = MIN_PUNTS = 30
MIN_FGA = 10


def ret_yards(text):
    m = RET_Y.search(text)
    if not m:
        return None
    v = m.group(1).lower()
    return 0 if v == 'no gain' else int(v)


def collect(pbp=PBP, games=GAMES):
    g = pd.read_csv(games, low_memory=False)[
        ['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    season_of = dict(zip(g['id'].astype(int), g['season'].astype(int)))
    sides = {int(r.id): (int(r.home_team_id), int(r.away_team_id))
             for r in g.itertuples()}
    acc = defaultdict(lambda: defaultdict(float))

    def other(gid, tid):
        p = sides.get(gid)
        return None if not p else (p[1] if p[0] == tid else p[0])

    keep = ['game_id', 'team_id', 'play_type_text', 'play_text',
            'special_teams_play']
    for chunk in pd.read_csv(pbp, usecols=keep, chunksize=400_000,
                             low_memory=False):
        chunk['season'] = chunk['game_id'].map(season_of)
        chunk['team_id'] = pd.to_numeric(chunk['team_id'], errors='coerce')
        chunk = chunk.dropna(subset=['season', 'team_id'])
        chunk['season'] = chunk['season'].astype(int)
        chunk['team_id'] = chunk['team_id'].astype(int)
        chunk['game_id'] = chunk['game_id'].astype(int)
        for r in chunk[chunk['special_teams_play'] == 1].itertuples():
            t, pt = str(r.play_text), str(r.play_type_text)
            low = t.lower()
            a = acc[(r.season, r.team_id)]
            if 'field goal' in pt.lower() and 'blocked' not in pt.lower():
                a['fga'] += 1
                a['fgm'] += int('Good' in pt)
                m = FG_D.search(t)
                if m:
                    dist = int(m.group(1))
                    a['fg_dist_sum'] += dist
                    a['fg_dist_n'] += 1
                    b = ('s' if dist < 30 else 'm' if dist < 40
                         else 'l' if dist < 50 else 'x')
                    a[f'fga_{b}'] += 1
                    a[f'fgm_{b}'] += int('Good' in pt)
            elif pt == 'Punt':
                a['punts'] += 1
                m = PUNT_Y.search(t)
                if m:
                    a['punt_yds'] += int(m.group(1))
                    a['punt_n'] += 1
                a['punt_tb'] += int('touchback' in low)
                a['punt_fc'] += int('fair catch' in low)
                a['punt_downed'] += int('downed' in low)
                ry = ret_yards(t)
                if ry is not None:
                    a['punt_ret_allowed'] += ry
                    a['punt_returned'] += 1
                    o = other(r.game_id, r.team_id)
                    if o is not None:
                        acc[(r.season, o)]['pret_yds'] += ry
                        acc[(r.season, o)]['pret_n'] += 1
            elif pt.startswith('Kickoff'):
                a['kicks'] += 1
                m = KICK_Y.search(t)
                if m:
                    a['kick_yds'] += int(m.group(1))
                    a['kick_n'] += 1
                a['kick_tb'] += int('touchback' in low)
                ry = ret_yards(t)
                if ry is not None:
                    a['kick_ret_allowed'] += ry
                    a['kick_returned'] += 1
                    o = other(r.game_id, r.team_id)
                    if o is not None:
                        acc[(r.season, o)]['kret_yds'] += ry
                        acc[(r.season, o)]['kret_n'] += 1
    return pd.DataFrame([dict(season=s, team_id=t, **a)
                         for (s, t), a in acc.items()]).fillna(0)


def rates(d):
    def sd(a, b):
        return a / b.replace(0, np.nan)

    d = d[d['fga'] + d['punts'] + d['kicks'] > 0].copy()
    d['fg_pct'] = sd(d['fgm'], d['fga'])
    d['fg_dist'] = sd(d['fg_dist_sum'], d['fg_dist_n'])
    d['punt_avg'] = sd(d['punt_yds'], d['punt_n'])
    d['punt_net'] = sd(d['punt_yds'] - d['punt_ret_allowed'], d['punt_n'])
    d['punt_ret_rate'] = sd(d['punt_returned'], d['punts'])
    d['kick_avg'] = sd(d['kick_yds'], d['kick_n'])
    d['kick_tb_rate'] = sd(d['kick_tb'], d['kicks'])
    d['kick_ret_allow'] = sd(d['kick_ret_allowed'], d['kick_returned'])
    d['pret_avg'] = sd(d['pret_yds'], d['pret_n'])
    d['kret_avg'] = sd(d['kret_yds'], d['kret_n'])
    for b in ('s', 'm', 'l', 'x'):
        if f'fga_{b}' in d.columns:
            d[f'fg_pct_{b}'] = sd(d.get(f'fgm_{b}', 0), d[f'fga_{b}'])
    return d.sort_values(['season', 'team_id'])


def stability(d, col, vcol, minv):
    x = d[['season', 'team_id', col, vcol]].copy()
    x = x[x[vcol] >= minv].dropna(subset=[col])
    g = x.groupby('season')[col]
    x['z'] = (x[col] - g.transform('mean')) / g.transform('std')
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

    d = rates(collect())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(d):,} team-seasons, "
          f"{int(d.season.min())}-{int(d.season.max())})")

    print("\n### parse coverage - all three should exceed 95% ###")
    for a, b, lab in (('fg_dist_n', 'fga', 'field goal distance'),
                      ('punt_n', 'punts', 'punt yards'),
                      ('kick_n', 'kicks', 'kickoff yards')):
        print(f"  {lab:<24}{d[a].sum() / d[b].sum():>7.1%}")

    print("\n### year-over-year repeatability ###")
    print("  adjusted line yards is 0.510; adjusted sack rate allowed 0.270\n")
    print(f"  {'measure':<22}{'r':>8}{'n':>8}")
    for c, v, mn in (('kick_tb_rate', 'kicks', MIN_KICKS),
                     ('kick_avg', 'kick_n', MIN_KICKS),
                     ('punt_ret_rate', 'punts', MIN_PUNTS),
                     ('punt_avg', 'punt_n', MIN_PUNTS),
                     ('punt_net', 'punt_n', MIN_PUNTS),
                     ('kret_avg', 'kret_n', 15),
                     ('kick_ret_allow', 'kick_returned', 15),
                     ('fg_pct', 'fga', MIN_FGA),
                     ('pret_avg', 'pret_n', 15),
                     ('fg_dist', 'fg_dist_n', MIN_FGA)):
        if c not in d.columns:
            continue
        r, n = stability(d, c, v, mn)
        print(f"  {c:<22}{r:>8.3f}{n:>8,}")
    print("\n  See the docstring before wiring any of this into the model.")


if __name__ == '__main__':
    main()
