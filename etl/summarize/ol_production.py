#!/usr/bin/env python3
"""Offensive line play, measured at team level because it cannot be measured
at player level.

THE DATA PROBLEM, STATED PLAINLY

The play-by-play carries no personnel. Seventy-five columns, one team id per
play and a text string; no lineup, no participation, no player list. Across
170,227 plays of 2025 the phrase "blocked by" appears 156 times and refers to
blocked kicks, and an offensive-line position word appears 28 times. CFBD's
player stats have six categories - defensive, kicking, passing, punting,
receiving, rushing - and no blocking category at all.

So there is no way to say what an individual lineman did. Everything here is a
team attribute, which is also how SP+ and FEI handle the line, for the same
reason. The only per-player information that exists for a lineman is his
recruiting grade and whether he is on the roster, both of which live in
talent_by_position.py.

WHAT IS MEASURED

Run blocking splits the credit for a carry between the line and the back, on
the standard convention that the first yards belong to the blocking and the
long gains belong to the runner:

    yards lost      120% to the line   (a run stopped behind the line is
                                        usually a blocking failure)
    0 to 4 yards    100%
    5 to 10 yards    50%
    11 and beyond     0%               (that is the back, not the blocking)

alongside stuff rate, the share of carries stopped at or behind the line, and
power success, short-yardage carries that convert. Pass protection is sack rate
and tackle-for-loss rate allowed.

All rates are opponent-adjusted by the same ridge used everywhere else, so a
line is measured against the fronts it actually faced.

Usage:
    python ol_production.py --out results/ol_production.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from pbp_cache import read_pbp

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

USECOLS = ['id', 'game_id', 'drive_id', 'team_id', 'play_text',
           'play_type_text', 'rushing_play', 'passing_play', 'offensive_play',
           'stat_yardage', 'down', 'distance', 'yards_to_goal', 'epa',
           'garbage_time_ind']
CHUNK = 400_000
ALPHA = 1.0
MIN_CARRIES = 200
MIN_CELLS = 200


def line_yards(gain):
    """Share of a carry credited to the blocking rather than the back."""
    g = np.asarray(gain, dtype=float)
    out = np.where(g < 0, g * 1.2,
          np.where(g <= 4, g,
          np.where(g <= 10, 4 + (g - 4) * 0.5, 7.0)))
    return out


def second_level_yards(gain):
    """Yards 5 through 10 - past the line, short of the open field."""
    g = np.asarray(gain, dtype=float)
    return np.clip(g - 5, 0, 5)


def open_field_yards(gain):
    """Yards past 10, which belong to the back and nobody else."""
    g = np.asarray(gain, dtype=float)
    return np.clip(g - 10, 0, None)


def collect(seasons_by_game):
    kept, scanned = [], 0
    for chunk in read_pbp(PBP, usecols=USECOLS, low_memory=False,
                             chunksize=CHUNK):
        chunk['game_id'] = pd.to_numeric(chunk['game_id'], errors='coerce')
        chunk['season'] = chunk['game_id'].map(seasons_by_game)
        chunk = chunk.dropna(subset=['season'])
        scanned += CHUNK
        if chunk.empty:
            continue
        for c in ('rushing_play', 'passing_play', 'offensive_play',
                  'stat_yardage', 'down', 'distance', 'yards_to_goal',
                  'team_id', 'epa'):
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk = chunk[chunk['offensive_play'] == 1]
        if chunk.empty:
            continue
        chunk['text_key'] = chunk['play_text'].astype(str)
        kept.append(chunk.drop(columns=['play_text']))
        print(f"  scanned {scanned:,} plays", end='\r')
    print()
    return pd.concat(kept, ignore_index=True)


def adjust(cells, value_col, weight_col, alpha=ALPHA):
    """Split a rate into an offence effect and a defence effect."""
    out = []
    for season, sd in cells.groupby('season'):
        sd = sd.dropna(subset=[value_col, weight_col, 'opponent_id'])
        sd = sd[sd[weight_col] > 0]
        if len(sd) < MIN_CELLS:
            continue
        offs = pd.Index(sorted(sd['team_id'].unique()))
        defs = pd.Index(sorted(sd['opponent_id'].unique()))
        oi = sd['team_id'].map({v: i for i, v in enumerate(offs)}).to_numpy()
        di = sd['opponent_id'].map({v: i for i, v in enumerate(defs)}).to_numpy()
        n = len(sd)
        rows = np.arange(n)
        X = sparse.hstack([
            sparse.csr_matrix((np.ones(n), (rows, oi)), shape=(n, len(offs))),
            sparse.csr_matrix((np.ones(n), (rows, di)), shape=(n, len(defs))),
        ]).tocsr()
        m = Ridge(alpha=alpha, fit_intercept=True).fit(
            X, sd[value_col].to_numpy(float),
            sample_weight=sd[weight_col].to_numpy(float))
        eff = pd.DataFrame({'team_id': offs, 'season': int(season),
                            f'adj_{value_col}': m.intercept_ + m.coef_[:len(offs)]})
        de = pd.Series(m.coef_[len(offs):], index=defs)
        faced = (sd.assign(_d=sd['opponent_id'].map(de)).groupby('team_id')
                 .apply(lambda x: np.average(x['_d'], weights=x[weight_col])))
        eff[f'front_faced_{value_col}'] = faced.reindex(offs).to_numpy()
        out.append(eff)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'ol_production.csv'))
    args = ap.parse_args()

    games = pd.read_csv(GAMES, low_memory=False)[
        ['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    for c in games.columns:
        games[c] = pd.to_numeric(games[c], errors='coerce')
    games = games.dropna().drop_duplicates('id')
    seasons_by_game = dict(zip(games['id'].astype(int),
                               games['season'].astype(int)))

    print("reading the play-by-play...")
    d = collect(seasons_by_game)
    before = len(d)
    d = d.drop_duplicates(subset=['game_id', 'drive_id', 'text_key'])
    print(f"  {before - len(d):,} duplicate play rows dropped")
    d['season'] = d['season'].astype(int)
    d = d.merge(games.rename(columns={'id': 'game_id'})[
        ['game_id', 'home_team_id', 'away_team_id']], on='game_id', how='left')
    d['opponent_id'] = np.where(d['team_id'] == d['home_team_id'],
                                d['away_team_id'], d['home_team_id'])
    d = d.dropna(subset=['opponent_id'])
    print(f"  {len(d):,} offensive plays")

    txt = d['play_type_text'].astype(str).str.lower()
    rush = d[d['rushing_play'] == 1].copy()
    rush['line_yards'] = line_yards(rush['stat_yardage'].fillna(0))
    rush['second_level'] = second_level_yards(rush['stat_yardage'].fillna(0))
    rush['open_field'] = open_field_yards(rush['stat_yardage'].fillna(0))
    rush['stuffed'] = (rush['stat_yardage'].fillna(0) <= 0).astype(float)
    rush['opportunity'] = (rush['stat_yardage'].fillna(0) >= 4).astype(float)
    # power situations are 3rd or 4th and 2 or less, plus 1st or 2nd and goal
    # from the 2 or in - the second clause is easy to forget and it is a fifth
    # of the sample
    goal_line = (rush['down'].isin([1, 2]) & (rush['distance'] <= 2)
                 & (rush['yards_to_goal'] <= 2))
    short = rush[(rush['down'].isin([3, 4]) & (rush['distance'] <= 2))
                 | goal_line].copy()
    short['converted'] = (short['stat_yardage'].fillna(0)
                          >= short['distance']).astype(float)

    pas = d[d['passing_play'] == 1].copy()
    pas['sack'] = txt.reindex(pas.index).str.contains('sack', na=False).astype(float)
    # a tackle for loss on a running play is the line's failure to move anyone
    rush['tfl'] = (rush['stat_yardage'].fillna(0) < 0).astype(float)

    def cell(df, col, name):
        c = df.groupby(['season', 'team_id', 'opponent_id'],
                       as_index=False).agg(**{name: (col, 'mean'),
                                              f'n_{name}': (col, 'size')})
        return c

    parts = []
    for df, col, name, wt in (
            (rush, 'line_yards', 'line_yards', 'n_line_yards'),
            (rush, 'second_level', 'second_level', 'n_second_level'),
            (rush, 'open_field', 'open_field', 'n_open_field'),
            (rush, 'stuffed', 'stuff_rate', 'n_stuff_rate'),
            (rush, 'opportunity', 'opportunity_rate', 'n_opportunity_rate'),
            (rush, 'tfl', 'tfl_rate_allowed', 'n_tfl_rate_allowed'),
            (pas, 'sack', 'sack_rate_allowed', 'n_sack_rate_allowed'),
            (short, 'converted', 'power_success', 'n_power_success')):
        c = cell(df, col, name)
        a = adjust(c, name, wt, args.alpha)
        if len(a):
            parts.append(a)

    season = rush.groupby(['season', 'team_id'], as_index=False).agg(
        carries=('stat_yardage', 'size'),
        rush_yards=('stat_yardage', 'sum'),
        line_yards_per_carry=('line_yards', 'mean'),
        second_level_yards=('second_level', 'mean'),
        open_field_yards=('open_field', 'mean'),
        stuff_rate=('stuffed', 'mean'),
        opportunity_rate=('opportunity', 'mean'),
        tfl_rate_allowed=('tfl', 'mean'))
    ps = pas.groupby(['season', 'team_id'], as_index=False).agg(
        dropbacks=('sack', 'size'), sack_rate_allowed=('sack', 'mean'))
    pw = short.groupby(['season', 'team_id'], as_index=False).agg(
        power_plays=('converted', 'size'), power_success=('converted', 'mean'))
    g = season.merge(ps, on=['season', 'team_id'], how='left').merge(
        pw, on=['season', 'team_id'], how='left')
    for a in parts:
        g = g.merge(a, on=['season', 'team_id'], how='left')
    g = g[g['carries'] >= MIN_CARRIES].copy()

    for c in ('line_yards_per_carry', 'stuff_rate', 'opportunity_rate',
              'sack_rate_allowed', 'power_success', 'tfl_rate_allowed',
              'adj_line_yards', 'adj_stuff_rate', 'adj_sack_rate_allowed'):
        if c in g.columns:
            asc = c in ('stuff_rate', 'sack_rate_allowed', 'tfl_rate_allowed',
                        'adj_stuff_rate', 'adj_sack_rate_allowed')
            g[f'{c}_pct'] = g.groupby('season')[c].rank(pct=True,
                                                        ascending=not asc)
    g = g.sort_values(['season', 'team_id'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    g.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(g)} team-seasons, "
          f"{int(g.season.min())}-{int(g.season.max())})")
    cols = [c for c in ('carries', 'line_yards_per_carry', 'adj_line_yards',
                        'stuff_rate', 'adj_stuff_rate', 'opportunity_rate',
                        'sack_rate_allowed', 'adj_sack_rate_allowed',
                        'power_success', 'tfl_rate_allowed') if c in g.columns]
    print(g[cols].describe().round(4).to_string())


if __name__ == '__main__':
    main()
