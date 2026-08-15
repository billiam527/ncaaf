#!/usr/bin/env python3
"""Receiver production from our own play-by-play: volume, yards, catches, scores.

Everything is counted from play_text, so nothing here depends on CFBD.

WHAT WAS TRIED FIRST, AND WHY IT WAS ABANDONED

A completed pass is one play with two people on it, so crediting the receiver
and the quarterback with the same EPA counts it twice and makes the two ratings
correlate by construction. The obvious fix is to fit them together,

    epa = quarterback effect + receiver effect + defense effect

and read the receiver's number as what he added on top of his quarterback. It
does not work, for a structural reason: a receiver catches from essentially one
quarterback, so within a team the two columns are nearly the same column and the
fit cannot tell them apart. Swept over alpha there is no setting that both
removes the leak and produces a stable measure:

    alpha       1       10       50      200     1000
    QB leak  -0.364   -0.089   +0.160   +0.322   +0.461
    stability 0.195    0.217    0.232    0.246    0.268

against 0.344 for doing nothing at all. A two-stage version, subtracting the
quarterback's own rating estimated from all his plays, lands at 0.333 - still
below raw, and it overcorrects to a -0.212 leak. Every corrected version was
LESS repeatable than the uncorrected one, which is the signature of a
correction adding noise rather than removing bias.

WHAT IS USED INSTEAD

Rates over a receiver's targets are noise. The median receiver sees 44 of them,
and the top of any per-target list fills with 27-target players who caught two
long balls. Volume is both more repeatable and cleaner:

    EPA per target        stability 0.344   QB leak +0.322
    total EPA             stability 0.435   QB leak +0.315
    target share          stability 0.519   QB leak -0.032

Target share is the most stable receiver statistic available here and is
essentially uncontaminated by quarterback quality, because it measures what the
coaching staff decided about the player rather than what his quarterback did
with the throw. So the counting statistics lead, and the per-target rate is
carried alongside as description rather than as the rating.

Usage:
    python receiver_production.py --out results/receiver_production.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from qb_production import VERBS, NOT_A_NAME, passer, name_keys, roster_lookup

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

USECOLS = ['id', 'game_id', 'drive_id', 'team_id', 'play_text',
           'play_type_text', 'passing_play', 'epa', 'stat_yardage',
           'garbage_time_ind']
CHUNK = 400_000
MIN_TARGETS = 25
ALPHA = 1.0

# Anything that can only follow the name, so the match stops there.
#
# "thrown" is the one that matters most and it is easy to miss: 14.6% of pass
# plays read "incomplete short left to #6 K.Gross thrown to BSU36", and without
# it the target parses as "K.Gross thrown", fails to resolve, and vanishes.
# Because every one of those is an incompletion, losing them inflated every
# catch rate on the page - Indiana's Charlie Becker read 34 of 35 rather than
# 34 of 47. A per-player catch rate far above the team's is the symptom to
# watch for if another of these turns up.
STOP = (r"for|and|to|at|in|on|as|is|was|then|caught|out|no|gain|loss|yards?|"
        r"yds?|thrown|overthrown|underthrown|hurried|deflected|batted|"
        r"defensed|dropped|intended|pressured|broken|blocked|"
        r"PENALTY|TOUCHDOWN|1ST|2ND|3RD|4TH")
_TOK = rf"(?!(?:{VERBS}|{STOP})\b)[A-Za-z][\w'\.\-]*"
# capped at three tokens: beyond that the match starts swallowing team names
_NAME = rf"{_TOK}(?:\s+{_TOK}){{0,2}}"
_JERSEY = r"(?:#\d+\s*)?"
_COMMA = rf"([A-Za-z][\w'\.\-]*),\s*([A-Za-z][\w'\.\-]*)"

# The target sits after "to", with three things able to get in the way: a jersey
# number in front of the name, direction qualifiers between the verb and "to"
# ("complete short left to"), and "caught" running on after it. Handling those
# moved receptions from 58% parsed to 100%.
P_TGT = re.compile(
    rf"\b(?:complete|incomplete|pass)\b[^,]{{0,28}}?\bto\s+{_JERSEY}({_NAME})",
    re.I)
P_TGT_COMMA = re.compile(
    rf"\b(?:complete|incomplete|pass)\b[^,]{{0,28}}?\bto\s+{_JERSEY}{_COMMA}",
    re.I)
P_TGT_TD = re.compile(rf"^{_JERSEY}({_NAME})\s+\d+\s*Yd\s+pass\s+from", re.I)

# Substring tests on play_type_text are a trap here: "incompletion" contains
# "completion", so a naive check counts every incompletion as a catch and puts
# the league catch rate at 99%. Match the types exactly.
CAUGHT_TYPES = {'pass reception', 'passing touchdown', 'pass completion',
                'receiving touchdown'}
NOT_CAUGHT = ('incomplet', 'interception', 'sack')


def target(text):
    """The intended receiver named on a pass play, or None."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    m = P_TGT_TD.match(t)
    if m:
        return m.group(1).strip()
    m = P_TGT_COMMA.search(t)
    if m:
        return f"{m.group(2)} {m.group(1)}"
    m = P_TGT.search(t)
    if m:
        v = m.group(1).strip()
        return v if len(v) > 1 and v.lower() not in NOT_A_NAME else None
    return None


def resolve(lookup, season, tid, raw):
    if not raw or pd.isna(tid):
        return (None, None, None)
    for k in name_keys(raw):
        hit = lookup.get((int(season), int(tid), k))
        if hit:
            return hit
    return (None, None, None)


def collect(seasons_by_game, lookup):
    """Every pass play with a resolved intended receiver."""
    kept, scanned = [], 0
    for chunk in pd.read_csv(PBP, usecols=USECOLS, low_memory=False,
                             chunksize=CHUNK):
        chunk['game_id'] = pd.to_numeric(chunk['game_id'], errors='coerce')
        chunk['season'] = chunk['game_id'].map(seasons_by_game)
        chunk = chunk.dropna(subset=['season'])
        scanned += CHUNK
        if chunk.empty:
            continue
        for col in ('passing_play', 'epa', 'stat_yardage', 'team_id'):
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk[chunk['passing_play'] == 1]
        if chunk.empty:
            continue
        chunk['play_text_key'] = chunk['play_text'].astype(str)
        chunk['tgt_raw'] = chunk['play_text'].map(target)
        chunk['qb_raw'] = chunk['play_text'].map(passer)
        chunk = chunk.dropna(subset=['tgt_raw', 'team_id'])
        if chunk.empty:
            continue
        tg = [resolve(lookup, s, t, r) for s, t, r in
              zip(chunk['season'], chunk['team_id'], chunk['tgt_raw'])]
        chunk['rec_id'] = [x[0] for x in tg]
        chunk['rec_who'] = [x[1] for x in tg]
        chunk['rec_pos'] = [x[2] for x in tg]
        qb = [resolve(lookup, s, t, r) for s, t, r in
              zip(chunk['season'], chunk['team_id'], chunk['qb_raw'])]
        chunk['qb_id'] = [x[0] for x in qb]
        kept.append(chunk.dropna(subset=['rec_id']).drop(columns=['play_text']))
        print(f"  scanned {scanned:,} plays", end='\r')
    print()
    return pd.concat(kept, ignore_index=True)


def opponent_adjust(plays, g, alpha=ALPHA):
    """Split each rate into a receiver effect and an opposing-defense effect.

    Only the rates are adjusted. Volume cannot be: a receiver's target share is
    decided by his own coaching staff, not by who lined up across from him, and
    "adjusting" it would be answering a question nobody asked.

    Weighted by targets, for the reason every other adjustment here is weighted
    by its denominator - a nine-target game and a two-target game are not equal
    evidence.
    """
    games = pd.read_csv(GAMES, low_memory=False)[
        ['id', 'home_team_id', 'away_team_id']].dropna()
    for c in games.columns:
        games[c] = pd.to_numeric(games[c], errors='coerce')
    games = games.dropna().drop_duplicates('id').rename(
        columns={'id': 'game_id'})
    p = plays.merge(games, on='game_id', how='left')
    p['opponent_id'] = np.where(p['team_id'] == p['home_team_id'],
                                p['away_team_id'], p['home_team_id'])
    p = p.dropna(subset=['opponent_id'])

    out = []
    for season, sd in p.groupby('season'):
        cell = sd.groupby(['rec_id', 'opponent_id'], as_index=False).agg(
            n=('epa', 'size'), ypt=('rec_yards', 'mean'),
            cr=('caught', 'mean'))
        if len(cell) < 200:
            continue
        recs = pd.Index(sorted(cell['rec_id'].unique()))
        defs = pd.Index(sorted(cell['opponent_id'].unique()))
        ri = cell['rec_id'].map({v: i for i, v in enumerate(recs)}).to_numpy()
        di = cell['opponent_id'].map(
            {v: i for i, v in enumerate(defs)}).to_numpy()
        n = len(cell)
        rows = np.arange(n)
        X = sparse.hstack([
            sparse.csr_matrix((np.ones(n), (rows, ri)), shape=(n, len(recs))),
            sparse.csr_matrix((np.ones(n), (rows, di)), shape=(n, len(defs))),
        ]).tocsr()
        w = cell['n'].to_numpy(float)
        res = {'rec_id': recs, 'season': int(season)}
        for col, name in (('ypt', 'adj_yards_per_target'),
                          ('cr', 'adj_catch_rate')):
            m = Ridge(alpha=alpha, fit_intercept=True).fit(
                X, cell[col].to_numpy(float), sample_weight=w)
            res[name] = m.intercept_ + m.coef_[:len(recs)]
            if col == 'ypt':
                de = pd.Series(m.coef_[len(recs):], index=defs)
                faced = (cell.assign(_d=cell['opponent_id'].map(de))
                         .groupby('rec_id')
                         .apply(lambda x: np.average(x['_d'], weights=x['n'])))
                res['defense_faced'] = faced.reindex(recs).to_numpy()
        out.append(pd.DataFrame(res))
    if not out:
        return g
    return g.merge(pd.concat(out, ignore_index=True),
                   on=['rec_id', 'season'], how='left')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-targets', type=int, default=MIN_TARGETS)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'receiver_production.csv'))
    args = ap.parse_args()

    games = pd.read_csv(GAMES, low_memory=False)[['id', 'season']].dropna()
    for c in games.columns:
        games[c] = pd.to_numeric(games[c], errors='coerce')
    games = games.dropna().drop_duplicates('id')
    seasons_by_game = dict(zip(games['id'].astype(int),
                               games['season'].astype(int)))

    print("building the roster lookup...")
    lookup = roster_lookup()
    print("reading the play-by-play...")
    d = collect(seasons_by_game, lookup)
    d['season'] = d['season'].astype(int)

    # A handful of games carry the same play many times over. Miami's 2015 game
    # 400756912 holds 1,493 rows for 183 distinct plays - one repeated 74 times
    # - which credited a running back with 151 catches for 4,674 yards.
    #
    # The copies carry sequential distinct play ids, so deduplicating on id
    # catches none of them. The key has to be the content. Play text embeds the
    # yard line and the gain, so the same text twice inside one drive is not
    # something that happens in a real game.
    before = len(d)
    d = d.drop_duplicates(subset=['game_id', 'drive_id', 'play_text_key'])
    print(f"  {before - len(d):,} duplicate play rows dropped")
    per_game = d.groupby(['game_id', 'team_id']).size()
    print(f"  targets per team-game after dedupe: median "
          f"{per_game.median():.0f}  max {per_game.max()}")
    print(f"  {len(d):,} targets resolved to a roster player")

    pt = d['play_type_text'].astype(str).str.lower().str.strip()
    d['caught'] = (pt.isin(CAUGHT_TYPES)
                   & ~pt.str.contains('|'.join(NOT_CAUGHT), na=False))
    d['td'] = pt.str.contains('touchdown', na=False) & d['caught']
    # yardage only counts on a catch; an incompletion carries none
    d['rec_yards'] = np.where(d['caught'], d['stat_yardage'].fillna(0), 0.0)
    # Two independent ways this has gone wrong, so both are checked here. A
    # figure near 99% means the play-type test is matching "incompletion" as
    # "completion". A figure well above the true rate means named incompletions
    # are being lost in the parse, which drops only failures and inflates
    # everyone. The true rate is computed from play types alone, needing no
    # target to be named at all.
    pt_all = d['play_type_text'].astype(str).str.lower().str.strip()
    n_caught = int(pt_all.isin(CAUGHT_TYPES).sum())
    n_inc = int(pt_all.str.contains('incomplet', na=False).sum())
    true_rate = n_caught / max(n_caught + n_inc, 1)
    print(f"  league catch rate {d['caught'].mean():.1%}"
          f"   (independent check from play types alone: {true_rate:.1%})")
    if abs(d['caught'].mean() - true_rate) > 0.03:
        print("  WARNING: those disagree by more than 3 points, which means "
              "targets are being lost asymmetrically")

    team = d.groupby(['team_id', 'season']).agg(
        team_targets=('epa', 'size'),
        team_yards=('rec_yards', 'sum')).reset_index()

    g = d.groupby(['rec_id', 'rec_who', 'rec_pos', 'team_id', 'season'],
                  as_index=False).agg(
        games=('game_id', 'nunique'), targets=('epa', 'size'),
        receptions=('caught', 'sum'), rec_yards=('rec_yards', 'sum'),
        touchdowns=('td', 'sum'), epa=('epa', 'sum'))
    g = g.merge(team, on=['team_id', 'season'], how='left')

    g['catch_rate'] = g['receptions'] / g['targets']
    g['yards_per_target'] = g['rec_yards'] / g['targets']
    g['yards_per_catch'] = g['rec_yards'] / g['receptions'].replace(0, np.nan)
    g['epa_per_target'] = g['epa'] / g['targets']
    # scoring on few catches marks a red-zone target rather than a volume one
    g['td_rate'] = g['touchdowns'] / g['receptions'].replace(0, np.nan)
    # share of what the offence threw, and of the yards it gained doing so -
    # the most repeatable receiver statistics available here
    g['target_share'] = g['targets'] / g['team_targets']
    g['yard_share'] = g['rec_yards'] / g['team_yards'].replace(0, np.nan)
    # team volume, so raw counting stats can be read against the system that
    # produced them: 60 catches on a team that threw 300 times is not 60 on a
    # team that threw 550
    g['team_pass_att'] = g['team_targets']
    g['team_pass_att_pct'] = g.groupby('season')['team_targets'].rank(pct=True)

    g = g[g['targets'] >= args.min_targets].copy()
    g = opponent_adjust(d, g)
    # how much of the rest of the offence a receiver has around him. A lone
    # good target draws the coverage a deep room would split.
    others = (g.groupby(['team_id', 'season'])['yard_share']
              .apply(lambda s: s.nlargest(3).iloc[1:].sum())
              .rename('other_top_share').reset_index())
    g = g.merge(others, on=['team_id', 'season'], how='left')

    for c in ('rec_yards', 'receptions', 'touchdowns', 'target_share',
              'yard_share', 'epa', 'adj_yards_per_target'):
        g[f'{c}_pct'] = g.groupby('season')[c].rank(pct=True)
    g['rank_yards'] = (g.groupby('season')['rec_yards']
                       .rank(ascending=False, method='min').astype('Int64'))
    g = g.sort_values(['season', 'rank_yards'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    g.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(g)} receiver-seasons, "
          f"{int(g.season.min())}-{int(g.season.max())})")
    cols = ['targets', 'receptions', 'rec_yards', 'touchdowns', 'catch_rate',
            'td_rate', 'target_share', 'yards_per_catch',
            'yards_per_target', 'adj_yards_per_target', 'defense_faced']
    print(g[[c for c in cols if c in g.columns]]
          .describe().round(3).to_string())


if __name__ == '__main__':
    main()
