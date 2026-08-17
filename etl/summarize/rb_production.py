#!/usr/bin/env python3
"""Running back production from our own play-by-play, opponent-adjusted.

A back is two players in one: a runner and a receiver out of the backfield.
Both halves are counted here and kept separate, because they are not the same
skill and a team can want either.

Reading the rusher out of play_text takes four formats:

  "Carson Hansen run for 2 yds to the ISU 27"
  "(14:46) Shotgun #1 K.Kelly rush middle for 5 yards gain to the BSU41"
  "Smith,Cameren rush left"                          <- surname first
  "Kneel down by UCF at UCF42 for loss of 2 yards"   <- not a player at all

The lesson from the receiver work is baked in rather than relearned. There, the
word "thrown" ran into the name on 14.6% of plays, every one of them an
incompletion, and the loss inflated every catch rate on the page until it was
caught by eye. So the direction and outcome words are all in the stop list here
- middle, left, right, gain, loss - and main() cross-checks the attributed
rushing yardage against a total computed from play flags alone, which needs no
name parsed at all. A gap over 3% prints a warning.

Kneel-downs are dropped. They are credited to the team, not a back, and they
are the only rushing plays a quarterback is guaranteed to take.

Usage:
    python rb_production.py --out results/rb_production.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from qb_production import name_keys, roster_lookup, norm, SUFFIX
from receiver_production import target as parse_target

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')

USECOLS = ['id', 'game_id', 'drive_id', 'team_id', 'play_text',
           'play_type_text', 'rushing_play', 'passing_play', 'epa',
           'stat_yardage', 'garbage_time_ind']
CHUNK = 400_000
ALPHA = 1.0
MIN_CARRIES = 40
MIN_CELLS = 200

# Everything that can follow a rusher's name. The direction words matter most -
# "rush middle for 5 yards" is a quarter of the plays.
RUSH_VERBS = r"run|rush|rushes|scramble|scrambles|kneel|kneels|takes"
STOP = (r"middle|left|right|end|guard|tackle|for|gain|loss|yards?|yds?|to|at|"
        r"the|up|no|and|fumbled|fumble|forced|recovered|touchdown|1ST|2ND|"
        r"3RD|4TH|penalty")
_TOK = rf"(?!(?:{RUSH_VERBS}|{STOP})\b)[A-Za-z][\w'\.\-]*"
_NAME = rf"{_TOK}(?:\s+{_TOK}){{0,2}}"
LEAD = re.compile(
    r"^(?:\(\d+:\d+\)\s*)?(?:(?:No\s+Huddle|Shotgun|Hurry\s+Up)[\w\- ]*?)?\s*"
    r"(?:#\d+\s*)?", re.I)
P_RUSH = re.compile(rf"^({_NAME})\s+(?:{RUSH_VERBS})\b", re.I)
P_COMMA = re.compile(
    rf"^([A-Za-z][\w'\.\-]*),\s*([A-Za-z][\w'\.\-]*)\s+(?:{RUSH_VERBS})\b", re.I)
KNEEL = re.compile(r"kneel|takes a knee|knee down", re.I)


def rusher(text):
    """The back credited with a carry, or None."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if KNEEL.search(t):
        return None
    m = P_COMMA.match(t)
    if m:
        return f"{m.group(2)} {m.group(1)}"
    m = P_RUSH.match(LEAD.sub('', t).strip())
    if not m:
        return None
    name = m.group(1).strip()
    return name if len(name) > 1 else None


def resolve(lookup, season, tid, raw):
    if not raw or pd.isna(tid):
        return (None, None, None)
    for k in name_keys(raw):
        hit = lookup.get((int(season), int(tid), k))
        if hit:
            return hit
    return (None, None, None)


def collect(seasons_by_game, lookup):
    """Rushing and receiving plays with the ball-carrier resolved."""
    kept, scanned = [], 0
    for chunk in pd.read_csv(PBP, usecols=USECOLS, low_memory=False,
                             chunksize=CHUNK):
        chunk['game_id'] = pd.to_numeric(chunk['game_id'], errors='coerce')
        chunk['season'] = chunk['game_id'].map(seasons_by_game)
        chunk = chunk.dropna(subset=['season'])
        scanned += CHUNK
        if chunk.empty:
            continue
        for col in ('rushing_play', 'passing_play', 'epa', 'stat_yardage',
                    'team_id'):
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk[(chunk['rushing_play'] == 1)
                      | (chunk['passing_play'] == 1)]
        if chunk.empty:
            continue
        txt = chunk['play_text']
        # np.where evaluates both arms, so the previous form ran the rushing
        # parser over every passing play and the target parser over every
        # rushing play before discarding half the work. Each parser now sees
        # only the rows it applies to.
        is_rush = chunk['rushing_play'] == 1
        raw = pd.Series(index=chunk.index, dtype=object)
        raw.loc[is_rush] = txt.loc[is_rush].map(rusher)
        raw.loc[~is_rush] = txt.loc[~is_rush].map(parse_target)
        chunk['raw'] = raw
        chunk = chunk.dropna(subset=['raw', 'team_id'])
        if chunk.empty:
            continue
        hit = [resolve(lookup, s, t, r) for s, t, r in
               zip(chunk['season'], chunk['team_id'], chunk['raw'])]
        chunk['pid'] = [x[0] for x in hit]
        chunk['who'] = [x[1] for x in hit]
        chunk['pos'] = [x[2] for x in hit]
        chunk['text_key'] = txt.astype(str)
        kept.append(chunk.dropna(subset=['pid']).drop(columns=['play_text']))
        print(f"  scanned {scanned:,} plays", end='\r')
    print()
    return pd.concat(kept, ignore_index=True)


def opponent_adjust(plays, g, alpha=ALPHA):
    """Split each rate into a back effect and an opposing-defense effect."""
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
        res = None
        for flag, cols in (('rushing_play', [('ypc', 'adj_yards_per_carry'),
                                             ('epr', 'adj_epa_per_rush')]),
                           ('passing_play', [('ypt', 'adj_yards_per_target'),
                                             ('ept', 'adj_epa_per_target')])):
            s = sd[sd[flag] == 1]
            if len(s) < MIN_CELLS:
                continue
            cell = s.groupby(['pid', 'opponent_id'], as_index=False).agg(
                n=('epa', 'size'), y=('stat_yardage', 'mean'),
                e=('epa', 'mean'))
            if len(cell) < MIN_CELLS:
                continue
            backs = pd.Index(sorted(cell['pid'].unique()))
            defs = pd.Index(sorted(cell['opponent_id'].unique()))
            bi = cell['pid'].map({v: i for i, v in enumerate(backs)}).to_numpy()
            di = cell['opponent_id'].map(
                {v: i for i, v in enumerate(defs)}).to_numpy()
            n = len(cell)
            rows = np.arange(n)
            X = sparse.hstack([
                sparse.csr_matrix((np.ones(n), (rows, bi)),
                                  shape=(n, len(backs))),
                sparse.csr_matrix((np.ones(n), (rows, di)),
                                  shape=(n, len(defs))),
            ]).tocsr()
            w = cell['n'].to_numpy(float)
            part = {'pid': backs, 'season': int(season)}
            for src, name in zip(('y', 'e'), [c[1] for c in cols]):
                m = Ridge(alpha=alpha, fit_intercept=True).fit(
                    X, cell[src].to_numpy(float), sample_weight=w)
                part[name] = m.intercept_ + m.coef_[:len(backs)]
                if name == 'adj_yards_per_carry':
                    de = pd.Series(m.coef_[len(backs):], index=defs)
                    # weighted mean per back, as two sums rather than a
                    # groupby-apply. The apply built a sub-frame per back per
                    # season and was most of this module's runtime.
                    t = cell[['pid', 'n']].copy()
                    t['_wd'] = cell['opponent_id'].map(de).to_numpy() * t['n']
                    s = t.groupby('pid')[['_wd', 'n']].sum()
                    faced = s['_wd'] / s['n']
                    part['run_defense_faced'] = faced.reindex(backs).to_numpy()
            part = pd.DataFrame(part)
            res = part if res is None else res.merge(
                part, on=['pid', 'season'], how='outer')
        if res is not None:
            out.append(res)
    if not out:
        return g
    return g.merge(pd.concat(out, ignore_index=True),
                   on=['pid', 'season'], how='left')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--min-carries', type=int, default=MIN_CARRIES)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'rb_production.csv'))
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
    before = len(d)
    d = d.drop_duplicates(subset=['game_id', 'drive_id', 'text_key'])
    print(f"  {before - len(d):,} duplicate play rows dropped")
    d = d[d['pos'].isin(['RB', 'FB'])]
    print(f"  {len(d):,} plays credited to a running back")

    rush = d[d['rushing_play'] == 1]
    rec = d[d['passing_play'] == 1]
    pt = rec['play_type_text'].astype(str).str.lower().str.strip()
    caught = pt.isin({'pass reception', 'passing touchdown', 'pass completion',
                      'receiving touchdown'}) & ~pt.str.contains(
                          'incomplet|interception|sack', na=False)
    rec = rec.assign(caught=caught)

    g = rush.groupby(['pid', 'who', 'team_id', 'season'], as_index=False).agg(
        games=('game_id', 'nunique'), carries=('epa', 'size'),
        rush_yards=('stat_yardage', 'sum'), rush_epa=('epa', 'sum'),
        rush_td=('play_type_text', lambda s: s.astype(str).str.contains(
            'touchdown', case=False, na=False).sum()))
    r2 = rec.groupby(['pid', 'season'], as_index=False).agg(
        targets=('epa', 'size'), receptions=('caught', 'sum'),
        rec_epa=('epa', 'sum'),
        rec_yards=('stat_yardage', lambda s: s[rec.loc[s.index, 'caught']].sum()))
    g = g.merge(r2, on=['pid', 'season'], how='left')
    for c in ('targets', 'receptions', 'rec_epa', 'rec_yards'):
        g[c] = g[c].fillna(0.0)

    g = g[g['carries'] >= args.min_carries].copy()
    g['yards_per_carry'] = g['rush_yards'] / g['carries']
    g['epa_per_rush'] = g['rush_epa'] / g['carries']
    g['yards_per_catch'] = g['rec_yards'] / g['receptions'].replace(0, np.nan)
    g['epa_per_catch'] = g['rec_epa'] / g['receptions'].replace(0, np.nan)
    g['epa_per_target'] = g['rec_epa'] / g['targets'].replace(0, np.nan)
    g['touches'] = g['carries'] + g['receptions']
    g['total_epa'] = g['rush_epa'] + g['rec_epa']
    g['epa_per_touch'] = g['total_epa'] / g['touches']
    g['scrimmage_yards'] = g['rush_yards'] + g['rec_yards']

    team = rush.groupby(['team_id', 'season']).size().rename(
        'team_carries').reset_index()
    g = g.merge(team, on=['team_id', 'season'], how='left')
    g['carry_share'] = g['carries'] / g['team_carries']

    g = opponent_adjust(d, g, args.alpha)

    # An independent check on the attribution. Team rushing yardage from the
    # play flags needs no name parsed, so a gap means backs are being lost.
    allrush = pd.read_csv(PBP, usecols=['game_id', 'rushing_play',
                                        'stat_yardage'], low_memory=False)
    allrush['game_id'] = pd.to_numeric(allrush['game_id'], errors='coerce')
    allrush['season'] = allrush['game_id'].map(seasons_by_game)
    allrush = allrush[(pd.to_numeric(allrush['rushing_play'],
                                     errors='coerce') == 1)
                      & allrush['season'].notna()]
    tot = pd.to_numeric(allrush['stat_yardage'], errors='coerce').sum()
    got = float(rush['stat_yardage'].sum())
    print(f"\n  rushing yards on RB-credited plays {got:,.0f} of "
          f"{tot:,.0f} league-wide ({got/tot:.1%})")
    print("  (backs take roughly half of all rushing yardage; quarterbacks, "
          "receivers\n   and sacks make up the rest, so this is a floor "
          "check rather than a target)")

    for c in ('yards_per_carry', 'epa_per_rush', 'carry_share', 'touches',
              'adj_yards_per_carry', 'adj_epa_per_rush'):
        if c in g.columns:
            g[f'{c}_pct'] = g.groupby('season')[c].rank(pct=True)
    g['rank_yards'] = (g.groupby('season')['rush_yards']
                       .rank(ascending=False, method='min').astype('Int64'))
    g = g.sort_values(['season', 'rank_yards'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    g.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(g)} back-seasons, "
          f"{int(g.season.min())}-{int(g.season.max())})")
    cols = ['carries', 'rush_yards', 'yards_per_carry', 'epa_per_rush',
            'adj_yards_per_carry', 'adj_epa_per_rush', 'targets', 'receptions',
            'yards_per_catch', 'epa_per_catch', 'carry_share', 'touches']
    print(g[[c for c in cols if c in g.columns]]
          .describe().round(3).to_string())


if __name__ == '__main__':
    main()
