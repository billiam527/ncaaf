#!/usr/bin/env python3
"""Quarterback production from our own play-by-play, opponent-adjusted.

Everything here is built from the play-by-play this repo already parses, so
what goes into the number is visible. CFBD's PPA is used nowhere; it is only
useful afterwards as an independent check, and the two agree at +0.85 on
EPA per play and +0.96 on volume.

There is no player id on a play. The passer has to be read out of play_text
and matched back to a roster row, which is most of the work:

  "Rocco Becht pass complete to Gabe Burkle for 10 yds"
  "Jordan Cooke 15 Yd pass from Michael Shulikov (Kick)"   <- passer follows 'from'
  "Shotgun #17 E.Grunkemeyer pass incomplete"              <- formation, jersey
  "R. Becht pass to D. Overby for 23 yds for a TD"         <- abbreviated

Two things that look like details and are not. A capitalised-word regex will
happily swallow the verb ("Caden Veltkamp pass") because the play text is not
consistently cased, and taking the last token as the surname breaks on "Tisdale
Jr." and "Del Rio-Wilson". Those two together were 88% of unmatched names, and
fixing them moved resolution from 76% to 98%. The same quarterback also appears
as both "C.Veltkamp" and "Caden Veltkamp", so matching is on surname plus first
initial rather than on the string.

A quarterback's plays are his dropbacks - passes and the sacks he took - plus
his own scrambles and designed runs. That means rushing quarterbacks score
higher than they would on a passing-only measure, which is intended: the
question is what the position produced, not what the passing produced.

The adjustment is the same ridge as everywhere else, weighted by plays:

    epa_per_play = quarterback effect + opposing defense effect + intercept

Weighting matters here for the reason it mattered for havoc - a quarterback can
throw 55 times one week and 12 the next, and an unweighted fit treats those as
equal evidence.

Usage:
    python qb_production.py --out results/qb_production.csv
"""

import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from pbp_cache import read_pbp

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

USECOLS = ['game_id', 'team_id', 'play_text', 'play_type_text', 'passing_play',
           'rushing_play', 'epa', 'stat_yardage', 'garbage_time_ind']
CHUNK = 400_000
ALPHA = 1.0
MIN_DROPBACKS = 100      # a season's worth of meaningful snaps
MIN_TEAM_GAMES = 200     # below this a season cannot support the ridge

# Words that can follow a name in play text. They must be excluded from the
# name pattern itself or the match runs past the name and takes the verb.
VERBS = r"pass|sacked|scramble|rush|run|kneel|incomplete|complete|for|to"
_TOK = rf"(?!(?:{VERBS})\b)[A-Za-z][\w'\.\-]*"
_NAME = rf"{_TOK}(?:\s+{_TOK})*"
LEAD = re.compile(
    r"^(?:\(\d+:\d+\)\s*)?(?:(?:No\s+Huddle|Shotgun|Hurry\s+Up)[\w\- ]*?)?\s*"
    r"(?:#\d+\s*)?", re.I)
P_LEAD = re.compile(rf"^({_NAME})\s+(?:{VERBS})\b", re.I)
P_FROM = re.compile(rf"\bpass\s+from\s+({_NAME})", re.I)
P_JERSEY = re.compile(rf"#\d+\s+({_NAME})\s+(?:{VERBS})\b", re.I)
NOT_A_NAME = set(VERBS.split('|'))
SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def passer(text):
    """The quarterback named on a play, or None."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    m = (P_FROM.search(t) or P_JERSEY.search(t)
         or P_LEAD.match(LEAD.sub('', t).strip()))
    if not m:
        return None
    name = m.group(1).strip()
    return None if name.lower() in NOT_A_NAME else name


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r"[^a-z]", '', s.lower())


def _tokens(name):
    parts = [x for x in re.split(r"[.\s]+", str(name)) if x]
    return [x for x in parts if norm(x) and norm(x) not in SUFFIX]


def name_keys(name):
    """Candidate (surname, first initial) keys, most specific first.

    The extra forms cover initials-as-first-name ("E.J. Warner") and compound
    surnames, which is what survives after the suffix handling.
    """
    t = _tokens(name)
    if len(t) < 2:
        return []
    initial = norm(t[0])[:1]
    keys = [(''.join(norm(x) for x in t[1:]), initial)]
    if len(t) > 2:
        keys.append((''.join(norm(x) for x in t[2:]), initial))
        keys.append((norm(t[-1]), initial))
    return keys


def roster_lookup():
    """(season, team_id, name key) -> (player id, name, position)."""
    roster = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_roster.csv'),
                         low_memory=False)
    teams = pd.read_csv(TEAMS)
    name_to_id = dict(zip(teams['location'], teams['id']))
    roster['tid'] = roster['team'].map(name_to_id)
    roster = roster.dropna(subset=['tid', 'lastName'])

    surname = roster['lastName'].map(
        lambda s: ''.join(norm(x) for x in re.split(r"[.\s]+", str(s))
                          if norm(x) and norm(x) not in SUFFIX))
    initial = roster['firstName'].map(lambda s: norm(s)[:1])
    roster['k'] = list(zip(surname, initial))

    # A quarterback wins any collision on surname-plus-initial, since that is
    # the position being attributed; failing that the first row in roster order
    # wins. Done with a stable sort and drop_duplicates rather than by walking
    # the groups: there are a quarter of a million of them, and materialising a
    # sub-frame for each cost 142 seconds per call - paid three times a pipeline
    # run, because qb, receiver and rb production all need this map.
    roster = roster.dropna(subset=['season'])
    roster['_qb'] = (roster['position'] == 'QB').astype(np.int8)
    pick = (roster.sort_values('_qb', ascending=False, kind='stable')
                  .drop_duplicates(subset=['season', 'tid', 'k'], keep='first'))
    names = (pick['firstName'].fillna('').astype(str) + ' '
             + pick['lastName'].fillna('').astype(str))
    return dict(zip(
        zip(pick['season'].astype(int), pick['tid'].astype(int), pick['k']),
        zip(pick['id'].astype(str), names, pick['position'])))


def collect_plays(seasons_by_game, lookup):
    """Every play attributable to a quarterback, across the whole file."""
    kept = []
    scanned = 0
    for chunk in read_pbp(PBP, usecols=USECOLS, low_memory=False,
                             chunksize=CHUNK):
        chunk['game_id'] = pd.to_numeric(chunk['game_id'], errors='coerce')
        chunk['season'] = chunk['game_id'].map(seasons_by_game)
        chunk = chunk.dropna(subset=['season'])
        scanned += CHUNK
        if chunk.empty:
            continue
        for col in ('passing_play', 'rushing_play', 'epa', 'stat_yardage',
                    'team_id', 'garbage_time_ind'):
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk[(chunk['passing_play'] == 1) |
                      (chunk['rushing_play'] == 1)]
        if chunk.empty:
            continue
        chunk['raw'] = chunk['play_text'].map(passer)
        chunk = chunk.dropna(subset=['raw', 'team_id'])
        if chunk.empty:
            continue

        pid, who, pos = [], [], []
        for season, tid, raw in zip(chunk['season'], chunk['team_id'],
                                    chunk['raw']):
            hit = None
            for k in name_keys(raw):
                hit = lookup.get((int(season), int(tid), k))
                if hit:
                    break
            pid.append(hit[0] if hit else None)
            who.append(hit[1] if hit else None)
            pos.append(hit[2] if hit else None)
        chunk['pid'], chunk['who'], chunk['pos'] = pid, who, pos
        kept.append(chunk[chunk['pos'] == 'QB'].drop(columns=['play_text']))
        print(f"  scanned {scanned:,} plays", end='\r')
    print()
    return pd.concat(kept, ignore_index=True)


def fbs_by_season():
    """Per-season FBS membership; schools move up, so this cannot be static."""
    path = os.path.join(PLAYER_DIR, 'cfbd_classification.csv')
    c = pd.read_csv(path, low_memory=False)
    c = c[c['fbs'] == 1]
    return {int(s): set(g['team_id']) for s, g in c.groupby('season')}


def adjust(per_game, alpha=ALPHA):
    """Play-weighted ridge, one season at a time, FBS opponents only."""
    fbs = fbs_by_season()
    out = []
    for season, sd in per_game.groupby('season'):
        sd = sd.dropna(subset=['opponent_id', 'epa_per_play'])
        # FCS defenses appear once or twice each and are easy to throw on, so
        # leaving them in the pool drags the reference point. They made the
        # unweighted mean opponent effect zero while the defenses quarterbacks
        # actually faced averaged -0.10, which put every adjusted figure a tenth
        # of a point above its raw one.
        members = fbs.get(int(season), set())
        if members:
            sd = sd[sd['team_id'].isin(members)
                    & sd['opponent_id'].isin(members)]
        if len(sd) < MIN_TEAM_GAMES:
            continue
        qbs = pd.Index(sorted(sd['pid'].unique()))
        opps = pd.Index(sorted(sd['opponent_id'].unique()))
        qi = sd['pid'].map({p: i for i, p in enumerate(qbs)}).to_numpy()
        oi = sd['opponent_id'].map({o: i for i, o in enumerate(opps)}).to_numpy()
        n = len(sd)
        rows = np.arange(n)
        X = sparse.hstack([
            sparse.csr_matrix((np.ones(n), (rows, qi)), shape=(n, len(qbs))),
            sparse.csr_matrix((np.ones(n), (rows, oi)), shape=(n, len(opps))),
        ]).tocsr()
        m = Ridge(alpha=alpha, fit_intercept=True).fit(
            X, sd['epa_per_play'].to_numpy(float),
            sample_weight=sd['plays'].to_numpy(float))

        # Centre on the defense actually faced, weighted by snaps, rather than
        # on the unweighted mean of the opponent coefficients. Those are not the
        # same number whenever some opponents are faced more than others, and
        # the difference is the whole level of the statistic.
        opp_series = pd.Series(m.coef_[len(qbs):], index=opps)
        opp_mean = float(np.average(
            sd['opponent_id'].map(opp_series).to_numpy(float),
            weights=sd['plays'].to_numpy(float)))
        eff = pd.DataFrame({
            'pid': qbs,
            'adj_epa_per_play': m.intercept_ + m.coef_[:len(qbs)] + opp_mean,
        })
        eff['season'] = int(season)
        # strength of the defenses faced, centred the same way so that zero
        # means an average schedule rather than an arbitrary offset
        opp_eff = opp_series - opp_mean
        # weighted mean per quarterback as two sums rather than a groupby-apply
        t = sd[['pid', 'plays']].copy()
        t['_wo'] = sd['opponent_id'].map(opp_eff).to_numpy() * t['plays']
        agg = t.groupby('pid')[['_wo', 'plays']].sum()
        faced = (agg['_wo'] / agg['plays']).rename('defense_faced')
        out.append(eff.merge(faced, on='pid', how='left'))
        print(f"  {int(season)}: {len(qbs)} quarterbacks", end='\r')
    print()
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--min-dropbacks', type=int, default=MIN_DROPBACKS)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'qb_production.csv'))
    # Reading and attributing the play-by-play takes minutes; the adjustment
    # takes seconds. Caching the attributed plays makes it practical to iterate
    # on the adjustment without re-parsing 2.7 million rows each time.
    ap.add_argument('--cache', default=os.path.join(_HERE, 'temp',
                                                    'qb_plays.pkl'))
    ap.add_argument('--reparse', action='store_true',
                    help='ignore the cache and re-read the play-by-play')
    args = ap.parse_args()

    games = pd.read_csv(GAMES, low_memory=False)[
        ['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    for c in games.columns:
        games[c] = pd.to_numeric(games[c], errors='coerce')
    games = games.dropna().drop_duplicates('id')
    seasons_by_game = dict(zip(games['id'].astype(int),
                               games['season'].astype(int)))

    if not args.reparse and os.path.exists(args.cache):
        plays = pd.read_pickle(args.cache)
        print(f"loaded {len(plays):,} attributed plays from {args.cache}")
    else:
        print("building the roster lookup...")
        lookup = roster_lookup()
        print(f"  {len(lookup):,} (season, team, name) keys")
        print("reading the play-by-play...")
        plays = collect_plays(seasons_by_game, lookup)
        print(f"  {len(plays):,} plays attributed to a quarterback")
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        plays.to_pickle(args.cache)
        print(f"  cached to {args.cache}")

    plays['season'] = plays['season'].astype(int)
    plays['game_id'] = plays['game_id'].astype(int)
    per_game = plays.groupby(['season', 'game_id', 'pid', 'who', 'team_id'],
                             as_index=False).agg(
        plays=('epa', 'size'), epa=('epa', 'sum'),
        dropbacks=('passing_play', 'sum'), rushes=('rushing_play', 'sum'),
        yards=('stat_yardage', 'sum'))
    per_game['epa_per_play'] = per_game['epa'] / per_game['plays']
    per_game = per_game.merge(
        games.rename(columns={'id': 'game_id'})[
            ['game_id', 'home_team_id', 'away_team_id']],
        on='game_id', how='left')
    per_game['opponent_id'] = np.where(
        per_game['team_id'] == per_game['home_team_id'],
        per_game['away_team_id'], per_game['home_team_id'])

    print("opponent-adjusting...")
    eff = adjust(per_game, alpha=args.alpha)

    season = per_game.groupby(['season', 'pid', 'who'], as_index=False).agg(
        team_id=('team_id', 'last'), games=('game_id', 'nunique'),
        plays=('plays', 'sum'), epa=('epa', 'sum'),
        dropbacks=('dropbacks', 'sum'), rushes=('rushes', 'sum'),
        yards=('yards', 'sum'))
    season['epa_per_play'] = season['epa'] / season['plays']
    season = season.merge(eff, on=['season', 'pid'], how='left')
    season = season[season['dropbacks'] >= args.min_dropbacks]
    season['adj_rank'] = (season.groupby('season')['adj_epa_per_play']
                          .rank(ascending=False, method='min').astype('Int64'))
    season = season.sort_values(['season', 'adj_rank'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    season.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(season)} quarterback-seasons, "
          f"{int(season.season.min())}-{int(season.season.max())})")
    print(season[['plays', 'epa_per_play', 'adj_epa_per_play',
                  'defense_faced']].describe().round(4).to_string())


if __name__ == '__main__':
    main()
