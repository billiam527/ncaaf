#!/usr/bin/env python3
"""Individual defensive production, parsed out of our own play-by-play.

WHY THIS EXISTS

Every defensive room in this model was ranked by recruiting grade, because
nothing measured what a defender actually did. That put Notre Dame's Leonard
Moore - a 0.8940 recruit, 432nd nationally, who led that secondary in passes
defensed - eleventh in his own room, behind three true freshmen who had never
played a snap. The grade is frozen the day a player signs and the room ranking
inherited that.

The play-by-play names defenders on the plays they make, and that has been
sitting unused:

    broken up by      3,420 plays in 2025   the coverage event
    sacked by         1,771
    hurried by        2,389
    recovered by      1,851
    intercepted by      750
    forced by           475

Tacklers are not named - "tackled by" appears zero times - so there are no
tackle counts here. That is less of a loss than it sounds. Tackles largely
measure how often the ball came your way; these measure disruption.

WHAT IS MEASURED, AND WHY IT IS A SHARE

There are still no snap counts for defenders, so a per-player rate cannot be
computed: the denominator does not exist. What can be computed is a player's
SHARE of his team's disruptive events, which is robust to not knowing how often
he was on the field, and is the same construction used for a receiver's target
share. That share is then scaled by the team's opponent-adjusted rate, so a
large share of a good defence outranks a large share of a bad one.

TWO HONEST LIMITS

Only 48% of interceptions name the defender - the rest read "pass intercepted"
with nobody credited - so interception counts here are a floor, and they are
weighted accordingly.

A shutdown corner is thrown at less, so coverage counting stats understate the
best ones. Moore's own line shows the shape of it: 11 passes defensed in 2024,
7 in 2025. That is either a step back or quarterbacks avoiding him, and nothing
here can tell those apart. Fixing it needs targets-allowed, which would mean
attributing every pass to a defender rather than to a receiver.

Usage:
    python defensive_production.py --out results/defensive_production.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

from qb_production import name_keys, roster_lookup
from pbp_cache import read_pbp

_HERE = os.path.dirname(os.path.abspath(__file__))
PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')

USECOLS = ['game_id', 'team_id', 'play_text', 'play_type_text',
           'passing_play', 'rushing_play', 'offensive_play']
CHUNK = 400_000

FIRST_ROSTER_SEASON = 2014

DB_POSITIONS = {'DB', 'CB', 'S', 'FS', 'SS', 'SAF', 'NB'}
FRONT_POSITIONS = {'DL', 'DE', 'DT', 'NT', 'EDGE', 'NG',
                   'LB', 'ILB', 'OLB', 'MLB', 'WLB', 'SLB'}

# Names arrive in two shapes - "Aamaris Brown" and "Robinson,Quincy" - and the
# second appears on 216 defensive phrases in 2025 alone, so both are matched.
# The name is capped at three tokens: past that the match starts eating the
# rest of the sentence.
_TOK = r"[A-Z][\w'\.\-]*"
_PLAIN = rf"{_TOK}(?:\s+{_TOK}){{0,2}}"
_COMMA = rf"{_TOK},\s*{_TOK}"
# a jersey number in front of the name is the third format, and skipping it
# cost 30% of hurries: "QB hurried by #4 J.Montgomery" parsed as nothing
_JERSEY = r"(?:#\d+\s*)?"
_NAME = rf"{_JERSEY}(?:{_COMMA}|{_PLAIN})"

# The phrase is matched case-insensitively because providers differ on it, but
# the NAME is not: with re.I on the whole pattern, [A-Z] matches lowercase and
# "hurried by the pressure" parses as a defender called The Pressure. Inline
# (?i:...) scopes the flag to the phrase alone. Missing this cost 77% of the
# hurries - 550 found where 2025 by itself holds 2,389.
EVENTS = {
    'breakup':   re.compile(rf"(?i:broken up by)\s+({_NAME})"),
    'sack':      re.compile(rf"(?i:sacked by)\s+({_NAME})"),
    'hurry':     re.compile(rf"(?i:hurried by)\s+({_NAME})"),
    'intercept': re.compile(rf"(?i:intercepted by)\s+({_NAME})"),
    'forced':    re.compile(rf"(?i:forced by)\s+({_NAME})"),
}
# "sacked by X and Y" splits the credit; 4% of sacks name two men
SECOND = re.compile(rf"\band\s+({_NAME})")


def defenders(text):
    """(event kind, raw name, credit) triples named on a play.

    A sack shared by two men is half a sack each, which is how the sport counts
    it - crediting both with a whole one would inflate every co-sack in the
    file. It applies to 4% of sacks.
    """
    if not isinstance(text, str):
        return []
    out = []
    for kind, pat in EVENTS.items():
        m = pat.search(text)
        if not m:
            continue
        tail = text[m.end():m.end() + 44].lstrip(' ,')
        m2 = SECOND.match(tail) if kind == 'sack' else None
        credit = 0.5 if m2 else 1.0
        out.append((kind, m.group(1).strip(), credit))
        if m2:
            out.append((kind, m2.group(1).strip(), credit))
    return out


def flip(raw):
    """Normalise the three formats to 'First Last'.

    'Robinson,Quincy' -> 'Quincy Robinson'; '#4 J.Montgomery' -> 'J.Montgomery'.
    """
    raw = re.sub(r'^#\d+\s*', '', raw).strip()
    if ',' in raw:
        a, b = raw.split(',', 1)
        return f"{b.strip()} {a.strip()}"
    return raw


def resolve(lookup, season, tid, raw):
    if not raw or pd.isna(tid):
        return (None, None, None)
    for k in name_keys(flip(raw)):
        hit = lookup.get((int(season), int(tid), k))
        if hit:
            return hit
    return (None, None, None)


def collect(seasons_by_game, lookup, games):
    """One row per named defensive event, with the defender resolved.

    The team_id on a play is the offence, so the defender belongs to the other
    side of that game - which is why the games table is needed here.
    """
    home = dict(zip(games['id'], games['home_team_id']))
    away = dict(zip(games['id'], games['away_team_id']))
    rows = []
    for chunk in read_pbp(PBP, usecols=USECOLS, low_memory=False,
                          chunksize=CHUNK):
        chunk = chunk.copy()
        chunk['game_id'] = pd.to_numeric(chunk['game_id'], errors='coerce')
        chunk['season'] = chunk['game_id'].map(seasons_by_game)
        chunk = chunk.dropna(subset=['season'])
        if chunk.empty:
            continue
        chunk['team_id'] = pd.to_numeric(chunk['team_id'], errors='coerce')
        txt = chunk['play_text'].astype(str)
        hit = txt.str.contains('broken up by|sacked by|hurried by|'
                               'intercepted by|forced by', regex=True,
                               case=False, na=False)
        sub = chunk[hit]
        if sub.empty:
            continue
        h = sub['game_id'].map(home)
        a = sub['game_id'].map(away)
        # defence is whichever side is not the offence on this play
        dtid = np.where(sub['team_id'] == h, a, h)
        for season, off, dt, text in zip(sub['season'], sub['team_id'], dtid,
                                         sub['play_text'].astype(str)):
            for kind, raw, credit in defenders(text):
                rows.append((int(season), dt, off, kind, raw, credit))
    ev = pd.DataFrame(rows, columns=['season', 'team_id', 'opponent_id',
                                     'event', 'raw', 'credit'])
    ev = ev.dropna(subset=['team_id'])
    ev['team_id'] = ev['team_id'].astype(int)
    got = [resolve(lookup, s, t, r) for s, t, r in
           zip(ev['season'], ev['team_id'], ev['raw'])]
    ev['pid'] = [x[0] for x in got]
    ev['who'] = [x[1] for x in got]
    ev['pos'] = [x[2] for x in got]
    return ev


def merge_box_scores(wide):
    """Official season counts alongside the parsed ones, and a reconciliation.

    Joined on the CFBD player id, which is the same id the roster carries, so
    this is an id join rather than a name match.
    """
    path = os.path.join(PLAYER_DIR, 'cfbd_stats.csv')
    if not os.path.exists(path):
        print("  no cfbd_stats.csv; box scores skipped")
        for c in ('pd_box', 'sack_box', 'hurry_box', 'tfl_box', 'tot_box'):
            wide[c] = np.nan
        return finalise(wide)
    st = pd.read_csv(path, low_memory=False)
    st = st[st['category'] == 'defensive'].copy()
    st['stat'] = pd.to_numeric(st['stat'], errors='coerce')
    st['playerId'] = st['playerId'].astype(str)
    box = st.pivot_table(index=['season', 'playerId'], columns='statType',
                         values='stat', aggfunc='sum').reset_index()
    box.columns.name = None
    ren = {'PD': 'pd_box', 'SACKS': 'sack_box', 'QB HUR': 'hurry_box',
           'TFL': 'tfl_box', 'TOT': 'tot_box'}
    box = box.rename(columns=ren)
    keep = ['season', 'playerId'] + [v for v in ren.values() if v in box.columns]
    wide['pid'] = wide['pid'].astype(str)
    wide = wide.merge(box[keep].rename(columns={'playerId': 'pid'}),
                      on=['season', 'pid'], how='left')

    print("\n  play-by-play against the official box score, where both exist:")
    for pbp_col, box_col, lab in (('breakup', 'pd_box', 'breakups / PD'),
                                  ('sack', 'sack_box', 'sacks'),
                                  ('hurry', 'hurry_box', 'hurries')):
        if box_col not in wide.columns:
            continue
        b = wide.dropna(subset=[box_col])
        b = b[(b[pbp_col] > 0) | (b[box_col] > 0)]
        if not len(b):
            continue
        print(f"    {lab:<16}pbp {b[pbp_col].sum():>9,.0f}   "
              f"box {b[box_col].sum():>9,.0f}   "
              f"pbp finds {b[pbp_col].sum() / max(b[box_col].sum(), 1):>5.0%}"
              f"   r {b[pbp_col].corr(b[box_col]):>+5.2f}")
    return finalise(wide)


def finalise(wide):
    """Prefer the official count, fall back to the parsed one."""
    for out, pbp_col, box_col in (('pd_best', 'breakup', 'pd_box'),
                                  ('sack_best', 'sack', 'sack_box'),
                                  ('hurry_best', 'hurry', 'hurry_box')):
        if box_col in wide.columns:
            wide[out] = wide[box_col].fillna(wide[pbp_col])
        else:
            wide[out] = wide[pbp_col]
    for c in ('tfl_box', 'tot_box'):
        if c not in wide.columns:
            wide[c] = 0.0
        wide[c] = wide[c].fillna(0.0)
    return wide


COVERAGE_SLOTS = 5      # nickel: two corners, a slot, two safeties
MIN_TACKLES = 10        # below this a defender did not play enough to credit


def coverage_value(wide):
    """Share out the yards a secondary saved against expectation.

    THE MEASURE

    Counting stats understate the best corners, because the best corners are
    thrown at least. What is not avoidable is the yardage the defence actually
    gave up, and the ridge in summarize_games already expresses that against
    expectation: adjusted_pass_yards_per_play_def is yards allowed given the
    offences a team faced, so the league mean minus a team's figure is yards
    saved per dropback against what was expected of it.

    WHAT IS TAKEN OUT FIRST

    Not all of that belongs to the secondary. The front seven predicts pass
    defence better than the secondary does - partial correlations of -0.220
    against +0.065 - because pressure is most of coverage. So the team figure is
    residualised against the front seven's play before anyone in the secondary
    is credited with it, exactly as defensive_backs.py does. What is left is the
    part the pass rush does not account for.

    WHO GETS IT

    Five men cover, and without snap counts we cannot know which five. Three
    candidate selectors, and only one of them is honest:

      recruiting grade   the thing this module exists to replace
      coverage events    reintroduces the bias being corrected - a shutdown
                         corner has few, because nobody throws at him
      tackles            accumulate mostly by being on the field, and are only
                         weakly related to coverage skill

    Tackles it is: the top five by tackles, credited equally. Equal shares
    rather than weighted ones because any weighting would have to come from the
    counting stats, which is the bias we just removed.

    The number travels with the player. A corner who saved yards in 2025 carries
    that into his 2026 rating, which is the whole point - it is a property of
    him, not of the team he happened to be on.
    """
    seasons = os.path.join(_HERE, 'results', 'season_summaries.csv')
    f7 = os.path.join(_HERE, 'results', 'front_seven.csv')
    if not os.path.exists(seasons):
        wide['cov_yards_value'] = np.nan
        return wide
    S = pd.read_csv(seasons, low_memory=False)
    col = 'adjusted_pass_yards_per_play_def'
    if col not in S.columns:
        wide['cov_yards_value'] = np.nan
        return wide
    t = S[['team_id', 'season', col]].dropna().copy()
    # higher is worse for this column - verified against points allowed - so
    # saved yards is the league mean minus the team figure
    t['saved'] = t.groupby('season')[col].transform('mean') - t[col]

    if os.path.exists(f7):
        F = pd.read_csv(f7, low_memory=False)[['team_id', 'season', 'f7_play']]
        t = t.merge(F, on=['team_id', 'season'], how='left')
        out = []
        for season, g in t.groupby('season'):
            m = g.dropna(subset=['saved', 'f7_play'])
            if len(m) < 30:
                g = g.assign(resid=g['saved'] - g['saved'].mean())
            else:
                A = np.column_stack([np.ones(len(m)), m['f7_play'].to_numpy()])
                b, *_ = np.linalg.lstsq(A, m['saved'].to_numpy(), rcond=None)
                g = g.assign(resid=g['saved']
                             - (b[0] + b[1] * g['f7_play'].fillna(
                                 g['f7_play'].mean())))
            out.append(g)
        t = pd.concat(out, ignore_index=True)
    else:
        t['resid'] = t['saved'] - t.groupby('season')['saved'].transform('mean')

    # A floor on tackles, because without one the top five by tackles on a
    # thinly covered roster is five men with two tackles between them, and the
    # leaderboard fills with FCS defenders who barely played.
    db = wide[(wide['group'] == 'DB') & (wide['tot_box'] >= MIN_TACKLES)].copy()
    db['_rank'] = db.groupby(['season', 'team_id'])['tot_box'].rank(
        ascending=False, method='first')
    # and a team needs a full complement before its figure is shared out at all
    full = db.groupby(['season', 'team_id'])['pid'].transform('size')
    db['credited'] = (db['_rank'] <= COVERAGE_SLOTS) & (full >= COVERAGE_SLOTS)
    db = db.merge(t[['team_id', 'season', 'resid']], on=['team_id', 'season'],
                  how='left')
    n = db[db['credited']].groupby(['season', 'team_id'])['pid'].transform(
        'size')
    db['cov_yards_value'] = np.where(db['credited'], db['resid'] / n, 0.0)
    wide = wide.merge(
        db[['season', 'team_id', 'pid', 'cov_yards_value', 'credited']],
        on=['season', 'team_id', 'pid'], how='left')
    # a defender below the tackle floor is uncredited, not missing
    wide['credited'] = wide['credited'].fillna(False).astype(bool)
    wide.loc[~wide['credited'], 'cov_yards_value'] = 0.0
    return wide


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'defensive_production.csv'))
    args = ap.parse_args()

    games = pd.read_csv(GAMES, low_memory=False)[
        ['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    for c in games.columns:
        games[c] = pd.to_numeric(games[c], errors='coerce')
    games = games.dropna().drop_duplicates('id')
    seasons_by_game = dict(zip(games['id'].astype(int),
                               games['season'].astype(int)))

    print("building the roster map...")
    lookup = roster_lookup()
    print("reading the play-by-play...")
    ev = collect(seasons_by_game, lookup, games)

    # The roster file starts in 2014; before that nothing resolves, because
    # there is nothing to resolve against. Reporting a rate across 2010-2013
    # made the parser look broken when it was only unsupplied.
    ev = ev[ev['season'] >= FIRST_ROSTER_SEASON]
    named = len(ev)
    got = ev.dropna(subset=['pid'])
    print(f"  {named:,} named events from {FIRST_ROSTER_SEASON}, "
          f"{len(got):,} resolved ({len(got) / max(named, 1):.0%})")
    print(f"  {'event':<12}{'named':>9}{'resolved':>10}")
    for k, sub in ev.groupby('event'):
        print(f"  {k:<12}{len(sub):>9,}{sub.pid.notna().mean():>10.0%}")
    print("  what does not resolve is overwhelmingly a defence whose roster we"
          " do not hold,\n  which is FCS opponents - the comma-name providers"
          " cover those games.")
    ev = got

    # summed on credit, not counted, so a shared sack lands as 0.5 each
    wide = (ev.pivot_table(index=['season', 'team_id', 'pid', 'who', 'pos'],
                           columns='event', values='credit', aggfunc='sum',
                           fill_value=0.0).reset_index())
    wide.columns.name = None
    for c in ('breakup', 'sack', 'hurry', 'intercept', 'forced'):
        if c not in wide.columns:
            wide[c] = 0.0

    # MERGE WITH THE BOX SCORES
    #
    # The two sources have opposite weaknesses. The play-by-play is game-level
    # and says who the offence was, so it can be opponent-adjusted, but the
    # crediting is inconsistent between providers and incomplete. The season box
    # score in cfbd_stats is the official count and misses nothing, but has no
    # opponent and no date. It also carries two things the play-by-play cannot:
    # tackles for loss and total tackles, because tacklers are never named in
    # the text.
    #
    # So the box score is the count of record wherever it exists, and the
    # play-by-play supplies what the box score lacks - interceptions, which are
    # not among its stat types at all, and the game-level structure.
    wide = merge_box_scores(wide)

    # Coverage and pressure are kept apart: they are different jobs, and a
    # secondary should not be credited for a pass rush. Interceptions carry
    # extra weight per event but come from the play-by-play alone, where only
    # 48% of them name the defender, so the count is a floor - weighting them 2
    # rather than the 4 or 5 their value would suggest keeps that undercount
    # from dominating.
    wide['coverage_events'] = wide['pd_best'] + 2 * wide['intercept']
    wide['pressure_events'] = (wide['sack_best'] + 0.5 * wide['hurry_best']
                               + wide['forced'] + 0.5 * wide['tfl_box'])

    tot = wide.groupby(['season', 'team_id'])[
        ['coverage_events', 'pressure_events']].transform('sum')
    # share of his own team's events - the only construction available without
    # snap counts, and robust to not knowing how often he was on the field
    wide['coverage_share'] = np.where(tot['coverage_events'] > 0,
                                      wide['coverage_events']
                                      / tot['coverage_events'], np.nan)
    wide['pressure_share'] = np.where(tot['pressure_events'] > 0,
                                      wide['pressure_events']
                                      / tot['pressure_events'], np.nan)
    wide['group'] = np.where(wide['pos'].isin(DB_POSITIONS), 'DB',
                     np.where(wide['pos'].isin(FRONT_POSITIONS), 'FRONT',
                              'OTHER'))

    # scale each share by how good the unit was, so a large share of a strong
    # defence outranks a large share of a weak one
    for f, col, out in (('defensive_backs.csv', 'db_rating', 'unit_db'),
                        ('front_seven.csv', 'f7_rating', 'unit_front')):
        p = os.path.join(_HERE, 'results', f)
        if os.path.exists(p):
            u = pd.read_csv(p, low_memory=False)[['team_id', 'season', col]]
            wide = wide.merge(u.rename(columns={col: out}),
                              on=['team_id', 'season'], how='left')
    wide['cov_value'] = wide['coverage_share'] * wide.get('unit_db', 0).fillna(0)
    wide['prs_value'] = wide['pressure_share'] * wide.get('unit_front',
                                                          0).fillna(0)
    wide = coverage_value(wide)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wide.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(wide):,} defender-seasons, "
          f"{int(wide.season.min())}-{int(wide.season.max())})")
    print(f"\n  {'group':<8}{'players':>9}{'PD':>9}{'sacks':>9}"
          f"{'hurries':>9}{'TFL':>8}{'ints':>7}")
    for gname, g in wide.groupby('group'):
        print(f"  {gname:<8}{len(g):>9,}{g.pd_best.sum():>9,.0f}"
              f"{g.sack_best.sum():>9,.1f}{g.hurry_best.sum():>9,.0f}"
              f"{g.tfl_box.sum():>8,.0f}{g.intercept.sum():>7,.0f}")
    half = (wide['sack'] % 1 != 0).sum()
    print(f"\n  defender-seasons carrying a half sack from the text: {half:,}")

    if 'cov_yards_value' in wide.columns:
        c = wide[wide.get('credited', False) == True]
        print(f"\n  yards-saved credit, split across the top {COVERAGE_SLOTS} "
              f"by tackles:")
        print(f"    defenders credited        {len(c):,}")
        print(f"    per-man value  mean {c.cov_yards_value.mean():+.4f}  "
              f"sd {c.cov_yards_value.std():.4f}")
        prev = wide[['season', 'pid', 'cov_yards_value']].copy()
        prev['season'] += 1
        prev = prev.rename(columns={'cov_yards_value': 'prior'})
        rep = wide.merge(prev, on=['season', 'pid'], how='inner').dropna(
            subset=['cov_yards_value', 'prior'])
        rep = rep[rep['cov_yards_value'] != 0]
        if len(rep) > 50:
            print(f"    repeats year to year      "
                  f"{rep.cov_yards_value.corr(rep.prior):+.3f}  "
                  f"(n={len(rep):,})")
            print("    that is what justifies carrying it forward at all")


if __name__ == '__main__':
    main()
