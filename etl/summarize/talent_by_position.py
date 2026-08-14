#!/usr/bin/env python3
"""Recruiting talent per team-season, split by position group and ranked.

roster_talent.py answers "how much talent does this team have"; this answers
"where is it". The difference that matters is granularity: roster_talent lumps
running backs, receivers and tight ends into one SKILL bucket, which hides the
thing you actually want to see - a team with elite receivers and nothing at
running back reads identically to the reverse.

Ratings come from the recruiting services, joined through the roster's
recruitIds. That makes this a measure of talent *acquired*, not talent
developed or deployed: a fifth-year starter and a redshirt freshman with the
same recruiting grade count the same, and a walk-on who became a star counts
not at all because he has no recruit id to join on. Roughly a third of roster
rows carry one.

Levels are not comparable across seasons - the services re-rate and the scale
drifts - so every measure is also published as a within-season percentile and
rank, which are what should be read.

Usage:
    python talent_by_position.py --out results/talent_by_position.csv
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')

# Finer than roster_talent.py's six. Splitting SKILL into RB, WR and TE is the
# whole point; DL keeps ends and tackles together because the services rate
# edge and interior inconsistently across years and splitting them would
# compare different definitions season to season.
POSITION_GROUPS = {
    'QB': 'QB',
    'RB': 'RB', 'FB': 'RB',
    'WR': 'WR', 'ATH': 'WR',
    'TE': 'TE',
    'OL': 'OL', 'C': 'OL', 'G': 'OL', 'OT': 'OL', 'OG': 'OL', 'T': 'OL',
    'DL': 'DL', 'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'EDGE': 'DL',
    'LB': 'LB', 'ILB': 'LB', 'OLB': 'LB', 'MLB': 'LB',
    'DB': 'DB', 'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB',
    'PK': 'ST', 'P': 'ST', 'K': 'ST', 'LS': 'ST',
}
GROUPS = ('QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'ST')

# A group needs enough rated bodies for a mean to mean anything. Quarterbacks
# and tight ends carry small rooms, so this is deliberately low; the count is
# published alongside so a thin room is visible rather than silently averaged.
MIN_IN_GROUP = 2
MIN_LINKED = 20


def load(name):
    path = os.path.join(PLAYER_DIR, f'cfbd_{name}.csv')
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run collect_cfbd_players.py first")
    return pd.read_csv(path, low_memory=False)


def first_recruit_id(value):
    """roster.recruitIds arrives as the string form of a list."""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'talent_by_position.csv'))
    args = ap.parse_args()

    roster = load('roster')
    recruits = load('recruits')
    teams = load('teams')
    name_to_id = dict(zip(teams['school'], teams['id']))

    # Conference has to come from the per-season classification file, not from
    # cfbd_teams. cfbd_teams carries only the CURRENT conference, which back-
    # dates every realignment: it puts Texas in the SEC in 2015 and USC in the
    # Big Ten in 2016, and erases the Pac-12 entirely. That mislabels 17% of
    # team-seasons, concentrated in 2017-2022.
    cls = load('classification')
    conf = {(r.team, int(r.season)): r.conference
            for r in cls.itertuples() if pd.notna(r.conference)}

    roster['rid'] = roster['recruitIds'].map(first_recruit_id)
    recruits['id'] = recruits['id'].astype(str)
    recruits['rating'] = pd.to_numeric(recruits['rating'], errors='coerce')
    recruits['stars'] = pd.to_numeric(recruits['stars'], errors='coerce')

    # Roster size before the join, so coverage can be reported. It is not a
    # footnote: within FBS the median team has 61% of its roster rated, but
    # the service academies sit near 10% because their players are largely
    # unrated by the services. A rating computed over 12 of 141 players is a
    # different kind of number to one computed over 90 of 110, and coverage
    # correlates +0.60 with the rating itself.
    size = (roster.groupby(['team', 'season']).size()
            .rename('roster_n').reset_index())

    j = roster.dropna(subset=['rid']).merge(
        recruits[['id', 'stars', 'rating']], left_on='rid', right_on='id',
        how='inner', suffixes=('', '_rec')).dropna(subset=['rating'])
    j['group'] = j['position'].map(POSITION_GROUPS)
    j['blue'] = j['stars'] >= 4
    print(f"roster {len(roster):,} rows, joined and rated {len(j):,} "
          f"({len(j)/len(roster):.1%})")
    unmapped = sorted(set(j.loc[j['group'].isna(), 'position'].dropna()))
    if unmapped:
        print(f"  positions with no group (dropped): {unmapped}")

    rows = []
    for (team, season), g in j.groupby(['team', 'season']):
        if len(g) < MIN_LINKED:
            continue
        row = {'team': team, 'season': int(season), 'linked': len(g)}
        for grp in GROUPS:
            sub = g[g['group'] == grp]
            n = len(sub)
            row[f'{grp}_n'] = n
            if n >= MIN_IN_GROUP:
                row[f'{grp}_rating'] = float(sub['rating'].mean())
                row[f'{grp}_blue'] = int(sub['blue'].sum())
                row[f'{grp}_blue_pct'] = float(sub['blue'].mean())
            else:
                row[f'{grp}_rating'] = np.nan
                row[f'{grp}_blue'] = np.nan
                row[f'{grp}_blue_pct'] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.merge(size, on=['team', 'season'], how='left')
    out['coverage'] = out['linked'] / out['roster_n']

    # Within-season rank and percentile. Levels drift as the services re-rate,
    # so the rank is the comparable number, not the rating.
    for grp in GROUPS:
        col = f'{grp}_rating'
        out[f'{grp}_rank'] = (out.groupby('season')[col]
                              .rank(ascending=False, method='min')
                              .astype('Int64'))
        out[f'{grp}_pct'] = out.groupby('season')[col].rank(pct=True)

    out['team_id'] = out['team'].map(name_to_id)
    out['conference'] = [conf.get((t, s)) for t, s in
                         zip(out['team'], out['season'])]
    print(f"team id match: {out['team_id'].notna().mean():.1%}")
    print(f"conference match: {out['conference'].notna().mean():.1%}")

    cols = ['team_id', 'team', 'conference', 'season',
            'linked', 'roster_n', 'coverage']
    for grp in GROUPS:
        cols += [f'{grp}_rating', f'{grp}_rank', f'{grp}_pct',
                 f'{grp}_n', f'{grp}_blue', f'{grp}_blue_pct']
    out = out[cols].sort_values(['season', 'team'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} team-seasons, "
          f"{int(out.season.min())}-{int(out.season.max())})")
    print(out[[f'{g}_rating' for g in GROUPS]].describe().round(4).to_string())


if __name__ == '__main__':
    main()
