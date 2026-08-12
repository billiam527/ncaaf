#!/usr/bin/env python3
"""Returning production by position group and starter/backup tier.

CFBD publishes one returning-production number per team, split only into
passing, rushing and receiving. This rebuilds that number from the player data
and then breaks it out by position group and by whether the player was a
starter or a backup, which is the part the API does not provide.

The base mechanic is validated against CFBD's own figure: weighting each
player's prior-season totalPPA and counting a player as returning when they
appear on the *same team's* roster the following year reproduces their
percentPPA at r = +0.999 with a mean absolute difference of 0.003. The
same-team condition is what makes it work - a player who transfers out still
appears in the next season's league-wide roster, so a team-agnostic test scores
a transfer-out as a return and inflates every team (Arkansas 2024: 0.995
against a true 0.527).

What each position group can be measured with:

  QB RB WR TE   usage share of team plays, from /player/usage
  DL LB DB      total tackles, as a proxy for time on the field
  ST            kick and punt attempts
  OL            nothing. There is no snap count, usage or efficiency metric
                for offensive linemen anywhere in the API, so OL returns are
                roster continuity only and are counted, not production-weighted

"Starter" is derived, not given - the API has no starts or snap-count field.
Within each team, season and position group, players are ranked by that group's
production metric and the top STARTER_SLOTS are treated as starters. That
follows usage rather than depth chart, so an injured starter replaced midseason
splits across both tiers rather than being misassigned to one.

Usage:
    python returning_production.py --out results/returning_production.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))      # etl/summarize
_ETL = os.path.dirname(_HERE)                           # etl
PLAYER_DIR = os.path.join(_ETL, 'collect', 'collect_cfbd_players', 'temp')

# Position strings vary by team and year (DL/DE/DT/NT/EDGE all appear), so they
# are folded into groups that correspond to how a team actually lines up.
POSITION_GROUPS = {
    'QB': 'QB',
    'RB': 'RB', 'FB': 'RB',
    'WR': 'WR',
    'TE': 'TE',
    'OL': 'OL', 'C': 'OL', 'G': 'OL', 'OT': 'OL', 'OG': 'OL', 'T': 'OL',
    'DL': 'DL', 'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'EDGE': 'DL',
    'LB': 'LB', 'ILB': 'LB', 'OLB': 'LB', 'MLB': 'LB',
    'DB': 'DB', 'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB',
    'PK': 'ST', 'P': 'ST', 'K': 'ST', 'LS': 'ST',
}

# Roughly how many of each group are on the field at once. Used to split each
# group into a starter tier and a backup tier by production rank.
STARTER_SLOTS = {'QB': 1, 'RB': 1, 'WR': 3, 'TE': 1, 'OL': 5,
                 'DL': 4, 'LB': 3, 'DB': 4, 'ST': 1}

SKILL_GROUPS = ('QB', 'RB', 'WR', 'TE')
DEFENSIVE_GROUPS = ('DL', 'LB', 'DB')


def load(name):
    path = os.path.join(PLAYER_DIR, f'cfbd_{name}.csv')
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run collect_cfbd_players.py first")
    return pd.read_csv(path, low_memory=False)


def build_player_seasons():
    """One row per (team, season, player) with a position group and a
    production weight comparable within that group."""
    roster = load('roster')
    usage = load('usage')
    stats = load('stats')

    for frame, col in ((roster, 'id'), (usage, 'id')):
        frame[col] = frame[col].astype(str)
    stats['playerId'] = stats['playerId'].astype(str)

    roster['group'] = roster['position'].map(POSITION_GROUPS)
    roster = roster.dropna(subset=['group', 'team', 'season'])
    roster['key'] = roster['team'] + '|' + roster['id']

    # skill production: share of team plays
    skill = usage[['season', 'team', 'id', 'position', 'usage_overall']].copy()
    skill['group'] = skill['position'].map(POSITION_GROUPS)
    skill = skill[skill['group'].isin(SKILL_GROUPS)]
    skill = skill.rename(columns={'usage_overall': 'production'})

    # defensive production: total tackles stand in for time on the field
    d = stats[(stats['category'] == 'defensive') & (stats['statType'] == 'TOT')]
    defence = d[['season', 'team', 'playerId', 'position', 'stat']].copy()
    defence = defence.rename(columns={'playerId': 'id', 'stat': 'production'})
    defence['group'] = defence['position'].map(POSITION_GROUPS)
    defence = defence[defence['group'].isin(DEFENSIVE_GROUPS)]

    # special teams: attempts
    st = stats[((stats['category'] == 'kicking') & (stats['statType'] == 'FGA'))
               | ((stats['category'] == 'punting') & (stats['statType'] == 'NO'))]
    special = st[['season', 'team', 'playerId', 'position', 'stat']].copy()
    special = special.rename(columns={'playerId': 'id', 'stat': 'production'})
    special['group'] = 'ST'

    produced = pd.concat([skill, defence, special], ignore_index=True)
    produced['production'] = pd.to_numeric(produced['production'], errors='coerce')
    produced = produced.dropna(subset=['production', 'team'])
    produced = produced[produced['production'] > 0]
    produced['id'] = produced['id'].astype(str)

    # Offensive linemen have no production metric at all, so they enter with a
    # weight of 1 each and their "returning share" is a headcount.
    ol = roster[roster['group'] == 'OL'][['season', 'team', 'id', 'position', 'group']].copy()
    ol['production'] = 1.0

    players = pd.concat([produced, ol], ignore_index=True)
    players['key'] = players['team'] + '|' + players['id']
    return players, roster


def assign_tiers(players):
    """Rank within team-season-group and split into starter and backup."""
    players = players.sort_values('production', ascending=False)
    players['rank'] = (players.groupby(['season', 'team', 'group'])
                       ['production'].rank(method='first', ascending=False))
    slots = players['group'].map(STARTER_SLOTS).fillna(1)
    players['tier'] = np.where(players['rank'] <= slots, 'starter', 'backup')
    return players


def returning_features(players, roster, season):
    """Features describing what returns FOR `season`, from season-1 production."""
    prior = players[players['season'] == season - 1]
    if prior.empty:
        return pd.DataFrame()

    current_keys = set(roster.loc[roster['season'] == season, 'key'])
    prior = prior.copy()
    prior['returns'] = prior['key'].isin(current_keys)

    rows = []
    for team, td in prior.groupby('team'):
        row = {'team': team, 'season': season}

        total = td['production'].sum()
        row['ret_overall'] = (td.loc[td.returns, 'production'].sum() / total
                              if total else np.nan)

        for group, gd in td.groupby('group'):
            gtot = gd['production'].sum()
            row[f'ret_{group}'] = (gd.loc[gd.returns, 'production'].sum() / gtot
                                   if gtot else np.nan)
            row[f'n_{group}_back'] = int(gd['returns'].sum())
            for tier, tdd in gd.groupby('tier'):
                ttot = tdd['production'].sum()
                row[f'ret_{group}_{tier}'] = (
                    tdd.loc[tdd.returns, 'production'].sum() / ttot
                    if ttot else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def add_ppa_baseline(features):
    """CFBD's own team-level figures, as the baseline any of this must beat."""
    ret = load('returning')
    keep = ['season', 'team', 'percentPPA', 'percentPassingPPA',
            'percentReceivingPPA', 'percentRushingPPA', 'usage',
            'passingUsage', 'receivingUsage', 'rushingUsage']
    ret = ret[[c for c in keep if c in ret.columns]]
    ret = ret.rename(columns={c: f'cfbd_{c}' for c in ret.columns
                              if c not in ('season', 'team')})
    return features.merge(ret, on=['season', 'team'], how='left')


def add_roster_churn(features):
    """Transfers in and out, and departures to the NFL draft."""
    portal = load('portal')
    draft = load('draft')

    out = (portal.groupby(['season', 'origin']).size()
           .rename('portal_out').reset_index()
           .rename(columns={'origin': 'team'}))
    into = (portal.groupby(['season', 'destination']).size()
            .rename('portal_in').reset_index()
            .rename(columns={'destination': 'team'}))
    picks = (draft.groupby(['year', 'collegeTeam']).size()
             .rename('draft_departures').reset_index()
             .rename(columns={'year': 'season', 'collegeTeam': 'team'}))

    for frame in (out, into, picks):
        features = features.merge(frame, on=['season', 'team'], how='left')
    for col in ('portal_out', 'portal_in', 'draft_departures'):
        if col in features.columns:
            features[col] = features[col].fillna(0)
    return features


def attach_team_ids(features):
    """CFBD team ids are ESPN team ids, so the join is exact once names are
    mapped through /teams."""
    teams = load('teams')
    name_to_id = dict(zip(teams['school'], teams['id']))
    features['team_id'] = features['team'].map(name_to_id)
    matched = features['team_id'].notna().mean()
    print(f"  team id match: {matched:.1%} "
          f"({int(features['team_id'].isna().sum())} unmatched rows)")
    if matched < 0.5:
        unmatched = sorted(features.loc[features['team_id'].isna(), 'team'].unique())
        print(f"  unmatched teams: {unmatched[:20]}")
    return features


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'returning_production.csv'))
    args = ap.parse_args()

    print("building player-seasons...")
    players, roster = build_player_seasons()
    players = assign_tiers(players)
    print(f"  {len(players)} player-seasons across "
          f"{players['season'].nunique()} seasons")
    print(f"  by group: "
          f"{players.groupby('group').size().sort_values(ascending=False).to_dict()}")

    seasons = sorted(int(s) for s in players['season'].dropna().unique())
    frames = []
    for season in range(min(seasons) + 1, max(seasons) + 1):
        f = returning_features(players, roster, season)
        if not f.empty:
            frames.append(f)
            print(f"  {season}: {len(f)} teams")
    if not frames:
        raise SystemExit("no returning-production features built")

    features = pd.concat(frames, ignore_index=True)
    features = add_ppa_baseline(features)
    features = add_roster_churn(features)
    features = attach_team_ids(features)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  {len(features)} team-seasons, {len(features.columns)} columns")

    ret_cols = [c for c in features.columns if c.startswith('ret_')]
    print(f"\n  coverage of the {len(ret_cols)} returning columns:")
    for c in sorted(ret_cols):
        print(f"    {c:<24} non-null {features[c].notna().mean():>6.1%}  "
              f"mean {features[c].mean():>6.3f}")


if __name__ == '__main__':
    main()
