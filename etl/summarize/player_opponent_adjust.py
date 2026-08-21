"""Weight a defender's production by the lines it was recorded against.

The front seven career term was raw counting stats: a sack against an FCS
line weighed what one against Texas weighed, and a Conference USA schedule
looked the same as an SEC one. That put Elijah Hill 3rd in the country on nine
sacks, 6.5 of which came against Merrimack, Jacksonville State and UTEP, and
none against the one Power 4 line he faced.

The team metrics have always been opponent-adjusted; only the player term was
not. This closes that, per event rather than per schedule:

    factor(game) = 1 + K * z(opponent line rating), clipped   for FBS
                 = FCS_FACTOR                                  otherwise
    prs_adj      = sum over games of prs(game) * factor(game)

CFBD carries no defensive box scores before 2016. Those seasons fall back to a
schedule-level factor - the same construction averaged over the opponents a
team faced - rather than being left unadjusted or, worse, treated as zero,
which would read as a man who had not played. With CAREER_DECAY at 0.5 a 2015
season is worth 6% of a 2019 one, so the fallback carries very little.

Writes results/player_opponent_adjust.csv, keyed (season, pid).
"""
import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, 'results')
COLLECT = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')

BOX = os.path.join(COLLECT, 'cfbd_game_stats.csv')
TEAMS = os.path.join(COLLECT, 'cfbd_teams.csv')
RATINGS = os.path.join(RESULTS, 'position_ratings.csv')
PRODUCTION = os.path.join(RESULTS, 'defensive_production.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')
OUT = os.path.join(RESULTS, 'player_opponent_adjust.csv')

# How far an FBS opponent may move a single event. K is set so two standard
# deviations of line quality is worth 30%, which is the same order as the
# team-level adjustment moves a rate.
K = 0.15
CLIP = (0.70, 1.30)
# A non-FBS game is not the bottom of the FBS band, it is a different kind of
# game, so it takes a flat half rather than being squeezed onto that scale.
FCS_FACTOR = 0.50
FIRST_BOX_SEASON = 2016

# The front seven key: sacks, then tackles for loss, then hurries.
W = {'SACKS': 1.0, 'TFL': 0.5, 'QB HUR': 0.25}


def team_name_to_id():
    """CFBD returns opponents by name; everything else here is keyed by id."""
    t = pd.read_csv(TEAMS)
    idc = 'id' if 'id' in t.columns else 'team_id'
    out = {}
    for c in ('school', 'team', 'name', 'alt_name1', 'abbreviation'):
        if c in t.columns:
            for nm, i in zip(t[c], t[idc]):
                if isinstance(nm, str) and nm not in out:
                    out[nm] = int(i)
    return out


def line_quality():
    """Opponent line rating, standardised within season."""
    pr = pd.read_csv(RATINGS)[['season', 'team_id', 'pf_ol']]
    st = pr.groupby('season')['pf_ol'].agg(['mean', 'std'])
    pr = pr.merge(st, left_on='season', right_index=True)
    pr['ol_z'] = (pr['pf_ol'] - pr['mean']) / pr['std']
    return pr[['season', 'team_id', 'ol_z']]


def per_event():
    """Adjusted production from the per-game box scores, 2016 on."""
    box = pd.read_csv(BOX, low_memory=False)
    box = box[(box['category'] == 'defensive') & box['statType'].isin(W)].copy()
    box['stat'] = pd.to_numeric(box['stat'], errors='coerce').fillna(0.0)
    box['prs'] = box['stat'] * box['statType'].map(W)

    box['opp_id'] = box['opponent'].map(team_name_to_id())
    box = box.merge(line_quality().rename(columns={'team_id': 'opp_id'}),
                    on=['season', 'opp_id'], how='left')
    box['is_fbs'] = box['ol_z'].notna()
    box['factor'] = np.where(box['is_fbs'],
                             (1 + K * box['ol_z']).clip(*CLIP),
                             FCS_FACTOR)
    box['prs_adj'] = box['prs'] * box['factor']

    out = box.groupby(['season', 'playerId'], as_index=False).agg(
        prs_raw=('prs', 'sum'), prs_adj=('prs_adj', 'sum'))
    out = out.rename(columns={'playerId': 'pid'})
    out['source'] = 'event'
    return out


def schedule_factor():
    """Team-season factor, for the seasons with no box score."""
    g = pd.read_csv(GAMES, low_memory=False)
    a = g[['season', 'home_team_id', 'away_team_id']].rename(
        columns={'home_team_id': 'team_id', 'away_team_id': 'opp_id'})
    b = g[['season', 'away_team_id', 'home_team_id']].rename(
        columns={'away_team_id': 'team_id', 'home_team_id': 'opp_id'})
    s = pd.concat([a, b], ignore_index=True).dropna().astype(int)

    s = s.merge(line_quality().rename(columns={'team_id': 'opp_id'}),
                on=['season', 'opp_id'], how='left')
    s['is_fbs'] = s['ol_z'].notna()
    s['factor'] = np.where(s['is_fbs'],
                           (1 + K * s['ol_z']).clip(*CLIP), FCS_FACTOR)
    return (s.groupby(['season', 'team_id'], as_index=False)['factor'].mean())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    ev = per_event()
    covered = set(ev['season'].unique())
    print(f"  per-event: {len(ev):,} player-seasons, "
          f"{min(covered)}-{max(covered)}")

    d = pd.read_csv(PRODUCTION, low_memory=False)
    d = d[d['group'] == 'FRONT'].copy()
    d['prs'] = (d['sack_best'].fillna(0) + 0.5 * d['tfl_box'].fillna(0)
                + 0.25 * d['hurry_best'].fillna(0))
    d = d.merge(schedule_factor(), on=['season', 'team_id'], how='left')
    d['factor'] = d['factor'].fillna(1.0)
    d['prs_sched'] = d['prs'] * d['factor']
    sched = d.groupby(['season', 'pid'], as_index=False).agg(
        prs_raw=('prs', 'sum'), prs_adj=('prs_sched', 'sum'))
    sched['source'] = 'schedule'

    ev['pid'] = ev['pid'].astype(str)
    sched['pid'] = sched['pid'].astype(str)

    # per-event wins wherever it exists; the schedule figure fills the rest
    keys = set(zip(ev['season'], ev['pid']))
    seen = np.array([(s, p) in keys for s, p in
                     zip(sched['season'], sched['pid'])])
    fill = sched[~seen]
    out = pd.concat([ev, fill], ignore_index=True)
    out = out.sort_values(['season', 'pid'])

    n_ev = (out['source'] == 'event').sum()
    print(f"  {len(out):,} player-seasons written: {n_ev:,} per-event, "
          f"{len(out) - n_ev:,} schedule-level")
    pre = (out['season'] < FIRST_BOX_SEASON).sum()
    print(f"  {pre:,} of them are before {FIRST_BOX_SEASON}, "
          f"where no box score exists")

    assert out['prs_adj'].notna().all(), 'a player-season lost its production'
    assert not out.duplicated(['season', 'pid']).any(), 'duplicate key'

    out.to_csv(args.out, index=False)
    print(f"  wrote {args.out}")

    big = out[out['prs_raw'] >= 5].copy()
    big['ratio'] = big['prs_adj'] / big['prs_raw']
    print(f"  among {len(big):,} player-seasons with 5+ raw production, "
          f"the factor runs {big['ratio'].min():.3f}-{big['ratio'].max():.3f} "
          f"(mean {big['ratio'].mean():.3f})")


if __name__ == '__main__':
    main()
