#!/usr/bin/env python3
"""Opponent-adjust the havoc rates, weighting each game by its denominator.

The ridge is the same one every other statistic goes through - each team-game
rate is split into a defense effect, an offense effect and an intercept, so

    observed = league mean + defense effect + offense effect + residual
    adjusted = league mean + defense effect

with the defense effect read back as the team's rating. What is different here
is the weighting, and it matters more for havoc than for anything else.

Efficiency statistics are measured over 60-80 snaps a game, so every game
carries roughly equal evidence and an unweighted fit is close to right. Havoc
rates are not: a defense can face 55 dropbacks one week and 13 the next, and
sacks are rare enough that a small denominator produces wild rates. Old
Dominion 2025 is the worked example - 7 sacks on 20 dropbacks against Troy
(35.0%) and 1 on 55 against App State (1.8%). Pooling their season gives
28/349 = 8.0%, but averaging the thirteen game rates with equal weight gives
10.1%, because their best rates happened to come in their lowest-volume games.
An unweighted fit reads that 10.1% as the team's true level and ranks them
among the best pass rushes in the country.

Weighting each game by the count its rate was measured over removes that: the
55-dropback game carries nearly three times the 20-dropback one, which is the
right relative evidence, and it shrinks small-denominator noise without any
explicit shrinkage term.

Each rate is weighted by its own denominator, since they differ within a single
game - dropbacks for sacks and coverage, rushing plays for tackles for loss,
third downs for third-down stops.

Usage:
    python adjust_havoc.py --out results/havoc_adjusted.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
HAVOC = os.path.join(_HERE, 'results', 'havoc.csv')
GAMES = os.path.join(_HERE, 'temp', 'games.csv')
CLASSIFICATION = os.path.join(
    _HERE, '..', 'collect', 'collect_cfbd_players', 'temp',
    'cfbd_classification.csv')

ALPHA = 1.0
MIN_TEAM_GAMES = 200

# each rate over the count it was actually measured over
WEIGHTS = {
    'tfl_rate': 'rush_plays',
    'sack_rate': 'pass_plays',
    'pass_defensed_rate': 'pass_plays',
    'interception_rate': 'pass_plays',
    'fumble_rate': 'def_plays',
    'third_down_stop_rate': 'third_downs',
    'fourth_down_stop_rate': 'fourth_downs',
    'redzone_stop_rate': 'rz_drives',
}


def load():
    h = pd.read_csv(HAVOC, low_memory=False)
    g = pd.read_csv(GAMES, low_memory=False)[
        ['id', 'season', 'home_team_id', 'away_team_id']].dropna()
    for c in g.columns:
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g = g.dropna().rename(columns={'id': 'game_id'}).drop_duplicates('game_id')
    h = h.merge(g[['game_id', 'home_team_id', 'away_team_id']],
                on='game_id', how='left')
    h['opponent_id'] = np.where(h['team_id'] == h['home_team_id'],
                                h['away_team_id'], h['home_team_id'])
    return h.dropna(subset=['opponent_id'])


def fbs_by_season(seasons):
    """Per-season FBS membership - schools move up, so this cannot be static."""
    c = pd.read_csv(CLASSIFICATION, low_memory=False)
    c = c[c['fbs'] == 1]
    return {s: set(c.loc[c['season'] == s, 'team_id']) for s in seasons}


def adjust_one(df, stat, weight_col, alpha=ALPHA, weighted=True):
    """Weighted ridge for a single stat in a single season."""
    d = df.dropna(subset=[stat, weight_col])
    d = d[d[weight_col] > 0]
    if len(d) < MIN_TEAM_GAMES or d[stat].std() < 1e-6:
        return None

    teams = pd.Index(sorted(set(d['team_id']) | set(d['opponent_id'])))
    idx = {t: i for i, t in enumerate(teams)}
    k, n = len(teams), len(d)
    rows = np.arange(n)
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (rows, d['team_id'].map(idx))),
                          shape=(n, k)),
        sparse.csr_matrix((np.ones(n), (rows, d['opponent_id'].map(idx))),
                          shape=(n, k)),
    ]).tocsr()

    w = d[weight_col].to_numpy(float) if weighted else None
    m = Ridge(alpha=alpha, fit_intercept=True).fit(
        X, d[stat].to_numpy(float), sample_weight=w)

    # pooled rate, weighted the same way the ridge is - numerator over
    # denominator across the season, not an average of the game rates
    num = (d[stat] * d[weight_col]).groupby(d['team_id']).sum()
    den = d.groupby('team_id')[weight_col].sum()
    return pd.DataFrame({
        'team_id': teams,
        f'adj_{stat}': m.intercept_ + m.coef_[:k],
        f'raw_{stat}': (num / den).reindex(teams).to_numpy(),
        f'opp_{stat}': pd.Series(m.coef_[k:], index=teams).to_numpy(),
    })


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--unweighted', action='store_true',
                    help='fit every game equally, ignoring its denominator - '
                         'kept only so the weighting can be measured against '
                         'itself in the walk-forward')
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'havoc_adjusted.csv'))
    args = ap.parse_args()

    h = load()
    fbs = fbs_by_season(sorted(h['season'].dropna().unique()))

    out = []
    for season, sd in h.groupby('season'):
        members = fbs.get(season, set())
        # the ridge needs both sides of every game present as teams in their
        # own right, so FCS opponents come out rather than being carried as
        # opponent-only columns the solve cannot identify
        sd = sd[sd['team_id'].isin(members) & sd['opponent_id'].isin(members)]
        if len(sd) < MIN_TEAM_GAMES:
            continue

        merged = None
        for stat, wcol in WEIGHTS.items():
            if stat not in sd.columns or wcol not in sd.columns:
                continue
            a = adjust_one(sd, stat, wcol, alpha=args.alpha,
                           weighted=not args.unweighted)
            if a is None:
                continue
            merged = a if merged is None else merged.merge(
                a, on='team_id', how='outer')
        if merged is None:
            continue
        merged.insert(1, 'season', int(season))
        merged['games'] = merged['team_id'].map(
            sd.groupby('team_id').size())
        out.append(merged)
        print(f"  {int(season)}: {len(merged)} teams", end='\r')
    print()

    if not out:
        raise SystemExit('no seasons adjusted')
    A = pd.concat(out, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    A.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(A)} team-seasons, "
          f"{int(A.season.min())}-{int(A.season.max())})")
    cols = [c for c in A.columns if c.startswith('adj_')]
    print(A[cols].describe().round(4).to_string())


if __name__ == '__main__':
    main()
