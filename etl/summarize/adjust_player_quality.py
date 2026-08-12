#!/usr/bin/env python3
"""Opponent-adjusted player quality from per-game PPA.

CFBD's season-level player PPA is a raw average: a receiver who faced a soft
schedule looks better than one who faced a hard one, and nothing in the figure
says which. That matters here because the returning-production features split a
roster into the players who were good and the players who were not, and an
unadjusted quality measure puts players in the wrong half.

The adjustment is the same shape as the team-level one in summarize_games.py.
For each season, solve

    ppa(player, game) ~ player_effect + opponent_effect

by ridge over a sparse design matrix, and keep the player coefficients. They are
then standardised within season and position group, because a quarterback's PPA
and a tight end's are not on the same scale.

Two limitations worth knowing. The per-game endpoint carries no play count, so
every game a player appears in weighs the same whether he took three snaps or
sixty - players are filtered by season usage instead, and by a minimum number of
games. And the opponent effect absorbs only defensive strength, not weather,
travel or game state.

Usage:
    python adjust_player_quality.py --out results/player_quality.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

_HERE = os.path.dirname(os.path.abspath(__file__))
_ETL = os.path.dirname(_HERE)
PLAYER_DIR = os.path.join(_ETL, 'collect', 'collect_cfbd_players', 'temp')

ALPHA = 25.0            # heavier than the team fit; each player has ~10 games
MIN_GAMES = 4           # below this a coefficient is mostly prior
MIN_USAGE = 0.02        # season share of team plays, to drop cameo appearances

GROUPS = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE'}


def load(name):
    path = os.path.join(PLAYER_DIR, f'cfbd_{name}.csv')
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run collect_cfbd_players.py "
                         f"--only-player-games first")
    return pd.read_csv(path, low_memory=False)


def adjust_season(pg, alpha=ALPHA):
    """Ridge-solve player and opponent effects for one season."""
    pg = pg.dropna(subset=['ppa_all', 'opponent']).copy()
    if pg.empty:
        return pd.DataFrame()

    pg['pkey'] = pg['team'].astype(str) + '|' + pg['id'].astype(str)

    counts = pg.groupby('pkey')['ppa_all'].size()
    keep = set(counts[counts >= MIN_GAMES].index)
    pg = pg[pg['pkey'].isin(keep)]
    if pg.empty:
        return pd.DataFrame()

    players = pd.Index(sorted(pg['pkey'].unique()))
    opps = pd.Index(sorted(pg['opponent'].astype(str).unique()))
    pi = pg['pkey'].map({p: i for i, p in enumerate(players)}).to_numpy()
    oi = pg['opponent'].astype(str).map(
        {o: i for i, o in enumerate(opps)}).to_numpy()

    n = len(pg)
    rows = np.arange(n)
    X = sparse.hstack([
        sparse.csr_matrix((np.ones(n), (rows, pi)), shape=(n, len(players))),
        sparse.csr_matrix((np.ones(n), (rows, oi)), shape=(n, len(opps))),
    ]).tocsr()
    y = pg['ppa_all'].to_numpy(float)

    model = Ridge(alpha=alpha, fit_intercept=True, solver='sparse_cg')
    model.fit(X, y)

    eff = pd.DataFrame({
        'pkey': players,
        'adj_quality': model.coef_[:len(players)],
    })
    eff['games'] = eff['pkey'].map(counts)
    raw = pg.groupby('pkey')['ppa_all'].mean().rename('raw_quality')
    eff = eff.merge(raw, on='pkey', how='left')
    # strength of schedule faced, in the same units
    opp_eff = pd.Series(model.coef_[len(players):], index=opps)
    pg['_o'] = pg['opponent'].astype(str).map(opp_eff)
    eff = eff.merge(pg.groupby('pkey')['_o'].mean().rename('opp_faced'),
                    on='pkey', how='left')
    return eff


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alpha', type=float, default=ALPHA)
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'player_quality.csv'))
    args = ap.parse_args()

    pg = load('player_games')
    usage = load('usage')
    pg['id'] = pg['id'].astype(str)
    usage['id'] = usage['id'].astype(str)
    pg['ppa_all'] = pd.to_numeric(pg['ppa_all'], errors='coerce')

    use = usage[['season', 'team', 'id', 'usage_overall', 'position']].copy()
    use['usage_overall'] = pd.to_numeric(use['usage_overall'], errors='coerce')
    use = use[use['usage_overall'] >= MIN_USAGE]
    # the usage file carries the odd repeated player-season; left un-deduped it
    # doubles rows through every merge below
    before = len(use)
    use = use.sort_values('usage_overall', ascending=False)
    use = use.drop_duplicates(subset=['season', 'team', 'id'], keep='first')
    if before != len(use):
        print(f"dropped {before - len(use)} duplicate player-seasons from usage")

    pg = pg.merge(use[['season', 'team', 'id']], on=['season', 'team', 'id'],
                  how='inner')
    print(f"player-games after the usage filter: {len(pg):,}")

    out = []
    for season, sd in pg.groupby('season'):
        eff = adjust_season(sd, args.alpha)
        if eff.empty:
            continue
        eff['season'] = int(season)
        out.append(eff)
        print(f"  {int(season)}: {len(eff):>5} players, "
              f"{sd['opponent'].nunique():>3} opponents, "
              f"{len(sd):>6} player-games")
    if not out:
        raise SystemExit("no player quality produced")

    q = pd.concat(out, ignore_index=True)
    q[['team', 'id']] = q['pkey'].str.split('|', n=1, expand=True)
    q = q.merge(use[['season', 'team', 'id', 'position', 'usage_overall']],
                on=['season', 'team', 'id'], how='left')
    q['group'] = q['position'].map(GROUPS)

    # Scale within season and position - a QB's PPA is not a receiver's.
    # Both inputs have long tails: a player who clears the games threshold with
    # a handful of extreme games can take a ridge coefficient several standard
    # deviations out, which would then dominate any median split. Winsorising
    # inside each group before standardising keeps the ordering while pulling
    # those tails back to something a split can use.
    def standardise(x):
        if len(x) < 8 or not x.std() or x.std() == 0:
            return pd.Series(0.0, index=x.index)
        clipped = x.clip(x.quantile(0.02), x.quantile(0.98))
        sd = clipped.std()
        if not sd or sd == 0:
            return pd.Series(0.0, index=x.index)
        return ((clipped - clipped.mean()) / sd).clip(-4, 4)

    for col in ('adj_quality', 'raw_quality'):
        q[f'z_{col.split("_")[0]}'] = (q.groupby(['season', 'group'])[col]
                                       .transform(standardise))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    q.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(q):,} player-seasons)")

    print("\n=== how much did the adjustment move things? ===")
    print(f"  raw vs adjusted quality correlation: "
          f"{q['raw_quality'].corr(q['adj_quality']):+.3f}")
    print(f"  z_raw vs z_adj correlation:          "
          f"{q['z_raw'].corr(q['z_adj']):+.3f}")
    moved = (q['z_adj'] - q['z_raw']).abs()
    print(f"  mean absolute move in z:             {moved.mean():.3f}")
    print(f"  players moving more than half an sd: {(moved > 0.5).mean():.1%}")
    print(f"  schedule faced, sd across players:   {q['opp_faced'].std():.4f}")

    print("\n=== biggest beneficiaries of the adjustment ===")
    q['delta'] = q['z_adj'] - q['z_raw']
    for _, r in q.nlargest(6, 'delta').iterrows():
        print(f"  {str(r['season'])[:4]} {str(r['team'])[:18]:<20}"
              f"{str(r['position']):<4} raw {r['z_raw']:+.2f} -> "
              f"adj {r['z_adj']:+.2f}  (faced {r['opp_faced']:+.3f})")


if __name__ == '__main__':
    main()
