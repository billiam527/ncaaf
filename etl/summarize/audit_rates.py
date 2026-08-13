#!/usr/bin/env python3
"""Check the summarised rates against a recomputation from the play-by-play.

This exists because a statistic can be wrong in a way that leaves the league
mean almost perfectly intact. _add_requested_stats used to read each rate off
the play-by-play column of the same name, not realising those columns hold
running season-to-date averages. The league mean barely moved - published
pass_yards_per_play averaged 7.03 against a true 6.95 - while the per-game
error ran to 1.47 yards and the correlation with the truth was only 0.80. No
summary statistic of the output would have caught that. Only recomputing the
input does.

Two checks, and they answer different questions:

  --recompute  rebuilds every rate from its numerator and denominator and
               compares team-game by team-game. This is the one that finds
               arithmetic bugs. Requires a pass over the play-by-play.

  --outliers   scans the season summaries for values many sigma from their
               season's mean. This finds individual bad rows - a team whose
               denominator collapsed, a join that half-failed - but it cannot
               see a bug that shifts every row by a similar amount.

Usage:
    python audit_rates.py --recompute --season 2025
    python audit_rates.py --outliers --sigma 3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from summarize_games import RATE_COMPONENTS  # noqa: E402

PBP = os.path.join(_HERE, 'temp', 'pbp.csv')
GBG = os.path.join(_HERE, 'results', 'game_by_game_summaries.csv')
SEASON = os.path.join(_HERE, 'results', 'season_summaries.csv')
CHUNK = 500_000

# The correlation below which a published rate is not tracking the truth.
# Set against the output's own precision rather than at 1.0: the summariser
# rounds on the way out, and for a rate averaging 0.032 that rounding alone
# caps the achievable correlation short of perfect. What this needs to catch
# is the 0.70-0.84 range the running-average bug produced, not rounding.
CORR_FLOOR = 0.99
# Relative bias in the league mean that counts as a real shift. Generous for
# the same reason - the bug this exists to catch barely moved the mean at all,
# so the correlation is the load-bearing test and this is a secondary check.
BIAS_TOL = 0.01


def needed_columns():
    cols = {'game_id', 'team_id', 'offensive_play', 'rushing_play',
            'passing_play'}
    for num, den in RATE_COMPONENTS.values():
        cols.add(num)
        if den != 'plays':
            cols.add(den)
    return sorted(cols)


def recompute(season):
    gbg = pd.read_csv(GBG, low_memory=False)
    gbg['game_id'] = pd.to_numeric(gbg['game_id'], errors='coerce')
    if season:
        gbg = gbg[gbg['season'] == season]
    gids = set(gbg['game_id'].dropna().astype(int))
    print(f"team-games in the summary file: {len(gbg):,}")

    use = needed_columns()
    acc, scanned = [], 0
    for chunk in pd.read_csv(PBP, usecols=use, low_memory=False,
                             chunksize=CHUNK):
        chunk['game_id'] = pd.to_numeric(chunk['game_id'], errors='coerce')
        hit = chunk[chunk['game_id'].isin(gids)]
        scanned += len(chunk)
        if len(hit):
            acc.append(hit)
        print(f"  scanned {scanned:,} plays", end='\r')
    print()
    if not acc:
        raise SystemExit('no matching plays found')

    d = pd.concat(acc, ignore_index=True)
    for c in use:
        if c != 'team_id':
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['team_id'] = d['team_id'].astype(str)
    off = d[d['offensive_play'] == 1]
    print(f"offensive plays: {len(off):,}")

    def one(g):
        out = {}
        for stat, (num, den) in RATE_COMPONENTS.items():
            if den == 'plays':
                denom, numer = len(g), g[num].sum()
            elif den in ('rushing_play', 'passing_play'):
                m = g[den] == 1
                denom, numer = int(m.sum()), g.loc[m, num].sum()
            else:
                denom, numer = g[den].sum(), g[num].sum()
            out[stat] = numer / denom if denom and denom > 0 else 0.0
        return pd.Series(out)

    truth = off.groupby(['game_id', 'team_id']).apply(one).reset_index()
    gbg['team_id'] = gbg['team_id'].astype(str)
    stats = [s for s in RATE_COMPONENTS if s in gbg.columns]
    m = truth.merge(gbg[['game_id', 'team_id'] + stats],
                    on=['game_id', 'team_id'], suffixes=('_true', '_pub'))
    print(f"matched: {len(m):,} team-games\n")

    hdr = (f"{'statistic':<24}{'published':>10}{'correct':>10}{'bias':>9}"
           f"{'mean|err|':>10}{'corr':>8}   verdict")
    print(hdr)
    print('-' * len(hdr))
    bad = 0
    for s in stats:
        a, b = m[f'{s}_true'], m[f'{s}_pub']
        ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 10:
            continue
        err = b - a
        corr = np.corrcoef(a, b)[0, 1]
        scale = max(abs(a.mean()), 1e-9)
        good = corr >= CORR_FLOOR and abs(err.mean()) / scale <= BIAS_TOL
        bad += 0 if good else 1
        print(f"{s:<24}{b.mean():>10.4f}{a.mean():>10.4f}{err.mean():>+9.4f}"
              f"{err.abs().mean():>10.4f}{corr:>8.3f}   "
              f"{'ok' if good else 'MISMATCH'}")
    print(f"\n{bad} of {len(stats)} statistics disagree with the play-by-play")
    return bad


def outliers(sigma):
    s = pd.read_csv(SEASON, low_memory=False)
    cols = [c for c in s.columns if c.startswith('adjusted_')]
    print(f"scanning {len(cols)} adjusted columns over {len(s):,} team-seasons"
          f" at {sigma} sigma\n")
    hits = []
    for c in cols:
        g = s.groupby('season')[c]
        z = (s[c] - g.transform('mean')) / g.transform('std')
        for i in s.index[z.abs() > sigma]:
            hits.append((abs(z[i]), int(s.at[i, 'season']),
                         s.at[i, 'team_id'], c, s.at[i, c], z[i]))
    if not hits:
        print('none')
        return 0
    hits.sort(reverse=True)
    print(f"{'season':>7} {'team_id':>9} {'column':<38}{'value':>10}{'z':>8}")
    print('-' * 74)
    for _, season, tid, c, v, z in hits[:40]:
        print(f"{season:>7} {str(tid):>9} {c:<38}{v:>10.4f}{z:>+8.2f}")
    if len(hits) > 40:
        print(f"... {len(hits) - 40} more")
    print(f"\n{len(hits)} values beyond {sigma} sigma "
          f"({len(hits) / (len(s) * len(cols)) * 100:.2f}% of cells; "
          f"a normal distribution gives {2 * (1 - 0.9987) * 100:.2f}% at 3)")
    return len(hits)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--outliers', action='store_true')
    ap.add_argument('--season', type=int, default=None,
                    help='limit --recompute to one season (much faster)')
    ap.add_argument('--sigma', type=float, default=3.0)
    args = ap.parse_args()
    if not args.recompute and not args.outliers:
        ap.error('choose --recompute, --outliers, or both')

    rc = 0
    if args.recompute:
        rc += recompute(args.season)
        print()
    if args.outliers:
        outliers(args.sigma)
    raise SystemExit(1 if rc else 0)


if __name__ == '__main__':
    main()
