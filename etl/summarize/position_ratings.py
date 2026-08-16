#!/usr/bin/env python3
"""Collect the six unit ratings into one preseason feature table.

One row per team-season, six columns, every one of them knowable BEFORE that
season is played. This is the file the projection model joins.

WHY THESE COLUMNS AND NOT THE OTHERS

Each of the six source modules produces two things: a rating for a season that
has been played, and a projection for the season that has not. Only the second
can be a preseason feature. Using ol_rating for season S to predict games in
season S would be feeding the model the answer.

    pf_qb   qb_projection.projected_z        grade blended with prior record
    pf_rb   rb_projection.backfield          top two backs
    pf_wr   receiver_projection.projected_corps
    pf_ol   ol_projection.proj_ol_rating     S-1 blocking, S roster
    pf_f7   front_seven.proj_f7_rating       S-1 front play, S roster
    pf_db   defensive_backs.proj_db_rating   S-1 secondary play, S roster

LEAKAGE CHECK, RUN ON EVERY COLUMN

A preseason feature for season S must track season S-1 outcomes more closely
than season S outcomes. If it tracks S better, it saw S. All six pass:

    pf_qb  vs pass offence   S-1 +0.481   S +0.413
    pf_rb  vs rush offence   S-1 +0.571   S +0.439
    pf_wr  vs pass offence   S-1 +0.507   S +0.434
    pf_ol  vs rush offence   S-1 +0.771   S +0.496
    pf_f7  vs defence        S-1 +0.800   S +0.574
    pf_db  vs defence        S-1 +0.524   S +0.435

The structural check agrees: all six are populated for 2026, which has not been
played. A column that needed season S could not have a 2026 value at all.
--check re-runs both tests.

WHAT THEY ARE WORTH

Differenced home minus away against actual margin over 6,135 games, 2017-2025,
alongside prior-season adjusted EPA:

    prior-season EPA alone      R2 0.226   MAE 14.36
    position ratings alone      R2 0.255   MAE 14.10
    both                        R2 0.270   MAE 13.96

so +0.044 R2 and 0.395 points of MAE on top of what the model already has. The
value is concentrated rather than spread: dropping the front seven costs 0.0458
R2 and dropping the line 0.0161, while quarterback, receiver and secondary cost
0.0018 apiece. The six correlate 0.47 to 0.61 with one another - they all partly
encode how good a programme is - so they do not add six features' worth.

Usage:
    python position_ratings.py --out results/position_ratings.csv
    python position_ratings.py --check
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, 'results')

# feature name -> (source file, column, what the model should read it as)
SOURCES = {
    'pf_qb': ('qb_projection.csv', 'projected_z', 'quarterback'),
    'pf_rb': ('rb_projection.csv', 'backfield', 'backfield'),
    'pf_wr': ('receiver_projection.csv', 'projected_corps', 'receiving corps'),
    'pf_ol': ('ol_projection.csv', 'proj_ol_rating', 'offensive line'),
    'pf_f7': ('front_seven.csv', 'proj_f7_rating', 'front seven'),
    'pf_db': ('defensive_backs.csv', 'proj_db_rating', 'secondary'),
}

# outcome each feature should predict, for the leakage check
LEAK_TARGET = {
    'pf_qb': 'adjusted_epa_per_pass_off',
    'pf_rb': 'adjusted_epa_per_rush_off',
    'pf_wr': 'adjusted_epa_per_pass_off',
    'pf_ol': 'adjusted_epa_per_rush_off',
    'pf_f7': 'adjusted_epa_per_play_def',
    'pf_db': 'adjusted_epa_per_play_def',
}


def build():
    base = None
    for feat, (fname, col, _) in SOURCES.items():
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        d = pd.read_csv(path, low_memory=False)
        if col not in d.columns:
            raise SystemExit(f"{fname} has no column {col}")
        keep = d[['team_id', 'season', col]].dropna(subset=[col])
        keep = keep.rename(columns={col: feat}).drop_duplicates(
            ['team_id', 'season'])
        base = keep if base is None else base.merge(
            keep, on=['team_id', 'season'], how='outer')
    base['team_id'] = base['team_id'].astype(int)
    base['season'] = base['season'].astype(int)
    return base.sort_values(['season', 'team_id'])


def check(base):
    """Re-run the leakage test rather than trusting the docstring."""
    S = pd.read_csv(os.path.join(RESULTS, 'season_summaries.csv'),
                    low_memory=False)
    print("### leakage check: a preseason feature must track S-1, not S ###")
    print(f"  {'feature':<9}{'vs S-1':>9}{'vs S':>8}{'':>4}verdict")
    bad = 0
    for feat, target in LEAK_TARGET.items():
        if feat not in base.columns or target not in S.columns:
            continue
        cur = S[['team_id', 'season', target]].rename(columns={target: 'cur'})
        prv = S[['team_id', 'season', target]].copy()
        prv['season'] += 1
        prv = prv.rename(columns={target: 'prev'})
        m = (base[['team_id', 'season', feat]]
             .merge(cur, on=['team_id', 'season'], how='inner')
             .merge(prv, on=['team_id', 'season'], how='inner').dropna())
        if len(m) < 100:
            continue
        rp, rc = m[feat].corr(m['prev']), m[feat].corr(m['cur'])
        ok = abs(rp) > abs(rc)
        bad += 0 if ok else 1
        print(f"  {feat:<9}{rp:>+9.3f}{rc:>+8.3f}    "
              f"{'ok' if ok else 'LEAKS - tracks S more closely'}")

    print("\n### structural check: the unplayed season must still populate ###")
    latest = int(base['season'].max())
    for feat in SOURCES:
        n = base.loc[base['season'] == latest, feat].notna().sum()
        flag = '' if n > 0 else '   <- cannot be a preseason feature'
        print(f"  {feat:<9}{latest} populated: {n:>4}{flag}")
        bad += 0 if n > 0 else 1
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='run the leakage and structural checks')
    ap.add_argument('--out', default=os.path.join(
        RESULTS, 'position_ratings.csv'))
    args = ap.parse_args()

    base = build()
    cols = list(SOURCES)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(base):,} team-seasons, "
          f"{int(base.season.min())}-{int(base.season.max())})")
    print(f"\n  {'feature':<9}{'rows':>8}{'mean':>8}{'std':>8}   source")
    for f in cols:
        src, col, _ = SOURCES[f]
        print(f"  {f:<9}{base[f].notna().sum():>8,}{base[f].mean():>8.3f}"
              f"{base[f].std():>8.3f}   {src}:{col}")
    full = base.dropna(subset=cols)
    print(f"\n  rows with all six: {len(full):,} "
          f"({int(full.season.min())}-{int(full.season.max())})")

    if args.check:
        print()
        bad = check(base)
        print(f"\n{'ALL CHECKS PASSED' if bad == 0 else f'{bad} CHECK(S) FAILED'}")
        raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
