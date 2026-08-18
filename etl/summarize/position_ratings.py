#!/usr/bin/env python3
"""Collect the eight unit ratings into one preseason feature table.

One row per team-season, eight columns, every one of them knowable BEFORE that
season is played. This is the file the projection model joins.

WHY THESE COLUMNS AND NOT THE OTHERS

Each source module produces two things: a rating for a season that has been
played, and a projection for the season that has not. Only the second can be a
preseason feature. Using ol_rating for season S to predict games in season S
would be feeding the model the answer.

    pf_qb   qb_projection.projected_z        grade blended with prior record
    pf_rb   rb_projection.backfield          top two backs
    pf_wr   receiver_projection.projected_wr wideouts alone
    pf_te   receiver_projection.projected_te tight ends alone
    pf_ol   ol_projection.proj_ol_rating     S-1 blocking, S roster
    pf_f7   front_seven.proj_f7_rating       S-1 front play, S roster
    pf_db   defensive_backs.proj_db_rating   S-1 secondary play, S roster
    pf_st   special_teams.proj_st_epaa       S-1 kicking, in points above
                                             the average kick from that spot

LEAKAGE CHECK, RUN ON EVERY COLUMN

A preseason feature for season S must track season S-1 outcomes more closely
than season S outcomes. If it tracks S better, it saw S. All eight pass:

    pf_qb  vs pass offence   S-1 +0.495   S +0.420
    pf_rb  vs rush offence   S-1 +0.571   S +0.436
    pf_wr  vs pass offence   S-1 +0.499   S +0.417
    pf_te  vs pass offence   S-1 +0.283   S +0.250
    pf_ol  vs rush offence   S-1 +0.642   S +0.468
    pf_f7  vs defence        S-1 +0.797   S +0.584
    pf_db  vs defence        S-1 +0.554   S +0.456
    pf_st  vs its own figure S-1 +0.986   S +0.248

Special teams is checked against its own realised figure rather than a season
summary column, because the season summariser never modelled kicking. Its S-1
correlation is near one by construction - the feature IS last season's figure,
standardized - and the S figure is simply how much special teams repeats.

The structural check agrees: all eight are populated for 2026, which has not
been played. A column that needed season S could not have a 2026 value at all.
--check re-runs both tests.

WHAT THEY ARE WORTH

Differenced home minus away against actual margin over 6,136 games, 2017-2025,
alongside prior-season adjusted EPA. In sample, as this table used to be built,
and then fitted on 2017-22 and scored on 2023-25, which is the figure to trust:

                                in sample        out of sample
    prior-season EPA alone   R2 0.226  MAE 14.37   0.194  14.38
    position ratings alone      0.257       14.09   0.260  13.84
    both                        0.274       13.94   0.256  13.87

Two things in that table are worth staring at. The position ratings beat prior
EPA outright out of sample, and adding prior EPA to them makes the model
slightly WORSE - it is fitted on six seasons and does not travel. That is a
model-level question and is left alone here.

What each column is worth, dropped from the full model:

                     in sample   out of sample
    pf_f7              +0.0054       +0.0088
    pf_ol              +0.0062       +0.0080
    pf_rb              +0.0023       +0.0040
    pf_st              +0.0016       +0.0023
    pf_qb              +0.0011       +0.0014
    pf_wr              +0.0004       +0.0007
    pf_te              +0.0006       -0.0010
    pf_db              +0.0008       -0.0011

The value is concentrated in the two lines of scrimmage. Special teams is
fourth, ahead of the quarterback, which is not where anyone would have guessed
it. Tight end and secondary now cost NOTHING out of sample - dropping either
slightly helps - which is a flag rather than an instruction; they are cheap to
keep and the sign is well inside noise.

The eight correlate 0.05 to 0.60 with one another. They all partly encode how
good a programme is, so they do not add eight features' worth.

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
# pf_wr is the wideouts alone, not the whole receiving corps. Tight end is split
# out as pf_te because the model needs to be able to lose one without losing the
# other - returning production carries a ret_TE_starter figure that nothing here
# could reproduce while the two sat inside one number. Both come off the same
# trimmed room, top four wideouts and top two tight ends, and both are already
# standardized within position, so a tight end is scored against tight ends.
SOURCES = {
    'pf_qb': ('qb_projection.csv', 'projected_z', 'quarterback'),
    'pf_rb': ('rb_projection.csv', 'backfield', 'backfield'),
    'pf_wr': ('receiver_projection.csv', 'projected_wr', 'wide receivers'),
    'pf_te': ('receiver_projection.csv', 'projected_te', 'tight ends'),
    'pf_ol': ('ol_projection.csv', 'proj_ol_rating', 'offensive line'),
    'pf_f7': ('front_seven.csv', 'proj_f7_rating', 'front seven'),
    'pf_db': ('defensive_backs.csv', 'proj_db_rating', 'secondary'),
    'pf_st': ('special_teams.csv', 'proj_st_epaa', 'special teams'),
}

# outcome each feature should predict, for the leakage check
LEAK_TARGET = {
    'pf_qb': 'adjusted_epa_per_pass_off',
    'pf_rb': 'adjusted_epa_per_rush_off',
    'pf_wr': 'adjusted_epa_per_pass_off',
    'pf_te': 'adjusted_epa_per_pass_off',
    'pf_ol': 'adjusted_epa_per_rush_off',
    'pf_f7': 'adjusted_epa_per_play_def',
    'pf_db': 'adjusted_epa_per_play_def',
}

# Special teams has no outcome in season_summaries to be checked against - the
# season summariser never modelled kicking - so its leakage target is its own
# realised figure, taken from the module that produces it.
LEAK_ALT = {'pf_st': ('special_teams.csv', 'st_epaa')}


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
    targets = {f: (S, t) for f, t in LEAK_TARGET.items()}
    for feat, (fname, col) in LEAK_ALT.items():
        path = os.path.join(RESULTS, fname)
        if os.path.exists(path):
            targets[feat] = (pd.read_csv(path, low_memory=False), col)
    for feat, (src, target) in targets.items():
        if feat not in base.columns or target not in src.columns:
            continue
        cur = src[['team_id', 'season', target]].rename(
            columns={target: 'cur'})
        prv = src[['team_id', 'season', target]].copy()
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
    print(f"\n  rows with all {len(cols)}: {len(full):,} "
          f"({int(full.season.min())}-{int(full.season.max())})")

    if args.check:
        print()
        bad = check(base)
        print(f"\n{'ALL CHECKS PASSED' if bad == 0 else f'{bad} CHECK(S) FAILED'}")
        raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
