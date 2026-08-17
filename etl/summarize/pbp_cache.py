#!/usr/bin/env python3
"""One decode of the play-by-play, shared by every module that reads it.

Six modules scan temp/pbp.csv independently - havoc, the four production
modules, drive_factors and run_pass_ratio - and rb_production reads it twice.
Each declares its own usecols, but pandas still tokenises the whole 1.3GB file
to find those columns, so the eight read sites paid roughly 45 seconds apiece
for the same decode: about six minutes of a twenty-minute rebuild spent parsing
the same text over and over.

This decodes it once into the union of what those modules ask for - 20 of the
75 columns - and writes a pickle beside it. Subsequent reads take 5.4 seconds
instead of 45.

Pickle rather than parquet because neither pyarrow nor fastparquet is installed
and this is a temp-directory cache regenerated whenever the play-by-play
changes, which is the one situation where pickle's portability problems do not
matter. qb_production already caches its attributed plays the same way.

read_pbp() is a drop-in for pd.read_csv on this file: same usecols and chunksize
arguments, same iteration behaviour. It falls back to reading the CSV whenever
the cache is missing, older than the play-by-play, or short of a requested
column, so a module works whether or not the cache was built.

Usage:
    python pbp_cache.py --pbp temp/pbp.csv          # build it
    python pbp_cache.py --pbp temp/pbp.csv --check  # report without building
"""

import argparse
import os

import pandas as pd

# The union of every module's USECOLS. A column missing here does not break
# anything - read_pbp falls back to the CSV for that caller - but it does mean
# that caller pays the full decode, so add to this list rather than working
# around it.
UNION = [
    'away_score', 'distance', 'down', 'drive_id', 'epa', 'game_id',
    'garbage_time_ind', 'home_score', 'id', 'offensive_play',
    'offensive_yards', 'passing_play', 'period', 'play_text',
    'play_type_text', 'rushing_play', 'scoring_play', 'stat_yardage',
    'team_id', 'yards_to_goal',
]

CACHE_NAME = 'pbp_cache.pkl'


def cache_path(pbp_path):
    return os.path.join(os.path.dirname(os.path.abspath(pbp_path)), CACHE_NAME)


def is_fresh(pbp_path):
    """A cache older than the play-by-play is not a cache, it is a trap."""
    cp = cache_path(pbp_path)
    if not os.path.exists(cp) or not os.path.exists(pbp_path):
        return False
    return os.path.getmtime(cp) >= os.path.getmtime(pbp_path)


def build(pbp_path, verbose=True):
    cols = pd.read_csv(pbp_path, nrows=0).columns
    want = [c for c in UNION if c in cols]
    missing = [c for c in UNION if c not in cols]
    if verbose and missing:
        print(f"  not in this play-by-play, skipped: {missing}")
    d = pd.read_csv(pbp_path, usecols=want, low_memory=False)
    cp = cache_path(pbp_path)
    d.to_pickle(cp)
    if verbose:
        print(f"  cached {len(d):,} plays x {len(want)} cols -> {cp} "
              f"({os.path.getsize(cp) / 1e9:.2f} GB)")
    return cp


def read_pbp(pbp_path, usecols=None, chunksize=None, **kwargs):
    """Drop-in for pd.read_csv against the play-by-play.

    kwargs is accepted and ignored so existing call sites can keep passing
    low_memory without a special case.
    """
    d = None
    if is_fresh(pbp_path):
        # read once, then decide - checking the columns by loading it and
        # loading it again afterwards would cost the saving twice over
        d = pd.read_pickle(cache_path(pbp_path))
        if usecols is not None and any(c not in d.columns for c in usecols):
            d = None

    if d is None:
        return pd.read_csv(pbp_path, usecols=usecols, chunksize=chunksize,
                           low_memory=kwargs.get('low_memory', False))

    if usecols is not None:
        d = d[list(usecols)]
    if chunksize is None:
        return d
    return (d.iloc[i:i + chunksize] for i in range(0, len(d), chunksize))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pbp', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'temp', 'pbp.csv'))
    ap.add_argument('--check', action='store_true',
                    help='report cache state without building')
    args = ap.parse_args()

    if not os.path.exists(args.pbp):
        raise SystemExit(f"no play-by-play at {args.pbp}")
    cp = cache_path(args.pbp)
    if args.check:
        print(f"  play-by-play {args.pbp} "
              f"({os.path.getsize(args.pbp) / 1e9:.2f} GB)")
        print(f"  cache        {cp} "
              f"{'fresh' if is_fresh(args.pbp) else 'missing or stale'}")
        return
    build(args.pbp)


if __name__ == '__main__':
    main()
