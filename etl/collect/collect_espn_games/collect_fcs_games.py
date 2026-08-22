"""Collect FCS games from ESPN, including FCS-vs-FCS.

The existing collector calls the scoreboard with no group, which returns the
FBS slate - so every game on file has at least one FBS team and there are zero
FCS-vs-FCS results. That leaves ~117 FCS teams rated on the one or two
crossover games they play a year, and four 2026 opponents with no record at
all.

Same endpoint, same parser, one extra parameter: groups=81 is Division I-AA.
Reusing transform_espn_ncaaf_game_data means the output is column-identical to
the FBS files and can go through the same format step.

Polite by construction: one request at a time, a delay between them, retries
with backoff, and a resume that skips any date already on disk so an interrupted
run does not re-fetch what it already has.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import pandas as pd

C = '/home/bill/ncaaf/etl/collect/collect_espn_games'
sys.path.insert(0, C)
from json_to_csv import transform_espn_ncaaf_game_data  # noqa: E402

OUT = f'{C}/temp/fcsjsons'
URL = ('http://site.api.espn.com/apis/site/v2/sports/football/'
       'college-football/scoreboard?dates={d}&groups=81&limit=200')

ap = argparse.ArgumentParser()
ap.add_argument('--start', type=int, default=2010)
ap.add_argument('--end', type=int, default=2025)
ap.add_argument('--delay', type=float, default=0.35)
a = ap.parse_args()

os.makedirs(OUT, exist_ok=True)


def season_dates(y):
    """Mid-August through mid-January covers the regular season and playoffs."""
    d, stop = date(y, 8, 15), date(y + 1, 1, 15)
    while d <= stop:
        yield d
        d += timedelta(days=1)


def fetch(url, tries=3):
    """Plain urllib, exactly as retrieve_game_data.py already does it.

    Setting a custom User-Agent gets a 403 from this host while the default
    is served normally - verified both ways on the same URL. So this matches
    the request the project already makes rather than dressing one up.
    """
    for i in range(tries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.getcode() != 200:
                    raise urllib.error.HTTPError(url, r.getcode(), '', {}, None)
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                print(f"    give up {url.split('dates=')[1][:8]}: {e}")
                return None
            time.sleep(2 ** i)
    return None


frames, stats = [], {'hit': 0, 'cached': 0, 'empty': 0, 'fail': 0}
for y in range(a.start, a.end + 1):
    got = 0
    for d in season_dates(y):
        ds = d.strftime('%Y%m%d')
        p = f'{OUT}/json_fcs_{ds}.json'
        if os.path.exists(p):
            with open(p) as f:
                js = json.load(f)
            stats['cached'] += 1
        else:
            js = fetch(URL.format(d=ds))
            if js is None:
                stats['fail'] += 1
                continue
            with open(p, 'w') as f:
                json.dump(js, f)
            stats['hit'] += 1
            time.sleep(a.delay)
        try:
            df = transform_espn_ncaaf_game_data(js)
        except Exception:
            df = None
        if df is not None and len(df):
            frames.append(df)
            got += len(df)
        else:
            stats['empty'] += 1
    print(f"  {y}: {got:>4} games", flush=True)

if not frames:
    print("no games collected")
    raise SystemExit(1)

g = pd.concat(frames, ignore_index=True)
g = g.drop_duplicates(subset='id')
dst = f'{C}/temp/games_fcs_{a.start}_to_{a.end}.csv'
g.to_csv(dst, index=False)
print(f"\n  requests {stats['hit']:,} new, {stats['cached']:,} cached, "
      f"{stats['fail']} failed, {stats['empty']:,} empty dates")
print(f"  wrote {dst}: {len(g):,} unique games, {len(g.columns)} columns")
