"""Collect venue metadata from CFBD: where each stadium is and what it is like.

Home-field advantage is currently a single constant the model applies to every
game. Testing whether it should vary needs something to vary it BY, and the
per-team version is already ruled out: research/hfa_power_analysis.sh finds a
true spread across teams of sd 0.00 with a split-half correlation of -0.002, so
each team's own advantage is unmeasurable from 88 games a side. Tier is ruled
out too - within-conference, where home and away schedules actually balance, P4
sits at 5.14 and G5 at 5.33, a difference of 0.19 against a standard error of
0.64.

What is left is features shared across teams, where every game contributes to
one coefficient rather than each team getting its own noisy estimate. Elevation
is one parameter fitted on 11,000 games instead of 134 separate ones.

This pulls the /venues endpoint once - about 300 rows - giving latitude,
longitude, elevation, capacity, dome and surface. Travel distance is then
computed locally from each team's own venue to the game's venue, so no further
requests are needed.

The key comes from $CFBD_API_KEY or ~/.cfbd_api_key, never from this file.

    python scrape_cfbd_venues.py
    python scrape_cfbd_venues.py --out somewhere.csv
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

from scrape_cfbd_data import load_cfbd_key

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, 'cfbd_venues.csv')
URL = 'https://api.collegefootballdata.com/venues'

# Columns worth keeping. CFBD returns a few more that are either free text or
# duplicated elsewhere.
KEEP = ['id', 'name', 'city', 'state', 'zip', 'country_code', 'timezone',
        'latitude', 'longitude', 'elevation', 'capacity', 'construction_year',
        'grass', 'dome']


def fetch(url, key, attempts=4):
    """One request, with backoff. A 429 here is the whole job failing."""
    req = urllib.request.Request(
        url, headers={'Authorization': f'Bearer {key}',
                      'accept': 'application/json'})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < attempts - 1:
                wait = 5 * (i + 1)
                print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    key = load_cfbd_key()
    if not key:
        raise SystemExit(
            "no CFBD key. Set $CFBD_API_KEY or write it to ~/.cfbd_api_key.")

    data = fetch(URL, key)
    if not data:
        raise SystemExit("no venues returned")

    df = pd.DataFrame(data)
    # CFBD has moved between camelCase and snake_case on this endpoint before.
    df.columns = [''.join('_' + c.lower() if c.isupper() else c for c in col).lstrip('_')
                  for col in df.columns]
    have = [c for c in KEEP if c in df.columns]
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        print(f"  not returned by the API: {missing}")
    df = df[have]

    for c in ('latitude', 'longitude', 'elevation', 'capacity',
              'construction_year'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} venues")
    print(f"  {'field':<20}{'non-null':>10}{'min':>12}{'max':>12}")
    for c in ('latitude', 'longitude', 'elevation', 'capacity'):
        if c in df.columns:
            s = df[c].dropna()
            print(f"  {c:<20}{len(s):>10}{s.min():>12.1f}{s.max():>12.1f}"
                  if len(s) else f"  {c:<20}{0:>10}")
    for c in ('dome', 'grass'):
        if c in df.columns:
            print(f"  {c:<20}{df[c].notna().sum():>10}   "
                  f"true on {int(df[c].fillna(False).astype(bool).sum())}")


if __name__ == '__main__':
    main()
