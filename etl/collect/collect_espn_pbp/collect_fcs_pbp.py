"""Play-by-play for FCS-vs-FCS games, so FCS teams can carry the same features.

The pbp collector walks game ids off the FBS scoreboard, so it has 98% of
crossover games and 3% of FCS-vs-FCS ones. That is why the 117 FCS teams in
season_summaries are built from one or two games a season while an FBS team
gets twelve - thin in a way the file does not show, because there is no volume
column to reveal it.

All twelve of the model's features are play-derived (rush/pass success,
explosive rates, EPA per rush and pass), so none of them can be reconstructed
from scores. This fetches the same summary?event= endpoint the existing pbp
collector uses and parses with its transform, so the output columns match the
existing 75 and the summarize step needs no special case.

Polite and resumable: one request at a time with a delay, retries with backoff,
each game's JSON cached so a re-run fetches only what is missing. Plain urllib
with no custom User-Agent, which is what this host serves.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import pandas as pd

R = '/home/bill/ncaaf'
C = f'{R}/etl/collect/collect_espn_pbp'
sys.path.insert(0, C)
from json_to_csv import transform_espn_ncaaf_data  # noqa: E402

CACHE = f'{C}/temp/fcspbpjsons'
URL = ('http://site.api.espn.com/apis/site/v2/sports/football/'
       'college-football/summary?event={g}')

ap = argparse.ArgumentParser()
ap.add_argument('--start', type=int, default=2010)
ap.add_argument('--end', type=int, default=2025)
ap.add_argument('--delay', type=float, default=0.35)
ap.add_argument('--limit', type=int, default=0, help='0 = no cap, for trials')
a = ap.parse_args()
os.makedirs(CACHE, exist_ok=True)

# which games: FCS vs FCS only. Crossover games already have pbp.
c = pd.read_csv(f'{R}/etl/collect/collect_cfbd_players/temp/cfbd_teams.csv',
                low_memory=False)
c['id'] = pd.to_numeric(c['id'], errors='coerce')
CLS = dict(zip(c['id'], c['classification'].astype(str).str.lower()))
g = pd.read_csv(f'{R}/etl/collect/collect_espn_games/temp/'
                'games_fcs_2010_to_2025.csv', low_memory=False)
for k in ('home_team_id', 'away_team_id'):
    g[k] = pd.to_numeric(g[k], errors='coerce')
g = g.dropna(subset=['home_team_id', 'away_team_id'])
g['hk'] = g['home_team_id'].map(CLS)
g['ak'] = g['away_team_id'].map(CLS)
ff = g[(g['hk'] == 'fcs') & (g['ak'] == 'fcs')]
ff = ff[(ff['season'] >= a.start) & (ff['season'] <= a.end)]
ids = sorted(pd.to_numeric(ff['id'], errors='coerce').dropna().astype(int))
if a.limit:
    ids = ids[:a.limit]
print(f"  {len(ids):,} FCS-vs-FCS games, {a.start}-{a.end}", flush=True)


def fetch(gid, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(URL.format(g=gid)), timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                return ('ERR', str(e)[:60])
            time.sleep(2 ** i)
    return None


frames, hit, cached, fail, empty = [], 0, 0, 0, 0
t0 = time.time()
for k, gid in enumerate(ids, 1):
    p = f'{CACHE}/pbp_{gid}.json'
    if os.path.exists(p):
        try:
            with open(p) as f:
                js = json.load(f)
            cached += 1
        except Exception:
            os.remove(p)
            js = None
    else:
        js = None
    if js is None:
        js = fetch(gid)
        if isinstance(js, tuple) or js is None:
            fail += 1
            continue
        with open(p, 'w') as f:
            json.dump(js, f)
        hit += 1
        time.sleep(a.delay)
    try:
        df = transform_espn_ncaaf_data(js)
    except Exception:
        df = None
    if df is not None and len(df):
        df = df.copy()
        df['game_id'] = gid
        frames.append(df)
    else:
        empty += 1
    if k % 250 == 0:
        el = time.time() - t0
        rate = k / el if el else 0
        print(f"    {k:>5,}/{len(ids):,}  {hit:,} new {cached:,} cached "
              f"{fail} failed  eta {(len(ids)-k)/rate/60:.0f} min", flush=True)

if not frames:
    print("  nothing parsed")
    raise SystemExit(1)
out = pd.concat(frames, ignore_index=True)
dst = f'{C}/temp/pbp_fcs_{a.start}_to_{a.end}.csv'
out.to_csv(dst, index=False)
print(f"\n  requests {hit:,} new, {cached:,} cached, {fail} failed, "
      f"{empty:,} parsed empty")
print(f"  wrote {dst}: {len(out):,} plays across "
      f"{out['game_id'].nunique():,} games, {len(out.columns)} columns")
