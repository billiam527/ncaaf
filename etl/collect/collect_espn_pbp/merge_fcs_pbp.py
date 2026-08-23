"""Step 3: fold the FCS plays into the per-season raw files.

The formatter resolves inputs by exact filename - play-by-play_YYYY-08-01_to_
YYYY+1-02-01.csv - and symlinks each to temp/pbp_YYYY.csv. A single combined
FCS file is invisible to it, so the plays are split by season and appended to
the season file they belong to. No formatter change, and the FCS plays are then
treated exactly like FBS ones.

Originals are backed up before any append, and the operation is idempotent:
plays already present by play id are not added twice, so a re-run is safe.
"""
import os
import shutil

import pandas as pd

R = '/home/bill/ncaaf'
RAW = f'{R}/etl/collect/collect_espn_pbp/temp'
FCS = f'{RAW}/pbp_fcs_2010_to_2025.csv'

games = pd.read_csv(f'{R}/etl/collect/collect_espn_games/temp/'
                    'games_fcs_2010_to_2025.csv', low_memory=False)
season = dict(zip(pd.to_numeric(games['id'], errors='coerce')
                  .dropna().astype(int), games['season'].astype(int)))

p = pd.read_csv(FCS, low_memory=False)
p['season'] = p['game_id'].map(season)
missing = int(p['season'].isna().sum())
print(f"  {len(p):,} FCS plays, {p['game_id'].nunique():,} games")
print(f"  plays whose game has no season: {missing:,}")
p = p.dropna(subset=['season'])
p['season'] = p['season'].astype(int)

print(f"\n  {'season':<8}{'FBS raw':>12}{'FCS to add':>12}{'already':>10}"
      f"{'result':>12}")
total_added = 0
for s, chunk in p.groupby('season'):
    f = f'{RAW}/play-by-play_{s}-08-01_to_{s + 1}-02-01.csv'
    if not os.path.exists(f):
        print(f"  {s:<8}{'(no FBS file)':>12}{len(chunk):>12}")
        continue
    bak = f + '.prefcs'
    if not os.path.exists(bak):
        shutil.copy2(f, bak)
    base = pd.read_csv(bak, low_memory=False)   # always from the original
    cols = list(base.columns)
    add = chunk[[c for c in cols if c in chunk.columns]].copy()
    for c in cols:
        if c not in add.columns:
            add[c] = pd.NA
    add = add[cols]
    have = set(pd.to_numeric(base['id'], errors='coerce').dropna())
    add = add[~pd.to_numeric(add['id'], errors='coerce').isin(have)]
    out = pd.concat([base, add], ignore_index=True)
    out.to_csv(f, index=False)
    total_added += len(add)
    print(f"  {s:<8}{len(base):>12,}{len(chunk):>12,}"
          f"{len(chunk) - len(add):>10,}{len(out):>12,}")

print(f"\n  added {total_added:,} plays across the season files")
print(f"  originals kept as *.prefcs")

print(f"\n  verifying one season end to end:")
s = 2024
f = f'{RAW}/play-by-play_{s}-08-01_to_{s + 1}-02-01.csv'
d = pd.read_csv(f, low_memory=False)
c = pd.read_csv(f'{R}/etl/collect/collect_cfbd_players/temp/cfbd_teams.csv',
                low_memory=False)
c['id'] = pd.to_numeric(c['id'], errors='coerce')
CLS = dict(zip(c['id'], c['classification'].astype(str).str.lower()))
k = pd.to_numeric(d['team_id'], errors='coerce').map(CLS).value_counts()
print(f"    {s}: {len(d):,} plays, {d['game_id'].nunique():,} games")
print(f"    plays by team class: {k.to_dict()}")
print(f"    columns still {len(d.columns)}: "
      f"{'unchanged' if len(d.columns) == 18 else 'CHANGED - check'}")
