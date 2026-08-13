#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf || exit 1
python - <<'PY'
import pandas as pd, numpy as np, json
T='etl/collect/collect_cfbd_players/temp/'

print("=== what talent measures exist right now ===")
t=pd.read_csv('etl/summarize/results/team_talent.csv',low_memory=False)
print(f"  team_talent.csv: {list(t.columns)}")
print(f"    talent_roll_pct  {t.talent_roll_pct.notna().mean():>6.1%}  "
      f"2014-2027   <- the one that ships")
print(f"    talent_pct       {t.talent_pct.notna().mean():>6.1%}  2015-2025")

print("\n=== how it is built, and what that misses ===")
r=pd.read_csv(T+'cfbd_recruiting.csv',low_memory=False)
print(f"  source: /recruiting/teams = one row per team-year")
print(f"    columns {list(r.columns)}")
print(f"  so talent = weighted sum of CLASS POINTS over 4 years.")
print(f"  points reward quantity as much as quality: a large class of")
print(f"  three-stars can outscore a small class of five-stars.")

print("\n=== is per-recruit data available? ===")
import requests, os
KEY=open(os.path.expanduser('~/.cfbd_api_key')).read().strip()
H={'Authorization':f'Bearer {KEY}'}
p=requests.get('https://api.collegefootballdata.com/recruiting/players',
               headers=H,params={'year':2023},timeout=60).json()
pr=pd.DataFrame(p)
print(f"  /recruiting/players 2023: {len(pr)} recruits")
print(f"    stars   {pr['stars'].value_counts().sort_index().to_dict()}")
print(f"    rating  mean {pd.to_numeric(pr['rating'],errors='coerce').mean():.4f}")
print(f"    positions: {pr['position'].nunique()} distinct")

print("\n=== THE KEY QUESTION: can recruits be joined to who is actually on the roster? ===")
ro=pd.read_csv(T+'cfbd_roster.csv',low_memory=False)
print(f"  roster columns include recruitIds? {'recruitIds' in ro.columns}")
if 'recruitIds' in ro.columns:
    s=ro['recruitIds'].dropna()
    print(f"  non-null recruitIds: {ro['recruitIds'].notna().mean():.1%} of "
          f"{len(ro):,} roster rows")
    print(f"  sample values: {list(s.head(4))}")
    import ast
    def parse(x):
        try:
            v=ast.literal_eval(str(x))
            return v if isinstance(v,list) else []
        except Exception:
            return []
    has=s.map(lambda x: len(parse(x))>0)
    print(f"  parse to a non-empty list: {has.mean():.1%}")

print("\n=== what a roster-based measure would fix ===")
print("  1 attrition   recruits who left are still counted today")
print("  2 transfers   a five-star portal arrival is invisible today")
print("  3 quality vs quantity   points conflate the two")
print("  4 position    a five-star QB counts the same as a five-star OL")
print("  5 blue-chip ratio   the standard measure, not derivable from points")
PY
