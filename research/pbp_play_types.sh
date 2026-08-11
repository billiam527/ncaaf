#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd, re
p = pd.read_csv('/home/bill/ncaaf/etl/data/pbp/formatted/pbp_edit.csv',
                nrows=60000, low_memory=False)

print("=== sample play_text by type ===")
for label, mask in [('pass', p.passing_play == 1), ('rush', p.rushing_play == 1),
                    ('special teams', p.special_teams_play == 1)]:
    s = p[mask]['play_text'].dropna()
    print(f"\n  --- {label} ({len(s)}) ---")
    for t in s.head(4):
        print(f"    {str(t)[:110]}")

print("\n=== can we extract names? ===")
txt = p['play_text'].dropna().astype(str)
pat = re.compile(r'\b([A-Z][a-z]+(?:\'[A-Za-z]+)?(?:\s+[A-Z][a-z]+)+)\b')
hits = txt.head(20000).apply(lambda s: len(pat.findall(s)))
print(f"  plays with >=1 capitalised name-like token: {(hits > 0).mean():.1%}")
print(f"  mean names per play: {hits.mean():.2f}")

print("\n=== structural markers that make parsing tractable ===")
for k in ['pass complete to', 'pass incomplete', 'run for', 'sacked by',
          'intercepted', 'fumble', 'tackle by', ' to the ']:
    print(f"    {k!r:<22} appears in {txt.str.contains(k, case=False, regex=False).mean():>6.1%}")

print("\n=== the hard part: who else was on the field? ===")
print("  no participation data - only players named in the description appear")
print("  so: ball carriers and tacklers are visible; OL, coverage, and")
print("  everyone off-ball are invisible")
PY
