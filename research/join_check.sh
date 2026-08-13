#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/etl/collect/collect_cfbd_players/temp || exit 1
python - <<'PY'
import pandas as pd, numpy as np, ast
ro=pd.read_csv('cfbd_roster.csv',low_memory=False)
rc=pd.read_csv('cfbd_recruits.csv',low_memory=False)
print(f"roster {len(ro):,}   recruits {len(rc):,}")
print(f"recruit columns: {list(rc.columns)}")

def first_id(x):
    try:
        v=ast.literal_eval(str(x))
        if isinstance(v,list) and v: return str(v[0])
    except Exception: pass
    return None
ro['rid']=ro['recruitIds'].map(first_id)
print(f"\n=== does roster.recruitIds resolve to recruits.id? ===")
have=ro['rid'].notna()
print(f"  roster rows with a recruit id: {have.mean():.1%}")
rc['id']=rc['id'].astype(str)
known=set(rc['id'])
hit=ro.loc[have,'rid'].isin(known)
print(f"  of those, found in cfbd_recruits: {hit.mean():.1%}")
print(f"  net linkable roster rows: {(have.mean()*hit.mean()):.1%}")

j=ro[have].copy(); j['rid']=j['rid'].astype(str)
j=j.merge(rc[['id','stars','rating','position','year','committedTo']],
          left_on='rid',right_on='id',how='inner',suffixes=('','_rec'))
print(f"\n=== the joined frame: {len(j):,} player-seasons ===")
print(f"  stars available: {j['stars'].notna().mean():.1%}")
print(f"  star distribution: {j['stars'].value_counts().sort_index().to_dict()}")
print(f"  rating mean {pd.to_numeric(j['rating'],errors='coerce').mean():.4f}")

print(f"\n=== sanity: do blue-chip shares look right by team? ===")
j['blue']=pd.to_numeric(j['stars'],errors='coerce')>=4
cur=j[j.season==2025]
g=cur.groupby('team').agg(n=('blue','size'),blue=('blue','sum'))
g=g[g.n>=25]; g['ratio']=g.blue/g.n
print(f"  {len(g)} teams with 25+ linked players in 2025")
print(f"  blue-chip ratio: mean {g.ratio.mean():.1%}  max {g.ratio.max():.1%}")
print("  top 8:")
for k,r in g.nlargest(8,'ratio').iterrows():
    print(f"    {str(k)[:22]:<24}{r['ratio']:>6.1%}   ({int(r['blue'])}/{int(r['n'])})")
print("  bottom 4:")
for k,r in g.nsmallest(4,'ratio').iterrows():
    print(f"    {str(k)[:22]:<24}{r['ratio']:>6.1%}   ({int(r['blue'])}/{int(r['n'])})")

print(f"\n=== does the roster link cover 2026? ===")
for y in (2023,2024,2025,2026):
    s=ro[ro.season==y]
    n=s['rid'].notna().sum()
    print(f"  {y}: {len(s):>6} roster rows, {n:>6} with a recruit id "
          f"({n/max(len(s),1):.0%})")
PY
