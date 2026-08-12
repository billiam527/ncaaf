#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd, numpy as np
f='/home/bill/ncaaf/etl/format/format_espn_pbp/temp/pbp_edit.csv'

print("=== what threshold does the code actually imply? ===")
for d in (20,22,25,28,29,30,35):
    p=1/(1+np.exp(-d/10)); norm=p/(p+(1-p)+0.05)
    print(f"  lead {d:>2}: home_win {norm:.4f}  {'FLAGGED' if norm>0.9 else '-'}")

tot=0; survive=0; by_lead={}
per_period={}
for ch in pd.read_csv(f, usecols=['period','home_score','away_score',
                                  'half_seconds_remaining','offensive_play'],
                      chunksize=500_000, low_memory=False):
    for c in ('period','home_score','away_score','half_seconds_remaining'):
        ch[c]=pd.to_numeric(ch[c],errors='coerce')
    lead=(ch['home_score']-ch['away_score']).abs()
    late=ch['period']>2
    tot+=len(ch)
    s=(late & (lead>28.44))
    survive+=int(s.sum())
    for lo,hi in ((22,29),(29,36),(36,99)):
        m=late&(lead>=lo)&(lead<hi)
        by_lead[(lo,hi)]=by_lead.get((lo,hi),0)+int(m.sum())
    for p in (3,4):
        per_period[p]=per_period.get(p,0)+int(((ch['period']==p)&(lead>28.44)).sum())

print(f"\n=== did the filter actually remove them? ===")
print(f"  total plays in the file:            {tot:,}")
print(f"  period>2 AND lead>28.44 SURVIVING:  {survive:,}")
print(f"  -> {'FILTER IS NOT WORKING' if survive>0 else 'filter did remove them'}")
print(f"  by period: {per_period}")

print(f"\n=== how much second-half blowout play remains in the data? ===")
for (lo,hi),n in sorted(by_lead.items()):
    print(f"  period>2, lead {lo}-{hi if hi<99 else '+'}: {n:>8,} plays ({n/tot:.2%})")

print(f"\n=== is half_seconds_remaining usable for a proper definition? ===")
d=pd.read_csv(f, usecols=['half_seconds_remaining','period'], nrows=400_000,
              low_memory=False)
v=pd.to_numeric(d['half_seconds_remaining'],errors='coerce')
print(f"  non-null {v.notna().mean():.1%}  min {v.min():.0f}  max {v.max():.0f}  "
      f"median {v.median():.0f}")
print(f"  distinct values: {v.nunique()}")
PY
