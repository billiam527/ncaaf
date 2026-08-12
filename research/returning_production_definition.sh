#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/etl/collect/collect_cfbd_players/temp || exit 1
python - <<'PY'
import pandas as pd, numpy as np

roster = pd.read_csv('cfbd_roster.csv', low_memory=False)
usage  = pd.read_csv('cfbd_usage.csv', low_memory=False)
ppa    = pd.read_csv('cfbd_ppa.csv', low_memory=False)
stats  = pd.read_csv('cfbd_stats.csv', low_memory=False)
ret    = pd.read_csv('cfbd_returning.csv', low_memory=False)
for d in (roster, usage, ppa):
    d['id'] = d['id'].astype(str)
stats['playerId'] = stats['playerId'].astype(str)

theirs = ret[ret.season == 2024].set_index('team')['percentPPA']

p23 = ppa[ppa.season == 2023].copy()
p23['tot'] = pd.to_numeric(p23['totalPPA_all'], errors='coerce')
p23 = p23[p23['tot'].notna() & (p23['tot'] != 0)]
p23['key'] = p23['team'] + '|' + p23['id']

def score(keys, label):
    d = p23.copy()
    d['ret'] = d['key'].isin(keys)
    d['w'] = d['tot'].abs()
    mine = d.groupby('team').apply(lambda x: x.loc[x.ret,'w'].sum()/x['w'].sum())
    cmp = pd.concat([mine.rename('mine'), theirs.rename('theirs')], axis=1).dropna()
    print(f"  {label:<44} r={cmp['mine'].corr(cmp['theirs']):+.3f}  "
          f"mean {cmp['mine'].mean():.3f} vs {cmp['theirs'].mean():.3f}  "
          f"|diff| {(cmp['mine']-cmp['theirs']).abs().mean():.3f}")
    return cmp

print("=== TEAM-AWARE membership ===")
r24 = set(roster.loc[roster.season==2024,'team'] + '|' +
          roster.loc[roster.season==2024,'id'])
score(r24, 'on the SAME team roster in 2024')

u24 = set(usage.loc[usage.season==2024,'team'] + '|' +
          usage.loc[usage.season==2024,'id'])
score(u24, 'recorded 2024 usage for the SAME team')

st24 = stats[stats.season==2024]
s24 = set(st24['team'] + '|' + st24['playerId'])
score(u24 | s24, 'any 2024 stat for the SAME team')

print("\n=== signed rather than absolute PPA weight ===")
d = p23.copy(); d['ret'] = d['key'].isin(r24)
mine = d.groupby('team').apply(lambda x: x.loc[x.ret,'tot'].sum()/x['tot'].sum())
cmp = pd.concat([mine.rename('mine'), theirs.rename('theirs')], axis=1).dropna()
print(f"  {'signed totalPPA, same-team roster':<44} r={cmp['mine'].corr(cmp['theirs']):+.3f}  "
      f"mean {cmp['mine'].mean():.3f} vs {cmp['theirs'].mean():.3f}  "
      f"|diff| {(cmp['mine']-cmp['theirs']).abs().mean():.3f}")

print("\n=== Arkansas again ===")
a = usage[(usage.season==2023) & (usage.team=='Arkansas')].copy()
a['key'] = a['team'] + '|' + a['id']
a = a.merge(p23[['id','tot']], on='id', how='left').nlargest(8,'usage_overall')
for _, r in a.iterrows():
    print(f"    {r['name'][:22]:<24} {r['position']:<3} usage {r['usage_overall']:.3f}  "
          f"back at Arkansas: {'YES' if r['key'] in r24 else 'no'}")
print(f"\n  CFBD says Arkansas returning percentPPA 2024 = {theirs.get('Arkansas'):.3f}")
PY
