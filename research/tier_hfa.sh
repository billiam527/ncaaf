#!/usr/bin/env bash
# Home advantage by tier, done properly.
#
# The earlier pass classified P4 as {SEC, Big Ten, ACC, Big 12, Pac-12}, which
# dropped Notre Dame and the other independents into G5. It also ran on
# walk-forward residuals, which exist for 4,020 games; the raw home-minus-away
# gap runs on every game with a conference label instead.
#
# A tier's home advantage is its mean margin AT HOME minus its mean margin ON
# THE ROAD. Comparing raw home margins across tiers would just measure that P4
# teams are better.
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf
python - <<'PY'
import numpy as np
import pandas as pd

R = '/home/bill/ncaaf'
g = pd.read_csv(f'{R}/etl/summarize/temp/games.csv', low_memory=False)
t = pd.read_csv(f'{R}/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
g = g.dropna(subset=['home_score_differential'])
g = g[g['neutral_site'] != 1]

s = pd.read_csv(f'{R}/etl/collect/collect_cfbd_games/cfbd_spread_data.csv',
                low_memory=False)
conf = s[['game_id', 'home_conference', 'away_conference']].drop_duplicates('game_id')
g = g.merge(conf, left_on='id', right_on='game_id', how='inner')
g = g.dropna(subset=['home_conference', 'away_conference'])

print("conference labels present:")
labels = sorted(set(g['home_conference']) | set(g['away_conference']))
for c in labels:
    n = (g['home_conference'] == c).sum() + (g['away_conference'] == c).sum()
    print(f"  {c:<26}{n:>6}")

P4 = {'SEC', 'Big Ten', 'ACC', 'Big 12'}
IND = {'FBS Independents', 'Independent', 'Independents'}

# The Pac-12 is kept as its own tier rather than assigned to either side. It was
# a power conference through 2023 and gutted from 2024, so folding it into P4
# mislabels the recent seasons and folding it into G5 mislabels the earlier
# ones. Reported separately, it contaminates neither.
PAC = {'Pac-12', 'Pac-10'}

TIERS = ('P4', 'Independent', 'Pac-12', 'G5')


def tier(c):
    if c in P4:
        return 'P4'
    if c in IND:
        return 'Independent'
    if c in PAC:
        return 'Pac-12'
    return 'G5'


g['ht'] = g['home_conference'].map(tier)
g['awt'] = g['away_conference'].map(tier)
print(f"\n{len(g):,} non-neutral FBS games with labels, "
      f"{int(g.season.min())}-{int(g.season.max())}")

# Each team-game from that team's own point of view.
rows = []
for _, r in g.iterrows():
    rows.append((r['ht'], 'home', r['home_score_differential'], r['awt']))
    rows.append((r['awt'], 'away', -r['home_score_differential'], r['ht']))
c = pd.DataFrame(rows, columns=['tier', 'where', 'perf', 'opp_tier'])

print("\n=== home advantage by tier ===")
print(f"  {'tier':<14}{'n home':>8}{'home':>9}{'n away':>8}{'away':>9}"
      f"{'HFA':>8}{'se':>7}{'t':>7}")
for name in TIERS:
    x = c[c.tier == name]
    hm, aw = x[x['where'] == 'home'], x[x['where'] == 'away']
    if len(hm) < 50:
        continue
    hfa = hm['perf'].mean() - aw['perf'].mean()
    se = np.sqrt(hm['perf'].var(ddof=1) / len(hm) + aw['perf'].var(ddof=1) / len(aw))
    print(f"  {name:<14}{len(hm):>8}{hm['perf'].mean():>9.2f}{len(aw):>8}"
          f"{aw['perf'].mean():>9.2f}{hfa:>8.2f}{se:>7.2f}{hfa / se:>7.2f}")

print("\n=== and against whom? home advantage by (host tier, visitor tier) ===")
print(f"  {'matchup':<26}{'n home':>8}{'HFA':>8}{'se':>7}{'t':>7}")
for ht in TIERS:
    for at in TIERS:
        hm = c[(c.tier == ht) & (c['where'] == 'home') & (c.opp_tier == at)]
        aw = c[(c.tier == ht) & (c['where'] == 'away') & (c.opp_tier == at)]
        if len(hm) < 80 or len(aw) < 80:
            continue
        hfa = hm['perf'].mean() - aw['perf'].mean()
        se = np.sqrt(hm['perf'].var(ddof=1) / len(hm)
                     + aw['perf'].var(ddof=1) / len(aw))
        print(f"  {ht + ' hosting ' + at:<26}{len(hm):>8}{hfa:>8.2f}{se:>7.2f}"
              f"{hfa / se:>7.2f}")

print("\n=== pairwise: is any tier's HFA different from another's? ===")
est = {}
for name in TIERS:
    x = c[c.tier == name]
    hm, aw = x[x['where'] == 'home'], x[x['where'] == 'away']
    if len(hm) < 50:
        continue
    est[name] = (hm['perf'].mean() - aw['perf'].mean(),
                 np.sqrt(hm['perf'].var(ddof=1) / len(hm)
                         + aw['perf'].var(ddof=1) / len(aw)))
names = list(est)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        d = est[a][0] - est[b][0]
        se = np.sqrt(est[a][1] ** 2 + est[b][1] ** 2)
        print(f"  {a} minus {b}: {d:+.2f}  se {se:.2f}  t {d / se:+.2f}")
PY
