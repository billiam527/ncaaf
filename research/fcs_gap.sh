#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd

t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
name = dict(zip(t['id'], t['location']))

print("=== 1. how much of the 2026 slate involves an FCS team? ===")
p = pd.read_csv('/home/bill/ncaaf/batch_prediction/prediction_file/predictions_with_distribution.csv',
                index_col=0)
p['home_fbs'] = p.home_team_id.isin(fbs)
p['away_fbs'] = p.away_team_id.isin(fbs)
p['kind'] = np.where(p.home_fbs & p.away_fbs, 'FBS v FBS',
             np.where(p.home_fbs ^ p.away_fbs, 'FBS v FCS', 'FCS v FCS'))
print(p['kind'].value_counts().to_string())
print(f"  predicted: {p.groupby('kind')['calibrated_margin'].apply(lambda s: s.notna().sum()).to_dict()}")

print("\n=== 2. how much data backs an FCS team's rating? ===")
ss = pd.read_csv('/home/bill/ncaaf/etl/summarize/results/season_summaries.csv')
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
g25 = g[g.season == 2025]
counts = pd.concat([g25.home_team_id, g25.away_team_id]).value_counts()
ss25 = ss[ss.season == 2025].copy()
ss25['is_fbs'] = ss25.team_id.isin(fbs)
ss25['games'] = ss25.team_id.map(counts).fillna(0)
for lab, sub in ss25.groupby('is_fbs'):
    print(f"  {'FBS' if lab else 'FCS':<5} teams rated: {len(sub):>4}   "
          f"median games in data: {sub['games'].median():>4.0f}   "
          f"1-3 games: {(sub['games'] <= 3).sum():>3}")

print("\n=== 3. do we have ANY out-of-sample measurement on FBS-v-FCS? ===")
h = pd.read_csv('/home/bill/ncaaf/analysis/backtest_expanding_preds.csv')
h['home_fbs'] = h.home_team_id.isin(fbs)
h['away_fbs'] = h.away_team_id.isin(fbs)
mixed = h[h.home_fbs ^ h.away_fbs]
print(f"  walk-forward games: {len(h)}")
print(f"  of which FBS v FCS: {len(mixed)}   <- the backtest filters these out")

print("\n=== 4. what do these games actually look like? ===")
gg = g.dropna(subset=['home_score_differential']).copy()
gg['home_fbs'] = gg.home_team_id.isin(fbs)
gg['away_fbs'] = gg.away_team_id.isin(fbs)
gg['kind'] = np.where(gg.home_fbs & gg.away_fbs, 'FBS v FBS',
              np.where(gg.home_fbs ^ gg.away_fbs, 'FBS v FCS', 'FCS v FCS'))
for k, s in gg.groupby('kind'):
    m = s['home_score_differential']
    print(f"  {k:<11} n={len(s):>6}  mean {m.mean():>+7.2f}  sd {m.std():>6.2f}  "
          f"|margin| {m.abs().mean():>5.2f}")

print("\n=== 5. FBS hosting FCS specifically (the common case) ===")
host = gg[(gg.home_fbs) & (~gg.away_fbs)]
print(f"  n={len(host)}  mean home margin {host['home_score_differential'].mean():+.2f}"
      f"  sd {host['home_score_differential'].std():.2f}")
print(f"  home team wins {(host['home_score_differential'] > 0).mean():.1%} of the time")
print(f"  share of all games: {len(host)/len(gg):.1%}")

print("\n=== 6. 2026 slate: which FCS teams are we predicting with thin data? ===")
mix26 = p[p['kind'] == 'FBS v FCS'].copy()
fcs_ids = np.where(mix26.home_fbs, mix26.away_team_id, mix26.home_team_id)
vc = pd.Series(fcs_ids).value_counts().head(8)
for tid, n in vc.items():
    print(f"    {str(name.get(tid, tid))[:24]:<26} appears {n}x  "
          f"2025 games in data: {int(counts.get(tid, 0))}")
PY
