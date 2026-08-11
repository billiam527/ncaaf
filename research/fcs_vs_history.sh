#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import numpy as np, pandas as pd
t = pd.read_csv('/home/bill/ncaaf/etl/collect/collect_espn_teams/temp/teams.csv')
fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
p = pd.read_csv('/home/bill/ncaaf/batch_prediction/prediction_file/predictions_with_distribution.csv',
                index_col=0)
p['home_fbs'] = p.home_team_id.isin(fbs); p['away_fbs'] = p.away_team_id.isin(fbs)
mix = p[(p.home_fbs) & (~p.away_fbs)].dropna(subset=['calibrated_margin'])
pure = p[p.home_fbs & p.away_fbs].dropna(subset=['calibrated_margin'])

print("=== what the model predicts vs what these games historically do ===")
print(f"{'':<26}{'n':>5}{'model mean':>13}{'historical':>13}{'gap':>8}")
print(f"{'FBS hosting FCS':<26}{len(mix):>5}{mix.calibrated_margin.mean():>13.1f}"
      f"{24.99:>13.1f}{mix.calibrated_margin.mean()-24.99:>+8.1f}")
print(f"{'FBS v FBS':<26}{len(pure):>5}{pure.calibrated_margin.mean():>13.1f}"
      f"{4.32:>13.1f}{pure.calibrated_margin.mean()-4.32:>+8.1f}")

print("\n=== the FCS games on the 2026 slate ===")
print(f"{'matchup':<24}{'calib':>8}{'P(win)':>9}{'mkt':>8}")
for _, r in mix.sort_values('calibrated_margin').head(14).iterrows():
    mkt = f"{r['market_margin']:+.1f}" if pd.notna(r.get('market_margin')) else '   -'
    print(f"{str(r['short_name'])[:23]:<24}{r['calibrated_margin']:>8.1f}"
          f"{r['p_home_win']:>9.0%}{mkt:>8}")

print(f"\n  model range: {mix.calibrated_margin.min():+.1f} to {mix.calibrated_margin.max():+.1f}")
print(f"  historical FBS-hosting-FCS: home wins 87.8%, mean +24.99")
print(f"  model says home wins on average: {mix.p_home_win.mean():.0%}")

print("\n=== does the pipeline even collect FCS games? ===")
g = pd.read_csv('/home/bill/ncaaf/etl/summarize/temp/games.csv', low_memory=False)
g['home_fbs'] = g.home_team_id.isin(fbs); g['away_fbs'] = g.away_team_id.isin(fbs)
print(f"  FCS v FCS games in 16 seasons: {(~g.home_fbs & ~g.away_fbs).sum()}")
print("  -> effectively no. FCS teams can only be rated from their 1-2 FBS games.")
PY
