"""Does training on seasons whose features do not exist hurt?

returning_production and position_ratings both begin in 2017; talent begins in
2014. Before that, preprocess fills every missing column with the column median
- and under the differential encoding, filling BOTH teams with the same median
makes the difference exactly zero. So 15 of the model's 28 columns are not
approximated for those seasons, they are dead.

    season  returning  talent  position
      2012          0       0         0
      2013          0       0         0
      2014          0     128         0
      2015          0     133         0
      2016          0     136         0
      2017        116     138       121

The shipped preseason model trains from 2015, so two of eleven seasons are like
that. The research harness uses TRAIN_START = 2012, so five of the seven
training seasons behind the 2019 fold were - which means every experiment run
through it inherits the question.

The in-season model faced this and moved its window to 2017, finding the choice
worth about 0.02 MAE. This asks the same of the preseason model, on later test
seasons so that a 2017 start still has history behind it.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

os.environ['DIFFERENTIAL_ENCODING'] = '0'
sys.path.insert(0, '/home/bill/ncaaf/research')
sys.path.insert(0, '/home/bill/ncaaf/batch_prediction')
import encoding_experiment as E  # noqa: E402

R = '/home/bill/ncaaf'
TEST_SEASONS = [2022, 2023, 2024, 2025]
STARTS = [2012, 2015, 2017]


def main():
    ss = pd.read_csv(f'{E.RESULTS}/season_summaries.csv')
    games = E.load_games()
    cache = {}
    for s in range(min(STARTS), max(TEST_SEASONS) + 1):
        b = E.build_season(s, ss, games)
        if b is not None:
            cache[s] = b
    cols = list(cache[max(cache)][0].columns)

    # How dead is each season's feature set, measured rather than assumed.
    print("  share of the 28 columns identically zero, by season:")
    for s in sorted(cache):
        X = E.transform(cache[s][0], 'diff+decay', cols)
        print(f"    {s}  {(X.abs().max() < 1e-9).mean():>5.0%}")

    lines = pd.read_csv(E.LINES, low_memory=False)
    lines['spread'] = pd.to_numeric(lines['spread'], errors='coerce')
    lines = lines.dropna(subset=['spread', 'game_id'])
    mk = (-lines.groupby('game_id')['spread'].median()).to_dict()

    rows = []
    for S in TEST_SEASONS:
        yte, ids = cache[S][1], cache[S][2]
        market = ids.map(mk)
        Xte = E.transform(cache[S][0], 'diff+decay', cols)
        for start in STARTS:
            tr = [s for s in cache if start <= s < S]
            Xtr = pd.concat([E.transform(cache[s][0], 'diff+decay', cols)
                             for s in tr], ignore_index=True)
            ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
            sc = StandardScaler().fit(Xtr)
            m = XGBRegressor(**E.PARAMS).fit(sc.transform(Xtr), ytr)
            pred = m.predict(sc.transform(Xte))

            ok = market.notna().to_numpy()
            cov = np.where(yte[ok] > market[ok], 1.0,
                           np.where(yte[ok] < market[ok], 0.0, np.nan))
            edge = pred[ok] - market[ok].to_numpy(float)
            won = np.where(edge > 0, cov, 1 - cov)
            rows.append({'season': S, 'start': start, 'n_train': len(Xtr),
                         'n': len(yte),
                         'mae': float(np.abs(pred - yte).mean()),
                         'n_bets': int(np.isfinite(won).sum()),
                         'ats': float(np.nanmean(won))})
        print(f"  {S} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f'{R}/analysis/train_start.csv', index=False)
    print(f"\n  {'train from':<12}{'train games':>13}{'MAE':>9}{'ATS':>9}")
    for start in STARTS:
        g = res[res.start == start]
        n, nb = g['n'].sum(), g['n_bets'].sum()
        print(f"  {start:<12}{int(g['n_train'].mean()):>13}"
              f"{(g['mae'] * g['n']).sum() / n:>9.3f}"
              f"{(g['ats'] * g['n_bets']).sum() / nb:>9.1%}")
    piv = res.pivot(index='season', columns='start', values='mae')
    print(f"\n{piv.round(3).to_string()}")
    print(f"\n  2017 start better than 2012 in "
          f"{int((piv[2017] < piv[2012]).sum())} of {len(piv)} seasons")


if __name__ == '__main__':
    main()
