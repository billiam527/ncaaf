"""Do the four returning-starter flags earn their place?

They looked dead in one game - UNC returns its running back and TCU does not,
the largest input gap on the board at -1.000, and the effect was -0.10 points.
One game is not evidence, so this is the same walk-forward harness the encoding
experiment used: seven held-out seasons, production hyperparameters, scored on
MAE and against the closing line.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, '/home/bill/ncaaf/research')
sys.path.insert(0, '/home/bill/ncaaf/batch_prediction')
import encoding_experiment as E  # noqa: E402

STARTERS = ['ret_QB_starter', 'ret_RB_starter', 'ret_WR_starter',
            'ret_TE_starter']

ss = pd.read_csv(f'{E.RESULTS}/season_summaries.csv')
games = E.load_games()
cache = {}
for s in range(E.TRAIN_START, max(E.TEST_SEASONS) + 1):
    b = E.build_season(s, ss, games)
    if b is not None:
        cache[s] = b
cols = list(cache[max(cache)][0].columns)

sample = E.transform(cache[max(cache)][0], 'diff+decay', cols)
drop = [c for c in sample.columns if any(s in c for s in STARTERS)]
print(f"{sample.shape[1]} columns; the starter flags are {drop}\n")
assert len(drop) == 4, 'expected exactly four starter columns'

lines = pd.read_csv(E.LINES, low_memory=False)
lines['spread'] = pd.to_numeric(lines['spread'], errors='coerce')
lines = lines.dropna(subset=['spread', 'game_id'])
mk = (-lines.groupby('game_id')['spread'].median()).to_dict()

rows = []
for S in E.TEST_SEASONS:
    tr = [s for s in cache if s < S]
    if len(tr) < 4:
        continue
    ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
    Xte_raw, yte, ids = cache[S]
    market = ids.map(mk)

    for label, keep in (('with starter flags', None),
                        ('without', drop)):
        Xtr = pd.concat([E.transform(cache[s][0], 'diff+decay', cols)
                         for s in tr], ignore_index=True)
        Xte = E.transform(Xte_raw, 'diff+decay', cols)
        if keep:
            Xtr, Xte = Xtr.drop(columns=keep), Xte.drop(columns=keep)
        sc = StandardScaler().fit(Xtr)
        m = XGBRegressor(**E.PARAMS).fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))

        ok = market.notna().to_numpy()
        cov = np.where(yte[ok] > market[ok], 1.0,
                       np.where(yte[ok] < market[ok], 0.0, np.nan))
        edge = pred[ok] - market[ok].to_numpy(float)
        won = np.where(edge > 0, cov, 1 - cov)
        big = np.abs(edge) >= 6
        rows.append({'season': S, 'variant': label, 'k': Xtr.shape[1],
                     'n': len(yte), 'mae': float(np.abs(pred - yte).mean()),
                     'n_bets': int(np.isfinite(won).sum()),
                     'ats': float(np.nanmean(won)),
                     'n_big': int(np.isfinite(won[big]).sum()),
                     'ats_big': float(np.nanmean(won[big]))})
    print(f"  {S} done", flush=True)

res = pd.DataFrame(rows)
print(f"\n  {'variant':<22}{'cols':>6}{'MAE':>9}{'ATS all':>10}{'ATS >=6':>10}")
for v in ('with starter flags', 'without'):
    g = res[res.variant == v]
    n, nb, nbg = g['n'].sum(), g['n_bets'].sum(), g['n_big'].sum()
    print(f"  {v:<22}{g['k'].iloc[0]:>6}{(g['mae'] * g['n']).sum() / n:>9.3f}"
          f"{(g['ats'] * g['n_bets']).sum() / nb:>10.1%}"
          f"{(g['ats_big'] * g['n_big']).sum() / nbg:>10.1%}")

piv = res.pivot(index='season', columns='variant', values='mae')
print(f"\n  {'season':>7}{'with':>9}{'without':>10}{'diff':>9}")
for s, r in piv.iterrows():
    print(f"  {s:>7}{r['with starter flags']:>9.3f}{r['without']:>10.3f}"
          f"{r['without'] - r['with starter flags']:>+9.3f}")
better = (piv['without'] < piv['with starter flags']).sum()
print(f"\n  dropping them is better in {better} of {len(piv)} seasons")
print("  (negative diff = the model is better WITHOUT them)")
