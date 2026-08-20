"""Which of the eight unit ratings earn their place?

pf_db prompted this: its attribution flips sign in 33% of games, against 2% for
pf_st and 8% for pf_ol, and that survived the move to differences unchanged. But
it is not the weak rating it looked like. Alone it is the third most predictive
of the eight, R2 0.148 behind pf_f7 at 0.187 and pf_ol at 0.172.

What is unusual about it is redundancy. Fitted alone its slope is 10.33 points
per unit; with the other seven present it keeps 1.59, or 15% - the least of any
of them. A team with a good secondary tends to be good everywhere, so most of
what pf_db knows is already in the others, and a feature the model can take or
leave is one whose attribution wanders.

Redundant is not the same as useless, though. Dropping it might cost nothing, or
might cost exactly the 15% it uniquely carries. One-at-a-time removal, walked
forward, is the only way to tell - and doing all eight rather than just pf_db
costs one more loop and answers the general question.
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
UNITS = ['pf_qb', 'pf_rb', 'pf_wr', 'pf_te', 'pf_ol', 'pf_f7', 'pf_db', 'pf_st']


def main():
    ss = pd.read_csv(f'{E.RESULTS}/season_summaries.csv')
    games = E.load_games()
    cache = {}
    for s in range(E.TRAIN_START, max(E.TEST_SEASONS) + 1):
        b = E.build_season(s, ss, games)
        if b is not None:
            cache[s] = b
    cols = list(cache[max(cache)][0].columns)

    lines = pd.read_csv(E.LINES, low_memory=False)
    lines['spread'] = pd.to_numeric(lines['spread'], errors='coerce')
    lines = lines.dropna(subset=['spread', 'game_id'])
    mk = (-lines.groupby('game_id')['spread'].median()).to_dict()

    variants = [('keep all', None)] + [(f'drop {u}', f'{u}_diff') for u in UNITS]

    rows = []
    for S in E.TEST_SEASONS:
        tr = [s for s in cache if s < S]
        if len(tr) < 4:
            continue
        ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
        yte, ids = cache[S][1], cache[S][2]
        market = ids.map(mk)
        base_tr = pd.concat([E.transform(cache[s][0], 'diff+decay', cols)
                             for s in tr], ignore_index=True)
        base_te = E.transform(cache[S][0], 'diff+decay', cols)

        for label, drop in variants:
            Xtr = base_tr if drop is None else base_tr.drop(columns=[drop])
            Xte = base_te if drop is None else base_te.drop(columns=[drop])
            sc = StandardScaler().fit(Xtr)
            m = XGBRegressor(**E.PARAMS).fit(sc.transform(Xtr), ytr)
            pred = m.predict(sc.transform(Xte))

            ok = market.notna().to_numpy()
            cov = np.where(yte[ok] > market[ok], 1.0,
                           np.where(yte[ok] < market[ok], 0.0, np.nan))
            edge = pred[ok] - market[ok].to_numpy(float)
            won = np.where(edge > 0, cov, 1 - cov)
            rows.append({'season': S, 'variant': label, 'n': len(yte),
                         'mae': float(np.abs(pred - yte).mean()),
                         'n_bets': int(np.isfinite(won).sum()),
                         'ats': float(np.nanmean(won))})
        print(f"  {S} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f'{R}/analysis/unit_rating_ablation.csv', index=False)
    piv = res.pivot(index='season', columns='variant', values='mae')
    base = piv['keep all']

    print(f"\n  {'variant':<16}{'MAE':>9}{'vs keep all':>13}{'ATS':>8}"
          f"{'seasons worse':>15}")
    order = ['keep all'] + [f'drop {u}' for u in UNITS]
    out = []
    for v in order:
        g = res[res.variant == v]
        n, nb = g['n'].sum(), g['n_bets'].sum()
        mae = (g['mae'] * g['n']).sum() / n
        ats = (g['ats'] * g['n_bets']).sum() / nb
        worse = int((piv[v] > base).sum()) if v != 'keep all' else 0
        out.append((mae, v, ats, worse))
    keep = next(m for m, v, a, w in out if v == 'keep all')
    for mae, v, ats, worse in out:
        delta = '' if v == 'keep all' else f'{mae - keep:>+13.3f}'
        ws = '' if v == 'keep all' else f'{worse} of {len(piv)}'
        print(f"  {v:<16}{mae:>9.3f}{delta}{ats:>8.1%}{ws:>15}")

    print("\n  A positive 'vs keep all' means removing that rating made the")
    print("  model worse, so the rating was earning its place.")


if __name__ == '__main__':
    main()
