"""Does talent_roll_pct earn its place, and is its response curve a problem?

Two observations prompted this. Its effect is most NEGATIVE when the two teams
have identical talent (-1.03 in the -0.02 to +0.02 band) and nearly zero when
the away team is far more talented (-0.06 in the -0.60 to -0.30 band), so the
learned response runs backwards across the whole negative half. And the raw
relationship it is supposed to represent is nothing like that shape - margin
runs -8.96, -5.21, -0.98, +2.53, +5.15, +10.58, +22.80, +32.42 across those same
bands, monotone throughout.

A flat residual is NOT evidence of redundancy - if the model uses a feature
properly the residual should be flat in it - so the only test that settles this
is removing the column and retraining.

Three variants, walk-forward, scored on MAE and against the closing line:

    keep                   as it ships
    drop                   the model loses its only measure of raw talent
    rank instead of pct    the percentile is compressed at the top, where
                           Alabama and the next four teams are separated by
                           hundredths; a rank spreads them out

The third is there because a percentile is a strange scale for this. Talent
points are extremely skewed - a handful of programmes are far above everyone -
and squashing that into 0..1 puts the biggest real gaps in the smallest
numerical space, which is where the response curve is worst behaved.
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
COL = 'talent_roll_pct_diff'


def rank_frame():
    """Within-season rank of talent, 0..1, spreading out the compressed top."""
    tal = pd.read_csv(f'{R}/etl/summarize/results/team_talent.csv')
    tal = tal.dropna(subset=['team_id', 'talent_roll'])
    tal['team_id'] = tal['team_id'].astype(int)
    tal['talent_rank'] = (tal.groupby('season')['talent_roll']
                          .rank(pct=True, ascending=True))
    return tal[['team_id', 'season', 'talent_rank']]


def main():
    ss = pd.read_csv(f'{E.RESULTS}/season_summaries.csv')
    games = E.load_games()
    ranks = rank_frame()

    cache = {}
    for s in range(E.TRAIN_START, max(E.TEST_SEASONS) + 1):
        b = E.build_season(s, ss, games)
        if b is not None:
            cache[s] = b
    cols = list(cache[max(cache)][0].columns)

    gm = games[['id', 'home_team_id', 'away_team_id', 'season']]
    rk = gm.merge(ranks.rename(columns={'team_id': 'home_team_id',
                                        'talent_rank': 'hr'}),
                  on=['home_team_id', 'season'], how='left')
    rk = rk.merge(ranks.rename(columns={'team_id': 'away_team_id',
                                        'talent_rank': 'ar'}),
                  on=['away_team_id', 'season'], how='left')
    rk['rank_diff'] = rk['hr'] - rk['ar']
    rk = rk.set_index('id')['rank_diff']

    lines = pd.read_csv(E.LINES, low_memory=False)
    lines['spread'] = pd.to_numeric(lines['spread'], errors='coerce')
    lines = lines.dropna(subset=['spread', 'game_id'])
    mk = (-lines.groupby('game_id')['spread'].median()).to_dict()

    def matrix(season, mode):
        X = E.transform(cache[season][0], 'diff+decay', cols)
        if mode == 'drop':
            return X.drop(columns=[COL])
        if mode == 'rank':
            ids = cache[season][2]
            return X.drop(columns=[COL]).assign(
                talent_rank_diff=ids.map(rk).to_numpy())
        return X

    rows = []
    for S in E.TEST_SEASONS:
        tr = [s for s in cache if s < S]
        if len(tr) < 4:
            continue
        ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
        yte, ids = cache[S][1], cache[S][2]
        market = ids.map(mk)

        for mode, label in (('keep', 'keep (as it ships)'),
                            ('drop', 'drop talent'),
                            ('rank', 'rank instead of pct')):
            Xtr = pd.concat([matrix(s, mode) for s in tr], ignore_index=True)
            Xte = matrix(S, mode)
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
    res.to_csv(f'{R}/analysis/talent_ablation.csv', index=False)
    order = ['keep (as it ships)', 'drop talent', 'rank instead of pct']
    print(f"\n  {'variant':<22}{'cols':>6}{'MAE':>9}{'ATS all':>10}{'ATS >=6':>10}")
    for v in order:
        g_ = res[res.variant == v]
        n, nb, nbg = g_['n'].sum(), g_['n_bets'].sum(), g_['n_big'].sum()
        print(f"  {v:<22}{g_['k'].iloc[0]:>6}"
              f"{(g_['mae'] * g_['n']).sum() / n:>9.3f}"
              f"{(g_['ats'] * g_['n_bets']).sum() / nb:>10.1%}"
              f"{(g_['ats_big'] * g_['n_big']).sum() / nbg:>10.1%}")
    piv = res.pivot(index='season', columns='variant', values='mae')[order]
    print(f"\n{piv.round(3).to_string()}")
    base = piv['keep (as it ships)']
    for v in order[1:]:
        print(f"\n  {v}: better than keeping it in "
              f"{int((piv[v] < base).sum())} of {len(piv)} seasons")


if __name__ == '__main__':
    main()
