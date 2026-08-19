"""Do venue features earn a place in the preseason model?

The in-sample regressions found travel at t=2.66 and one-sided elevation at
t=2.30 against the walk-forward residual. That was across roughly eight
candidate features, so one at t=2.3 is about what chance produces. This is the
decisive version: add them as real features, retrain per season on earlier
seasons only, and score on MAE and against the closing line.

FORMULATED AS DIFFERENCES, WHICH THE IN-SAMPLE PASS DID NOT DO

Everything else the model sees is home minus away, and the venue features should
be too:

    travel_diff  how much further the away team travelled than the home team
    climb_diff   how much further the away team climbed than the home team,
                 one-sided because thin air hurts the side going up and coming
                 down is not an advantage

At a true home site the home team's travel and climb are zero, so these reduce
to the away team's own figures - the quantities tested in-sample. At a NEUTRAL
site both sides travel, and the differences handle it correctly on their own
rather than needing the separate adjustment the pipeline currently applies.
That is the version worth testing, not the one-sided original.
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
VENUES = f'{R}/etl/collect/collect_cfbd_games/cfbd_venues.csv'
EARTH_MI = 3958.8


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_MI * np.arcsin(np.sqrt(a))


def venue_frame():
    """One row per game: how much further the away side travelled and climbed."""
    g = pd.read_csv(E.GAMES, low_memory=False)
    t = pd.read_csv(E.TEAMS)
    v = pd.read_csv(VENUES)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
    g = g.dropna(subset=['venue_id']).copy()
    g['venue_id'] = g['venue_id'].astype(int)

    home = g[g['neutral_site'] != 1]
    own = (home.groupby(['home_team_id', 'season'])['venue_id']
           .agg(lambda s: s.mode().iloc[0]).reset_index()
           .rename(columns={'home_team_id': 'team_id', 'venue_id': 'own_venue'}))

    vv = v.rename(columns={'id': 'venue_id'})[
        ['venue_id', 'latitude', 'longitude', 'elevation']]
    g = g.merge(vv, on='venue_id', how='left')

    for side in ('home', 'away'):
        g = g.merge(own.rename(columns={'team_id': f'{side}_team_id',
                                        'own_venue': f'{side}_own'}),
                    on=[f'{side}_team_id', 'season'], how='left')
        g = g.merge(vv.rename(columns={
            'venue_id': f'{side}_own', 'latitude': f'{side}_lat',
            'longitude': f'{side}_lon', 'elevation': f'{side}_elev'}),
            on=f'{side}_own', how='left')
        g[f'{side}_travel'] = haversine(g[f'{side}_lat'], g[f'{side}_lon'],
                                        g['latitude'], g['longitude'])
        g[f'{side}_climb'] = (g['elevation'] - g[f'{side}_elev']).clip(lower=0)

    g['travel_diff'] = g['away_travel'] - g['home_travel']
    g['climb_diff'] = g['away_climb'] - g['home_climb']
    return g[['id', 'travel_diff', 'climb_diff']]


VARIANTS = {
    'base (28 columns)': [],
    '+ travel': ['travel_diff'],
    '+ climb': ['climb_diff'],
    '+ both': ['travel_diff', 'climb_diff'],
}


def main():
    ss = pd.read_csv(f'{E.RESULTS}/season_summaries.csv')
    games = E.load_games()
    vf = venue_frame().set_index('id')
    print(f"venue features on {len(vf):,} games; "
          f"travel {vf['travel_diff'].notna().mean():.0%}, "
          f"climb {vf['climb_diff'].notna().mean():.0%} populated")

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

    def matrix(season, extra):
        X = E.transform(cache[season][0], 'diff+decay', cols)
        ids = cache[season][2]
        for c in extra:
            X = X.assign(**{c: ids.map(vf[c]).to_numpy()})
        return X

    rows = []
    for S in E.TEST_SEASONS:
        tr = [s for s in cache if s < S]
        if len(tr) < 4:
            continue
        ytr = pd.concat([cache[s][1] for s in tr], ignore_index=True)
        yte, ids = cache[S][1], cache[S][2]
        market = ids.map(mk)

        for label, extra in VARIANTS.items():
            Xtr = pd.concat([matrix(s, extra) for s in tr], ignore_index=True)
            Xte = matrix(S, extra)
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
    res.to_csv(f'{R}/analysis/venue_feature_experiment.csv', index=False)

    print(f"\n  {'variant':<20}{'cols':>6}{'MAE':>9}{'ATS all':>10}{'ATS >=6':>10}")
    for v in VARIANTS:
        g_ = res[res.variant == v]
        n, nb, nbg = g_['n'].sum(), g_['n_bets'].sum(), g_['n_big'].sum()
        print(f"  {v:<20}{g_['k'].iloc[0]:>6}{(g_['mae'] * g_['n']).sum() / n:>9.3f}"
              f"{(g_['ats'] * g_['n_bets']).sum() / nb:>10.1%}"
              f"{(g_['ats_big'] * g_['n_big']).sum() / nbg:>10.1%}")
    print("\n  break-even at -110 is 52.4%")

    piv = res.pivot(index='season', columns='variant', values='mae')
    print(f"\n{piv.round(3).to_string()}")
    base = piv['base (28 columns)']
    print(f"\n  {'variant':<20}{'seasons better than base':>26}")
    for v in VARIANTS:
        if v == 'base (28 columns)':
            continue
        print(f"  {v:<20}{int((piv[v] < base).sum()):>16} of {len(piv)}")


if __name__ == '__main__':
    main()
