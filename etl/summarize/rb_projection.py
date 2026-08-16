#!/usr/bin/env python3
"""Project a team's backfield: the two backs expected to carry it.

A room is the top two by projection, not the whole depth chart. Teams field 2.4
qualifying backs a season and the top two take the overwhelming share of the
carries, so a third body adds noise rather than offence.

WHAT CARRIES FORWARD, AND WHAT DOES NOT

Volume repeats and rates do not, the same shape found at receiver:

    touches            0.461      adj yards per carry   0.315
    rushing yards      0.459      adj EPA per rush      0.313
    carries            0.453      yards per carry       0.265
    carry share        0.438      EPA per rush          0.261
                                  yards per catch       0.140
                                  EPA per catch         0.090

A back's receiving rate is the least repeatable thing measured anywhere in this
repo. Ten catches out of the backfield tell you almost nothing about the next
ten, so the receiving half is carried on volume and given little weight.

The opponent adjustment earns its place here, unlike at some other positions:
it lifts yards per carry from 0.265 to 0.315 and EPA per rush from 0.261 to
0.313, because a back's schedule of run defences varies more than a receiver's
schedule of secondaries.

RECRUITING

Tested rather than assumed, since it works for quarterbacks and not for
receivers. The answer is printed when this runs, and the weight it earns is
fitted rather than chosen.

Usage:
    python rb_projection.py --season 2026 --from-season 2017
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from qb_projection import first_recruit_id, load

_HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTION = os.path.join(_HERE, 'results', 'rb_production.csv')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

ROOM_SIZE = 2
ROOM_POSITIONS = ('RB', 'FB')
# share of a class that ever reaches 40 carries, by star band; measured below
PLAY_RATE = {5: 0.80, 4: 0.55, 3: 0.25, 2: 0.08}


def build_value(prod):
    """EPA above an average back, on the same carries and catches."""
    rush_lg = prod.groupby('season')['adj_epa_per_rush'].transform('mean')
    prod['rush_value'] = prod['carries'] * (prod['adj_epa_per_rush'] - rush_lg)
    rec_lg = prod.groupby('season')['epa_per_catch'].transform('mean')
    prod['rec_value'] = (prod['receptions']
                         * (prod['epa_per_catch'].fillna(rec_lg) - rec_lg))
    prod['value'] = prod['rush_value'] + prod['rec_value']
    prod['z_value'] = prod.groupby('season')['value'].transform(
        lambda s: (s - s.mean()) / s.std())
    return prod


def measure_play_rate(prod, recruits):
    """Share of each star band that ever reaches a qualifying season.

    `prod` already carries the recruit id; re-joining the roster here produced
    rid_x and rid_y and a KeyError.
    """
    played = set(prod['rid'].dropna())
    rc = recruits.copy()
    rc['id'] = rc['id'].astype(str)
    rc['rating'] = pd.to_numeric(rc['rating'], errors='coerce')
    rc['year'] = pd.to_numeric(rc['year'], errors='coerce')
    old = rc[(rc['year'].between(2014, 2021))
             & (rc['position'].isin(ROOM_POSITIONS))].copy()
    if old.empty:
        return PLAY_RATE
    old['ever'] = old['id'].isin(played)

    def band(r):
        if pd.isna(r):
            return None
        return 5 if r >= 0.9833 else 4 if r >= 0.89 else 3 if r >= 0.7969 else 2
    old['band'] = old['rating'].map(band)
    out = {}
    print("  share of each band that ever reaches 40 carries:")
    for b in (5, 4, 3, 2):
        s = old[old['band'] == b]
        if len(s) >= 15:
            out[b] = float(s['ever'].mean())
            print(f"    {b} star  n={len(s):>5}  {s['ever'].mean():>5.0%}")
        else:
            out[b] = PLAY_RATE[b]
            print(f"    {b} star  n={len(s):>5}  too few, using {PLAY_RATE[b]:.0%}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2026)
    ap.add_argument('--from-season', type=int, default=2017)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'rb_projection.csv'))
    args = ap.parse_args()

    prod = pd.read_csv(PRODUCTION, low_memory=False)
    prod['pid'] = prod['pid'].astype(str)
    prod = build_value(prod)

    roster = load('roster')
    roster['pid'] = roster['id'].astype(str)
    recruits = load('recruits')
    recruits['id'] = recruits['id'].astype(str)
    for c in ('rating', 'stars'):
        recruits[c] = pd.to_numeric(recruits[c], errors='coerce')
    classif = load('classification')
    teams = pd.read_csv(TEAMS)
    name_to_id = dict(zip(teams['location'], teams['id']))
    id_to_name = {v: k for k, v in name_to_id.items()}
    prod['team'] = prod['team_id'].map(id_to_name)

    roster['rid'] = roster['recruitIds'].map(first_recruit_id)
    link = roster.dropna(subset=['rid']).drop_duplicates('pid')[['pid', 'rid']]
    prod = prod.merge(link, on='pid', how='left').merge(
        recruits[['id', 'rating']].rename(columns={'id': 'rid'}),
        on='rid', how='left')
    mu, sd = recruits['rating'].mean(), recruits['rating'].std()
    prod['rating_z'] = (prod['rating'] - mu) / sd

    prod = prod.sort_values(['pid', 'season'])
    prod['next'] = prod.groupby('pid')['z_value'].shift(-1)
    prod['n_season'] = prod.groupby('pid')['season'].shift(-1)
    P = prod[prod['n_season'] == prod['season'] + 1].dropna(
        subset=['z_value', 'next', 'carries'])
    P = P.assign(lc=np.log(P['carries']), zc=P['z_value'] * np.log(P['carries']))

    print("does the recruiting grade add anything for a back with a record?")
    tr, te = P[P['season'] <= 2022], P[P['season'] > 2022]
    for lab, cols in (('production only', ['z_value', 'lc', 'zc']),
                      ('recruiting only', ['rating_z']),
                      ('both', ['z_value', 'lc', 'zc', 'rating_z'])):
        a = tr.dropna(subset=cols)
        b = te.dropna(subset=cols)
        if len(a) < 50 or len(b) < 20:
            continue
        m = LinearRegression().fit(a[cols], a['next'])
        r = np.corrcoef(m.predict(b[cols]), b['next'])[0, 1]
        extra = f"   rating weight {m.coef_[-1]:+.3f}" if lab == 'both' else ''
        print(f"  {lab:<20}holdout r = {r:+.3f}{extra}")

    # The recruiting grade earns a place here, unlike at receiver: it lifts the
    # holdout from +0.436 to +0.472 even for backs who already have a record.
    # Backs whose grade is missing fall back to the production-only fit rather
    # than being dropped.
    X = ['z_value', 'lc', 'zc', 'rating_z']
    Xp = ['z_value', 'lc', 'zc']
    Pr = P.dropna(subset=['rating_z'])
    model = LinearRegression().fit(Pr[X], Pr['next'])
    model_prod = LinearRegression().fit(P[Xp], P['next'])
    print(f"\n  with a grade   : next = {model.intercept_:+.3f} "
          f"{model.coef_[0]:+.3f}*value {model.coef_[1]:+.3f}*log(carries) "
          f"{model.coef_[2]:+.3f}*interaction {model.coef_[3]:+.3f}*rating")
    print(f"  without a grade: next = {model_prod.intercept_:+.3f} "
          f"{model_prod.coef_[0]:+.3f}*value {model_prod.coef_[1]:+.3f}"
          f"*log(carries) {model_prod.coef_[2]:+.3f}*interaction")

    print("\nhow often does a recruited back ever play?")
    play_rate = measure_play_rate(prod, recruits)
    first = prod.sort_values('season').drop_duplicates('pid').dropna(
        subset=['rating_z', 'z_value'])
    mf = LinearRegression().fit(first[['rating_z']], first['z_value'])
    print(f"  first qualifying season from the grade: "
          f"z = {mf.intercept_:+.3f} {mf.coef_[0]:+.3f}*rating_z")

    out, players = [], []
    for season in range(args.from_season, args.season + 1):
        fbs = set(classif.loc[(classif['season'] == season)
                              & (classif['fbs'] == 1), 'team'])
        hist = prod[prod['season'] < season]
        if hist.empty:
            continue
        last = (hist.sort_values('season').groupby('pid')
                .agg(prev_team=('team', 'last'), z_value=('z_value', 'last'),
                     carries=('carries', 'last'),
                     prev_yards=('rush_yards', 'last')).reset_index())
        r = roster[(roster['season'] == season)
                   & (roster['position'].isin(ROOM_POSITIONS))].copy()
        r = r[r['team'].isin(fbs)]
        if r.empty:
            continue
        r = r.merge(recruits[['id', 'rating', 'stars']].rename(
            columns={'id': 'rid'}), on='rid', how='left')
        r = r.merge(last, on='pid', how='left')
        r['who'] = r['firstName'].astype(str) + ' ' + r['lastName'].astype(str)
        r['rating_z'] = (r['rating'] - mu) / sd

        has = r['z_value'].notna() & r['carries'].notna()
        r['projected'] = np.nan
        if has.any():
            d = r[has].assign(lc=np.log(r.loc[has, 'carries']))
            d['zc'] = d['z_value'] * d['lc']
            graded = d['rating_z'].notna()
            if graded.any():
                r.loc[d.index[graded], 'projected'] = model.predict(
                    d.loc[graded, X])
            if (~graded).any():
                r.loc[d.index[~graded], 'projected'] = model_prod.predict(
                    d.loc[~graded, Xp])
        # no record: the grade, discounted by the chance he ever carries it
        nr = ~has & r['rating_z'].notna() & r['stars'].notna()
        if nr.any():
            p = r.loc[nr, 'stars'].astype(int).map(play_rate).fillna(0.08)
            r.loc[nr, 'p_play'] = p
            r.loc[nr, 'if_plays'] = (mf.intercept_
                                     + mf.coef_[0] * r.loc[nr, 'rating_z'])
            r.loc[nr, 'projected'] = p * r.loc[nr, 'if_plays']
        r['basis'] = np.where(has, 'record',
                              np.where(nr, 'recruiting', None))
        r = r.dropna(subset=['projected'])
        if r.empty:
            continue
        r['_rk'] = r.groupby('team')['projected'].rank(ascending=False,
                                                       method='first')
        room = r[r['_rk'] <= ROOM_SIZE].copy()
        room['moved'] = (room['prev_team'].notna()
                         & (room['prev_team'] != room['team']))
        agg = room.groupby('team', as_index=False).agg(
            backfield=('projected', 'sum'), best_back=('projected', 'max'),
            n_back=('projected', 'size'), arrivals=('moved', 'sum'))
        by = (room.groupby(['team', 'basis'])['projected'].sum()
              .unstack(fill_value=0.0).reset_index())
        for c, nm in (('record', 'room_record'),
                      ('recruiting', 'room_recruiting')):
            agg[nm] = agg['team'].map(
                dict(zip(by['team'], by[c]))) if c in by.columns else 0.0
        agg['season'] = season
        agg['team_id'] = agg['team'].map(name_to_id)
        out.append(agg)
        players.append(room.assign(season=season))
        print(f"  {season}: {len(agg)} backfields", end='\r')
    print()

    R = pd.concat(out, ignore_index=True)
    for c in ('backfield', 'best_back'):
        R[f'{c}_pct'] = R.groupby('season')[c].rank(pct=True)
    R['backfield_rank'] = (R.groupby('season')['backfield']
                           .rank(ascending=False, method='min').astype('Int64'))
    R = R.sort_values(['season', 'backfield_rank'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    R.to_csv(args.out, index=False)

    PL = pd.concat(players, ignore_index=True)
    keep = ['season', 'team', 'pid', 'firstName', 'lastName', 'position',
            'basis', 'prev_team', 'prev_yards', 'carries', 'z_value',
            'stars', 'rating', 'p_play', 'if_plays', 'projected', 'moved']
    PL = PL[[c for c in keep if c in PL.columns]].sort_values(
        ['season', 'projected'], ascending=[True, False])
    ppath = args.out.replace('.csv', '_players.csv')
    PL.to_csv(ppath, index=False)
    print(f"wrote {args.out}  ({len(R)} team-seasons)")
    print(f"wrote {ppath}  ({len(PL)} backs)")


if __name__ == '__main__':
    main()
