"""Does home-field advantage depend on the venue, rather than the team?

Per-team is already ruled out: hfa_power_analysis finds a true spread across
teams of sd 0.00 and a split-half correlation of -0.002. Tier is ruled out too -
within-conference, where home and away schedules balance, P4 sits at 5.14 and G5
at 5.33, a gap of 0.19 against a standard error of 0.64.

Both of those fail the same way: each team gets its own estimate from 88 games a
side, and that is not enough to see anything smaller than about 3 points. A
venue FEATURE avoids it. Altitude is one coefficient fitted on every game at
once, not 134 separate noisy numbers.

Four candidates, in rough order of prior plausibility:

    elevation gap   how much higher the venue is than the visitor's own home,
                    which is the Air Force / Wyoming / Utah State effect
    travel          great-circle miles from the visitor's home venue
    capacity        crowd size, the folk explanation for home advantage
    dome            indoors removes weather as a leveller

Measured against the walk-forward residual, so team quality is already accounted
for: what is left is what the model does not know. Positive means the home team
beat the model's expectation.
"""
import os
import sys

import numpy as np
import pandas as pd

R = '/home/bill/ncaaf'
GAMES = f'{R}/etl/summarize/temp/games.csv'
TEAMS = f'{R}/etl/collect/collect_espn_teams/temp/teams.csv'
VENUES = f'{R}/etl/collect/collect_cfbd_games/cfbd_venues.csv'
HISTORY = f'{R}/analysis/backtest_expanding_preds.csv'
OUT = f'{R}/analysis/venue_features.csv'

EARTH_MI = 3958.8


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_MI * np.arcsin(np.sqrt(a))


def build():
    g = pd.read_csv(GAMES, low_memory=False)
    t = pd.read_csv(TEAMS)
    v = pd.read_csv(VENUES)
    fbs = set(t.loc[t['fbs_ind'] == 1.0, 'id'])
    g = g[g.home_team_id.isin(fbs) & g.away_team_id.isin(fbs)]
    g = g.dropna(subset=['home_score_differential', 'venue_id'])
    g['venue_id'] = g['venue_id'].astype(int)

    # Each team's own home venue: the one it hosts in most often that season.
    # Taken per season because programmes do move.
    home = g[g['neutral_site'] != 1]
    own = (home.groupby(['home_team_id', 'season'])['venue_id']
           .agg(lambda s: s.mode().iloc[0]).reset_index()
           .rename(columns={'home_team_id': 'team_id', 'venue_id': 'own_venue'}))

    vv = v.rename(columns={'id': 'venue_id'})[
        ['venue_id', 'latitude', 'longitude', 'elevation', 'capacity', 'dome']]
    g = g.merge(vv, on='venue_id', how='left')
    g = g.merge(own.rename(columns={'team_id': 'away_team_id',
                                    'own_venue': 'away_home_venue'}),
                on=['away_team_id', 'season'], how='left')
    g = g.merge(vv.rename(columns={
        'venue_id': 'away_home_venue', 'latitude': 'away_lat',
        'longitude': 'away_lon', 'elevation': 'away_elev',
        'capacity': 'away_cap', 'dome': 'away_dome'}),
        on='away_home_venue', how='left')

    g['travel_mi'] = haversine(g['away_lat'], g['away_lon'],
                               g['latitude'], g['longitude'])
    g['elev_gap'] = g['elevation'] - g['away_elev']
    g['dome'] = g['dome'].fillna(False).astype(bool)
    return g


def main():
    g = build()
    h = pd.read_csv(HISTORY)
    h = h[h.week_num < 90].dropna(subset=['home_score_differential',
                                          'preseason_model_preds'])
    d = g.merge(h[['id', 'preseason_model_preds']], on='id', how='inner')
    d = d[d['neutral_site'] != 1]
    d['resid'] = d['home_score_differential'] - d['preseason_model_preds']
    print(f"{len(d):,} non-neutral games with a walk-forward prediction, "
          f"{int(d.season.min())}-{int(d.season.max())}")
    for c in ('travel_mi', 'elev_gap', 'capacity'):
        print(f"  {c:<12}{d[c].notna().sum():>6} non-null "
              f"({d[c].notna().mean():.0%})")
    print(f"  {'dome':<12}{int(d['dome'].sum()):>6} games indoors")

    print("\n=== each feature on its own, against the residual ===")
    print(f"  {'feature':<26}{'n':>6}{'slope':>12}{'se':>9}{'t':>7}"
          f"{'per unit':>22}")
    tests = [
        ('elevation gap', 'elev_gap', 1000, 'points per 1000 ft higher'),
        ('travel distance', 'travel_mi', 1000, 'points per 1000 miles'),
        ('capacity', 'capacity', 10000, 'points per 10k seats'),
    ]
    for label, col, unit, desc in tests:
        x = d.dropna(subset=[col, 'resid'])
        if len(x) < 200:
            print(f"  {label:<26}{len(x):>6}  too few")
            continue
        X = np.c_[np.ones(len(x)), x[col].to_numpy(float)]
        y = x['resid'].to_numpy(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        s2 = (resid ** 2).sum() / (len(x) - 2)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        print(f"  {label:<26}{len(x):>6}{b[1] * unit:>12.3f}{se * unit:>9.3f}"
              f"{b[1] / se:>7.2f}  {desc:<20}")

    x = d.dropna(subset=['resid'])
    a, b_ = x[x['dome']]['resid'], x[~x['dome']]['resid']
    if len(a) > 50:
        diff = a.mean() - b_.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))
        print(f"  {'dome vs outdoors':<26}{len(a):>6}{diff:>12.3f}{se:>9.3f}"
              f"{diff / se:>7.2f}  {'points, indoors':<20}")

    print("\n=== elevation gap, in bands ===")
    x = d.dropna(subset=['elev_gap', 'resid'])
    bands = [(-99999, -2000), (-2000, -500), (-500, 500), (500, 2000),
             (2000, 99999)]
    print(f"  {'gap (feet)':<20}{'n':>7}{'mean resid':>13}{'se':>8}")
    for lo, hi in bands:
        s = x[(x.elev_gap >= lo) & (x.elev_gap < hi)]
        if len(s) < 40:
            continue
        se = s['resid'].std(ddof=1) / np.sqrt(len(s))
        print(f"  {f'{lo:+,} to {hi:+,}'[:19]:<20}{len(s):>7}"
              f"{s['resid'].mean():>13.2f}{se:>8.2f}")

    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
