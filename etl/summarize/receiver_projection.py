#!/usr/bin/env python3
"""Project a team's receiving corps from returning production.

Deliberately NOT a blend of recruiting grade and production, which is what the
quarterback projection does. For receivers the recruiting grade carries almost
nothing:

    correlation with production, by year on campus
                        yr1     yr2     yr3     yr4
    target share      +0.023  +0.005  -0.085  -0.047
    receiving yards   +0.179  +0.103  +0.036  +0.059
    quarterbacks      +0.441  +0.398  +0.299  +0.250

Five-star receivers average a 16.6% target share; two-stars average 16.5%. The
band means are indistinguishable. Fit against a prior season the grade holds a
holdout correlation of +0.045 on its own, prior production holds +0.427, and
adding the grade to production moves nothing - the fitted coefficient on it
goes slightly negative. So it is left out, and this rates on production alone.

Why the two positions differ is not settled here, but the shape of it is: a
quarterback's job is close to fixed across teams, while a receiver's production
is largely a coaching decision about how often to throw at him. Target share is
the most repeatable receiver statistic precisely because it measures that
decision, and no recruiting service is forecasting it.

A team takes the sum of its returning receivers' target share, which answers
"how much of last season's passing game is still here", and its best returning
receiver separately, since one dominant target and three interchangeable ones
are different things.

Usage:
    python receiver_projection.py --season 2026 --from-season 2017
"""

import argparse
import os

import numpy as np
import pandas as pd

from qb_projection import first_recruit_id, load

_HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTION = os.path.join(_HERE, 'results', 'receiver_production.csv')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

MIN_TARGETS = 25


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2026)
    ap.add_argument('--from-season', type=int, default=2017)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'receiver_projection.csv'))
    args = ap.parse_args()

    prod = pd.read_csv(PRODUCTION, low_memory=False)
    prod['rec_id'] = prod['rec_id'].astype(str)
    # standardise within season so seasons are comparable
    for c, z in (('target_share', 'z_share'), ('rec_yards', 'z_yards'),
                 ('adj_yards_per_target', 'z_ypt')):
        prod[z] = prod.groupby('season')[c].transform(
            lambda s: (s - s.mean()) / s.std())

    roster = load('roster')
    roster['pid'] = roster['id'].astype(str)
    classif = load('classification')
    teams = pd.read_csv(TEAMS)
    name_to_id = dict(zip(teams['location'], teams['id']))

    out = []
    for season in range(args.from_season, args.season + 1):
        fbs = set(classif.loc[(classif['season'] == season)
                              & (classif['fbs'] == 1), 'team'])
        r = roster[(roster['season'] == season)
                   & (roster['position'].isin(['WR', 'TE']))].copy()
        r = r[r['team'].isin(fbs)]
        if r.empty:
            continue
        # what each man on this year's roster did BEFORE this season
        hist = prod[prod['season'] < season]
        if hist.empty:
            continue
        career = (hist.sort_values('season')
                  .groupby('rec_id')
                  .agg(prior_share=('z_share', 'last'),
                       prior_yards=('z_yards', 'last'),
                       prior_ypt=('z_ypt', 'last'),
                       prior_target_share=('target_share', 'last'),
                       prior_rec_yards=('rec_yards', 'last'),
                       last_season=('season', 'max')).reset_index())
        r = r.merge(career, left_on='pid', right_on='rec_id', how='left')
        r['returning'] = r['prior_share'].notna()

        g = r.groupby('team', as_index=False).agg(
            returning_receivers=('returning', 'sum'),
            corps_share=('prior_target_share', 'sum'),
            corps_yards=('prior_rec_yards', 'sum'),
            best_share=('prior_share', 'max'),
            best_yards=('prior_yards', 'max'),
            best_ypt=('prior_ypt', 'max'))
        g['season'] = season
        g['team_id'] = g['team'].map(name_to_id)
        out.append(g)
        print(f"  {season}: {len(g)} teams, "
              f"{g['returning_receivers'].mean():.1f} returning receivers each",
              end='\r')
    print()

    R = pd.concat(out, ignore_index=True)
    for c in ('corps_share', 'corps_yards', 'best_share', 'best_yards'):
        R[f'{c}_pct'] = R.groupby('season')[c].rank(pct=True)
    R['corps_rank'] = (R.groupby('season')['corps_share']
                       .rank(ascending=False, method='min').astype('Int64'))
    R = R.sort_values(['season', 'corps_rank'])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    R.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(R)} team-seasons, "
          f"{int(R.season.min())}-{int(R.season.max())})")
    print(R[['returning_receivers', 'corps_share', 'best_share']]
          .describe().round(3).to_string())


if __name__ == '__main__':
    main()
