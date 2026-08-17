#!/usr/bin/env python3
"""Project quarterback quality by blending recruiting grade with production.

A recruiting grade is the best information available about a quarterback who
has never played, and steadily worse information about one who has. Measured
against opponent-adjusted production, the correlation decays with every year on
campus:

    year 1  +0.441      year 3  +0.299
    year 2  +0.398      year 4  +0.250

and a quarterback's own freshman season predicts his sophomore year better
(+0.537) than his recruiting grade does (+0.398). So the sensible projection
leans on the grade until there is a record, then leans on the record.

Fitting that on seasons through 2022 and testing on 2023-2025 gives

    first qualifying season   z = -0.085 + 0.380 * rating          r = +0.321
    with a record            z = +0.107 + 0.185 * rating
                                        + 0.399 * prior            r = +0.468

so production carries about 68% of the weight once it exists, and the split
moves further with experience - 47/53 in year two, 23/77 by year three.

Worth being honest about one result: production ALONE tests at +0.477 on the
same rows, marginally better than the blend. The recruiting term is not earning
its place for quarterbacks with a real record, and is kept only because it
steadies year-two cases where the record is one partial season. Anyone tempted
to weight recruiting more heavily should look at that number first.

The output is per quarterback and per team, the team taking its best projected
arm - the same convention talent_by_position.py uses for QB1, but ranking on
expected production rather than on stars.

Usage:
    python qb_projection.py --season 2026 --out results/qb_projection.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTION = os.path.join(_HERE, 'results', 'qb_production.csv')
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

# Below this a season is not a record, it is a cameo.
MIN_PRIOR_PLAYS = 100

# Reliability shrinkage was tried here and rejected on the evidence.
#
# The motivation was sound: a bad record is by construction a short one, since a
# quarterback playing badly gets benched, so the downgrades rest on 120-220
# plays while the upgrades rest on 800-1,300. Weighting the record by
# plays/(plays+K) should have corrected that.
#
# It does not help. Swept over K on a walk-forward where each season is
# predicted from strictly earlier seasons, mean correlation falls monotonically:
#
#     K        none    200     400     800    2000    4000
#     mean r  0.4407  0.4399  0.4379  0.4350  0.4307  0.4280
#
# An earlier sweep appeared to favour K=800, but its splits (>2020, >2021,
# >2022, >2023) share test rows with one another, so the same seasons were
# scored repeatedly and the apparent gain was selection on reused data. The
# likely reason it adds nothing: the fitted coefficient on the record already
# absorbs its average unreliability, and the career figure is play-weighted
# within a player, so longer records already count for more. Shrinking on top of
# that only attenuates real signal.
#
# Left documented rather than deleted so it is not re-derived from first
# principles and re-added.


def load(name):
    return pd.read_csv(os.path.join(PLAYER_DIR, f'cfbd_{name}.csv'),
                       low_memory=False)


def first_recruit_id(value):
    """roster.recruitIds arrives as the string form of a list."""
    import ast
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    return None


def career_to_date(prod, before):
    """Play-weighted mean z over every season strictly before `before`.

    Weighted by snaps so a 500-play season counts for more than a 110-play one,
    which is the same reason the havoc adjustment weights by its denominator.
    """
    h = prod[prod['season'] < before]
    h = h[h['plays'] >= MIN_PRIOR_PLAYS]
    if h.empty:
        return pd.DataFrame(columns=['pid', 'prior_z', 'prior_plays'])
    g = h.assign(_w=h['z'] * h['plays']).groupby('pid')
    out = g.agg(_num=('_w', 'sum'), prior_plays=('plays', 'sum')).reset_index()
    out['prior_z'] = out['_num'] / out['prior_plays']
    return out[['pid', 'prior_z', 'prior_plays']]


def fit_blend(prod, cutoff):
    """Fit both models on seasons up to `cutoff`, and report holdout skill."""
    frames = []
    for season in sorted(prod['season'].unique()):
        cur = prod[prod['season'] == season][['pid', 'season', 'z', 'rating']]
        merged = cur.merge(career_to_date(prod, season), on='pid', how='left')
        # the first season has no prior at all, so the join yields an all-NA
        # column; give it a dtype rather than letting concat guess
        for c in ('prior_z', 'prior_plays'):
            merged[c] = pd.to_numeric(merged.get(c), errors='coerce')
        frames.append(merged)
    D = pd.concat(frames, ignore_index=True).dropna(subset=['z'])
    D['rating_z'] = (D['rating'] - D['rating'].mean()) / D['rating'].std()

    train, test = D[D['season'] <= cutoff], D[D['season'] > cutoff]
    report = {}

    a = train[train['prior_z'].isna()].dropna(subset=['rating_z'])
    m_new = LinearRegression().fit(a[['rating_z']], a['z'])
    at = test[test['prior_z'].isna()].dropna(subset=['rating_z'])
    report['new'] = (len(a), len(at),
                     np.corrcoef(m_new.predict(at[['rating_z']]),
                                 at['z'])[0, 1] if len(at) > 5 else np.nan)

    b = train.dropna(subset=['rating_z', 'prior_z'])
    m_exp = LinearRegression().fit(b[['rating_z', 'prior_z']], b['z'])
    bt = test.dropna(subset=['rating_z', 'prior_z'])
    report['experienced'] = (len(b), len(bt),
                             np.corrcoef(m_exp.predict(bt[['rating_z', 'prior_z']]),
                                         bt['z'])[0, 1] if len(bt) > 5 else np.nan)
    return m_new, m_exp, D, report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, default=2026)
    # Defaults to the full history, matching rb_projection and
    # receiver_projection. It used to default to None, which produced the
    # --season row alone: bare `python qb_projection.py` wrote 137 team-seasons
    # where its siblings wrote 1,300, and position_ratings then carried a pf_qb
    # column that was empty for every season before the last one. Nothing
    # complained, because a projection file with one season in it is a perfectly
    # valid file.
    ap.add_argument('--from-season', type=int, default=2017,
                    help='project every season from here to --season, each '
                         'fitted only on what came before it')
    ap.add_argument('--fit-cutoff', type=int, default=2022,
                    help='fit on seasons up to here, hold out the rest, for '
                         'the reported skill figures only')
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'qb_projection.csv'))
    args = ap.parse_args()

    prod = pd.read_csv(PRODUCTION, low_memory=False)
    prod = prod.dropna(subset=['adj_epa_per_play'])
    prod['pid'] = prod['pid'].astype(str)
    # standardise within season so seasons are comparable
    prod['z'] = prod.groupby('season')['adj_epa_per_play'].transform(
        lambda s: (s - s.mean()) / s.std())

    roster = load('roster')
    recruits = load('recruits')
    recruits['id'] = recruits['id'].astype(str)
    for c in ('rating', 'stars'):
        recruits[c] = pd.to_numeric(recruits[c], errors='coerce')

    roster['pid'] = roster['id'].astype(str)
    roster['rid'] = roster['recruitIds'].map(first_recruit_id)
    link = roster.dropna(subset=['rid']).drop_duplicates('pid')[['pid', 'rid']]
    prod = prod.merge(link, on='pid', how='left').merge(
        recruits[['id', 'rating']].rename(columns={'id': 'rid'}),
        on='rid', how='left')

    m_new, m_exp, D, report = fit_blend(prod, args.fit_cutoff)
    print(f"fit on seasons <= {args.fit_cutoff}, held out the rest")
    for k, (n_tr, n_te, r) in report.items():
        print(f"  {k:<12} train {n_tr:>4}  holdout {n_te:>4}  r = {r:+.3f}")
    print(f"  first season : z = {m_new.intercept_:+.3f} "
          f"{m_new.coef_[0]:+.3f}*rating")
    print(f"  with a record: z = {m_exp.intercept_:+.3f} "
          f"{m_exp.coef_[0]:+.3f}*rating {m_exp.coef_[1]:+.3f}*prior")
    share = abs(m_exp.coef_[1]) / (abs(m_exp.coef_[0]) + abs(m_exp.coef_[1]))
    print(f"  production carries {share:.0%} of the weight")

    mu, sd = D['rating'].mean(), D['rating'].std()
    classif = load('classification')
    teams = pd.read_csv(TEAMS)
    name_to_id = dict(zip(teams['location'], teams['id']))

    seasons = ([args.season] if args.from_season is None
               else list(range(args.from_season, args.season + 1)))

    all_players = []
    for season in seasons:
        # Refit on everything strictly before this season. Using one global fit
        # would let a 2015 projection borrow coefficients estimated partly from
        # 2020, which is the kind of leak that makes a backtest look better than
        # the thing it is testing.
        if len(seasons) > 1:
            hist = prod[prod['season'] < season]
            if hist['season'].nunique() < 3:
                continue
            mn, me, _, _ = fit_blend(hist, hist['season'].max())
        else:
            mn, me = m_new, m_exp

        fbs = set(classif.loc[(classif['season'] == season)
                              & (classif['fbs'] == 1), 'team'])
        r = roster[(roster['season'] == season)
                   & (roster['position'] == 'QB')].copy()
        r = r[r['team'].isin(fbs)]
        if r.empty:
            continue
        r = r.merge(recruits[['id', 'rating', 'stars']].rename(
            columns={'id': 'rid'}), on='rid', how='left')
        r = r.merge(career_to_date(prod, season), on='pid', how='left')
        r['who'] = r['firstName'].astype(str) + ' ' + r['lastName'].astype(str)
        r['rating_z'] = (r['rating'] - mu) / sd

        def project(row):
            has_rec = pd.notna(row['prior_z'])
            has_rat = pd.notna(row['rating_z'])
            if has_rec and has_rat:
                return (me.intercept_ + me.coef_[0] * row['rating_z']
                        + me.coef_[1] * row['prior_z'])
            if has_rec:
                return me.intercept_ + me.coef_[1] * row['prior_z']
            if has_rat:
                return mn.intercept_ + mn.coef_[0] * row['rating_z']
            return np.nan

        r['projected_z'] = r.apply(project, axis=1)
        # what stars alone would have said, so the two can be compared directly
        r['recruiting_only_z'] = np.where(
            r['rating_z'].notna(),
            mn.intercept_ + mn.coef_[0] * r['rating_z'], np.nan)
        r['basis'] = np.where(r['prior_z'].notna(), 'production', 'recruiting')
        r['team_id'] = r['team'].map(name_to_id)
        cols = ['team_id', 'team', 'pid', 'who', 'rating', 'stars', 'prior_z',
                'prior_plays', 'recruiting_only_z',
                'projected_z', 'basis']
        part = r.dropna(subset=['projected_z'])[cols].copy()
        part.insert(0, 'season', season)
        all_players.append(part)
        print(f"  {season}: {len(r):>4} roster QBs, "
              f"{r['projected_z'].notna().mean():>4.0%} projectable, "
              f"{r['prior_z'].notna().mean():>4.0%} with a record")

    players = pd.concat(all_players, ignore_index=True).sort_values(
        ['season', 'projected_z'], ascending=[True, False])

    # the team takes its best projected arm, as QB1 takes its best rated one
    best = players.drop_duplicates(['season', 'team']).copy()
    best['qb_rank'] = (best.groupby('season')['projected_z']
                       .rank(ascending=False, method='min').astype(int))
    best['qb_pct'] = best.groupby('season')['projected_z'].rank(pct=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best.to_csv(args.out, index=False)
    players.to_csv(args.out.replace('.csv', '_players.csv'), index=False)
    print(f"wrote {args.out}  ({len(best)} teams)")
    print(f"wrote {args.out.replace('.csv', '_players.csv')} "
          f"({len(players)} quarterbacks)")

    show = best[best['season'] == max(seasons)].nsmallest(12, 'qb_rank')
    print(f"\n  {max(seasons)}")
    print(f"  {'#':>3} {'team':<20}{'quarterback':<22}{'proj':>7}"
          f"{'stars say':>11}{'basis':>13}")
    for _, x in show.iterrows():
        ro_ = f"{x.recruiting_only_z:+.2f}" if pd.notna(
            x.recruiting_only_z) else "   -"
        print(f"  {x.qb_rank:>3} {str(x.team)[:18]:<20}{str(x.who)[:20]:<22}"
              f"{x.projected_z:>+7.2f}{ro_:>11}{x.basis:>13}")


if __name__ == '__main__':
    main()

