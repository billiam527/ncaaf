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

PLAYER PROJECTION

Each returning receiver is carried forward the same way the quarterback model
works, on his own record plus the year he is entering:

    next = -0.312 + 0.733*current + 0.156*log(targets)
                  - 0.073*current*log(targets) - 0.083*experience

Two results in there are worth stating because they contradict what seems
obvious. Volume does NOT make a season more predictive - fitted separately by
target count the slope is flat, 0.47 at 25-35 targets against 0.43 at 95+, and
the correlation between target count and slope is -0.160. A 26-target season
carries forward about as well as a 106-target one, which is why shrinking thin
seasons toward the mean does nothing here: value already scales with targets,
so the weighting is present before anything is shrunk. And experience carries a
NEGATIVE coefficient, so an older receiver projects slightly worse than a
younger one who produced the same, the reverse of the quarterback curve.

Holdout, fitted through 2022 and tested on 2023-2025:

    flat slope                          +0.336
    slope varying with volume           +0.345
    flat slope + experience             +0.352
    volume-varying slope + experience   +0.365

Usage:
    python receiver_projection.py --season 2026 --from-season 2017
"""

import argparse
import os

import numpy as np
import pandas as pd

from qb_projection import first_recruit_id, load  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTION = os.path.join(_HERE, 'results', 'receiver_production.csv')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

MIN_TARGETS = 25

# A receiving room is a depth chart, not a roster. The median team fields three
# qualified receivers and one tight end, the top three take 85% of the targets
# and the top four take 95%, so counting a sixth body adds noise rather than
# offence. Trimming also tracks the team's passing production slightly better
# than counting everyone: against adjusted EPA per pass, top 3 WR + 1 TE
# correlates 0.712 and top 4 WR + 2 TE 0.704, against 0.706 for the full room.
# Four and two is used - it holds 95% of the targets and is more forgiving of a
# team that spreads the ball unusually wide.
ROOM_WR, ROOM_TE = 4, 2

# Running backs catch passes but are not receivers: 8.9 yards a catch against a
# receiver's 13.6, out of the backfield rather than a route tree. They and the
# stray quarterbacks, linemen and defensive backs picked up on trick plays are
# excluded - 13% of the rows before filtering.
ROOM_POSITIONS = ('WR', 'TE')


def fit_projection(prod, cutoff=2022):
    """next season's value from this one, volume and experience.

    Reported holdout skill is from a fit through `cutoff` tested on what
    follows; the returned model is refitted on everything, since the projection
    itself should use all the evidence available.
    """
    from sklearn.linear_model import LinearRegression
    d = prod.sort_values(['rec_id', 'season']).copy()
    for c in ('z_value', 'season', 'exp'):
        d['n_' + c] = d.groupby('rec_id')[c].shift(-1)
    P = d[d['n_season'] == d['season'] + 1].dropna(
        subset=['z_value', 'n_z_value', 'targets', 'exp'])
    P = P.assign(lt=np.log(P['targets']),
                 zt=P['z_value'] * np.log(P['targets']))
    X = ['z_value', 'lt', 'zt', 'exp']
    tr, te = P[P['season'] <= cutoff], P[P['season'] > cutoff]
    skill = np.nan
    if len(te) > 30:
        m0 = LinearRegression().fit(tr[X], tr['n_z_value'])
        skill = np.corrcoef(m0.predict(te[X]), te['n_z_value'])[0, 1]
    m = LinearRegression().fit(P[X], P['n_z_value'])
    return m, skill, len(P)


# A recruit's chance of ever reaching a qualifying season, by band. Measured on
# the 2014-2021 classes, whose careers are finished. This is the larger half of
# the uncertainty about a freshman: whether he plays at all, not how well.
PLAY_RATE = {5: 0.84, 4: 0.47, 3: 0.18, 2: 0.06}
# A tight end reaches a qualifying season less often than a wide receiver
# holding the recruiting grade fixed - 39% against 50% at four stars, 11%
# against 20% at three - so the rate is scaled rather than shared. Weighted
# across bands on the finished 2014-2021 classes the ratio is 0.64.
TE_PLAY_FACTOR = 0.64
# first qualifying season from the recruiting grade alone, holdout r = +0.174
FRESH_A, FRESH_B = -0.042, 0.187


def project_freshmen(roster, recruits, season, rating_mu, rating_sd):
    """Expected value from receivers with no record, mostly incoming recruits.

    Two things have to be multiplied, and using either alone is wrong. A
    five-star receiver reaches a qualifying season 84% of the time and a
    two-star 6%, so most of a class contributes nothing; and conditional on
    playing, the grade predicts his first season only weakly (+0.174). The
    product is a small number for everyone, which is the honest answer - a
    signing class is worth much less to next season than a returning starter.
    """
    r = roster[(roster['season'] == season)
               & (roster['position'].isin(['WR', 'TE']))].copy()
    r = r.merge(recruits[['id', 'rating', 'stars']].rename(
        columns={'id': 'rid'}), on='rid', how='left')
    r = r.dropna(subset=['rating', 'stars'])
    if r.empty:
        return pd.DataFrame()
    r['p_play'] = r['stars'].astype(int).map(PLAY_RATE).fillna(0.06)
    r.loc[r['position'] == 'TE', 'p_play'] *= TE_PLAY_FACTOR
    rz = (r['rating'] - rating_mu) / rating_sd
    r['if_plays'] = FRESH_A + FRESH_B * rz
    r['projected'] = r['p_play'] * r['if_plays']
    return r


def trim_to_room(df, value_col, pos_col='position'):
    """Keep only the depth chart: the best few receivers and tight ends.

    Ranked on projected value, so the room is who a team expects to throw to,
    not who happens to be listed.
    """
    d = df[df[pos_col].isin(ROOM_POSITIONS)].copy()
    if d.empty:
        return d
    d['_rk'] = (d.groupby(['team', pos_col])[value_col]
                .rank(ascending=False, method='first'))
    limit = d[pos_col].map({'WR': ROOM_WR, 'TE': ROOM_TE})
    return d[d['_rk'] <= limit].drop(columns=['_rk'])


def project_players(prod, roster, recruits, season, model):
    """Every returning receiver's expected value for `season`."""
    hist = prod[prod['season'] < season]
    if hist.empty:
        return pd.DataFrame()
    last = (hist.sort_values('season').groupby('rec_id')
            .agg(prev_team=('team', 'last'), prev_season=('season', 'max'),
                 z_value=('z_value', 'last'), targets=('targets', 'last'),
                 prev_yards=('rec_yards', 'last'),
                 prev_exp=('exp', 'last')).reset_index())
    r = roster[(roster['season'] == season)
               & (roster['position'].isin(['WR', 'TE']))].copy()
    r = r.merge(last, left_on='pid', right_on='rec_id', how='inner')
    if r.empty:
        return pd.DataFrame()
    # a year older than his last recorded season, not than his last on a roster
    r['exp'] = r['prev_exp'] + (season - r['prev_season'])
    r = r.dropna(subset=['z_value', 'targets', 'exp'])
    r['lt'] = np.log(r['targets'])
    r['zt'] = r['z_value'] * r['lt']
    r['projected'] = model.predict(r[['z_value', 'lt', 'zt', 'exp']])
    r['moved'] = r['prev_team'] != r['team']
    return r


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
    id_to_name = {v: k for k, v in name_to_id.items()}

    # value above an average receiver on the same targets, which is what the
    # projection carries forward
    lg = prod.groupby('season')['adj_yards_per_target'].mean().rename('lg')
    prod = prod.merge(lg, on='season', how='left')
    prod['value'] = prod['targets'] * (prod['adj_yards_per_target'] - prod['lg'])
    # Standardised within season AND position. A tight end does a smaller job:
    # 41.6 targets against 54.9, a 12.7% target share against 16.5%, 12.0 yards
    # a catch against 13.6. On a shared scale they are 13% of the population and
    # 0% of the top hundred, which says nothing about tight ends and everything
    # about the comparison. They are not a different kind of player to model,
    # though - fitted separately the slope is +0.397 against +0.392 for wide
    # receivers and stability +0.363 against +0.343 - so the same projection
    # runs on both and only the yardstick changes.
    prod['z_value'] = prod.groupby(['season', 'rec_pos'])['value'].transform(
        lambda s: (s - s.mean()) / s.std())
    prod['team'] = prod['team_id'].map(id_to_name)

    recruits = load('recruits')
    recruits['id'] = recruits['id'].astype(str)
    recruits['year'] = pd.to_numeric(recruits['year'], errors='coerce')
    roster['rid'] = roster['recruitIds'].map(first_recruit_id)
    link = roster.dropna(subset=['rid']).drop_duplicates('pid')[['pid', 'rid']]
    prod = prod.merge(link, left_on='rec_id', right_on='pid', how='left').merge(
        recruits[['id', 'year']].rename(columns={'id': 'rid',
                                                 'year': 'class_year'}),
        on='rid', how='left')
    prod['exp'] = prod['season'] - prod['class_year']

    rating_mu = float(prod['rating'].mean()) if 'rating' in prod.columns else None
    if rating_mu is None:
        rr = recruits[['id', 'rating']].copy()
        rr['rating'] = pd.to_numeric(rr['rating'], errors='coerce')
        rating_mu = float(rr['rating'].mean())
        rating_sd = float(rr['rating'].std())
    else:
        rating_sd = float(prod['rating'].std())

    model, skill, n = fit_projection(prod)
    print(f"projection fitted on {n:,} consecutive pairs, "
          f"holdout r = {skill:+.3f}")
    print(f"  next = {model.intercept_:+.3f} "
          f"{model.coef_[0]:+.3f}*value {model.coef_[1]:+.3f}*log(targets) "
          f"{model.coef_[2]:+.3f}*value:log(targets) {model.coef_[3]:+.3f}*exp")

    out, players = [], []
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

        # every returning receiver and every incoming one, projected, then cut
        # to a depth chart. Both sources compete for the same places: a
        # five-star freshman can displace a marginal returner, which is what
        # actually happens.
        pl = project_players(prod, roster, recruits, season, model)
        pl = pl[pl['team'].isin(fbs)] if len(pl) else pl
        if len(pl):
            pl = pl.assign(basis='record')
        proven = set(pl['pid']) if len(pl) else set()
        fr = project_freshmen(roster, recruits, season, rating_mu, rating_sd)
        if len(fr):
            fr = fr[fr['team'].isin(fbs) & ~fr['pid'].isin(proven)]
            if len(fr):
                fr = fr.assign(basis='recruiting', moved=False)
        both = pd.concat([x for x in (pl, fr) if len(x)], ignore_index=True) \
            if (len(pl) or len(fr)) else pd.DataFrame()
        if len(both):
            room = trim_to_room(both, 'projected')
            if len(room):
                agg = room.groupby('team', as_index=False).agg(
                    projected_corps=('projected', 'sum'),
                    projected_best=('projected', 'max'),
                    projected_n=('projected', 'size'),
                    arrivals=('moved', 'sum'))
                by = (room.groupby(['team', 'basis'])['projected'].sum()
                      .unstack(fill_value=0.0).reset_index())
                for c, nm in (('record', 'room_record'),
                              ('recruiting', 'room_freshman')):
                    agg[nm] = agg['team'].map(
                        dict(zip(by['team'], by[c]))) if c in by.columns else 0.0
                g = g.merge(agg, on='team', how='left')
                players.append(room.assign(season=season, in_room=True))
        g['team_id'] = g['team'].map(name_to_id)
        out.append(g)
        print(f"  {season}: {len(g)} teams, "
              f"{g['returning_receivers'].mean():.1f} returning receivers each",
              end='\r')
    print()

    R = pd.concat(out, ignore_index=True)
    for c in ('projected_corps', 'room_record', 'room_freshman'):
        if c in R.columns:
            R[c] = R[c].fillna(0.0)
    if 'projected_corps' in R.columns:
        R['projected_total'] = R['projected_corps']
    cols = ['corps_share', 'corps_yards', 'best_share', 'best_yards']
    for c in ('projected_corps', 'projected_best', 'projected_total'):
        if c in R.columns:
            cols.append(c)
    for c in cols:
        R[f'{c}_pct'] = R.groupby('season')[c].rank(pct=True)

    if players:
        PL = pd.concat(players, ignore_index=True)
        keep = ['season', 'team', 'pid', 'firstName', 'lastName', 'position',
                'basis', 'prev_team', 'prev_season', 'prev_yards', 'targets',
                'exp', 'z_value', 'stars', 'rating', 'p_play', 'if_plays',
                'projected', 'moved']
        PL = PL[[c for c in keep if c in PL.columns]]
        PL = PL.sort_values(['season', 'projected'], ascending=[True, False])
        ppath = args.out.replace('.csv', '_players.csv')
        PL.to_csv(ppath, index=False)
        print(f"wrote {ppath}  ({len(PL)} receiver-seasons projected)")
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
