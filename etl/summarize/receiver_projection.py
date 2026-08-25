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

    next = a + b*current + c*log(receptions)
             + d*current*log(receptions) + e*experience

with the fitted values printed at build time rather than quoted here, because
every version of this docstring that hard-coded them went stale within weeks.

The volume term is RECEPTIONS. It was log(targets) until 23 Aug 2026, and the
whole value definition rested on targets before that. Targets turned out to be
unusable across most of the panel: ESPN stopped naming the intended receiver on
incompletions during 2021-2024, so only 70.8% of 2024 pass plays name anybody
against 98.6% in 2025, and measured against CFBD pass attempts our team target
totals run at 0.679 of the truth in 2024 against 0.933 in 2025. Only
incompletions vanish, so catch rate, yards per target and target share all
inflate. Receptions are unaffected and match CFBD box scores at 0.95-0.99 in
every season.

Experience carries a NEGATIVE coefficient, so an older receiver projects
slightly worse than a younger one who produced the same - the reverse of the
quarterback curve. Whether volume makes a season more predictive is a question
this rebuild reopens: on target counts the slope looked flat, but that was
measured on the damaged panel.

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

# Last season, the volume behind it, their interaction, the year the man is
# entering, his EPA per catch, and what he had done BEFORE last season.
#
# The two share terms that used to sit here are gone. They correlate +0.902
# with each other and were worth +0.0017 between them; drop either and the
# survivor flips sign, which is what a pair of collinear terms fitting noise
# does. They remain the right measure of a receiver on their own - reception
# share is the heaviest term in the season rating - but conditional on the
# value and the volume they add nothing.
#
# The career replaces them and is worth ten times as much. Note it is the best
# of the seasons BEFORE the last one, not the career best: career best includes
# the last season and equals it for 85% of pairs, so putting both in drove
# z_value to -0.135 and made it swing -0.04/-0.03/-0.21 across training windows.
# Split that way both terms stay stable and interpretable.
#
# Two splits, since a gain has to survive both:
#                                     test 23-25   test 21-25
#   last season only                    +0.3478      +0.3881
#   + both shares (previous)            +0.3495      +0.3889
#   + career best and count             +0.3625      +0.3946   (unstable)
#   + best-so-far and count (used)      +0.3628      +0.3966
FEATURES = ['z_value', 'lt', 'zt', 'exp', 'z_adj_epa_per_catch',
            'prior_max', 'has_prior', 'car_n']
Z_EXTRAS = ('reception_share', 'yard_share', 'adj_epa_per_catch')


def add_career(d, group='rec_id', col='z_value'):
    """Best season before this one, whether there is one, and how many.

    A man in his first qualifying season has no prior. That is not a bad prior,
    so it is flagged rather than filled with a number that would read as one.
    """
    g = d.groupby(group)[col]
    d['prior_max'] = g.transform(lambda s: s.shift().expanding().max())
    d['has_prior'] = d['prior_max'].notna().astype(float)
    d['prior_max'] = d['prior_max'].fillna(0.0)
    d['car_n'] = d.groupby(group).cumcount() + 1
    return d


def fit_projection(prod, cutoff=2022):
    """next season's value from this one, volume and experience.

    Reported holdout skill is from a fit through `cutoff` tested on what
    follows; the returned model is refitted on everything, since the projection
    itself should use all the evidence available.
    """
    from sklearn.linear_model import LinearRegression
    d = prod.sort_values(['rec_id', 'season']).copy()
    d = add_career(d)
    for c in ('z_value', 'season', 'exp'):
        d['n_' + c] = d.groupby('rec_id')[c].shift(-1)
    P = d[d['n_season'] == d['season'] + 1].dropna(
        subset=['z_value', 'n_z_value', 'receptions', 'exp'])
    # The volume term is receptions, not targets: targets are undercounted by a
    # season-varying amount (0.679 of the truth in 2024, 0.933 in 2025), so a
    # log(targets) term would read a receiver's workload partly off how well his
    # games happened to be recorded.
    P = P.assign(lt=np.log(P['receptions'].clip(lower=1)),
                 zt=P['z_value'] * np.log(P['receptions'].clip(lower=1)))
    P = P.dropna(subset=FEATURES)
    tr, te = P[P['season'] <= cutoff], P[P['season'] > cutoff]
    skill = np.nan
    if len(te) > 30:
        m0 = LinearRegression().fit(tr[FEATURES], tr['n_z_value'])
        skill = np.corrcoef(m0.predict(te[FEATURES]), te['n_z_value'])[0, 1]
    m = LinearRegression().fit(P[FEATURES], P['n_z_value'])
    return m, skill, len(P)


# Fallbacks only. These were hardcoded constants for a long time and went stale
# without anything noticing, twice over: they describe the chance of reaching a
# QUALIFYING season, and the gate moved from 25 targets to 16 receptions when
# targets stopped being trustworthy, so the population they were measured on no
# longer exists. FRESH_A/FRESH_B predicted a target-based value that has since
# been replaced by a catch-based one.
#
# fit_freshman() now derives all three from the data on every build and prints
# them. These are used only if a band has too few finished careers to fit.
PLAY_RATE = {5: 0.84, 4: 0.47, 3: 0.18, 2: 0.06}
TE_PLAY_FACTOR = 0.64
FRESH_A, FRESH_B = -0.042, 0.187
# Classes signed by LAST_FINISHED_CLASS have had four seasons to appear and are
# treated as finished; later ones are still in progress and would read as
# failures. The lower bound matters just as much and is easier to miss: the
# production panel starts in 2014, so a man signed in 2012 who qualified in
# 2013 is invisible to it and would be counted as never having played. Without
# the bound the five-star band read 44 men where 2014-2021 holds 32.
FIRST_FINISHED_CLASS = 2014
LAST_FINISHED_CLASS = 2021
MIN_BAND = 30


# The counting line, projected alongside the rating. The rating is a
# standardised number nobody can read off a page; catches, yards per catch and
# EPA per catch are what a reader checks a room against.
#
# Each is fitted on the same features as the rating plus its own current value,
# which is the strongest predictor of itself. Holdout on 2023-2025:
#
#   receptions          +0.513   mean error 12.1 catches
#   adj yards per catch +0.625   mean error 2.11
#   adj EPA per catch   see build output
#
# Only men with a record get one. A freshman has nothing to carry forward and
# an invented line would read as knowledge.
LINE = {'receptions': 'proj_receptions',
        'adj_yards_per_catch': 'proj_ypc',
        'adj_epa_per_catch': 'proj_epc'}


def fit_line(prod, cutoff=2022):
    """One model per counting statistic, and its holdout skill."""
    from sklearn.linear_model import LinearRegression
    d = prod.sort_values(['rec_id', 'season']).copy()
    # lt and zt are built inside fit_projection, so they have to be rebuilt
    # here rather than assumed present on prod
    d['lt'] = np.log(d['receptions'].clip(lower=1))
    d['zt'] = d['z_value'] * d['lt']
    for c in LINE:
        d['n_' + c] = d.groupby('rec_id')[c].shift(-1)
    d['n_season'] = d.groupby('rec_id')['season'].shift(-1)
    P = d[d['n_season'] == d['season'] + 1]
    out = {}
    for src, name in LINE.items():
        X = FEATURES + [src]
        q = P.dropna(subset=X + ['n_' + src])
        if len(q) < 200:
            continue
        tr, te = q[q['season'] <= cutoff], q[q['season'] > cutoff]
        skill, err = np.nan, np.nan
        if len(te) > 30:
            m0 = LinearRegression().fit(tr[X], tr['n_' + src])
            pr = m0.predict(te[X])
            skill = np.corrcoef(pr, te['n_' + src])[0, 1]
            err = float(np.abs(pr - te['n_' + src]).mean())
        out[name] = (LinearRegression().fit(q[X], q['n_' + src]), X,
                     skill, err, src)
    print("projected line, holdout on the seasons after "
          f"{cutoff}:")
    for name, (_, _, sk, er, src) in out.items():
        print(f"  {src:<22}r = {sk:+.3f}   mean error {er:>6.2f}")
    return out


def fit_freshman(prod, roster, recruits, rating_mu, rating_sd):
    """Derive the no-record model from finished careers.

    Both halves have to describe THE SAME SEASON. An earlier version paired
    P(ever reaches a qualifying season) with the value of whichever season that
    turned out to be, which is not a projection of anything: a five-star who
    first qualifies as a junior sits in the 81% who ever qualify and contributes
    nothing to next year. That pairing understated a five-star signee five-fold,
    +0.046 against a measured +0.227, while making a three-star twice too
    negative.

    So this is a hazard: given he is on a roster this season and has no
    qualifying season behind him, what is the chance he qualifies NOW, and how
    good is he if he does. Both terms condition on the grade and on years since
    signing, because an incoming freshman and a redshirt junior with the same
    grade are different cases and project_freshmen sees both.

    Returns (play_model, val_model, skill).
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    rc = recruits.dropna(subset=['class_year', 'rating']).copy()
    rc = rc[rc['class_year'].between(FIRST_FINISHED_CLASS,
                                     LAST_FINISHED_CLASS)]
    if rc.empty or 'pid' not in rc.columns:
        return None, None, {}, np.nan

    # the season each man first qualified, and how good it was
    first = (prod.sort_values('season').groupby('rec_id')
             .agg(first_season=('season', 'first'),
                  first_z=('z_value', 'first'),
                  # the line he posted in that first season, so the same fit
                  # can produce a readable projection and not only a rating
                  receptions=('receptions', 'first'),
                  adj_yards_per_catch=('adj_yards_per_catch', 'first'),
                  adj_epa_per_catch=('adj_epa_per_catch', 'first'))
             .reset_index())
    rc = rc.merge(first, left_on='pid', right_on='rec_id', how='left')

    # One row per season a man is on a roster with no qualifying season yet -
    # exactly the population project_freshmen is applied to. A man who has
    # already qualified leaves the risk set; one who never qualifies stays in
    # it for as long as he is rostered, which is what makes the rate a hazard
    # rather than a career total.
    ros = roster[roster['position'].isin(ROOM_POSITIONS)].copy()
    ros = ros.dropna(subset=['rid'])[['rid', 'season', 'position']]
    panel = ros.merge(rc[['rid', 'rating', 'stars', 'class_year',
                          'first_season', 'first_z', 'receptions',
                          'adj_yards_per_catch', 'adj_epa_per_catch']],
                      on='rid', how='inner')
    panel['age'] = panel['season'] - panel['class_year']
    panel = panel[panel['age'].between(0, 4)]
    at_risk = panel[panel['first_season'].isna()
                    | (panel['season'] <= panel['first_season'])].copy()
    at_risk['qualifies'] = (at_risk['season'] == at_risk['first_season'])
    at_risk['rz'] = (at_risk['rating'] - rating_mu) / rating_sd
    at_risk['is_te'] = (at_risk['position'] == 'TE').astype(float)
    # a squared term because the grade bends at the top: fitted straight, the
    # 2,000-odd three-stars set the slope and the five-stars are missed
    at_risk['rz2'] = at_risk['rz'] ** 2

    XP = ['rz', 'rz2', 'age', 'is_te']
    play_model = val_model = None
    fit = at_risk.dropna(subset=XP + ['qualifies'])
    if len(fit) > 500 and fit['qualifies'].nunique() > 1:
        play_model = LogisticRegression(max_iter=2000).fit(
            fit[XP].to_numpy(float), fit['qualifies'].astype(int).to_numpy())

    # value given he qualifies THIS season, on the same conditioning
    q = fit[fit['qualifies']].dropna(subset=['first_z'])
    skill = np.nan
    if len(q) > 100:
        cut = q['class_year'].quantile(0.7)
        tr, te_ = q[q['class_year'] <= cut], q[q['class_year'] > cut]
        if len(te_) > 30:
            m0 = LinearRegression().fit(tr[XP].to_numpy(float),
                                        tr['first_z'].to_numpy())
            skill = np.corrcoef(m0.predict(te_[XP].to_numpy(float)),
                                te_['first_z'])[0, 1]
        val_model = LinearRegression().fit(q[XP].to_numpy(float),
                                           q['first_z'].to_numpy())

    # The stat line a man posts IF he breaks through, on the same features.
    # Conditional by construction: multiplying by a 12% chance of playing would
    # print four catches, which is a number nobody posts - the first qualifying
    # season is 25-odd catches or it does not happen.
    #
    # The grade is a weak instrument here: r = +0.15 on receptions against
    # +0.50 for a returning man's own record. So these land close to the band
    # median and should be read as "what a recruit of this grade posts when he
    # breaks through", not as a forecast of the individual.
    line_models = {}
    for src, name in LINE.items():
        q2 = q.dropna(subset=XP + [src]) if src in q.columns else q.iloc[:0]
        if len(q2) > 200:
            line_models[name] = LinearRegression().fit(
                q2[XP].to_numpy(float), q2[src].to_numpy())

    print(f"no-record model: {len(fit):,} rostered seasons at risk, "
          f"{int(fit['qualifies'].sum()):,} qualified "
          f"(classes {FIRST_FINISHED_CLASS}-{LAST_FINISHED_CLASS})")
    if play_model is not None and val_model is not None:
        print("  a signee entering year one, by band:")
        for st in (5, 4, 3, 2):
            g = fit[(fit['stars'] == st) & (fit['is_te'] == 0)]
            if len(g) < 10:
                continue
            rz = float(g['rz'].mean())
            x = np.array([[rz, rz ** 2, 0.0, 0.0]])
            pp = float(play_model.predict_proba(x)[0, 1])
            vv = float(val_model.predict(x)[0])
            raw = g[g['age'] == 0]
            print(f"    {st}*  P(qualifies) {pp:>5.0%}   if he does {vv:>+6.2f}"
                  f"   product {pp*vv:>+6.3f}"
                  + (f"   (raw {raw['qualifies'].mean():.0%})" if len(raw)
                     else ''))
        print(f"  value model holdout r = {skill:+.3f}")
    return play_model, val_model, line_models, skill


def project_freshmen(roster, recruits, season, rating_mu, rating_sd,
                     fresh=None):
    """Expected value from receivers with no record, mostly incoming recruits.

    Two things have to be multiplied, and using either alone is wrong. Most of
    a signing class never reaches a qualifying season, so a grade on its own
    badly overstates what a class is worth; and among those who do play, the
    grade says only a little about how good the first season is. The product is
    a small number for everyone, which is the honest answer - a signing class is
    worth much less to next season than one returning starter.

    `fresh` carries the rates derived from finished careers by fit_freshman.
    Without it the module-level fallbacks are used, which is only correct if
    neither the qualification gate nor the value definition has moved.
    """
    play_model, val_model, line_models = (
        fresh if fresh is not None else (None, None, {}))
    r = roster[(roster['season'] == season)
               & (roster['position'].isin(ROOM_POSITIONS))].copy()
    r = r.merge(recruits[['id', 'rating', 'stars', 'year']].rename(
        columns={'id': 'rid', 'year': 'class_year'}), on='rid', how='left')
    r = r.dropna(subset=['rating', 'stars'])
    if r.empty:
        return pd.DataFrame()
    rz = ((r['rating'] - rating_mu) / rating_sd).to_numpy(float)
    # years since signing. Both halves condition on it, because a man with no
    # record entering year one and one entering year four are different bets:
    # the first is unproven, the second has had chances and not taken them.
    age = pd.to_numeric(r['class_year'], errors='coerce')
    age = (season - age).clip(lower=0, upper=4).fillna(0).to_numpy(float)
    is_te = (r['position'] == 'TE').to_numpy(float)
    X = np.column_stack([rz, rz ** 2, age, is_te])

    if play_model is not None and val_model is not None:
        r['p_play'] = play_model.predict_proba(X)[:, 1]
        r['if_plays'] = val_model.predict(X)
    else:
        r['p_play'] = r['stars'].astype(int).map(PLAY_RATE).fillna(0.06)
        r.loc[r['position'] == 'TE', 'p_play'] *= TE_PLAY_FACTOR
        r['if_plays'] = FRESH_A + FRESH_B * rz
    r['projected'] = r['p_play'] * r['if_plays']
    # the line he posts IF he breaks through - not multiplied by p_play, since
    # 12% of a season is a number nobody posts. Flagged so a page can say so.
    for name, m in (line_models or {}).items():
        r[name] = m.predict(X)
    if 'proj_receptions' in r.columns:
        r['proj_receptions'] = r['proj_receptions'].clip(lower=0)
    r['line_if_plays'] = bool(line_models)
    return r


def trim_to_room(df, value_col, pos_col='position'):
    """Keep only the depth chart: the best few receivers and tight ends.

    A man with a record outranks a man without one, and only then does
    projected value decide. Sorting on value alone answered the wrong question.

    The projection is an expected value and it correctly discounts an unproven
    recruit by his small chance of playing at all - so a five-star at +0.10 can
    sit above a returner at -0.19 and be properly priced. But a depth chart is
    not a list of expected values, it is a list of who will be on the field.
    Measured over the 706 cases where a recruit was projected above a returner
    on the same team:

        reached a qualifying season      recruit 14%     returner 58%
        mean z, a non-season counted 0   recruit -0.02   returner -0.06

    The recruit is the marginally better bet on value and a quarter as likely
    to appear. Ohio State's room put two recruits above Brandon Inniss, who was
    third on the team in catches and returns to a room that lost Carnell Tate.
    """
    d = df[df[pos_col].isin(ROOM_POSITIONS)].copy()
    if d.empty:
        return d
    tier = np.where(d['basis'].eq('record'), 0, 1) if 'basis' in d.columns \
        else np.zeros(len(d))
    d['_tier'] = tier
    order = d.sort_values(['_tier', value_col], ascending=[True, False])
    d['_rk'] = order.groupby(['team', pos_col]).cumcount().add(1).reindex(
        d.index)
    limit = d[pos_col].map({'WR': ROOM_WR, 'TE': ROOM_TE})
    return d[d['_rk'] <= limit].drop(columns=['_rk', '_tier'])


def project_players(prod, roster, recruits, season, model, line=None):
    """Every returning receiver's expected value for `season`."""
    hist = prod[prod['season'] < season]
    if hist.empty:
        return pd.DataFrame()
    # the career behind his last recorded season, on the same definition the
    # fit used: best of everything strictly before it, and how many he has
    hist = add_career(hist.sort_values(['rec_id', 'season']).copy())
    last = (hist.sort_values('season').groupby('rec_id')
            .agg(prev_team=('team', 'last'), prev_season=('season', 'max'),
                 z_value=('z_value', 'last'), targets=('targets', 'last'),
                 receptions=('receptions', 'last'),
                 prev_yards=('rec_yards', 'last'),
                 z_reception_share=('z_reception_share', 'last'),
                 z_yard_share=('z_yard_share', 'last'),
                 z_adj_epa_per_catch=('z_adj_epa_per_catch', 'last'),
                 prior_max=('prior_max', 'last'),
                 has_prior=('has_prior', 'last'),
                 car_n=('car_n', 'last'),
                 # the raw rates the counting line is projected from
                 adj_yards_per_catch=('adj_yards_per_catch', 'last'),
                 adj_epa_per_catch=('adj_epa_per_catch', 'last'),
                 prev_exp=('exp', 'last')).reset_index())
    r = roster[(roster['season'] == season)
               & (roster['position'].isin(ROOM_POSITIONS))].copy()
    r = r.merge(last, left_on='pid', right_on='rec_id', how='inner')
    if r.empty:
        return pd.DataFrame()
    # The recruiting grade, which project_freshmen merges because it needs it
    # and this never did - so every receiver with a record carried a null one
    # and the column read empty for exactly the men most likely to be looked
    # up. It carries no weight in the projection; it is here to be shown.
    if 'rid' in r.columns:
        r = r.merge(recruits[['id', 'stars', 'rating']].rename(
            columns={'id': 'rid'}), on='rid', how='left')
    # a year older than his last recorded season, not than his last on a roster
    r['exp'] = r['prev_exp'] + (season - r['prev_season'])
    # Experience comes from the recruiting class year, reached through
    # roster.recruitIds. 950 of the 2026 receivers have no such link - walk-ons,
    # juco arrivals, anyone whose recruiting row never matched - and for them
    # exp was NaN, so the dropna below removed them from the room no matter how
    # much production they had. That cost 79 of 463 returning receivers,
    # including men with 224 and 146 career catches. A recruiting gap was
    # deciding a production question.
    #
    # The roster's own eligibility year covers 100% of them and is the right
    # fallback: year 1 is a first-year player, so exp is year - 1.
    if 'year' in r.columns:
        r['exp'] = r['exp'].fillna(
            pd.to_numeric(r['year'], errors='coerce') - 1)
    r = r.dropna(subset=['z_value', 'receptions', 'exp'])
    r['lt'] = np.log(r['receptions'].clip(lower=1))
    r['zt'] = r['z_value'] * r['lt']
    # a receiver missing one of the extras keeps his projection rather than
    # dropping out of the room: the extras are standardised, so 0 is the
    # position's own average for that season and is the right neutral fill
    for _c in FEATURES:
        if _c not in r.columns:
            r[_c] = 0.0
    r[FEATURES] = r[FEATURES].fillna(0.0)
    r['projected'] = model.predict(r[FEATURES])
    # the readable line, for men who have one. A projected reception count is
    # clipped at zero because a linear fit will occasionally go under.
    for name, (m, cols, _sk, _er, src) in (line or {}).items():
        ok = r[src].notna()
        r[name] = np.nan
        if ok.any():
            r.loc[ok, name] = m.predict(r.loc[ok, cols])
    if 'proj_receptions' in r.columns:
        r['proj_receptions'] = r['proj_receptions'].clip(lower=0)
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

    # Value above an average receiver on the same CATCHES, which is what the
    # projection carries forward.
    #
    # This was targets x (adj_yards_per_target - league) until 23 Aug 2026. It
    # had to change because targets are not trustworthy: ESPN stopped naming the
    # intended receiver on incompletions through 2021-2024, so only 70.8% of
    # 2024 pass plays name anyone against 98.6% in 2025, and our team target
    # totals run at 0.679 of the truth in 2024 against 0.933 in 2025. Both terms
    # of the old definition were affected - the multiplier directly, and the
    # rate through its denominator.
    #
    # Receptions survive: they match CFBD box scores at 0.95-0.99 in every
    # season. Fitted on clean seasons only, the per-catch family predicts next
    # season at +0.376 against the target family's +0.351 on the per-catch
    # yardstick and +0.321 against +0.354 on the per-target one - within about
    # 0.03 either way - so this costs almost nothing and works in all twelve
    # seasons rather than only 2014-2020 and 2025.
    # Total opponent-adjusted receiving yards. Subtracting a league mean first
    # was tried and is worse: it repeats at 0.374 across the whole panel where
    # the plain total repeats at 0.415, and it drops holdout from +0.344 to
    # +0.273, because the subtraction re-introduces a season-level quantity the
    # measure then has to carry. A counting statistic is also what the team
    # aggregate needs, since a room's total is the sum of its receivers and
    # shares sum to 1 by construction.
    prod['value'] = prod['receptions'] * prod['adj_yards_per_catch']
    # The target-based value is kept alongside for the seasons where targets
    # are sound, and so the two can go on being compared.
    lgt = prod.groupby('season')['adj_yards_per_target'].mean().rename('lg_t')
    prod = prod.merge(lgt, on='season', how='left')
    prod['value_target'] = prod['targets'] * (prod['adj_yards_per_target']
                                              - prod['lg_t'])
    # Standardised within season AND position. A tight end does a smaller job:
    # 43.4 targets against 57.8, a 12.8% target share against 16.5%, 12.1 yards
    # a catch against 13.7. On a shared scale they are 13% of the population and
    # 0% of the top hundred, which says nothing about tight ends and everything
    # about the comparison. They are not a different kind of player to model,
    # though - fitted separately the slope is +0.397 against +0.392 for wide
    # receivers and stability +0.363 against +0.343 - so the same projection
    # runs on both and only the yardstick changes.
    prod['z_value'] = prod.groupby(['season', 'rec_pos'])['value'].transform(
        lambda s: (s - s.mean()) / s.std())
    # the extras the fit reads, on the same season-and-position scale
    for _c in Z_EXTRAS:
        prod['z_' + _c] = prod.groupby(['season', 'rec_pos'])[_c].transform(
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

    # every receiver recruit, linked to a roster id where one exists so the
    # unlinked ones stay in the denominator as men who never played
    _rec = recruits.copy()
    _rec['rid'] = _rec['id'].astype(str)
    _rec['class_year'] = pd.to_numeric(_rec['year'], errors='coerce')
    for _c in ('rating', 'stars'):
        _rec[_c] = pd.to_numeric(_rec[_c], errors='coerce')
    _rec = _rec[_rec['position'].isin(ROOM_POSITIONS)]
    finished = _rec.merge(link, on='rid', how='left')

    rating_mu = float(prod['rating'].mean()) if 'rating' in prod.columns else None
    if rating_mu is None:
        rr = recruits[['id', 'rating']].copy()
        rr['rating'] = pd.to_numeric(rr['rating'], errors='coerce')
        rating_mu = float(rr['rating'].mean())
        rating_sd = float(rr['rating'].std())
    else:
        rating_sd = float(prod['rating'].std())

    _pm, _vm, _lm, _fskill = fit_freshman(prod, roster, finished,
                                          rating_mu, rating_sd)
    FRESH = (_pm, _vm, _lm)

    model, skill, n = fit_projection(prod)
    LINE_MODELS = fit_line(add_career(prod.sort_values(['rec_id', 'season'])))
    print(f"projection fitted on {n:,} consecutive pairs, "
          f"holdout r = {skill:+.3f}")
    # printed off FEATURES rather than by position, so adding a term cannot
    # silently mislabel the ones after it
    LABEL = {'z_value': 'value', 'lt': 'log(receptions)',
             'zt': 'value:log(receptions)', 'exp': 'exp',
             'z_reception_share': 'reception_share',
             'z_yard_share': 'yard_share',
             'z_adj_epa_per_catch': 'adj_epa_per_catch',
             'prior_max': 'best_prior_season', 'has_prior': 'has_prior',
             'car_n': 'seasons_played'}
    terms = ' '.join(f"{c:+.3f}*{LABEL.get(f, f)}"
                     for f, c in zip(FEATURES, model.coef_))
    print(f"  next = {model.intercept_:+.3f} {terms}")

    # The fitted equation, written beside the results. Any page that shows it
    # then renders from this rather than from a figure typed into a template,
    # which is how every other quoted coefficient in this project went stale.
    import json as _json
    mpath = os.path.join(os.path.dirname(args.out),
                         'receiver_projection_model.json')
    with open(mpath, 'w') as fh:
        _json.dump({'intercept': float(model.intercept_),
                    'terms': [{'feature': f, 'label': LABEL.get(f, f),
                               'coef': float(c)}
                              for f, c in zip(FEATURES, model.coef_)],
                    'holdout_r': None if np.isnan(skill) else float(skill),
                    'pairs': int(n)}, fh, indent=1)
    print(f"  wrote {mpath}")

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
        pl = project_players(prod, roster, recruits, season, model,
                             line=LINE_MODELS)
        pl = pl[pl['team'].isin(fbs)] if len(pl) else pl
        if len(pl):
            pl = pl.assign(basis='record')
        proven = set(pl['pid']) if len(pl) else set()
        fr = project_freshmen(roster, recruits, season, rating_mu, rating_sd,
                              fresh=FRESH)
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
                # The corps figure sums wideouts and tight ends together, which
                # is right for "how good is the receiving room" and wrong for
                # anything that needs them apart - a model cannot dock a team
                # for losing its tight end if the tight end is inside a single
                # number. Both are already standardized within position, so a
                # tight end is scored against tight ends and the two sums live
                # on comparable scales.
                for pos, nm in (('WR', 'projected_wr'), ('TE', 'projected_te')):
                    s = (room[room['position'] == pos]
                         .groupby('team')['projected'].sum())
                    agg[nm] = agg['team'].map(s).fillna(0.0)
                    n = (room[room['position'] == pos]
                         .groupby('team')['projected'].size())
                    agg[f'{nm}_n'] = agg['team'].map(n).fillna(0).astype(int)
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
    for c in ('projected_corps', 'room_record', 'room_freshman',
              'projected_wr', 'projected_te'):
        if c in R.columns:
            R[c] = R[c].fillna(0.0)
    if 'projected_corps' in R.columns:
        R['projected_total'] = R['projected_corps']
    cols = ['corps_share', 'corps_yards', 'best_share', 'best_yards']
    for c in ('projected_corps', 'projected_best', 'projected_total',
              'projected_wr', 'projected_te'):
        if c in R.columns:
            cols.append(c)
    for c in cols:
        R[f'{c}_pct'] = R.groupby('season')[c].rank(pct=True)

    if players:
        PL = pd.concat(players, ignore_index=True)
        # every input the projection reads is written out alongside its
        # output, so a page or an audit can show what produced a number
        # without re-deriving it from the production file
        keep = ['season', 'team', 'pid', 'firstName', 'lastName', 'position',
                'basis', 'prev_team', 'prev_season', 'prev_yards',
                'receptions', 'targets',
                'exp', 'z_value', 'z_reception_share', 'z_yard_share',
                'z_adj_epa_per_catch', 'prior_max', 'has_prior', 'car_n',
                'proj_receptions', 'proj_ypc', 'proj_epc', 'line_if_plays',
                'stars', 'rating', 'p_play', 'if_plays',
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
