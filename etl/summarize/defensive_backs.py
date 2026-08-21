#!/usr/bin/env python3
"""One secondary rating per team-season, and a projection of the next one.

THE PROBLEM THIS UNIT HAS

Pass defence is not mostly the secondary's. Holding the other unit's recruiting
fixed, across 613 team-seasons:

    pass EPA allowed        front seven -0.220   secondary +0.065
    pass success allowed    front seven +0.195   secondary -0.049

The front seven predicts pass defence better than it predicts rush defence, and
better than the secondary predicts its own phase. Pressure is doing the work: a
quarterback with two seconds throws worse than a quarterback with four, whoever
is covering. So a secondary rating built on raw pass defence would mostly be
re-reading the pass rush.

This module therefore rates the secondary on pass defence AFTER the front seven
is accounted for. Each pass-defence measure is residualized against the front-
seven rating, and the secondary is graded on what is left. That number is
smaller and less flattering than a raw one, and it is the honest one.

A CAUTION ABOUT PASSES DEFENSED

Pass-defensed rate is oriented HIGHER = WORSE against points allowed (r +0.115),
which is not a mistake in the data. It is a volume artifact: a defence that
cannot stop the run, or that trails, faces more throws and racks up more passes
defensed. It is carried here on the same footing as everything else - orientation
verified against points allowed, not assumed from the name - and the fit is
allowed to price it at zero if that is what it is worth.

Interception rate repeats year over year at 0.198, close to a coin flip, so it
is fitted rather than assumed important.

Usage:
    python defensive_backs.py --out results/defensive_backs.csv
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
HAVOC = os.path.join(_HERE, 'results', 'havoc_adjusted.csv')
SEASONS = os.path.join(_HERE, 'results', 'season_summaries.csv')
TALENT = os.path.join(_HERE, 'results', 'talent_by_position.csv')
FRONT7 = os.path.join(_HERE, 'results', 'front_seven.csv')
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

DB_POSITIONS = {'DB', 'CB', 'S', 'FS', 'SS', 'SAF', 'NB'}
ROOM_SIZE = 5           # nickel is the base defence now, so five, not four
MIN_GRADED = 2

# Orientation verified against points allowed over 2,154 team-seasons, never
# inferred from the column name. See front_seven.py for the full table - the
# EPA columns are stored as defensive credit, everything else as allowed.
INVERTED = ('adjusted_pass_success_def', 'adjusted_explosive_pass_rate_def',
            'adj_pass_defensed_rate')

COVER_PARTS = ('adjusted_epa_per_pass_def', 'adjusted_pass_success_def',
               'adjusted_explosive_pass_rate_def')
BALL_PARTS = ('adj_interception_rate', 'adj_pass_defensed_rate',
              'adj_fumble_rate')

# The tier rule lives in tiers.py, not in a frozen set here. A copy of that set
# in this file read Notre Dame as group-of-five and the 2026 Pac-12 as power.
from tiers import power_series  # noqa: E402

TEAM_NAME = {}          # filled in main(), used by the tier rule



# Recency and class handling for the career figure. A flat sum treats a season
# three years ago as fully as last season, which put a man whose last year was
# two coverage events at the top of the file. CAREER_DECAY=1.0 restores the
# flat behaviour.
# Blend of the three career measures. Ball events only is what ships,
# and a walk-forward is why. A grid over 55 mixes put 0.6/0.1/0.3
# highest at +0.0113 correlation with next season's pass defence,
# t +1.96 - but scored on game predictions, each season fitted on
# prior seasons only, that mix came out 0.009 MAE WORSE (t +1.71,
# 5,192 games, ATS unchanged at 50.2%). The in-sample gain was the
# maximum of a grid and did not survive. Kept switchable so the
# result can be reproduced, not because it is a candidate.
CAREER_BALL = float(os.environ.get('CAREER_BALL', '1.0'))
CAREER_TKL = float(os.environ.get('CAREER_TKL', '0.0'))
CAREER_DISRUPT = float(os.environ.get('CAREER_DISRUPT', '0.0'))

CAREER_DECAY = 0.5
CAREER_CLASS_CAP = 4        # fifth and sixth-year men count as seniors


def decay_sum(out, col, lam=CAREER_DECAY):
    """Running per-player sum with each earlier season weighted lam ** gap.

    Decays by the real year gap, not by row position - a man who missed a
    season should lose two years of weight, not one.
    """
    if lam >= 1.0:
        return out.groupby('pid')[col].cumsum()
    run, prev_pid, prev_s, acc = [], None, None, 0.0
    for pid, s, v in zip(out['pid'], out['season'], out[col]):
        acc = 0.0 if pid != prev_pid else acc * lam ** (s - prev_s)
        acc += v
        run.append(acc)
        prev_pid, prev_s = pid, s
    return run


def class_key(out):
    """Standardise within class year, seniors and beyond pooled."""
    if not out['class_yr'].notna().any():
        return ['season']
    out['class_bin'] = out['class_yr'].clip(upper=CAREER_CLASS_CAP)
    return ['season', 'class_bin']

def first_recruit_id(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return str(parsed[0]) if isinstance(parsed, list) and parsed else None


def room_members(size=ROOM_SIZE, rated_only=False, order='production'):
    """The five men, one row each, with their recruiting grade.

    Five because nickel is the base defence in this era, not four. As everywhere
    else on this side of the ball there are no snap counts, so this cannot know
    who starts.

    Two consumers want different things from this, and serving both with one
    selection measurably costs:

      rated_only=False   who is on the field, for the coverage carry. An
                         unrated man's saved yards are as real as anyone's.
      rated_only=True    the best five RATED men, for the recruiting grade.
                         Letting unrated men take places here computes the mean
                         over three or four grades instead of five and makes it
                         noisier - worth -0.024 on the predictive criterion,
                         more than a standard error, for no gain against Steele.

    The ORDER splits the same way, which was not obvious and cost real accuracy
    to miss. Ordering the room by production was the fix that put Leonard Moore
    in his own secondary, and it was applied to the recruiting grade as well,
    where it is wrong:

        grade by    terms by      predicts   vs Steele's 68   Notre Dame
        production  production      0.328         +0.728          3rd
        rating      production      0.340         +0.770          1st
        rating      rating          0.323         +0.676          4th
        production  rating          0.305         +0.619          4th

    The grade asks how much talent a room holds, which the most talented men
    answer. The carry and career terms ask who is on the field, which the most
    productive answer. So order='rating' for the grade, order='production' for
    the player terms - and the room the page displays stays production-ordered,
    because that question is 'who plays', not 'who was signed'.
    """
    roster = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_roster.csv'),
                         low_memory=False)
    recruits = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_recruits.csv'),
                           low_memory=False)[['id', 'rating']]
    recruits['id'] = recruits['id'].astype(str)
    recruits['rating'] = pd.to_numeric(recruits['rating'], errors='coerce')

    r = roster[roster['position'].isin(DB_POSITIONS)].copy()
    r['rid'] = r['recruitIds'].map(first_recruit_id)
    r = r.merge(recruits.rename(columns={'id': 'rid'}), on='rid', how='left')

    teams = pd.read_csv(TEAMS)
    r['team_id'] = r['team'].map({v: k for k, v in zip(teams['id'],
                                                       teams['location'])})
    if 'teamId' in r.columns:
        r['team_id'] = r['team_id'].fillna(
            pd.to_numeric(r['teamId'], errors='coerce'))
    r['season'] = pd.to_numeric(r['season'], errors='coerce')
    # NOT dropna on rating. Membership and grading are two different jobs and
    # this line used to do both at once: a man with no recruiting record was
    # dropped before selection, so no room could hold him whatever he did.
    # 35% of defensive backs on a 2026 roster have no composite, and 22% of
    # those with 30 or more tackles - Miami's Zechariah Poyser, second on that
    # team in tackles and carrying a full share of its saved yards, lost his
    # place to men with 16 and 11. The burden falls hardest on G5 rosters,
    # whose players are less likely to be in the 247 database at all, so the
    # old rule quietly thinned exactly the rooms it should have been reading.
    r = r.dropna(subset=['team_id', 'season'])
    r['team_id'] = r['team_id'].astype(int)
    r['season'] = r['season'].astype(int)
    r['pid'] = r['id'].astype(str)

    # A man who has played outranks a man who has not, whatever they were
    # graded out of high school. Ranking on the recruiting composite alone put
    # Notre Dame's Leonard Moore eleventh in his own 2026 room - a 0.8940
    # recruit who led that secondary in passes defensed - behind three true
    # freshmen who have never taken a snap. The grade is frozen at signing; the
    # production is not.
    #
    # Prior-season production decides the order among players who have any, and
    # the recruiting grade orders the rest - unrated men sort last within a tie,
    # so a graded freshman still outranks an ungraded one when neither played.
    if rated_only:
        r = r[r['rating'].notna()]
    r = r.merge(prior_production(), on=['pid', 'season'], how='left')
    r['played'] = r['prod_events'].notna().astype(int)
    r['prod_events'] = r['prod_events'].fillna(0.0)
    keys = (['rating'] if order == 'rating'
            else ['played', 'prod_events', 'rating'])
    return (r.sort_values(keys, ascending=False)
             .groupby(['team_id', 'season']).head(size)
             [['team_id', 'season', 'pid', 'rating', 'played']])


def secondary_room(size=ROOM_SIZE):
    """Mean recruiting grade of the top five defensive backs.

    Selected among rated players only, so this is a mean of five grades rather
    than of however many of the five happen to be in the 247 database. The men
    with no grade are not being judged badly here - they are simply graded
    somewhere else, by their coverage carry, which is the number that knows
    what they actually did.
    """
    top = room_members(size, rated_only=True, order='rating')
    out = top.groupby(['team_id', 'season'], as_index=False).agg(
        DB_rating_top=('rating', 'mean'),      # skips NaN
        DB_n=('rating', 'count'),              # count is rated members only
        DB_size=('rating', 'size'),
        DB_played=('played', 'sum'))
    out['DB_unrated'] = out['DB_size'] - out['DB_n']
    out.loc[out['DB_n'] < MIN_GRADED, 'DB_rating_top'] = np.nan
    return out.drop(columns=['DB_size'])


def coverage_carry(size=ROOM_SIZE, path=None):
    """What this room's five men saved against expectation last season.

    defensive_production.coverage_value shares a team's yards-saved-per-dropback
    out among the five defensive backs who tackled most, and the share travels
    with the man. Summing those shares over the room gives a team-season figure
    that is high when a secondary returns men off a defence that gave up little,
    zero when it returns nobody, and - unlike anything else in this module -
    responds to who actually left.

    Preseason-safe by construction: the production is last season's, the roster
    publishes before this one. So this belongs to the season it is computed for
    and must NOT be lagged again alongside the measured play.
    """
    path = path or os.path.join(_HERE, 'results', 'defensive_production.csv')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['team_id', 'season', 'cov_carry',
                                     'cov_n'])
    p = pd.read_csv(path, low_memory=False)
    if 'cov_yards_value' not in p.columns:
        return pd.DataFrame(columns=['team_id', 'season', 'cov_carry',
                                     'cov_n'])
    p = p[p['group'] == 'DB'][['pid', 'season', 'cov_yards_value']].copy()
    p['pid'] = p['pid'].astype(str)
    p['season'] += 1                        # carried into the season it informs
    m = room_members(size).merge(p, on=['pid', 'season'], how='left')
    m['cov_yards_value'] = m['cov_yards_value'].fillna(0.0)
    return m.groupby(['team_id', 'season'], as_index=False).agg(
        cov_carry=('cov_yards_value', 'sum'),
        cov_n=('cov_yards_value', lambda s: int((s != 0).sum())))


# Measured at 0.00 / 0.15 / 0.30 / 0.45 / 0.60 and monotonically decreasing -
# 0.326, 0.326, 0.325, 0.324, 0.323 - so career tackles buy nothing at team
# level. The grid is kept short enough to notice if that ever changes.
#
# I do not have a verified reason why. The obvious one - that a room's tackle
# total barely varies between teams - is wrong on checking: across 2026 rooms
# the coefficient of variation is 0.54 for tackles against 0.53 for ball
# events, effectively identical. A likelier story is that tackles carry two
# opposed signals at team level, since a defence on the field more often
# records more of them, but that is untested and stated as a guess.
#
# What the result does NOT say is that tackles are uninformative about a
# PLAYER. They plainly are - carrying them moves Ty Benefield from 36th to 7th
# on a player list and takes safeties from 2 of the top 25 to 4. This fit
# scores team-seasons and cannot adjudicate that question either way.
CAREER_TKL_SHARES = (0.0, 0.15, 0.30)


def career_production(path=None, tkl_share=0.0, g5_discount=0.0):
    # g5_discount subtracts a fixed number of ball events from every season a
    # man spent outside a power conference. The size is not a guess: 53 players
    # who moved from G5 to power lost 1.08 ball events while 936 who stayed in
    # G5 gained 0.71, a difference of -1.79 with se 0.58, t -3.07. Against a G5
    # mean near 4.4 that is a 40% overstatement. Conventional schedule strength
    # finds none of this - average opponent pass EPA correlates +0.008 with
    # player ball events - so whatever the tier line captures, it is not the
    # quality of the offences faced. Partly it is role: the same movers lose
    # 7.4 tackles, so they are playing less on deeper rosters. Both readings
    # argue for the discount on a cross-team measure; neither tells us the mix.
    """Career production, judged against a player's own class.

    Reading only last season throws away that a man did it twice, and comparing
    a junior to seniors understates him - seniors have simply had longer to
    accumulate. So this sums production over every prior season, then
    standardises WITHIN class year and season, so a junior is measured against
    juniors.

    TWO KINDS OF PRODUCTION, AND WHY THERE IS NO POSITION TERM

    Ball events - passes defensed and interceptions - are a corner's job.
    Counting only those buries safeties: they are 34% of the population but
    take 8% of the top 25, averaging 48 tackles against a corner's 37 and one
    fewer ball event. Boise State's Ty Benefield had 105 tackles in 2025 and
    scored nothing for any of them.

    The obvious repair is to standardise within position, and it is a trap.
    The CB/S label is a REPORTING CONVENTION, not a fact about the player: 48
    programs label every defensive back 'DB' and 48 label none of them that
    way, and a man's label never changes across his career (100% consistent
    over 1,590 checks, 0% of ambiguous cases recoverable from his own history).
    Adjusting only where the label exists would treat Notre Dame's secondary
    differently from Miami's for reasons of paperwork, in a rating whose whole
    job is comparing teams.

    So no position term. Instead both kinds of production are carried, each
    standardised in its own right, and tkl_share - how much of the blend is
    tackles - is chosen by the same fit that chooses everything else. A safety
    earns through tackles and a corner through ball events, and no label is
    needed. The cost is that tackles are partly a playing-time proxy, so the
    two cases pull against each other: weight them and Benefield rises while
    Moore, who had 31 tackles because nobody threw at him, falls. The fit
    adjudicates rather than me.

    What that surfaces: Notre Dame's Leonard Moore enters 2026 first of 168
    juniors at +4.79, further clear of second than second is of fifteenth.
    Nothing else in this module can see him - his counting stats are ordinary
    because offences stopped throwing at him, and his coverage credit is a team
    share identical to his four team-mates'. Only two other defensive backs
    since 2016 produced at his level as both a freshman and a sophomore, and
    one of them went 50th in the 2026 draft.

    Preseason-safe: strictly prior seasons, and class year comes from the
    recruiting class rather than the roster field, which carries a calendar
    season on a quarter of its rows. See class_year.py.
    """
    path = path or os.path.join(_HERE, 'results', 'defensive_production.csv')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['pid', 'season', 'z_car'])
    d = pd.read_csv(path, low_memory=False)
    d = d[d['group'] == 'DB'].copy()
    d['pid'] = d['pid'].astype(str)
    d['ball'] = d['pd_best'].fillna(0) + 2 * d['intercept'].fillna(0)
    d['tkl'] = d['tot_box'].fillna(0)
    # Tackles for loss and sacks: a defensive back has to make one of
    # these, where a tackle is largely a proxy for time on the field.
    d['disrupt'] = (d['tfl_box'].fillna(0)
                    + 2 * d['sack_box'].fillna(0))
    if g5_discount:
        T = pd.read_csv(TALENT, low_memory=False)[
            ['team_id', 'season', 'conference']].drop_duplicates(
            ['team_id', 'season'])
        T['team'] = T['team_id'].map(TEAM_NAME)
        T['power'] = power_series(T)
        d = d.merge(T[['team_id', 'season', 'power']],
                    on=['team_id', 'season'], how='left')
        d['ball'] = np.where(d['power'].fillna(False).astype(bool), d['ball'],
                             np.maximum(d['ball'] - g5_discount, 0.0))
    out = d.groupby(['pid', 'season'], as_index=False)[
        ['ball', 'tkl', 'disrupt']].sum()
    out = out.sort_values(['pid', 'season'])
    out['car_ball'] = decay_sum(out, 'ball')
    out['car_tkl'] = decay_sum(out, 'tkl')
    out['car_disrupt'] = decay_sum(out, 'disrupt')
    out['season'] += 1

    try:
        from class_year import class_years
        cy = class_years()
        cy['pid'] = cy['pid'].astype(str)
        out = out.merge(cy[['pid', 'season', 'class_yr']],
                        on=['pid', 'season'], how='left')
    except Exception:
        out['class_yr'] = np.nan
    # within class where we know it, within season where we do not
    key = class_key(out)
    for src, dst in (('car_ball', 'z_ball'), ('car_tkl', 'z_tkl'),
                     ('car_disrupt', 'z_disrupt')):
        g = out.groupby(key, dropna=False)[src]
        out[dst] = (((out[src] - g.transform('mean'))
                     / g.transform('std').replace(0, np.nan)).fillna(0.0))
    # Three-way blend. CAREER_BALL/TKL/DISRUPT default to the shipped
    # ball-only figure; set them to test another mix.
    if CAREER_DISRUPT or CAREER_BALL != 1.0:
        tot = CAREER_BALL + CAREER_TKL + CAREER_DISRUPT
        out['z_car'] = ((CAREER_BALL * out['z_ball']
                         + CAREER_TKL * out['z_tkl']
                         + CAREER_DISRUPT * out['z_disrupt']) / tot)
    else:
        out['z_car'] = ((1 - tkl_share) * out['z_ball']
                        + tkl_share * out['z_tkl'])
    return out[['pid', 'season', 'car_ball', 'car_tkl', 'z_ball',
                'z_tkl', 'z_disrupt', 'z_car']]


def career_room(size=ROOM_SIZE, tkl_share=0.0, members=None, career=None):
    """Class-adjusted career production the room brings in, summed.

    members and career are accepted so a sweep over tkl_share can reuse one
    room and one career table instead of rebuilding both each time.
    """
    c = career_production(tkl_share=tkl_share) if career is None else career
    if not len(c):
        return pd.DataFrame(columns=['team_id', 'season', 'car_sum'])
    if career is not None:
        # Same rule as career_production, or a three-way blend set
        # there is thrown away here and the caller sees no change.
        if CAREER_DISRUPT or CAREER_BALL != 1.0:
            tot = CAREER_BALL + CAREER_TKL + CAREER_DISRUPT
            c = c.assign(z_car=(CAREER_BALL * c['z_ball']
                                + CAREER_TKL * c['z_tkl']
                                + CAREER_DISRUPT * c['z_disrupt'])
                         / tot)
        else:
            c = c.assign(z_car=(1 - tkl_share) * c['z_ball']
                         + tkl_share * c['z_tkl'])
    m = (room_members(size) if members is None else members).merge(
        c[['pid', 'season', 'z_car']], on=['pid', 'season'], how='left')
    m['z_car'] = m['z_car'].fillna(0.0)
    return m.groupby(['team_id', 'season'], as_index=False).agg(
        car_sum=('z_car', 'sum'))


def prior_production(path=None, career=True):
    """Production a player brings into a season, stamped onto that season.

    Shifted forward a year so the 2026 room is ordered by what these players
    did through 2025, which is knowable before 2026 is played.

    career=True accumulates every prior season rather than reading only the
    last one. Ordering a room, that is the better claim: a man with two years
    behind him is likelier to be on the field than one with a single good
    season, and reading only last year forgets that Notre Dame's Leonard Moore
    produced as a freshman as well as a sophomore. Raw here, not adjusted for
    class - inside one team a senior having done more than a freshman is the
    point, not a bias. The class adjustment belongs to career_production(),
    which compares men across teams.

    Known gap: a man who misses a season entirely has no row for the season
    after it, so an injury year erases his career here rather than carrying it
    across. Filling that forward is untested and is left alone deliberately.
    """
    path = path or os.path.join(_HERE, 'results', 'defensive_production.csv')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['pid', 'season', 'prod_events'])
    d = pd.read_csv(path, low_memory=False)
    d = d[d['group'] == 'DB']
    d['pid'] = d['pid'].astype(str)
    # tackles carry the playing-time signal, coverage events the skill one
    d['prod_events'] = (d['coverage_events'].fillna(0)
                        + 0.1 * d['tot_box'].fillna(0))
    out = d.groupby(['pid', 'season'], as_index=False)['prod_events'].sum()
    if career:
        out = out.sort_values(['pid', 'season'])
        out['prod_events'] = out.groupby('pid')['prod_events'].cumsum()
    out['season'] += 1
    return out


def zscore(df, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        g = out.groupby('season')[c]
        s = g.transform('std').replace(0, np.nan)
        out[f'z_{c}'] = (out[c] - g.transform('mean')) / s
        if c in INVERTED:
            out[f'z_{c}'] *= -1
    return out


def residualize(d, cols, against):
    """Strip out whatever the front seven already explains.

    Each measure is regressed on the front-seven rating within season and
    replaced by its residual, so what remains is the part of pass defence the
    pass rush does not account for. Without this the secondary rating is largely
    a second reading of the front.
    """
    out = d.copy()
    for c in cols:
        if c not in out.columns:
            continue
        res = pd.Series(np.nan, index=out.index)
        for _, idx in out.groupby('season').groups.items():
            g = out.loc[idx]
            m = g[[c, against]].dropna()
            if len(m) < 30:
                res.loc[idx] = g[c] - g[c].mean()
                continue
            A = np.column_stack([np.ones(len(m)), m[against].to_numpy()])
            b, *_ = np.linalg.lstsq(A, m[c].to_numpy(), rcond=None)
            pred = b[0] + b[1] * g[against]
            res.loc[idx] = g[c] - pred
        out[f'r_{c}'] = res
    return out


def fit_weights(d, parts, target, prefix='r_z_'):
    cols = [f'{prefix}{p}' for p in parts if f'{prefix}{p}' in d.columns]
    x = d.dropna(subset=cols + [target])
    if len(x) < 100:
        return {c: 1.0 / max(len(cols), 1) for c in cols}, np.nan, len(x)
    y = x[target].to_numpy(float)
    y = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(x))] + [x[c].to_numpy(float) for c in cols])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    w = np.clip(b[1:], 0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(len(cols), 1.0 / len(cols))
    return dict(zip(cols, w)), r2, len(x)


def blend(d, wc, wb, cover_share, rec_share):
    cov = sum(d[c] * w for c, w in wc.items())
    ball = sum(d[c] * w for c, w in wb.items())
    play = cover_share * cov + (1 - cover_share) * ball
    out = d.copy()
    out['coverage_z'] = cov
    out['ball_skills'] = ball
    out['db_play'] = play
    rec = d['z_DB_rating_top']
    out['db_rating'] = np.where(rec.notna(),
                                (1 - rec_share) * play + rec_share * rec,
                                play)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-season', type=int, default=None)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'defensive_backs.csv'))
    args = ap.parse_args()

    global TEAM_NAME
    _t = pd.read_csv(TEAMS)
    TEAM_NAME = dict(zip(_t['id'], _t['location']))

    H = pd.read_csv(HAVOC, low_memory=False)
    S = pd.read_csv(SEASONS, low_memory=False)
    T = pd.read_csv(TALENT, low_memory=False)
    # f7_play, not f7_rating. The full front-seven rating is 40% recruiting, and
    # a program that recruits linemen well recruits defensive backs well, so
    # residualizing against it strips out program quality alongside pass rush
    # and leaves the secondary looking emptier than it is. The play component is
    # what the pass rush actually did.
    F = pd.read_csv(FRONT7, low_memory=False)[
        ['team_id', 'season', 'f7_play', 'f7_rating']]

    scols = ['team_id', 'season'] + [c for c in COVER_PARTS if c in S.columns]
    if 'adjusted_epa_per_play_def' in S.columns:
        scols.append('adjusted_epa_per_play_def')
    d = H.merge(S[scols], on=['team_id', 'season'], how='inner')
    tcols = [c for c in ('team_id', 'season', 'conference') if c in T.columns]
    d = d.merge(T[tcols].drop_duplicates(['team_id', 'season']),
                on=['team_id', 'season'], how='left')
    d = d.merge(secondary_room(), on=['team_id', 'season'], how='left')
    d = d.merge(F, on=['team_id', 'season'], how='left')

    d = zscore(d, list(COVER_PARTS) + list(BALL_PARTS) + ['DB_rating_top'])
    zc = [f'z_{p}' for p in COVER_PARTS] + [f'z_{p}' for p in BALL_PARTS]
    d = residualize(d, zc, 'f7_play')

    TGT = 'adjusted_epa_per_play_def'
    # residualize() writes r_z_<name>, and fit_weights prepends 'r_z_', so the
    # raw part names go in here - not the z_ ones
    wc, r2c, nc = fit_weights(d, COVER_PARTS, TGT)
    wb, r2b, nb_ = fit_weights(d, BALL_PARTS, TGT)
    print("### weights fitted on what the front seven does NOT explain ###")
    print(f"  coverage    (R2 {r2c:.3f}, n={nc:,})")
    for c, w in sorted(wc.items(), key=lambda x: -x[1]):
        print(f"    {c[4:]:<36}{w:>6.3f}")
    print(f"  ball skills (R2 {r2b:.3f}, n={nb_:,})")
    for c, w in sorted(wb.items(), key=lambda x: -x[1]):
        print(f"    {c[4:]:<36}{w:>6.3f}")

    # The mix is chosen on the job the rating actually does: standing before a
    # season and saying how that secondary will play. Written out,
    #
    #   proj(S) = (1-rec) * [ (1-cov-car) * db_play(S-1)
    #                         + cov * z_cov_carry(S) + car * z_career(S) ]
    #             + rec * z_recruit(S)
    #
    # scored against what the defence did in S. Note where the two player terms
    # sit: both are built from production through S-1 carried by the S roster,
    # so they are ALREADY preseason quantities for S. Lagging them alongside the
    # measured play would make them two years stale, which measurably costs
    # (holdout 0.212 that way against 0.242 this way).
    carry = coverage_carry()
    room = secondary_room()
    # one room and one career table, reused across every tackle share
    _mem, _car = room_members(), career_production()
    careers = {i: career_room(tkl_share=t, members=_mem, career=_car)
               .rename(columns={'car_sum': f'car_sum_{i}'})
               for i, t in enumerate(CAREER_TKL_SHARES)}

    def projection_frame(cs):
        b = blend(d, wc, wb, cs, 0.0)
        prior = b[['team_id', 'season', 'db_play']].copy()
        prior['season'] += 1
        prior = prior.rename(columns={'db_play': 'prior_play'})
        y = room[['team_id', 'season']].merge(
            d[['team_id', 'season', 'conference', TGT]],
            on=['team_id', 'season'], how='outer')
        y = y.merge(prior, on=['team_id', 'season'], how='left')
        y = y.merge(carry, on=['team_id', 'season'], how='left')
        y = y.merge(room, on=['team_id', 'season'], how='left')
        for i in careers:
            y = y.merge(careers[i], on=['team_id', 'season'], how='left')
        g = y.groupby('season')
        pairs = [('cov_carry', 'z_cov_carry'), ('DB_rating_top', 'zrec')]
        pairs += [(f'car_sum_{i}', f'z_career_{i}') for i in careers]
        for src, dst in pairs:
            y[dst] = g[src].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
                if s.std(ddof=0) else np.nan)
        # a room returning nobody carries nothing, which is the signal, not a gap
        y['z_cov_carry'] = y['z_cov_carry'].fillna(0.0)
        for i in careers:
            y[f'z_career_{i}'] = y[f'z_career_{i}'].fillna(0.0)
        # the tier rule needs the team name, not just the id - Notre Dame is
        # told apart from the other independents by name
        y['team'] = y['team_id'].map(TEAM_NAME)
        y['power'] = power_series(y)
        return y

    # Five shares now, so the loop is ~300,000 evaluations. Doing that with a
    # DataFrame copy each time takes hours; pulling the columns out to arrays
    # once per cover share makes it seconds, and the arithmetic is identical.
    # Every tackle share is merged into the same frame, so varying it costs a
    # different column rather than a rebuild.
    def arrays(cs, cache={}):
        if cs in cache:
            return cache[cs]
        y = projection_frame(cs)
        y = y[y[TGT].notna()]
        a = dict(
            prior=y['prior_play'].to_numpy(float),
            cov=y['z_cov_carry'].to_numpy(float),
            car={i: y[f'z_career_{i}'].to_numpy(float) for i in careers},
            rec=y['zrec'].to_numpy(float),
            tgt=y[TGT].to_numpy(float),
            p4=y['power'].to_numpy(bool))
        cache[cs] = a
        return a

    def score(cs, cov, car, rc, ti):
        a = arrays(cs)
        cz = a['car'][ti]
        play = (1 - cov - car) * a['prior'] + cov * a['cov'] + car * cz
        play = np.where(np.isnan(play), cov * a['cov'] + car * cz, play)
        has = ~np.isnan(a['rec'])
        p = np.where(has, (1 - rc) * play + rc * np.nan_to_num(a['rec']), play)
        ok = ~np.isnan(p) & ~np.isnan(a['tgt'])
        rs, n = [], int(ok.sum())
        for m in (a['p4'] & ok, ~a['p4'] & ok):
            if m.sum() >= 100:
                rs.append(np.corrcoef(p[m], a['tgt'][m])[0, 1])
        return (float(np.mean(rs)) if rs else np.nan), n

    grid = []
    for cs in np.arange(0.0, 1.01, 0.05):
        for cov in np.arange(0.0, 0.81, 0.05):
            for car in np.arange(0.0, 0.51, 0.05):
                if cov + car > 0.9:
                    continue
                for ti in careers:
                    for rc in np.arange(0.0, 0.75, 0.05):
                        r, n = score(cs, cov, car, rc, ti)
                        if not np.isnan(r):
                            grid.append((r, cs, cov, car, rc, n, ti))
    top = max(grid)

    # The peak of this surface is flat and it sits at a high recruiting share,
    # which is the one input that cannot respond to anything that happens after
    # signing day. Rather than spend the last thousandth of fit on it, take the
    # standard error of a correlation at this sample size,
    #
    #     se = (1 - r^2) / sqrt(n)     ~ 0.020 at n = 2,054
    #
    # and among every mix within one of those of the best, keep the one that
    # leans least on recruiting. Two reasons beyond the statistics. This rating
    # exists to be adjusted when a corner goes down, and a rating that is mostly
    # frozen high-school grades barely moves when he does. And the mixes this
    # rule prefers are the ones that agree with the published preseason room
    # rankings - against Phil Steele's 2026 top 68, Spearman +0.751 here against
    # +0.685 at the unconstrained peak, for 0.016 of correlation.
    # Half a standard error, not a whole one - see front_seven.py, where a full
    # band chose a mix that was worse on both criteria at once.
    se = 0.5 * (1 - top[0] ** 2) / np.sqrt(top[5])
    close = [g for g in grid if g[0] >= top[0] - se]
    r, cover_share, cov_share, car_share, rec_share, nb2, tidx = min(
        close, key=lambda g: (round(g[4], 4), -g[0]))
    tkl_share = CAREER_TKL_SHARES[tidx]
    no_carry, _ = score(cover_share, 0.0, car_share, rec_share, tidx)
    no_car, _ = score(cover_share, cov_share, 0.0, rec_share, tidx)
    print("\n### mix chosen by predicting the season it stands before ###")
    print("  scored within conference tier, then averaged")
    print(f"  coverage share of measured play   {cover_share:.2f}")
    print(f"  ball skills share                 {1 - cover_share:.2f}")
    print(f"  carried coverage yards, of play   {cov_share:.2f}")
    print(f"  career production, of play        {car_share:.2f}")
    print(f"  last season's play, of play       "
          f"{1 - cov_share - car_share:.2f}")
    print(f"  tackles' share of career          {tkl_share:.2f}"
          f"   (ball events {1 - tkl_share:.2f})")
    print(f"  recruiting share of rating        {rec_share:.2f}")
    print(f"  within-tier correlation           {r:.3f}   (n={nb2:,})")
    print(f"  without the carried yards         {no_carry:.3f}   "
          f"({r - no_carry:+.3f})")
    print(f"  without career production         {no_car:.3f}   "
          f"({r - no_car:+.3f})")
    print(f"  unconstrained peak                {top[0]:.3f}  at rec "
          f"{top[4]:.2f}, given up to keep recruiting low (1 se = {se:.3f})")
    print("\n  what each tackle share is worth, holding the rest fixed:")
    for i, t in enumerate(CAREER_TKL_SHARES):
        v, _ = score(cover_share, cov_share, car_share, rec_share, i)
        mark = '  <- chosen' if i == tidx else ''
        print(f"    tackles {t:.2f}   {v:.3f}{mark}")

    career = careers[tidx].rename(columns={f'car_sum_{tidx}': 'car_sum'})

    d = blend(d, wc, wb, cover_share, rec_share)

    proj = d[['team_id', 'season', 'coverage_z', 'ball_skills',
              'db_play']].copy()
    proj['season'] += 1
    proj = proj.rename(columns={'coverage_z': 'proj_coverage',
                                'ball_skills': 'proj_ball_skills',
                                'db_play': 'proj_prior_play'})
    d = d.merge(proj, on=['team_id', 'season'], how='outer')
    d = d.merge(carry, on=['team_id', 'season'], how='left')
    d = d.merge(career, on=['team_id', 'season'], how='left')

    # The frame above is built by an inner join on havoc and season summaries,
    # both of which stop at the last played season, so the room and conference
    # merged onto seasons that exist there. The outer merge just created rows
    # for the season being projected, and they arrived with no room attached -
    # which left proj_db_rating equal to proj_db_play for every team and quietly
    # dropped recruiting, 70% of the rating, out of the projection entirely.
    # Backfill the room for those rows and re-standardise across the whole
    # frame, so the projected season is scored on the same scale as the rest.
    fill = d[['team_id', 'season']].merge(room, on=['team_id', 'season'],
                                          how='left')
    d['DB_rating_top'] = d['DB_rating_top'].fillna(
        pd.Series(fill['DB_rating_top'].to_numpy(), index=d.index))
    # the carry needs no backfill here: it is merged onto d after the outer
    # merge above, so the projected season already has it
    if 'conference' in d.columns:
        conf = T[['team_id', 'season', 'conference']].drop_duplicates(
            ['team_id', 'season'])
        cf = d[['team_id', 'season']].merge(conf, on=['team_id', 'season'],
                                            how='left')
        d['conference'] = d['conference'].fillna(
            pd.Series(cf['conference'].to_numpy(), index=d.index))
    g = d.groupby('season')['DB_rating_top']
    d['z_DB_rating_top'] = ((d['DB_rating_top'] - g.transform('mean'))
                            / g.transform('std').replace(0, np.nan))
    for src, dst in (('cov_carry', 'z_cov_carry'), ('car_sum', 'z_career')):
        gg = d.groupby('season')[src]
        d[dst] = ((d[src] - gg.transform('mean'))
                  / gg.transform('std').replace(0, np.nan)).fillna(0.0)

    # last season's measured play, this season's carried credit and the career
    # the room brings in, mixed as the sweep chose. A team with no measured
    # prior season - a new arrival, or the first season in the file - still has
    # a room, so it is projected off the player terms rather than dropped.
    d['proj_db_play'] = ((1 - cov_share - car_share) * d['proj_prior_play']
                         + cov_share * d['z_cov_carry']
                         + car_share * d['z_career'])
    d['proj_db_play'] = d['proj_db_play'].fillna(
        cov_share * d['z_cov_carry'] + car_share * d['z_career'])
    d['proj_db_rating'] = np.where(
        d['z_DB_rating_top'].notna(),
        (1 - rec_share) * d['proj_db_play'] + rec_share * d['z_DB_rating_top'],
        d['proj_db_play'])
    d.loc[d['proj_prior_play'].isna() & (d['cov_n'].fillna(0) == 0),
          'proj_db_rating'] = np.nan

    if args.from_season:
        d = d[d['season'] == args.from_season]
    for c in ('db_rating', 'coverage_z', 'ball_skills', 'proj_db_rating'):
        if c in d.columns:
            d[f'{c}_rank'] = d.groupby('season')[c].rank(ascending=False,
                                                         method='min')
    t = pd.read_csv(TEAMS)
    d['team'] = d['team_id'].map(dict(zip(t['id'], t['location'])))
    d = d.sort_values(['season', 'db_rating'], ascending=[True, False])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(d)} rows, "
          f"{d['db_rating'].notna().sum()} rated)")

    # How much does the secondary add once the front seven is known? This is
    # the number that says whether the split was worth making.
    e = d.dropna(subset=['db_rating', 'f7_rating', TGT])
    y = e[TGT].to_numpy()
    def r2(cols):
        A = np.column_stack([np.ones(len(e))] + [e[c].to_numpy() for c in cols])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        return 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    a, b2, c2 = r2(['f7_rating']), r2(['db_rating']), r2(['f7_rating',
                                                          'db_rating'])
    print(f"\n### does the secondary add anything? (n={len(e):,}) ###")
    print(f"  front seven alone        R2 {a:.3f}")
    print(f"  secondary alone          R2 {b2:.3f}")
    print(f"  both                     R2 {c2:.3f}   "
          f"secondary adds {c2 - a:+.3f}")

    last = int(d.loc[d['db_rating'].notna(), 'season'].max())
    x = d[(d.season == last) & d.db_rating.notna()]
    print(f"\n### {last} best secondaries ###")
    print(f"  {'':4}{'team':<22}{'rating':>8}{'cover':>8}{'ball':>8}{'recruit':>9}")
    for i, (_, row) in enumerate(x.nlargest(12, 'db_rating').iterrows(), 1):
        rec = row.get('z_DB_rating_top', np.nan)
        rs = '-' if pd.isna(rec) else f'{rec:.2f}'
        print(f"  {i:<4}{str(row.team)[:20]:<22}{row.db_rating:>8.2f}"
              f"{row.coverage_z:>8.2f}{row.ball_skills:>8.2f}{rs:>9}")


if __name__ == '__main__':
    main()
