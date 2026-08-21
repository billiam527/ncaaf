#!/usr/bin/env python3
"""One front-seven rating per team-season, and a projection of the next one.

WHAT GOES IN

Havoc the front seven creates - sacks and tackles for loss - alongside rush
defence, which is very largely their job, blended with the recruiting grades of
the top four linemen and top three linebackers. Everything is opponent-adjusted
before it gets here: the havoc rates by the play-weighted ridge in
adjust_havoc.py, the rush figures by the ridge in summarize_games.py. A front
seven is therefore measured against the offences it actually faced.

RUSH DEFENCE IS A FRONT-SEVEN JOB - BUT SO IS PASS DEFENCE

Raw correlations do not separate the units at all, because DL, LB and DB grades
move together; a program that recruits one well recruits all three. Rush EPA
allowed correlates -0.446 with front-seven talent and -0.423 with secondary
talent, which says almost nothing. Partial correlations, holding the other unit
fixed, do separate them:

    rush EPA allowed        front seven -0.157   secondary +0.010
    explosive rush allowed  front seven +0.195   secondary -0.089
    pass EPA allowed        front seven -0.220   secondary +0.065

So rush defence is theirs, as expected. The surprise is the third line: the
front seven predicts PASS defence better than it predicts rush defence, and the
secondary's own coefficient is near zero with the wrong sign. Pressure is doing
the work. Pass defence is therefore not handed wholly to the secondary anywhere
in this model, and a defensive_backs.py that assumes otherwise would be wrong.

SACKS ARE THE NOISY HALF OF HAVOC

Year over year on the same defence, across 1,913 pairs, TFL rate repeats at
0.462 and sack rate at 0.196. Sacks are close to a coin flip, for the same
reason sack rate allowed is on the offensive line: a sack needs a quarterback
who holds the ball. Both are in the rating because both are real, but the
weights are fitted rather than assumed, and sacks earn less than TFLs.

Usage:
    python front_seven.py --out results/front_seven.csv
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
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

DL_POSITIONS = {'DL', 'DE', 'DT', 'NT', 'EDGE', 'NG'}
LB_POSITIONS = {'LB', 'ILB', 'OLB', 'MLB', 'WLB', 'SLB'}
N_DL, N_LB = 4, 3

# THE DEFENSIVE COLUMNS DO NOT SHARE A SIGN CONVENTION. Verified against points
# allowed across 2,154 team-seasons, not inferred from the column names:
#
#   adjusted_epa_per_*_def        r -0.78 / -0.67 / -0.69   higher = BETTER
#   adjusted_*_success_def        r +0.63 / +0.55 / +0.60   higher = WORSE
#   adjusted_*_yards_per_play_def r +0.74 / +0.67 / +0.64   higher = WORSE
#   adjusted_explosive_*_def      r +0.61 / +0.56 / +0.52   higher = WORSE
#   adj_sack_rate, adj_tfl_rate   r -0.36 / -0.25           higher = BETTER
#
# The three EPA columns are stored as defensive credit; everything else in
# season_summaries is stored as allowed. Assuming one convention for all of them
# is what put Air Force, on 30.3 points a game, top of this table and Ohio State,
# on 9.3, near the bottom. Anything added here must be checked the same way.
INVERTED = ('adjusted_rush_success_def', 'adjusted_explosive_rush_rate_def')

HAVOC_PARTS = ('adj_tfl_rate', 'adj_sack_rate')
RUSH_PARTS = ('adjusted_epa_per_rush_def', 'adjusted_rush_success_def',
              'adjusted_explosive_rush_rate_def')

MIN_COVERAGE = 0.5
# The tier rule lives in tiers.py, not in a frozen set here.
from tiers import tier_series  # noqa: E402



# Recency and class handling for the career figure. A flat sum treats a season
# three years ago as fully as last season, which put a man whose last year was
# two coverage events at the top of the file. CAREER_DECAY=1.0 restores the
# flat behaviour.
CAREER_DECAY = 0.5
CAREER_CLASS_CAP = 4        # fifth and sixth-year men count as seniors


# Opponent adjustment for the career term, off unless asked for. The team
# metrics have always been opponent-adjusted; the player term never was, which
# let a sack against an FCS line weigh what one against Texas weighed.
# Walked forward 2019-2025 this costs +0.0138 MAE (t +2.05, worse in 6 of 7
# seasons) - a tenth of a percent, accepted knowingly for ratings that survive
# being looked at. F7_OPP_ADJ=0 restores the raw career term.
OPP_ADJ = os.environ.get('F7_OPP_ADJ', '1') == '1'
OPP_PATH = os.path.join(_HERE, 'results', 'player_opponent_adjust.csv')


def _opponent_adjust(out):
    """Swap raw pressure for the figure weighted by the lines it came against.

    A player-season with no adjusted row keeps its raw value rather than
    becoming a zero - the adjustment is a reweighting, not a filter.
    """
    if not OPP_ADJ:
        return out
    if not os.path.exists(OPP_PATH):
        print(f"  F7_OPP_ADJ set but {OPP_PATH} is missing; using raw")
        return out
    a = pd.read_csv(OPP_PATH)
    a['pid'] = a['pid'].astype(str).str.replace(r'\.0$', '', regex=True)
    out = out.merge(a[['season', 'pid', 'prs_adj']],
                    on=['season', 'pid'], how='left')
    hit = out['prs_adj'].notna().sum()
    print(f"  opponent-adjusted career: {hit:,} of {len(out):,} "
          f"player-seasons matched")
    out['prs'] = out['prs_adj'].fillna(out['prs'])
    return out.drop(columns=['prs_adj'])


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


PRODUCTION = os.path.join(_HERE, 'results', 'defensive_production.csv')


def prior_production(path=None, career=True):
    """Career production a front-seven player brings into a season.

    Sacks, hurries and tackles for loss are what this unit is for, so the key
    is pressure_events plus a light weight on total tackles for playing time -
    the mirror of the secondary's key, with pressure where coverage sits there.
    """
    path = path or PRODUCTION
    if not os.path.exists(path):
        return pd.DataFrame(columns=['pid', 'season', 'prod_events'])
    d = pd.read_csv(path, low_memory=False)
    d = d[d['group'] == 'FRONT'].copy()
    d['pid'] = d['pid'].astype(str)
    d['prod_events'] = (d['pressure_events'].fillna(0)
                        + 0.1 * d['tot_box'].fillna(0))
    out = d.groupby(['pid', 'season'], as_index=False)['prod_events'].sum()
    if career:
        out = out.sort_values(['pid', 'season'])
        out['prod_events'] = out.groupby('pid')['prod_events'].cumsum()
    out['season'] += 1
    return out


def career_production(path=None):
    """Career pressure production, judged against a player's own class.

    Same construction as the secondary's: cumulate over prior seasons, then
    standardise within class year so a junior is measured against juniors
    rather than against seniors who have had longer to accumulate.
    """
    path = path or PRODUCTION
    if not os.path.exists(path):
        return pd.DataFrame(columns=['pid', 'season', 'z_car'])
    d = pd.read_csv(path, low_memory=False)
    d = d[d['group'] == 'FRONT'].copy()
    d['pid'] = d['pid'].astype(str)
    d['prs'] = (d['sack_best'].fillna(0) + 0.5 * d['tfl_box'].fillna(0)
                + 0.25 * d['hurry_best'].fillna(0))
    out = d.groupby(['pid', 'season'], as_index=False)['prs'].sum()
    out = _opponent_adjust(out)
    out = out.sort_values(['pid', 'season'])
    out['car_prs'] = decay_sum(out, 'prs')
    out['season'] += 1
    try:
        import sys as _s
        _s.path.insert(0, _HERE)
        from class_year import class_years
        cy = class_years()
        cy['pid'] = cy['pid'].astype(str)
        out = out.merge(cy[['pid', 'season', 'class_yr']],
                        on=['pid', 'season'], how='left')
    except Exception:
        out['class_yr'] = np.nan
    key = class_key(out)
    g = out.groupby(key, dropna=False)['car_prs']
    out['z_car'] = (((out['car_prs'] - g.transform('mean'))
                     / g.transform('std').replace(0, np.nan)).fillna(0.0))
    return out[['pid', 'season', 'car_prs', 'z_car']]


def room_members(rated_only=False, order='production'):
    """The seven men, one row each - four linemen and three backers.

    TWO THINGS THIS USED TO GET WRONG

    It ranked on the high-school composite alone, which is frozen at signing,
    so Ohio State's entire 2026 linebacker room was three men with zero career
    production between them while Christian Alliegro - 29.9 career production
    and four sacks, but a 0.8550 recruit - was not in it. Notre Dame's line
    held two men who have never played while Boubacar Traore, 6.5 sacks, was
    left out. Ordering by career production instead replaces 60% of the places
    in this file.

    And it dropped every player with no recruiting record before selecting,
    which is 37% of front-seven players and 32% of those with 20 or more
    tackles. Membership and grading are different jobs: rated_only=False picks
    who is on the field, rated_only=True picks the best rated men for the
    recruiting grade, which stays a mean of real grades rather than of however
    many of the seven happen to be in the 247 database.

    ORDER MATTERS THE SAME WAY, AND THIS WAS NOT OBVIOUS

    Two consumers want two different rooms, and giving both the production
    ordering measured worse than giving both the recruiting one:

        grade by    terms by      predicts   vs Steele's DL rooms
        rating      rating           0.508          +0.687
        rating      production       0.518          +0.700
        production  rating           0.500          +0.623
        production  production       0.513          +0.650

    The recruiting grade is asking how much talent a room holds, which the most
    talented men answer. The carried pressure and career terms are asking who
    is on the field, which the most productive answer. So order='rating' for
    the grade and order='production' for the player terms - the same split as
    rated_only, one level down.
    """
    roster = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_roster.csv'),
                         low_memory=False)
    recruits = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_recruits.csv'),
                           low_memory=False)[['id', 'rating']]
    recruits['id'] = recruits['id'].astype(str)
    recruits['rating'] = pd.to_numeric(recruits['rating'], errors='coerce')

    r = roster[roster['position'].isin(DL_POSITIONS | LB_POSITIONS)].copy()
    r['rid'] = r['recruitIds'].map(first_recruit_id)
    r = r.merge(recruits.rename(columns={'id': 'rid'}), on='rid', how='left')

    teams = pd.read_csv(TEAMS)
    r['team_id'] = r['team'].map({v: k for k, v in zip(teams['id'],
                                                       teams['location'])})
    if 'teamId' in r.columns:
        r['team_id'] = r['team_id'].fillna(
            pd.to_numeric(r['teamId'], errors='coerce'))
    r['season'] = pd.to_numeric(r['season'], errors='coerce')
    r = r.dropna(subset=['team_id', 'season'])
    if rated_only:
        r = r[r['rating'].notna()]
    r['team_id'] = r['team_id'].astype(int)
    r['season'] = r['season'].astype(int)
    r['pid'] = r['id'].astype(str)
    r = r.merge(prior_production(), on=['pid', 'season'], how='left')
    r['played'] = r['prod_events'].notna().astype(int)
    r['prod_events'] = r['prod_events'].fillna(0.0)
    r['grp'] = np.where(r['position'].isin(DL_POSITIONS), 'DL', 'LB')
    keys = (['rating'] if order == 'rating'
            else ['played', 'prod_events', 'rating'])
    # under 'production', career production decides the order and the composite
    # breaks ties among men who have never played; unrated men sort last
    r = r.sort_values(keys, ascending=False)
    out = []
    for pos, n in ((DL_POSITIONS, N_DL), (LB_POSITIONS, N_LB)):
        g = r[r['position'].isin(pos)]
        out.append(g.groupby(['team_id', 'season']).head(n))
    return pd.concat(out, ignore_index=True)[
        ['team_id', 'season', 'pid', 'rating', 'played', 'prod_events', 'grp']]


def front_seven_room():
    """Recruiting grade of the room, weighted four linemen to three backers.

    Ordered by the grade itself: this number asks how much talent a room holds.
    """
    top = room_members(rated_only=True, order='rating')
    # Requiring a full four AND a full three left only 28% of team-seasons with
    # a grade, because the two gates compound. Each group needs two graded
    # players and the mean is taken over however many of the top n are actually
    # graded, which lifts coverage to roughly two thirds.
    MIN_GRADED = 2
    out = []
    for tag, n in (('DL', N_DL), ('LB', N_LB)):
        g = top[top['grp'] == tag]
        agg = g.groupby(['team_id', 'season'], as_index=False).agg(
            **{f'{tag}_rating': ('rating', 'mean'),
               f'{tag}_n': ('rating', 'count')})
        agg.loc[agg[f'{tag}_n'] < MIN_GRADED, f'{tag}_rating'] = np.nan
        out.append(agg)
    m = out[0].merge(out[1], on=['team_id', 'season'], how='outer')
    # the front is four linemen and three backers, so weight the mean that way,
    # falling back to whichever group is present when only one is
    dl, lb = m['DL_rating'], m['LB_rating']
    wd = np.where(dl.notna(), N_DL, 0.0)
    wl = np.where(lb.notna(), N_LB, 0.0)
    tot = wd + wl
    m['F7_rating'] = np.where(
        tot > 0, (np.nan_to_num(dl) * wd + np.nan_to_num(lb) * wl)
        / np.where(tot > 0, tot, 1), np.nan)
    m['F7_n'] = m['DL_n'].fillna(0) + m['LB_n'].fillna(0)
    return m


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


def fit_weights(d, parts, target):
    """Standardized coefficients on the target, normalized to sum to one.

    The target is adjusted_epa_per_play_def, which is already oriented so that
    higher means a better defence - see the note on INVERTED. Every z_ column
    reaching here is oriented the same way, so honest coefficients are positive
    and the clip at zero means what it is meant to: drop a measure that only
    looks harmful because it is collinear with a better one.
    """
    cols = [f'z_{p}' for p in parts if f'z_{p}' in d.columns]
    x = d.dropna(subset=cols + [target])
    if len(x) < 100:
        return {c: 1.0 / len(cols) for c in cols}, np.nan, len(x)
    y = x[target].to_numpy(float)
    y = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(x))] + [x[c].to_numpy(float) for c in cols])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    w = np.clip(b[1:], 0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(len(cols), 1.0 / len(cols))
    return dict(zip(cols, w)), r2, len(x)


def pressure_carry(path=None):
    """What this room's men were worth as pass rushers last season.

    defensive_production.prs_value is a player's share of his team's pressure,
    computed there and until now consumed by nothing. Summing it over the room
    gives a team-season figure that responds to who actually left, which no
    team rate does.

    Preseason-safe: last season's production on this season's roster, so it
    belongs to the season it is computed for and must not be lagged again.
    """
    path = path or PRODUCTION
    if not os.path.exists(path):
        return pd.DataFrame(columns=['team_id', 'season', 'prs_carry'])
    p = pd.read_csv(path, low_memory=False)
    if 'prs_value' not in p.columns:
        return pd.DataFrame(columns=['team_id', 'season', 'prs_carry'])
    p = p[p['group'] == 'FRONT'][['pid', 'season', 'prs_value']].copy()
    p['pid'] = p['pid'].astype(str)
    p['season'] += 1
    m = room_members().merge(p, on=['pid', 'season'], how='left')
    m['prs_value'] = m['prs_value'].fillna(0.0)
    return m.groupby(['team_id', 'season'], as_index=False).agg(
        prs_carry=('prs_value', 'sum'))

# Walked forward, summing the raw career figure instead of the class-relative
# one is a testable alternative rather than an opinion: standardising within
# class year measurably costs prediction on the player leaderboard, so the
# model's career term is worth the same test. Default off; the switch exists so
# the result stays reproducible either way.
_RAW_CAREER = os.environ.get('F7_CAREER_RAW', '0') == '1'


def _career_col(c):
    """Which column career_room sums. z_car unless the switch is set."""
    if not _RAW_CAREER:
        return 'z_car'
    for cand in ('car_prs', 'car_prs', 'car_ball', 'car'):
        if cand in c.columns:
            return cand
    return 'z_car'


def career_room():
    """Class-adjusted career pressure production the room brings in."""
    c = career_production()
    if not len(c):
        return pd.DataFrame(columns=['team_id', 'season', 'car_sum'])
    col = _career_col(c)
    m = room_members().merge(c[['pid', 'season', col]],
                             on=['pid', 'season'], how='left')
    m[col] = m[col].fillna(0.0)
    return m.groupby(['team_id', 'season'], as_index=False).agg(
        car_sum=(col, 'sum'))


def blend(d, wh, wr, havoc_share, rec_share):
    hav = sum(d[c] * w for c, w in wh.items())
    rsh = sum(d[c] * w for c, w in wr.items())
    play = havoc_share * hav + (1 - havoc_share) * rsh
    out = d.copy()
    out['havoc'] = hav
    out['run_stop'] = rsh
    out['f7_play'] = play
    # A team with no recruiting grade keeps its play rating untouched. Filling
    # the missing grade with zero instead would shrink a third of the table
    # toward the mean and make any recruiting weight look worse than it is -
    # which is exactly why an earlier run picked a recruiting share of zero.
    rec = d['z_F7_rating']
    out['f7_rating'] = np.where(rec.notna(),
                                (1 - rec_share) * play + rec_share * rec,
                                play)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-season', type=int, default=None)
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'front_seven.csv'))
    args = ap.parse_args()

    H = pd.read_csv(HAVOC, low_memory=False)
    S = pd.read_csv(SEASONS, low_memory=False)
    T = pd.read_csv(TALENT, low_memory=False)

    scols = ['team_id', 'season'] + [c for c in RUSH_PARTS if c in S.columns]
    for extra in ('adjusted_epa_per_play_def', 'adjusted_epa_per_pass_def'):
        if extra in S.columns:
            scols.append(extra)
    # inner: a row needs both havoc and season figures to be rated at all, and
    # season_summaries carries team-seasons havoc does not
    d = H.merge(S[scols], on=['team_id', 'season'], how='inner')

    tcols = [c for c in ('team_id', 'season', 'conference', 'coverage')
             if c in T.columns]
    d = d.merge(T[tcols].drop_duplicates(['team_id', 'season']),
                on=['team_id', 'season'], how='left')
    # No roster-coverage gate here. talent_by_position's coverage figure is the
    # share of the WHOLE roster carrying a recruiting rating, and gating on it
    # held this table to 29% of team-seasons even though 70% have a gradeable
    # front. The room carries its own gate - two graded linemen and two graded
    # backers - which is the check that actually matters for this number.
    d = d.merge(front_seven_room(), on=['team_id', 'season'], how='left')

    d = zscore(d, list(HAVOC_PARTS) + list(RUSH_PARTS) + ['F7_rating'])

    # Each half is fitted against overall defensive EPA - what a front seven is
    # ultimately for - rather than against the half it belongs to, which would
    # be circular.
    TGT = 'adjusted_epa_per_play_def'
    wh, r2h, nh = fit_weights(d, HAVOC_PARTS, TGT)
    wr, r2r, nr = fit_weights(d, RUSH_PARTS, TGT)
    print("### weights fitted against adjusted defensive EPA ###")
    print(f"  havoc      (R2 {r2h:.3f}, n={nh:,})")
    for c, w in sorted(wh.items(), key=lambda x: -x[1]):
        print(f"    {c[2:]:<34}{w:>6.3f}")
    print(f"  run stopping (R2 {r2r:.3f}, n={nr:,})")
    for c, w in sorted(wr.items(), key=lambda x: -x[1]):
        print(f"    {c[2:]:<34}{w:>6.3f}")

    # Shares chosen by predicting NEXT season's defence, scored within
    # conference tier and averaged - pooled scoring lets recruiting be paid for
    # flagging which tier a team is in, which it did on the offensive line.
    # Scored the way the secondary is: on the projection standing before a
    # season, written out, rather than on the rating correlated with the next
    # one. The two are the same quantity, but the explicit form is what lets
    # the two player-carried terms in - both are built from production through
    # S-1 on the S roster, so they are ALREADY preseason quantities for S and
    # must not be lagged alongside the measured play.
    #
    #   proj(S) = (1-rec) * [ (1-prs-car) * f7_play(S-1)
    #                         + prs * z_prs_carry(S) + car * z_career(S) ]
    #             + rec * z_recruit(S)
    room = front_seven_room()
    carry = pressure_carry()
    career = career_room()
    _tm = pd.read_csv(TEAMS)
    NAME = dict(zip(_tm['id'], _tm['location']))

    def projection_frame(hs):
        b = blend(d, wh, wr, hs, 0.0)
        prior = b[['team_id', 'season', 'f7_play']].copy()
        prior['season'] += 1
        prior = prior.rename(columns={'f7_play': 'prior_play'})
        y = room[['team_id', 'season']].merge(
            d[['team_id', 'season', 'conference', TGT]],
            on=['team_id', 'season'], how='outer')
        y = y.merge(prior, on=['team_id', 'season'], how='left')
        y = y.merge(carry, on=['team_id', 'season'], how='left')
        y = y.merge(career, on=['team_id', 'season'], how='left')
        y = y.merge(room, on=['team_id', 'season'], how='left')
        g = y.groupby('season')
        for src, dst in (('prs_carry', 'z_prs'), ('car_sum', 'z_car'),
                         ('F7_rating', 'zrec')):
            y[dst] = g[src].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
                if s.std(ddof=0) else np.nan)
        for c in ('z_prs', 'z_car'):
            y[c] = y[c].fillna(0.0)
        y['team'] = y['team_id'].map(NAME)
        y['power'] = tier_series(y) == 'P4'
        return y

    def arrays(hs, cache={}):
        if hs in cache:
            return cache[hs]
        y = projection_frame(hs)
        y = y[y[TGT].notna()]
        cache[hs] = dict(prior=y['prior_play'].to_numpy(float),
                         prs=y['z_prs'].to_numpy(float),
                         car=y['z_car'].to_numpy(float),
                         rec=y['zrec'].to_numpy(float),
                         tgt=y[TGT].to_numpy(float),
                         p4=y['power'].to_numpy(bool))
        return cache[hs]

    def score(hs, ps, cs_, rc):
        a = arrays(hs)
        play = ((1 - ps - cs_) * a['prior'] + ps * a['prs'] + cs_ * a['car'])
        play = np.where(np.isnan(play), ps * a['prs'] + cs_ * a['car'], play)
        has = ~np.isnan(a['rec'])
        p = np.where(has, (1 - rc) * play + rc * np.nan_to_num(a['rec']), play)
        ok = ~np.isnan(p) & ~np.isnan(a['tgt'])
        rs, n = [], int(ok.sum())
        for m in (a['p4'] & ok, ~a['p4'] & ok):
            if m.sum() >= 100:
                rs.append(np.corrcoef(p[m], a['tgt'][m])[0, 1])
        return (float(np.mean(rs)) if rs else np.nan), n

    grid = []
    for hs in np.arange(0.0, 1.01, 0.05):
        for ps in np.arange(0.0, 0.81, 0.05):
            for cs_ in np.arange(0.0, 0.51, 0.05):
                if ps + cs_ > 0.9:
                    continue
                for rc in np.arange(0.0, 0.65, 0.05):
                    v, n = score(hs, ps, cs_, rc)
                    if not np.isnan(v):
                        grid.append((v, hs, ps, cs_, rc, n))
    top = max(grid)
    # Same rule as the secondary: the surface is flat at the top and peaks at a
    # high recruiting share, the one input that cannot respond to anything after
    # signing day, so among mixes close to the best keep the one leaning least
    # on recruiting.
    #
    # HALF a standard error, not a whole one. At a full se this module chose
    # recruiting 0.20 against a peak at 0.40, and that was worse on BOTH
    # criteria - 0.507 against 0.519 predicting, and +0.678 against +0.700 on
    # Steele's defensive-line rooms. A band wide enough to be indifferent
    # between those is too wide to be measuring anything.
    #
    # It is not free next door: the secondary moves from recruiting 0.35 to
    # 0.40, gaining 0.008 of correlation and losing 0.016 of Steele agreement.
    # Narrowing is still right - prediction is the criterion these are chosen
    # on and both modules gain there - but it is a trade, not a free lunch.
    se = 0.5 * (1 - top[0] ** 2) / np.sqrt(top[5])
    close = [g for g in grid if g[0] >= top[0] - se]
    r, havoc_share, prs_share, car_share, rec_share, nb = min(
        close, key=lambda g: (round(g[4], 4), -g[0]))
    no_prs, _ = score(havoc_share, 0.0, car_share, rec_share)
    no_car, _ = score(havoc_share, prs_share, 0.0, rec_share)
    print("\n### mix chosen by predicting the season it stands before ###")
    print("  scored within conference tier, then averaged")
    print(f"  havoc share of measured play  {havoc_share:.2f}")
    print(f"  run stopping share            {1 - havoc_share:.2f}")
    print(f"  carried pressure, of play     {prs_share:.2f}")
    print(f"  career production, of play    {car_share:.2f}")
    print(f"  last season's play, of play   {1 - prs_share - car_share:.2f}")
    print(f"  recruiting share of rating    {rec_share:.2f}")
    print(f"  within-tier correlation       {r:.3f}   (n={nb:,})")
    print(f"  without the carried pressure  {no_prs:.3f}   ({r - no_prs:+.3f})")
    print(f"  without career production     {no_car:.3f}   ({r - no_car:+.3f})")
    print(f"  unconstrained peak            {top[0]:.3f}  at rec {top[4]:.2f}"
          f"  (1 se = {se:.3f})")

    d = blend(d, wh, wr, havoc_share, rec_share)

    proj = d[['team_id', 'season', 'havoc', 'run_stop', 'f7_play']].copy()
    proj['season'] += 1
    proj = proj.rename(columns={'havoc': 'proj_havoc',
                                'run_stop': 'proj_run_stop',
                                'f7_play': 'proj_prior_play'})
    d = d.merge(proj, on=['team_id', 'season'], how='outer')
    d = d.merge(carry, on=['team_id', 'season'], how='left')
    d = d.merge(career, on=['team_id', 'season'], how='left')

    # The frame above comes from an inner join on havoc and season summaries,
    # both of which stop at the last played season, so the room and conference
    # merged only onto seasons that exist there. The outer merge has just
    # created rows for the season being projected and they carry no room, which
    # left proj_f7_rating equal to proj_f7_play for every team and dropped
    # recruiting out of the projection entirely. Backfill the room for those
    # rows and re-standardise across the whole frame.
    fill = d[['team_id', 'season']].merge(room[['team_id', 'season',
                                                'F7_rating']],
                                          on=['team_id', 'season'], how='left')
    d['F7_rating'] = d['F7_rating'].fillna(
        pd.Series(fill['F7_rating'].to_numpy(), index=d.index))
    if 'conference' in d.columns:
        conf = T[['team_id', 'season', 'conference']].drop_duplicates(
            ['team_id', 'season'])
        cf = d[['team_id', 'season']].merge(conf, on=['team_id', 'season'],
                                            how='left')
        d['conference'] = d['conference'].fillna(
            pd.Series(cf['conference'].to_numpy(), index=d.index))
    g = d.groupby('season')['F7_rating']
    d['z_F7_rating'] = ((d['F7_rating'] - g.transform('mean'))
                        / g.transform('std').replace(0, np.nan))
    for src, dst in (('prs_carry', 'z_prs_carry'), ('car_sum', 'z_career')):
        gg = d.groupby('season')[src]
        d[dst] = ((d[src] - gg.transform('mean'))
                  / gg.transform('std').replace(0, np.nan)).fillna(0.0)

    # last season's measured play, the pressure this room carries and the career
    # it brings in, mixed as the sweep chose. A team with no measured prior
    # season still has a room, so it is projected off the player terms.
    d['proj_f7_play'] = ((1 - prs_share - car_share) * d['proj_prior_play']
                         + prs_share * d['z_prs_carry']
                         + car_share * d['z_career'])
    d['proj_f7_play'] = d['proj_f7_play'].fillna(
        prs_share * d['z_prs_carry'] + car_share * d['z_career'])
    d['proj_f7_rating'] = np.where(
        d['z_F7_rating'].notna(),
        (1 - rec_share) * d['proj_f7_play'] + rec_share * d['z_F7_rating'],
        d['proj_f7_play'])
    d.loc[d['proj_prior_play'].isna() & (d['prs_carry'].isna()),
          'proj_f7_rating'] = np.nan

    if args.from_season:
        d = d[d['season'] == args.from_season]

    for c in ('f7_rating', 'havoc', 'run_stop', 'proj_f7_rating'):
        if c in d.columns:
            d[f'{c}_rank'] = d.groupby('season')[c].rank(ascending=False,
                                                         method='min')
    t = pd.read_csv(TEAMS)
    d['team'] = d['team_id'].map(dict(zip(t['id'], t['location'])))
    d = d.sort_values(['season', 'f7_rating'], ascending=[True, False])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(d)} rows, "
          f"{d['f7_rating'].notna().sum()} rated)")

    last = int(d.loc[d['f7_rating'].notna(), 'season'].max())
    x = d[(d.season == last) & d.f7_rating.notna()]
    print(f"\n### {last} best fronts ###")
    print(f"  {'':4}{'team':<22}{'rating':>8}{'havoc':>8}{'run':>8}{'recruit':>9}")
    for i, (_, row) in enumerate(x.nlargest(12, 'f7_rating').iterrows(), 1):
        rec = row.get('z_F7_rating', np.nan)
        rs = '-' if pd.isna(rec) else f'{rec:.2f}'
        print(f"  {i:<4}{str(row.team)[:20]:<22}{row.f7_rating:>8.2f}"
              f"{row.havoc:>8.2f}{row.run_stop:>8.2f}{rs:>9}")


if __name__ == '__main__':
    main()
