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

P4 = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12'}


def first_recruit_id(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return str(parsed[0]) if isinstance(parsed, list) and parsed else None


def room_members(size=ROOM_SIZE):
    """The five men, one row each, with their recruiting grade.

    Five because nickel is the base defence in this era, not four. As everywhere
    else on this side of the ball there are no snap counts, so this cannot know
    who starts.
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
    r = r.dropna(subset=['team_id', 'season', 'rating'])
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
    # the recruiting grade orders the rest. The room's rating is still the mean
    # composite of whoever is picked, so the scale is unchanged - what changes
    # is which five are in it.
    r = r.merge(prior_production(), on=['pid', 'season'], how='left')
    r['played'] = r['prod_events'].notna().astype(int)
    r['prod_events'] = r['prod_events'].fillna(0.0)
    return (r.sort_values(['played', 'prod_events', 'rating'], ascending=False)
             .groupby(['team_id', 'season']).head(size)
             [['team_id', 'season', 'pid', 'rating', 'played']])


def secondary_room(size=ROOM_SIZE):
    """Mean recruiting grade of the top five defensive backs."""
    top = room_members(size)
    out = top.groupby(['team_id', 'season'], as_index=False).agg(
        DB_rating_top=('rating', 'mean'), DB_n=('rating', 'size'),
        DB_played=('played', 'sum'))
    out.loc[out['DB_n'] < MIN_GRADED, 'DB_rating_top'] = np.nan
    return out


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


def prior_production(path=None):
    """Last season's defensive production, stamped onto the season it informs.

    Shifted forward a year so the 2026 room is ordered by what these players
    did in 2025, which is knowable before 2026 is played.
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
    #   proj(S) = (1-rec) * [ (1-cov) * db_play(S-1) + cov * z_cov_carry(S) ]
    #             + rec * z_recruit(S)
    #
    # scored against what the defence did in S. Note where the carried credit
    # sits: it is built from S-1 production carried by the S roster, so it is
    # ALREADY a preseason quantity for S. Lagging it alongside the measured play
    # would make it two years stale, which measurably costs (holdout 0.212 that
    # way against 0.242 this way).
    carry = coverage_carry()
    room = secondary_room()

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
        g = y.groupby('season')
        for src, dst in (('cov_carry', 'z_cov_carry'),
                         ('DB_rating_top', 'zrec')):
            y[dst] = g[src].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0)
                if s.std(ddof=0) else np.nan)
        # a room returning nobody carries nothing, which is the signal, not a gap
        y['z_cov_carry'] = y['z_cov_carry'].fillna(0.0)
        return y

    def project(y, cov, rc):
        play = (1 - cov) * y['prior_play'] + cov * y['z_cov_carry']
        play = play.fillna(cov * y['z_cov_carry'])
        return np.where(y['zrec'].notna(),
                        (1 - rc) * play + rc * y['zrec'], play)

    def score(cs, cov, rc, frame_cache={}):
        if cs not in frame_cache:
            frame_cache[cs] = projection_frame(cs)
        y = frame_cache[cs].copy()
        y['proj'] = project(y, cov, rc)
        y = y.dropna(subset=['proj', TGT])
        y['tier'] = np.where(y['conference'].isin(P4), 'P4', 'G5')
        rs = [g['proj'].corr(g[TGT]) for _, g in y.groupby('tier')
              if len(g) >= 100]
        return (np.mean(rs) if rs else np.nan), len(y)

    grid = []
    for cs in np.arange(0.0, 1.01, 0.05):
        for cov in np.arange(0.0, 0.81, 0.05):
            for rc in np.arange(0.0, 0.75, 0.05):
                r, n = score(cs, cov, rc)
                if not np.isnan(r):
                    grid.append((r, cs, cov, rc, n))
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
    se = (1 - top[0] ** 2) / np.sqrt(top[4])
    close = [g for g in grid if g[0] >= top[0] - se]
    r, cover_share, cov_share, rec_share, nb2 = min(
        close, key=lambda g: (round(g[3], 4), -g[0]))
    base, _ = score(cover_share, 0.0, rec_share)
    print("\n### mix chosen by predicting the season it stands before ###")
    print("  scored within conference tier, then averaged")
    print(f"  coverage share of measured play   {cover_share:.2f}")
    print(f"  ball skills share                 {1 - cover_share:.2f}")
    print(f"  carried coverage yards, of play   {cov_share:.2f}")
    print(f"  recruiting share of rating        {rec_share:.2f}")
    print(f"  within-tier correlation           {r:.3f}   (n={nb2:,})")
    print(f"  same mix without the carry        {base:.3f}   "
          f"({r - base:+.3f})")
    print(f"  unconstrained peak                {top[0]:.3f}  at rec "
          f"{top[3]:.2f}, given up to keep recruiting low (1 se = {se:.3f})")

    d = blend(d, wc, wb, cover_share, rec_share)

    proj = d[['team_id', 'season', 'coverage_z', 'ball_skills',
              'db_play']].copy()
    proj['season'] += 1
    proj = proj.rename(columns={'coverage_z': 'proj_coverage',
                                'ball_skills': 'proj_ball_skills',
                                'db_play': 'proj_prior_play'})
    d = d.merge(proj, on=['team_id', 'season'], how='outer')
    d = d.merge(carry, on=['team_id', 'season'], how='left')

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
    gc = d.groupby('season')['cov_carry']
    d['z_cov_carry'] = ((d['cov_carry'] - gc.transform('mean'))
                        / gc.transform('std').replace(0, np.nan)).fillna(0.0)

    # last season's measured play and this season's carried credit, mixed as the
    # sweep chose. A team with no measured prior season - a new arrival, or the
    # first season in the file - still has a carry and a room, so it is projected
    # off those rather than dropped.
    d['proj_db_play'] = ((1 - cov_share) * d['proj_prior_play']
                         + cov_share * d['z_cov_carry'])
    d['proj_db_play'] = d['proj_db_play'].fillna(
        cov_share * d['z_cov_carry'])
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
