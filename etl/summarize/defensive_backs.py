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


def secondary_room(size=ROOM_SIZE):
    """Recruiting grade of the top five defensive backs.

    Five because nickel is the base defence in this era, not four. As everywhere
    else on this side of the ball there are no snap counts, so this ranks on the
    high-school composite and cannot know who starts.
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

    top = r.sort_values('rating', ascending=False).groupby(
        ['team_id', 'season']).head(size)
    out = top.groupby(['team_id', 'season'], as_index=False).agg(
        DB_rating_top=('rating', 'mean'), DB_n=('rating', 'size'))
    out.loc[out['DB_n'] < MIN_GRADED, 'DB_rating_top'] = np.nan
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

    lag = d[['team_id', 'season', TGT]].copy()
    lag['season'] -= 1
    lag = lag.rename(columns={TGT: 'next_def'})
    ev = d.merge(lag, on=['team_id', 'season'], how='inner')
    ev['tier'] = np.where(ev.get('conference',
                                 pd.Series(index=ev.index)).isin(P4),
                          'P4', 'G5')

    def score(cs, rc):
        b = blend(ev, wc, wb, cs, rc).dropna(subset=['db_rating', 'next_def'])
        rs = [g['db_rating'].corr(g['next_def'])
              for _, g in b.groupby('tier') if len(g) >= 100]
        return (np.mean(rs) if rs else np.nan), len(b)

    best = None
    for cs in np.arange(0.0, 1.01, 0.05):
        for rc in np.arange(0.0, 0.75, 0.05):
            r, n = score(cs, rc)
            if np.isnan(r):
                continue
            if best is None or r > best[0]:
                best = (r, cs, rc, n)
    r, cover_share, rec_share, nb2 = best
    print("\n### mix chosen by predicting NEXT season's defence ###")
    print("  scored within conference tier, then averaged")
    print(f"  coverage share of play        {cover_share:.2f}")
    print(f"  ball skills share             {1 - cover_share:.2f}")
    print(f"  recruiting share of rating    {rec_share:.2f}")
    print(f"  within-tier correlation       {r:.3f}   (n={nb2:,})")

    d = blend(d, wc, wb, cover_share, rec_share)

    proj = d[['team_id', 'season', 'coverage_z', 'ball_skills',
              'db_play']].copy()
    proj['season'] += 1
    proj = proj.rename(columns={'coverage_z': 'proj_coverage',
                                'ball_skills': 'proj_ball_skills',
                                'db_play': 'proj_db_play'})
    d = d.merge(proj, on=['team_id', 'season'], how='outer')
    d['proj_db_rating'] = np.where(
        d['z_DB_rating_top'].notna(),
        (1 - rec_share) * d['proj_db_play'] + rec_share * d['z_DB_rating_top'],
        d['proj_db_play'])
    d.loc[d['proj_db_play'].isna(), 'proj_db_rating'] = np.nan

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
