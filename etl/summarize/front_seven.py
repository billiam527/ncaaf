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
P4 = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12'}


def first_recruit_id(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return str(parsed[0]) if isinstance(parsed, list) and parsed else None


def front_seven_room():
    """Recruiting grade of the top four linemen and top three linebackers.

    Seven men play the front, so the room is seven. As on the offensive line
    there is no snap count for any of them, so this ranks on the high-school
    composite and cannot know who actually starts - a grade frozen at signing.
    The same test done for the offensive line applies: age-discounting the
    grade measured worse, not better, so it is left flat.
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
    r = r.dropna(subset=['team_id', 'season', 'rating'])
    r['team_id'] = r['team_id'].astype(int)
    r['season'] = r['season'].astype(int)
    r = r.sort_values('rating', ascending=False)

    # Requiring a full four AND a full three left only 28% of team-seasons with
    # a grade, because the two gates compound. Each group needs two graded
    # players and the mean is taken over however many of the top n are actually
    # graded, which lifts coverage to roughly two thirds.
    MIN_GRADED = 2
    out = []
    for pos, n, tag in ((DL_POSITIONS, N_DL, 'DL'), (LB_POSITIONS, N_LB, 'LB')):
        g = r[r['position'].isin(pos)]
        top = g.groupby(['team_id', 'season']).head(n)
        agg = top.groupby(['team_id', 'season'], as_index=False).agg(
            **{f'{tag}_rating': ('rating', 'mean'),
               f'{tag}_n': ('rating', 'size')})
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
    lag = d[['team_id', 'season', TGT]].copy()
    lag['season'] -= 1
    lag = lag.rename(columns={TGT: 'next_def'})
    ev = d.merge(lag, on=['team_id', 'season'], how='inner')
    ev['tier'] = np.where(ev.get('conference',
                                 pd.Series(index=ev.index)).isin(P4),
                          'P4', 'G5')

    def score(hs, rc):
        b = blend(ev, wh, wr, hs, rc).dropna(subset=['f7_rating', 'next_def'])
        rs = [g['f7_rating'].corr(g['next_def'])
              for _, g in b.groupby('tier') if len(g) >= 100]
        return (np.mean(rs) if rs else np.nan), len(b)

    best = None
    for hs in np.arange(0.0, 1.01, 0.05):
        for rc in np.arange(0.0, 0.65, 0.05):
            r, n = score(hs, rc)
            if np.isnan(r):
                continue
            if best is None or r > best[0]:
                best = (r, hs, rc, n)
    r, havoc_share, rec_share, nb = best
    print("\n### mix chosen by predicting NEXT season's defence ###")
    print("  scored within conference tier, then averaged")
    print(f"  havoc share of play           {havoc_share:.2f}")
    print(f"  run stopping share            {1 - havoc_share:.2f}")
    print(f"  recruiting share of rating    {rec_share:.2f}")
    print(f"  within-tier correlation       {r:.3f}   (n={nb:,})")

    d = blend(d, wh, wr, havoc_share, rec_share)

    proj = d[['team_id', 'season', 'havoc', 'run_stop', 'f7_play']].copy()
    proj['season'] += 1
    proj = proj.rename(columns={'havoc': 'proj_havoc',
                                'run_stop': 'proj_run_stop',
                                'f7_play': 'proj_f7_play'})
    d = d.merge(proj, on=['team_id', 'season'], how='outer')
    d['proj_f7_rating'] = np.where(
        d['z_F7_rating'].notna(),
        (1 - rec_share) * d['proj_f7_play'] + rec_share * d['z_F7_rating'],
        d['proj_f7_play'])
    d.loc[d['proj_f7_play'].isna(), 'proj_f7_rating'] = np.nan

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
