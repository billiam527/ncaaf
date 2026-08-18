#!/usr/bin/env python3
"""One offensive-line rating per team-season, and a projection of the next one.

Two things are combined here: what the line did, from ol_production.py, and who
the line is, from the recruiting grades in talent_by_position.py. Both matter.
Recruiting correlates 0.43 with adjusted line yards and adds R2 +0.061 on top of
last season's blocking, which is far more than it did for receivers (0.02) and
about what it did for backs.

WEIGHTS ARE FITTED, NOT CHOSEN

The six blocking measures are not equally important and guessing their weights
would be the whole rating. Instead they are regressed on adjusted rush EPA - the
thing a line exists to produce - and the standardized coefficients become the
weights. Whatever the line contributes to running the ball, it contributes in
the proportions the data gives.

PASS PROTECTION IS DELIBERATELY LIGHT

Sack rate allowed repeats year over year at 0.27 across 1,986 team pairs, against
0.51 for adjusted line yards. That is close to noise, and it matches the long-
standing finding that sack rate belongs more to the quarterback - how long he
holds it, whether he throws it away - than to the five men blocking. It is in the
rating because pass protection is real, but it is not allowed to dominate, and
the run-blocking and pass-protection halves are also reported separately so the
distinction stays visible.

THE PROJECTION IS MIXED SEPARATELY FROM THE RATING

There are two mixes here, not one, and an earlier version fitted the first and
reused it for the second. ol_rating combines THIS season's blocking with THIS
season's roster; proj_ol_rating combines LAST season's blocking with this
season's roster. Carried blocking is much the weaker of the two - within tier it
predicts next season's offense at 0.25 against 0.49 for the contemporaneous
figure - so the share left over for recruiting should be larger in the
projection, and fitting it on the blend held it down. Tuned on its own problem
the projection's recruiting share is 0.60 rather than 0.30, and it is better on
both criteria: within-tier correlation on the 2023-25 holdout rises 0.230 to
0.252, and seven of the eleven lines Yahoo and CBS agree on land in our top
fifteen instead of four.

The run share is left where the blend fits it. Across the selection band the
projection is indifferent to it - anything from 0.50 to 1.00 sits inside half a
standard error - so there is nothing to choose on and no reason to move it.

WHAT IS NOT HERE

No yards before contact, no time to throw. Both are charted by a human watching
film and neither exists in play-by-play or in CFBD. Line yards is the standard
rules-based stand-in for the first, on gain bands rather than on contact; there
is no stand-in for the second. See ol_production.py.

And no player-carried term. Every other unit has one: the secondary and the
front seven carry production forward on the men who produced it, because those
men appear in the play text. Linemen never do. Two stand-ins were built and both
came back null, measured on 2019-22 and confirmed on 2023-25:

  continuity, how much of last year's line is still on the roster, matched on a
  player id that survives both a new season and a transfer. On its own it is
  -0.055 on the holdout, and blended into carried play at any weight it makes
  the holdout worse. Note the first pass at this looked unstable - continuity
  swinging from 0.73 in 2021 to 0.32 in 2023 - which turned out to be the roster
  file rather than football; see top_room below.

  experience, seasons already on a roster, counted from roster history rather
  than the `year` label, which does not survive a redshirt. Also null: -0.001
  tuning and 0.001 on the holdout. This extends what was already known here.
  Every age DISCOUNT tried made prediction worse; experience does no better as
  its own additive term than it did as a multiplier on the grade.

So the room grade is the only player-level information this unit has, which is
part of why it earns a larger share here than anywhere else.

Usage:
    python ol_projection.py --out results/ol_projection.csv
    python ol_projection.py --from-season 2026
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(_HERE, 'results', 'ol_production.csv')
TALENT = os.path.join(_HERE, 'results', 'talent_by_position.csv')
SEASONS = os.path.join(_HERE, 'results', 'season_summaries.csv')
PLAYER_DIR = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp')
TEAMS = os.path.join(_HERE, '..', 'collect', 'collect_espn_teams', 'temp',
                     'teams.csv')

OL_POSITIONS = {'OL', 'OT', 'OG', 'C', 'G', 'T', 'OC', 'LT', 'RT', 'LG', 'RG'}

# lower is better for these, so they are flipped before standardizing
INVERTED = ('adj_stuff_rate', 'adj_sack_rate_allowed', 'adj_tfl_rate_allowed')

# Every part is the opponent-adjusted column. ol_production writes both a raw
# season mean and an adjusted one, under names that do not match - adj_second_level
# against second_level_yards - and an earlier version of this file asked for the
# raw name for three of the seven parts, which quietly put 47% of the run weight
# and 26% of the pass weight on unadjusted numbers.
RUN_PARTS = ('adj_line_yards', 'adj_stuff_rate', 'adj_opportunity_rate',
             'adj_power_success', 'adj_second_level')
PASS_PARTS = ('adj_sack_rate_allowed', 'adj_tfl_rate_allowed')

MIN_COVERAGE = 0.5
ROOM_SIZE = 5


def first_recruit_id(value):
    """roster.recruitIds arrives as the string form of a list."""
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return str(parsed[0]) if isinstance(parsed, list) and parsed else None


def fbs_mask(df):
    """True where (team_id, season) was an FBS team that season."""
    c = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_classification.csv'),
                    low_memory=False)
    fbs = {int(s): set(g.loc[g['fbs'] == 1, 'team_id'])
           for s, g in c.groupby('season')}
    return pd.Series(
        [int(t) in fbs.get(int(s), set())
         for s, t in zip(df['season'], df['team_id'])], index=df.index)


def top_room(size=ROOM_SIZE):
    """Recruiting grade of the best `size` linemen on the roster.

    Five men block, so the room is the top five. A whole-group mean is a
    different quantity - it rewards a team for carrying twelve well-regarded
    backups over one that starts five good linemen and nothing behind them.
    Depth still matters, but it matters when somebody goes down, which is a
    game-level question and does not belong in a season-level rating.

    The five are the five highest recruiting grades, flat, with no discount for
    class. That looks wrong on the page - a true freshman outranks a developed
    senior, because the 247 Composite is frozen at signing and never ages - and
    it cannot be fixed here. There are no snap counts for linemen in any source
    we hold, so nothing downstream knows who actually played. Every age discount
    tried made prediction worse, not better: flat correlates 0.419 with adjusted
    line yards, a 0.6/0.85/1/1 step 0.407, a geometric 0.85 curve 0.392, and that
    last one promotes a two-star senior over a 0.98 freshman. Without snaps the
    grade is not identifying starters, it is indexing how good the room is, and
    a freshman counts toward that honestly.

    Restricted to FBS. teams.csv carries FCS schools, so any of them on the
    roster file gets an id and reaches this function - and the roster file holds
    77 non-FBS schools in 2022 and 84 in 2025 against two to seven in every other
    season. Since the grade is standardized WITHIN season, a crowd of weak-
    recruiting schools appearing in two seasons only drops the mean in exactly
    those years and inflates every FBS z-score there, by about 0.12 of a standard
    deviation in 2025. That is a season-specific distortion of a quantity that is
    supposed to mean rank against the year, and it also broke the first
    continuity measurement: a man on a 2022 FCS roster who is not on a 2023 FBS
    roster read as a departure.
    """
    roster = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_roster.csv'),
                         low_memory=False)
    recruits = pd.read_csv(os.path.join(PLAYER_DIR, 'cfbd_recruits.csv'),
                           low_memory=False)[['id', 'rating']]
    recruits['id'] = recruits['id'].astype(str)
    recruits['rating'] = pd.to_numeric(recruits['rating'], errors='coerce')

    ol = roster[roster['position'].isin(OL_POSITIONS)].copy()
    ol['rid'] = ol['recruitIds'].map(first_recruit_id)
    ol = ol.merge(recruits.rename(columns={'id': 'rid'}), on='rid', how='left')

    teams = pd.read_csv(TEAMS)
    by_name = {v: k for k, v in zip(teams['id'], teams['location'])}
    ol['team_id'] = ol['team'].map(by_name)
    if 'teamId' in ol.columns:
        ol['team_id'] = ol['team_id'].fillna(
            pd.to_numeric(ol['teamId'], errors='coerce'))
    ol['season'] = pd.to_numeric(ol['season'], errors='coerce')
    ol = ol.dropna(subset=['team_id', 'season', 'rating'])
    ol['team_id'] = ol['team_id'].astype(int)
    ol['season'] = ol['season'].astype(int)
    ol = ol[fbs_mask(ol)]

    ol = ol.sort_values('rating', ascending=False)
    top = ol.groupby(['team_id', 'season']).head(size)
    out = top.groupby(['team_id', 'season'], as_index=False).agg(
        OL5_rating=('rating', 'mean'), OL5_n=('rating', 'size'))
    # fewer than five graded linemen means the mean is of something else
    out.loc[out['OL5_n'] < size, 'OL5_rating'] = np.nan
    return out


def zscore(df, cols):
    """Standardize within season, so a rating means rank against that year."""
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
    """Standardized coefficients on adjusted rush EPA, normalized to sum to 1.

    Negative coefficients are floored at zero: a blocking measure that appears
    to hurt rushing is collinearity talking, not football.
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


def blend(d, wr, wp, run_share, rec_share):
    run = sum(d[c] * w for c, w in wr.items())
    pas = sum(d[c] * w for c, w in wp.items())
    play = run_share * run + (1 - run_share) * pas
    rec = d['z_OL_rating'].fillna(0)
    d = d.copy()
    d['run_block'] = run
    d['pass_protect'] = pas
    d['ol_play'] = play
    d['ol_rating'] = (1 - rec_share) * play + rec_share * rec
    return d


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-season', type=int, default=None,
                    help='project this season from the one before it')
    ap.add_argument('--out', default=os.path.join(
        _HERE, 'results', 'ol_projection.csv'))
    args = ap.parse_args()

    O = pd.read_csv(PROD, low_memory=False)
    T = pd.read_csv(TALENT, low_memory=False)
    keep = [c for c in ('team_id', 'season', 'OL_rating', 'OL_n', 'OL_blue',
                        'coverage', 'conference') if c in T.columns]
    # outer, not left: the season being projected has a roster but no plays yet,
    # and its recruiting is the whole point of projecting it
    d = O.merge(T[keep], on=['team_id', 'season'], how='outer')
    if 'coverage' in d.columns:
        d.loc[d['coverage'] < MIN_COVERAGE, 'OL_rating'] = np.nan

    # the top five, not the whole group
    d = d.merge(top_room(ROOM_SIZE), on=['team_id', 'season'], how='left')
    # The fallback is a DIFFERENT quantity, not a missing value of the same one:
    # a mean over five men against a mean over the whole two-deep, and the second
    # runs low because it includes the bottom of the roster. Filling one with the
    # other and standardizing the result treats them as one column, so every team
    # short of five graded linemen is pushed down the rating for the shape of its
    # roster file rather than the quality of its line. Each is standardized on
    # its own below and the z-scores are combined instead.
    full = d['OL_rating'].copy()
    d['OL_group'] = full
    both = d.dropna(subset=['OL5_rating', 'adj_line_yards'])
    fb = d.dropna(subset=['adj_line_yards'])
    fb = fb[full.reindex(fb.index).notna()]
    print(f"### room: top {ROOM_SIZE} rather than the whole group ###")
    print(f"  top {ROOM_SIZE} mean grade  r {both['OL5_rating'].corr(both['adj_line_yards']):.3f}"
          f"   (n={len(both):,})")
    if len(fb):
        print(f"  whole-group mean    r "
              f"{full.reindex(fb.index).corr(fb['adj_line_yards']):.3f}"
              f"   (n={len(fb):,})")

    S = pd.read_csv(SEASONS, low_memory=False)
    # each half is fitted against the phase it actually serves - run blocking
    # against rushing, pass protection against passing. Fitting protection
    # against rush EPA, as an earlier pass did, just rediscovers that a
    # tackle-for-loss is a running play.
    RUSH_T, PASS_T, ALL_T = ('adjusted_epa_per_rush_off',
                             'adjusted_epa_per_pass_off',
                             'adjusted_epa_per_play_off')
    have = [c for c in (RUSH_T, PASS_T, ALL_T) if c in S.columns]
    d = d.merge(S[['team_id', 'season'] + have], on=['team_id', 'season'],
                how='left')

    d = zscore(d, list(RUN_PARTS) + list(PASS_PARTS)
               + ['OL5_rating', 'OL_group'])
    # top five where we have five, the whole group standardized separately where
    # we do not, so the two scales never meet
    d['z_OL_rating'] = d['z_OL5_rating'].fillna(d['z_OL_group'])
    d['OL_rating'] = d['OL5_rating'].fillna(d['OL_group'])
    nfb = int(d['z_OL5_rating'].isna().sum() - d['z_OL_rating'].isna().sum())
    print(f"  fell back to the whole-group mean on {nfb:,} team-seasons")

    wr, r2r, nr = fit_weights(d, RUN_PARTS, RUSH_T)
    wp, r2p, npp = fit_weights(d, PASS_PARTS, PASS_T)
    print("### weights fitted against the phase each half serves ###")
    print(f"  run blocking  vs rush EPA   (R2 {r2r:.3f}, n={nr:,})")
    for c, w in sorted(wr.items(), key=lambda x: -x[1]):
        print(f"    {c[2:]:<26}{w:>6.3f}")
    print(f"  pass protect  vs pass EPA   (R2 {r2p:.3f}, n={npp:,})")
    for c, w in sorted(wp.items(), key=lambda x: -x[1]):
        print(f"    {c[2:]:<26}{w:>6.3f}")

    # The mix is chosen by predicting next season's OFFENSE, not next season's
    # blocking. Tuning against next year's line yards would be circular - it
    # would hand the run-blocking half the whole rating by construction, which
    # is exactly what happened before this was fixed.
    lag = d[['team_id', 'season', ALL_T]].copy()
    lag['season'] -= 1
    lag = lag.rename(columns={ALL_T: 'next_off'})
    ev = d.merge(lag, on=['team_id', 'season'], how='inner')

    # Scored WITHIN conference tier, then averaged. Pooled scoring hands
    # recruiting far too much weight, because the teams whose rosters link
    # cleanly to recruiting records are 65% power-conference against 36% of the
    # whole set - so a pooled fit rewards recruiting partly for flagging which
    # tier a team is in. Within P4 alone, recruiting predicts next year's
    # offense at 0.397 and blocking at 0.432; pooled, recruiting appeared to
    # win 0.571 to 0.493. Scoring inside each tier removes that shortcut.
    # The tier rule lives in tiers.py. A frozen set here read Notre Dame as G5
    # and the rebuilt 2026 Pac-12 as power, which is the shortcut this scoring
    # exists to close.
    import sys as _s
    _s.path.insert(0, _HERE)
    from tiers import tier_series
    ev = ev.copy()
    if 'team' not in ev.columns:
        _tm = pd.read_csv(TEAMS)
        ev['team'] = ev['team_id'].map(dict(zip(_tm['id'], _tm['location'])))
    ev['tier'] = tier_series(ev)

    def score(rs, rc):
        b = blend(ev, wr, wp, rs, rc).dropna(subset=['ol_rating', 'next_off'])
        rs_ = [g['ol_rating'].corr(g['next_off'])
               for _, g in b.groupby('tier') if len(g) >= 100]
        return (np.mean(rs_) if rs_ else np.nan), len(b)

    best = None
    for rs in np.arange(0.30, 1.01, 0.05):
        for rc in np.arange(0.0, 0.65, 0.05):
            r, n = score(rs, rc)
            if np.isnan(r):
                continue
            if best is None or r > best[0]:
                best = (r, rs, rc, n)
    r, run_share, rec_share, nb = best
    pooled = blend(ev, wr, wp, run_share, rec_share).dropna(
        subset=['ol_rating', 'next_off'])
    print(f"\n### mix chosen by predicting NEXT season's offense ###")
    print(f"  scored within conference tier, then averaged")
    print(f"  run blocking share of play    {run_share:.2f}")
    print(f"  recruiting share of rating    {rec_share:.2f}")
    print(f"  within-tier correlation       {r:.3f}   (n={nb:,})")
    print(f"  pooled, for reference         "
          f"{pooled['ol_rating'].corr(pooled['next_off']):.3f}")

    d = blend(d, wr, wp, run_share, rec_share)

    # The projection carries LAST season's play forward, because that is all
    # we know about the blocking, but pairs it with THIS season's recruiting,
    # because the roster for the season being projected is already known -
    # the transfers have transferred and the freshmen have signed. Carrying
    # the old recruiting forward too would throw that away.
    proj = d[['team_id', 'season', 'run_block', 'pass_protect',
              'ol_play']].copy()
    proj['season'] += 1
    proj = proj.rename(columns={'run_block': 'proj_run_block',
                                'pass_protect': 'proj_pass_protect',
                                'ol_play': 'proj_ol_play'})
    d = d.merge(proj, on=['team_id', 'season'], how='outer')

    # The projection gets its OWN recruiting share, fitted on its own problem.
    # Reusing rec_share from the blend above is a category error: that share was
    # chosen against THIS season's blocking, which is a far stronger signal than
    # the carried figure, so it leaves recruiting less to do than the projection
    # needs. Chosen on 2019-22 and confirmed on 2023-25, never the reverse.
    #
    # The selection band - half a standard error, as next door - runs from 0.35
    # to 0.70, which is wide, so it is worth saying where inside it the published
    # preseason lists sit. Yahoo's top 25 and CBS's top 10 agree on membership
    # and not much else: nine of CBS's ten are in Yahoo's, but their rank
    # correlation on the teams they share is only +0.50. Counting how many of the
    # eleven lines they jointly like land in our top fifteen, the band's lower
    # end gets five and everything from 0.45 up gets seven. The tuning peak falls
    # at 0.55, inside the part both criteria prefer, so there is nothing to trade
    # here and the peak is taken as it stands.
    if 'next_off' not in d.columns:
        d = d.merge(lag, on=['team_id', 'season'], how='left')
    if 'team' not in d.columns:
        _tm2 = pd.read_csv(TEAMS)
        d['team'] = d['team_id'].map(dict(zip(_tm2['id'], _tm2['location'])))
    d['tier'] = tier_series(d)
    tune = d[d['season'].between(2019, 2022)]
    hold = d[d['season'].between(2023, 2025)]

    def proj_score(frame, rc):
        x = frame.assign(_t=(1 - rc) * frame['proj_ol_play']
                         + rc * frame['z_OL_rating'].fillna(0))
        x = x.dropna(subset=['_t', 'next_off', 'tier'])
        rs_ = [g['_t'].corr(g['next_off']) for _, g in x.groupby('tier')
               if len(g) >= 60]
        return (float(np.mean(rs_)) if rs_ else np.nan), len(x)

    cand = [(proj_score(tune, rc)[0], rc) for rc in np.arange(0.0, 0.86, 0.05)]
    cand = [c for c in cand if not np.isnan(c[0])]
    proj_rec, nt = max(cand)[1], proj_score(tune, max(cand)[1])[1]
    print("\n### the projection's own mix, fitted on the carried figure ###")
    print(f"  recruiting share of projection {proj_rec:.2f}"
          f"   (the blend uses {rec_share:.2f})")
    print(f"  tuning 2019-22                 {max(cand)[0]:.3f}  (n={nt:,})")
    print(f"  holdout 2023-25                {proj_score(hold, proj_rec)[0]:.3f}"
          f"   against {proj_score(hold, rec_share)[0]:.3f} at the blend's share")

    d['proj_ol_rating'] = ((1 - proj_rec) * d['proj_ol_play']
                           + proj_rec * d['z_OL_rating'].fillna(0))
    # with no prior play at all there is nothing to carry, so drop the row
    d.loc[d['proj_ol_play'].isna(), 'proj_ol_rating'] = np.nan

    if args.from_season:
        d = d[d['season'] == args.from_season]

    for c in ('ol_rating', 'run_block', 'pass_protect', 'proj_ol_rating'):
        if c in d.columns:
            d[f'{c}_rank'] = d.groupby('season')[c].rank(ascending=False,
                                                         method='min')

    t = pd.read_csv(TEAMS)
    d['team'] = d['team_id'].map(dict(zip(t['id'], t['location'])))
    d = d.sort_values(['season', 'ol_rating'], ascending=[True, False])

    # next_off is season S's OWN offense, joined only to fit the mix above. It
    # must not reach the file: everything here is meant to stand before the
    # season, and a column holding the answer is one careless join away from
    # being used as a feature.
    d = d.drop(columns=[c for c in ('next_off',) if c in d.columns])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d.to_csv(args.out, index=False)
    n = d['ol_rating'].notna().sum()
    print(f"\nwrote {args.out}  ({len(d)} rows, {n} rated)")

    last = int(d.loc[d['ol_rating'].notna(), 'season'].max())
    x = d[(d.season == last) & d.ol_rating.notna()]
    print(f"\n### {last} best lines ###")
    print(f"  {'':4}{'team':<22}{'rating':>8}{'run':>8}{'pass':>8}{'recruit':>9}")
    for i, (_, row) in enumerate(x.nlargest(12, 'ol_rating').iterrows(), 1):
        rec = row.get('z_OL_rating', np.nan)
        print(f"  {i:<4}{str(row.team)[:20]:<22}{row.ol_rating:>8.2f}"
              f"{row.run_block:>8.2f}{row.pass_protect:>8.2f}"
              f"{rec:>9.2f}" if pd.notna(rec) else
              f"  {i:<4}{str(row.team)[:20]:<22}{row.ol_rating:>8.2f}"
              f"{row.run_block:>8.2f}{row.pass_protect:>8.2f}{'-':>9}")


if __name__ == '__main__':
    main()
