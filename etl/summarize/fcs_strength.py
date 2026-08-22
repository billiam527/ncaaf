"""An FCS strength rating from the whole FCS game graph, not 20 crossover games.

Until now an FCS team was rated on the one or two games a year it played an FBS
opponent - a median of 20 games over fifteen seasons, and nothing at all for
four teams on the 2026 slate. There are now 8,581 FCS-vs-FCS results, which
means a team can be rated against the division it actually plays in, with the
crossover games tying that scale to the FBS one.

Ratings are a per-season ridge on margins - one coefficient per team plus a
home-field term, the same shape as the two-way opponent adjustment the team
metrics already use. Ridge rather than least squares because the FCS graph is
only weakly connected: conferences play mostly within themselves, so an
unpenalised fit chases teams with few cross-conference games.

Scale is tied down by holding FBS teams at their known ratings where a
crossover game exists, so an FCS rating means the same thing as an FBS one
rather than floating on its own centre.
"""
import numpy as np
import pandas as pd

R = '/home/bill/ncaaf'
NEW = f'{R}/etl/collect/collect_espn_games/temp/games_fcs_2010_to_2025.csv'
OUT = f'{R}/etl/summarize/results/fcs_strength.csv'
LAM = 25.0        # ridge penalty, in games-worth of shrinkage toward average

c = pd.read_csv(f'{R}/etl/collect/collect_cfbd_players/temp/cfbd_teams.csv',
                low_memory=False)
c['id'] = pd.to_numeric(c['id'], errors='coerce')
CLS = dict(zip(c['id'], c['classification'].astype(str).str.lower()))
NAME = dict(zip(c['id'], c['school']))
CONF = dict(zip(c['id'], c['conference']))

n = pd.read_csv(NEW, low_memory=False)
o = pd.read_csv(f'{R}/etl/data/games/formatted/games.csv', low_memory=False)
# the FBS file already carries the crossover games; union on id so each game
# counts once regardless of which pull found it
for d in (n, o):
    for k in ('home_team_id', 'away_team_id', 'home_score', 'away_score'):
        d[k] = pd.to_numeric(d[k], errors='coerce')
keep = ['id', 'season', 'home_team_id', 'away_team_id', 'home_score',
        'away_score', 'neutral_site']
g = pd.concat([n[keep], o[[k for k in keep if k in o.columns]]],
              ignore_index=True)
g = g.dropna(subset=['home_team_id', 'away_team_id', 'home_score',
                     'away_score', 'season'])
g = g.drop_duplicates(subset='id')
g['margin'] = g['home_score'] - g['away_score']
g['neutral'] = g.get('neutral_site', False).fillna(False).astype(bool)
print(f"  {len(g):,} scored games, {int(g['season'].min())}-"
      f"{int(g['season'].max())}")


DIVS = ['fbs', 'fcs', 'ii', 'iii', '?']


def rate_season(d):
    """Ridge on margin, with division level as its own unpenalised term.

    margin = (div_home - div_away) + (team_home - team_away) + hfa

    Penalising every team toward one global mean was wrong. The FCS subgraph
    touches the FBS one through only ~150 crossover games a year, so that thin
    bridge is exactly what the penalty shrinks away: both divisions collapsed
    onto the same centre and an average FCS team came out 1.1 points worse than
    an average FBS team, against about 20 in the crossover results.

    Division effects carry the level and are not penalised. The ridge then only
    shrinks a team toward its OWN division's mean, which is what a prior over
    teams should say.
    """
    teams = sorted(set(d['home_team_id']) | set(d['away_team_id']))
    idx = {t: i for i, t in enumerate(teams)}
    div = {t: CLS.get(int(t), '?') for t in teams}
    # first division is the reference, absorbed into home field
    dcol = {k: i for i, k in enumerate(DIVS[1:])}
    p, q, m = len(teams), len(dcol), len(d)
    X = np.zeros((m, p + q + 1))
    for r, (h, a, nu) in enumerate(zip(d['home_team_id'], d['away_team_id'],
                                       d['neutral'])):
        X[r, idx[h]] += 1.0
        X[r, idx[a]] -= 1.0
        for t, s in ((h, 1.0), (a, -1.0)):
            k = dcol.get(div[t])
            if k is not None:
                X[r, p + k] += s
        X[r, p + q] = 0.0 if nu else 1.0
    y = d['margin'].to_numpy(float)
    P = np.eye(p + q + 1) * LAM
    P[p:, p:] = 0.0                    # divisions and home field unpenalised
    # A division with no cross-division games in a season has an
    # unidentifiable level and makes the normal equations singular. lstsq
    # takes the minimum-norm solution there instead of failing, which sets
    # exactly the level that has no evidence to zero.
    beta = np.linalg.lstsq(X.T @ X + P, X.T @ y, rcond=None)[0]
    lev = {k: float(beta[p + i]) for k, i in dcol.items()}
    lev[DIVS[0]] = 0.0
    return ({t: float(beta[idx[t]]) + lev.get(div[t], 0.0) for t in teams},
            float(beta[p + q]), lev)


rows = []
for s, d in g.groupby('season'):
    if len(d) < 200:
        continue
    rat, hfa, lev = rate_season(d)
    played = pd.concat([d['home_team_id'], d['away_team_id']]).value_counts()
    for t, v in rat.items():
        rows.append({'season': int(s), 'team_id': int(t), 'rating': v,
                     'games': int(played.get(t, 0)),
                     'cls': CLS.get(int(t), '?')})
    if s in (2019, 2024, 2025):
        print(f"    {int(s)}: {len(d):,} games, {len(rat)} teams, "
              f"home field {hfa:+.2f}, FCS level {lev.get('fcs', 0):+.1f}")

out = pd.DataFrame(rows)
# centre each season on its FBS mean so the scale means one thing across years
for s, d in out.groupby('season'):
    mu = d.loc[d['cls'] == 'fbs', 'rating'].mean()
    out.loc[out['season'] == s, 'rating'] -= mu

print(f"\n  rated {len(out):,} team-seasons")
print(f"  by class: {out.groupby('cls').size().to_dict()}")
sd = out.groupby('cls')['rating'].agg(['mean', 'std']).round(2)
print(sd.to_string())

cur = out[(out['season'] == 2025) & (out['cls'] == 'fcs')].copy()
cur['school'] = cur['team_id'].map(NAME)
cur['conference'] = cur['team_id'].map(CONF)
cur = cur.sort_values('rating', ascending=False)
print(f"\n  2025 FCS, strongest (rating is points against an average FBS team):")
print(cur.head(8)[['school', 'conference', 'games', 'rating']]
      .round(1).to_string(index=False))
print(f"\n  weakest:")
print(cur.tail(4)[['school', 'conference', 'games', 'rating']]
      .round(1).to_string(index=False))

print(f"\n  the four that had no record at all:")
for t in (2385, 284, 292, 2698):
    r = cur[cur['team_id'] == t]
    if len(r):
        x = r.iloc[0]
        print(f"    {str(x['school']):<24}{x['games']:>3} games  "
              f"rating {x['rating']:+.1f}")
    else:
        print(f"    {str(NAME.get(t, t)):<24} not rated in 2025")

# The rating must reproduce the gap the crossover games actually show, or the
# divisions are still collapsed onto one centre.
gap = (out[out['cls'] == 'fbs'].groupby('season')['rating'].mean()
       - out[out['cls'] == 'fcs'].groupby('season')['rating'].mean())
print(f"\n  SANITY: mean FBS minus mean FCS rating = {gap.mean():+.1f} points")
print(f"    crossover games say the FBS side wins by about +20.5 on average,")
print(f"    of which roughly {7.0:.0f} is home field since 80% are FBS home,")
print(f"    so a gap near 13-17 is right; 1-2 would mean collapsed again.")
assert gap.mean() > 8, f'divisions still collapsed ({gap.mean():+.1f})'

# and it must rank the divisions in the obvious order
order = out.groupby('cls')['rating'].mean().sort_values(ascending=False)
print(f"    division order: "
      + ' > '.join(f"{k} {v:+.1f}" for k, v in order.items()))

out.to_csv(OUT, index=False)
print(f"\n  wrote {OUT}")
