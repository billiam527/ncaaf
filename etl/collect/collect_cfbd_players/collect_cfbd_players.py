#!/usr/bin/env python3
"""Collect player-level data from CFBD for returning-production features.

CFBD's own /player/returning endpoint gives a team-level summary only: one
percentPPA and one usage figure per team, split three ways into passing,
rushing and receiving. That is the off-the-shelf product and it is collected
here as a baseline, but it cannot answer "which positions return, and are they
starters or backups" - so the per-player sources are collected alongside it and
the position/tier structure is built in etl/summarize/returning_production.py.

What each source can and cannot support:

  roster              every player with a position, including OL and defense
  player/usage        share of team plays - QB/RB/WR/TE only
  ppa/players/season  per-player PPA - QB/RB/WR/TE only
  stats/player/season season totals by category; 'defensive' reaches DL/LB/DB
  player/returning    team-level baseline to beat
  player/portal       transfers in and out
  draft/picks         departures to the NFL

There is no snap count and no starts field anywhere in the API, so "starter"
has to be derived from production rank within a position group. There is also
no usage or efficiency metric of any kind for offensive linemen - roster
continuity is the only OL signal available.

Team ids are ESPN team ids: all 136 FBS teams in /teams/fbs match
etl/collect/collect_espn_teams/temp/teams.csv exactly, so no name matching is
needed. Player-level endpoints return team names rather than ids, so the name
to id map is written out from /teams for the transform to use.

Usage:
    python collect_cfbd_players.py --start-year 2014 --end-year 2026
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

BASE = 'https://api.collegefootballdata.com'
KEY_FILE = os.path.expanduser('~/.cfbd_api_key')

# Usage and PPA begin in 2014; rosters go back further but are not useful on
# their own. Asking for earlier years returns empty lists, not errors.
FIRST_USABLE_YEAR = 2014

# 'interceptions' is a category of its own - the 'defensive' one
# carries TOT/SOLO/TFL/SACKS/PD/QB HUR but no picks. Leaving it out
# forced defensive_production to parse them from play text, which
# counts penalty-nullified plays and duplicated rows.
STAT_CATEGORIES = ('passing', 'rushing', 'receiving',
                   'defensive', 'interceptions', 'kicking', 'punting')


def load_cfbd_key():
    """CFBD API key from $CFBD_API_KEY or ~/.cfbd_api_key. None if absent."""
    key = os.environ.get('CFBD_API_KEY', '').strip()
    if key:
        return key
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as fh:
            key = fh.read().strip()
        if key:
            return key
    return None


def fetch(path, headers, params, attempts=4):
    """GET one endpoint, retrying on rate limits and transient failures.

    Returns a list of records. Raises after the last attempt rather than
    returning an empty list, so a failed pull cannot be mistaken for a year
    that genuinely has no data.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(BASE + path, headers=headers,
                             params=params, timeout=120)
        except requests.RequestException as exc:
            if attempt == attempts:
                raise
            print(f"    {path} {params}: {type(exc).__name__}, "
                  f"retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
            print(f"    {path} {params}: HTTP {r.status_code}, "
                  f"retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"{path} {params} failed: "
                           f"HTTP {r.status_code} {r.text[:200]}")
    raise RuntimeError(f"{path} {params}: exhausted {attempts} attempts")


def flatten_usage(records):
    """player/usage nests its numbers under a 'usage' dict."""
    rows = []
    for rec in records:
        row = {k: v for k, v in rec.items() if k != 'usage'}
        for k, v in (rec.get('usage') or {}).items():
            row[f'usage_{k}'] = v
        rows.append(row)
    return rows


def flatten_ppa(records):
    """ppa/players/season nests averagePPA and totalPPA dicts."""
    rows = []
    for rec in records:
        row = {k: v for k, v in rec.items()
               if k not in ('averagePPA', 'totalPPA')}
        for prefix in ('averagePPA', 'totalPPA'):
            for k, v in (rec.get(prefix) or {}).items():
                row[f'{prefix}_{k}'] = v
        rows.append(row)
    return rows


def collect_player_games(year, headers, pause=0.3):
    """Per-game player PPA, which carries the opponent faced.

    The season-level PPA endpoint gives one figure per player with no way to
    tell a soft schedule from a hard one. This one names the opponent for every
    game, which is what an opponent adjustment needs.

    The endpoint requires a week when no team is given, so a season is pulled a
    week at a time. Note there is no play-count field: a player who took three
    snaps in a game appears the same as one who took sixty, so the adjustment
    downstream has to lean on season usage to weight players instead.
    """
    frames = []
    for season_type in ('regular', 'postseason'):
        weeks = range(1, 17) if season_type == 'regular' else range(1, 6)
        for week in weeks:
            recs = fetch('/ppa/players/games', headers,
                         {'year': year, 'week': week, 'seasonType': season_type})
            if not recs:
                continue
            rows = []
            for r in recs:
                row = {k: v for k, v in r.items() if k != 'averagePPA'}
                for k, v in (r.get('averagePPA') or {}).items():
                    row[f'ppa_{k}'] = v
                rows.append(row)
            frames.append(pd.DataFrame(rows))
            time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_game_box(year, headers, pause=0.3):
    """Per-game box scores, which is season stats with the opponent attached.

    /stats/player/season gives a defender's nine sacks as one number. This
    gives the same nine split by game, so each one can be weighted by the line
    it came against. Shape is game -> team -> category -> statType -> athlete,
    flattened here to one row per athlete-stat with both sides of the fixture
    named, because the opponent is the whole point of collecting it.

    A week at a time, like the PPA pull: the endpoint wants a week when no team
    is given.
    """
    rows = []
    for season_type in ('regular', 'postseason'):
        weeks = range(1, 17) if season_type == 'regular' else range(1, 6)
        for week in weeks:
            recs = fetch('/games/players', headers,
                         {'year': year, 'week': week,
                          'seasonType': season_type})
            if not recs:
                continue
            for game in recs:
                teams = game.get('teams') or []
                sides = [t.get('team') for t in teams]
                for t in teams:
                    opp = next((s for s in sides if s != t.get('team')), None)
                    for cat in (t.get('categories') or []):
                        for typ in (cat.get('types') or []):
                            for a in (typ.get('athletes') or []):
                                rows.append({
                                    'season': year,
                                    'week': week,
                                    'seasonType': season_type,
                                    'game_id': game.get('id'),
                                    'team': t.get('team'),
                                    'opponent': opp,
                                    'homeAway': t.get('homeAway'),
                                    'category': cat.get('name'),
                                    'statType': typ.get('name'),
                                    'playerId': a.get('id'),
                                    'player': a.get('name'),
                                    'stat': a.get('stat'),
                                })
            time.sleep(pause)
    return pd.DataFrame(rows)


def collect_year(year, headers):
    """Every per-year source for one season, as {name: DataFrame}."""
    out = {}

    out['roster'] = pd.DataFrame(fetch('/roster', headers, {'year': year}))
    out['usage'] = pd.DataFrame(
        flatten_usage(fetch('/player/usage', headers, {'year': year})))
    out['ppa'] = pd.DataFrame(
        flatten_ppa(fetch('/ppa/players/season', headers, {'year': year})))
    out['returning'] = pd.DataFrame(
        fetch('/player/returning', headers, {'year': year}))
    out['portal'] = pd.DataFrame(fetch('/player/portal', headers, {'year': year}))
    out['draft'] = pd.DataFrame(fetch('/draft/picks', headers, {'year': year}))

    stats = []
    for category in STAT_CATEGORIES:
        recs = fetch('/stats/player/season', headers,
                     {'year': year, 'category': category})
        stats.extend(recs)
    out['stats'] = pd.DataFrame(stats)

    # roster carries no season of its own
    for name, frame in out.items():
        if not frame.empty and 'season' not in frame.columns:
            frame.insert(0, 'season', year)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start-year', type=int, default=FIRST_USABLE_YEAR)
    ap.add_argument('--end-year', type=int, default=2026)
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'temp'))
    ap.add_argument('--pause', type=float, default=0.4,
                    help='seconds between requests')
    ap.add_argument('--player-games', action='store_true',
                    help='also pull per-game player PPA (about 20 requests per '
                         'season; needed for the opponent adjustment)')
    ap.add_argument('--only-player-games', action='store_true',
                    help='pull only the per-game PPA, leaving other files alone')
    ap.add_argument('--game-box', action='store_true',
                    help='also pull per-game box scores (about 20 requests per '
                         'season; gives every stat an opponent)')
    ap.add_argument('--only-game-box', action='store_true',
                    help='pull only the per-game box scores')
    args = ap.parse_args()

    key = load_cfbd_key()
    if not key:
        print("ERROR: no CFBD API key.")
        print(f"   Set CFBD_API_KEY, or put the key in {KEY_FILE}")
        print("   Get one from https://collegefootballdata.com/key")
        sys.exit(1)
    headers = {'Authorization': f'Bearer {key}'}

    if args.start_year < FIRST_USABLE_YEAR:
        print(f"NOTE: player usage and PPA begin in {FIRST_USABLE_YEAR}; "
              f"earlier years will come back empty")

    os.makedirs(args.out_dir, exist_ok=True)

    # The team map is what lets the transform turn CFBD team names back into
    # ESPN team ids. Collected once rather than per year.
    # /teams rather than /teams/fbs: the player endpoints return FCS teams too,
    # and mapping only the 136 FBS schools leaves 40% of rows without an id.
    teams = pd.DataFrame(fetch('/teams', headers, {'year': args.end_year}))
    if 'location' in teams.columns:
        venues = pd.json_normalize(teams['location']).add_prefix('venue_')
        teams = pd.concat([teams.drop(columns=['location']), venues], axis=1)
    teams = teams.drop(columns=[c for c in ('logos', 'alternateNames')
                                if c in teams.columns])
    teams.to_csv(os.path.join(args.out_dir, 'cfbd_teams.csv'), index=False)
    print(f"teams: {len(teams)} FBS teams -> cfbd_teams.csv")

    collected = {}
    for year in range(args.start_year, args.end_year + 1):
        print(f"\n{year}")
        if not args.only_player_games and not args.only_game_box:
            try:
                year_data = collect_year(year, headers)
            except RuntimeError as exc:
                print(f"  FAILED: {exc}")
                raise
            for name, frame in year_data.items():
                print(f"  {name:<10} {len(frame):>7} rows")
                collected.setdefault(name, []).append(frame)
        if args.player_games or args.only_player_games:
            pg = collect_player_games(year, headers, args.pause)
            print(f"  {'player_gm':<10} {len(pg):>7} rows"
                  f"  ({pg['opponent'].nunique() if len(pg) else 0} opponents)")
            if len(pg):
                collected.setdefault('player_games', []).append(pg)
        if args.game_box or args.only_game_box:
            gb = collect_game_box(year, headers, args.pause)
            games = gb['game_id'].nunique() if len(gb) else 0
            print(f"  {'game_box':<10} {len(gb):>7} rows  ({games} games)")
            if len(gb):
                collected.setdefault('game_stats', []).append(gb)
        time.sleep(args.pause)

    print()
    for name, frames in collected.items():
        combined = pd.concat(frames, ignore_index=True)
        path = os.path.join(args.out_dir, f'cfbd_{name}.csv')
        combined.to_csv(path, index=False)
        seasons = sorted(combined['season'].dropna().unique()) \
            if 'season' in combined.columns else []
        print(f"wrote {os.path.basename(path):<22} {len(combined):>8} rows"
              f"  seasons {int(min(seasons))}-{int(max(seasons))}"
              if seasons else
              f"wrote {os.path.basename(path):<22} {len(combined):>8} rows")


if __name__ == '__main__':
    main()
