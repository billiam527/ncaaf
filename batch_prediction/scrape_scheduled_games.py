"""Pull the remaining scheduled games from ESPN's scoreboard.

ASK FOR A DATE RANGE, NEVER A SINGLE DATE

The scoreboard endpoint silently caps a single-date query at 25 events and
ignores `limit` entirely while doing it. A Saturday in September has 60 to 90
games, so the schedule this wrote held 471 of them - FBS teams averaged 6.7
scheduled games each against a real 12, and the shortfall was invisible because
every game present was correct.

    dates=20260905&limit=900          25 events
    dates=20260905-20260905&limit=900 68 events   same day, range form
    dates=20260901-20260907&limit=900 91 events

The range form does honour `limit` (at limit=25 it returns 25), so both parts
of the old call were wrong together. Weekly windows over the 2026 season return
905 rows before de-duplication against ~870 real games.

Two consequences of the range form to keep in mind. ESPN snaps a window out to
its own week boundaries, so consecutive windows overlap and the results must be
de-duplicated on game id. And an event's date can no longer be taken from the
query - it comes from the event itself, converted out of UTC, or a Saturday
night kickoff on the west coast lands on Sunday.

NOTE: etl/collect/collect_espn_games/run.py builds the same single-date URL with
`&limit=100` and has the same cap. The historical table it produced is complete
(up to 74 games on a single date), so the cap postdates that collection - but a
re-run today would quietly return a third of each season.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import urllib.request
import json
import pandas as pd

# ESPN groups games by US Eastern calendar date, which is what the historical
# games table stores and what the week labels are built around.
GAME_TZ = ZoneInfo('America/New_York')


def date_list_generation(start_date: str,
                         end_date: str) -> list:
    """
    Generate a list of dates.

    param start_date: (string) starting date in yyyy-mm-dd format
    param end_date: (string) ending date in yyyy-mm-dd format

    return date_list: (list) inclusive list of strings between the start and end date
    """

    assert isinstance(start_date, str), 'start_date must be a string'
    assert isinstance(end_date, str), 'end_date must be a string'

    start_date = datetime.strptime(start_date, '%Y-%m-%d')  # start date
    end_date = datetime.strptime(end_date, '%Y-%m-%d')  # end date

    delta = end_date - start_date  # as timedelta

    date_generated = [start_date + timedelta(days=x) for x in range(0, delta.days)]
    date_list = []
    for i in date_generated:
        year = start_date.year
        # today's date is after Feb but before August,
        # pull from August of current year to Jan of next year
        if datetime(year, 8, 1) >= start_date > datetime(year, 2, 1):
            if datetime.strptime(str(year) + '-08-01', '%Y-%m-%d') < i < \
                    datetime.strptime(str(year + 1) + '-02-01', '%Y-%m-%d'):
                date_list.append(i)
        # if today's date is after August but before Jan,
        # pull from today until end of Jan
        elif datetime(year, 12, 31) >= start_date > datetime(year, 8, 1):
            if start_date < i < datetime.strptime(str(year + 1) + '-02-01', '%Y-%m-%d'):
                date_list.append(i)
        # if today's date is between Jan and Feb,
        # pull from today until end of Mar
        else:
            if start_date < i < datetime.strptime(str(year) + '-02-01', '%Y-%m-%d'):
                date_list.append(i)

    date_strs = []
    for i in date_list:
        date_strs.append(i.strftime('%Y%m%d'))

    return date_strs


def date_windows(date_strs: list, size: int = 7) -> list:
    """Group a list of yyyymmdd strings into 'yyyymmdd-yyyymmdd' windows.

    One request per window instead of per day, because the single-date form of
    the scoreboard endpoint caps at 25 events. Windows are built from the day
    list rather than from the calendar so the season-boundary logic in
    date_list_generation stays the one place that decides what to pull.
    """
    assert isinstance(date_strs, list), 'date_strs must be a list'
    assert size > 0, 'size must be positive'

    return [f"{chunk[0]}-{chunk[-1]}"
            for chunk in (date_strs[i:i + size]
                          for i in range(0, len(date_strs), size))]


def create_urls(prefix, suffix, data):

    """
    Create a list of string URLs using a prefix, suffix, and any data that needs to be looped through.

    param prefix: (str) string for URLs
    param suffix: (str) string to come after looped data
    param data: (list) list of string to be looped through and added to prefix and suffix

    return URLs: (list) list of URL strings
    """

    assert isinstance(prefix, str), 'prefix must be a string'
    assert isinstance(suffix, str), 'suffix must be a string'
    assert isinstance(data, list), 'data must be a list'
    for string in data:
        assert isinstance(string, str), 'each value in data must be a string'

    urls = [prefix + j + suffix for j in data]

    return urls


def retrieve_espn_game_data(url: str) -> dict:

    """
    Retrieve dict of specific game id from ESPN api.

    param url: (str) url from espn api.

    return data: (dict) play by play dictionary.
    """

    assert isinstance(url, str), 'urls must be strings'

    try:
        with urllib.request.urlopen(url) as web_data:
            data = json.loads(web_data.read().decode())

            return data

    except urllib.error.HTTPError as e:
        print('HTTP error', e.code, 'for', url)
    except urllib.error.URLError as e:
        print('URL error', e.reason, 'for', url)

    return None


def transform_espn_ncaaf_game_data(json_data):

    """
    param json_data: (json) data pulled from espn's ncaaf api at the game level

    return df: (pd.DataFrame) data in the relational table form
    """

    # game data
    id = []
    date = []
    week = []
    name = []
    short_name = []
    season = []
    status = []
    venue_id = []
    neutral_site = []

    # team info
    home_team_id = []
    away_team_id = []

    col_names = ['id',
                 'date',
                 'week',
                 'name',
                 'short_name',
                 'season',
                 'status',
                 'venue_id',
                 'neutral_site',
                 'home_team_id',
                 'away_team_id']

    # a failed request or a date with no games both yield an empty frame, so the
    # caller never has to distinguish None from a DataFrame
    if json_data is not None:
        for game in json_data['events']:
            id.append(game['id'])
            date.append(game['date'])
            week.append(game['week']['number'])
            name.append(game['name'])
            short_name.append(game['shortName'])
            season.append(game['season']['year'])
            status.append(game['status']['type']['name'])
            competition = game['competitions'][0]
            try:
                venue_id.append(competition['venue']['id'])
            except KeyError as e:
                # print('Error:', e, 'venue_id', game['shortName'])
                venue_id.append(999)
            neutral_site.append(competition['neutralSite'])
            home_team_id.append(competition['competitors'][0]['id'])
            away_team_id.append(competition['competitors'][1]['id'])

    columns = list(zip(id,
                       date,
                       week,
                       name,
                       short_name,
                       season,
                       status,
                       venue_id,
                       neutral_site,
                       home_team_id,
                       away_team_id))

    df = pd.DataFrame(columns, columns=col_names)

    return df


if __name__ == '__main__':

    dates = date_list_generation(datetime.today().strftime('%Y-%m-%d'),
                                 (datetime.today() + timedelta(days=365)).strftime('%Y-%m-%d'))
    windows = date_windows(dates)

    date_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
    date_suffix = '&limit=900'

    date_urls = create_urls(date_prefix, date_suffix, windows)

    dfs = []
    failed = []
    for window, url in zip(windows, date_urls):
        json_data = retrieve_espn_game_data(url)
        if json_data is None:
            failed.append(window)
            continue
        pd_data = transform_espn_ncaaf_game_data(json_data)
        if pd_data.empty:
            continue
        dfs.append(pd_data)

    # a partial scrape produces a schedule that silently omits games, so refuse
    # to write one rather than let it reach the prediction file
    if failed:
        print('Failed to retrieve', len(failed), 'of', len(windows), 'windows')
        if len(failed) > len(windows) * 0.05:
            raise RuntimeError('Too many windows failed to download: ' + ', '.join(failed))

    if not dfs:
        raise RuntimeError('No scheduled games retrieved for any of the ' +
                           str(len(windows)) + ' windows searched')

    df = pd.concat(dfs)
    # ESPN snaps each window out to its own week boundaries, so consecutive
    # windows overlap and the same game arrives more than once.
    before = len(df)
    df = df.drop_duplicates(subset='id')
    print('Retrieved', before, 'rows,', len(df), 'distinct games')

    # The date is the event's own kickoff, not the window that found it, taken
    # in US Eastern so a late west-coast Saturday does not become Sunday.
    df['date'] = (pd.to_datetime(df['date'], format='ISO8601', utc=True)
                    .dt.tz_convert(GAME_TZ).dt.normalize().dt.tz_localize(None))

    df = df.loc[df['status'] == 'STATUS_SCHEDULED']
    df = df[df['name'].str.contains('TBD') == False]
    df = df[df['short_name'].str.contains('TBD') == False]
    df.sort_values('date', inplace=True)

    print(len(df), 'scheduled games from', df['date'].min().date(),
          'to', df['date'].max().date())
    df.to_csv('scheduled_games.csv')
