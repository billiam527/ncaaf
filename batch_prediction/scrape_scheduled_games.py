from datetime import datetime, timedelta
import urllib.request
import json
import pandas as pd
from datetime import datetime


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
        pass
    except urllib.error.URLError as e:
        pass


def transform_espn_ncaaf_game_data(json_data):

    """
    param json_data: (json) data pulled from espn's ncaaf api at the game level

    return df: (pd.DataFrame) data in the relational table form
    """

    # game data
    id = []
    date = []
    name = []
    short_name = []
    season = []
    status = []
    venue_id = []
    neutral_site = []

    # team info
    home_team_id = []
    away_team_id = []

    if json_data is not None:
        for game in json_data['events']:
            id.append(game['id'])
            date.append(game['date'])
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
                           name,
                           short_name,
                           season,
                           status,
                           venue_id,
                           neutral_site,
                           home_team_id,
                           away_team_id))

        col_names = ['id',
                     'date',
                     'name',
                     'short_name',
                     'season',
                     'status',
                     'venue_id',
                     'neutral_site',
                     'home_team_id',
                     'away_team_id']

        df = pd.DataFrame(columns, columns=col_names)

        return df


if __name__ == '__main__':

    dates = date_list_generation(datetime.today().strftime('%Y-%m-%d'),
                                 (datetime.today() + timedelta(days=365)).strftime('%Y-%m-%d'))

    date_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
    date_suffix = '&limit=900'

    date_urls = create_urls(date_prefix, date_suffix, dates)

    dfs = []
    for url in date_urls:
        json_data = retrieve_espn_game_data(url)
        pd_data = transform_espn_ncaaf_game_data(json_data)
        dfs.append(pd_data)

    df = pd.concat(dfs)
    df = df.loc[df['status'] == 'STATUS_SCHEDULED']
    df = df[df['name'].str.contains('TBD') == False]
    df = df[df['short_name'].str.contains('TBD') == False]
    df.sort_values('date', inplace=True)

    df.to_csv('/home/bill/ncaaf/batch_prediction/prediction_file/scheduled_games.csv')
