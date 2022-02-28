import os
import json
import pandas as pd
import datetime
import argparse
from date_generation import date_list_generation
from create_date_api_url_strings import create_urls as dt_create_urls
from retrieve_game_ids import retrieve_espn_game_ids
from create_game_id_api_url_strings import create_urls as gid_create_urls
from retrieve_pbp_data import retrieve_espn_play_by_play_data
from json_to_csv import transform_espn_ncaaf_data

parser = argparse. \
    ArgumentParser(description='Input game dates (oldest to newest) to pull NCAA Football play by play records')
today = (datetime.datetime.today()).strftime('%Y-%m-%d')
end_date = (datetime.datetime.today()).strftime('%Y-%m-%d')
start_date = (datetime.datetime.today() - datetime.timedelta(6*365/12)).strftime('%Y-%m-%d')
parser.add_argument('--start_date', default=start_date, help='Start date to pull data from')
parser.add_argument('--end_date', default=end_date, help='End date to pull data from')
args = parser.parse_args()
start_date = args.start_date
end_date = args.end_date


def main(start_date=start_date, end_date=end_date):

    # Date generation
    dates = date_list_generation(start_date, end_date)
    file_name = 'dates_'
    if os.path.exists('temp'):
        with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
            for item in dates:
                file.write('%s\n' % item)
    else:
        os.mkdir('temp')
        with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
            for item in dates:
                file.write('%s\n' % item)

    # Create date api urls
    date_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
    date_suffix = '&limit=900'
    file_name = 'date_urls_'
    date_urls = dt_create_urls(date_prefix, date_suffix, dates)
    with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
        for item in date_urls:
            file.write('%s\n' % item)

    # Retrieve game ids
    file_name = 'game_ids_'
    game_ids = retrieve_espn_game_ids(date_urls)
    with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
        for item in game_ids:
            file.write('%s\n' % item)

    # Create game id api urls
    gameid_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event='
    gameid_suffix = '&limit=900'
    file_name = 'gameid_urls_'
    gameid_urls = gid_create_urls(gameid_prefix, gameid_suffix, game_ids)
    with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
        for item in gameid_urls:
            file.write('%s\n' % item)

    # Retrieve json play by play data
    dfs = []
    file_name = 'json_pbp'
    for i in range(len(gameid_urls)):
        try:
            json_data = retrieve_espn_play_by_play_data(gameid_urls[i])
            beg_substring = 'event='
            end_substring = '&'
            game_id = (gameid_urls[i].split(beg_substring)[1]).split(end_substring)[0]
            # logging.info('{}: {} scraped and saved'.format(str(i + 1), gameid_urls[i]))
            if os.path.exists('temp/pbpjsons'):
                with open('temp/pbpjsons/' + file_name + '_' + str(game_id) + '.json', 'w') as file:
                    json.dump(json_data, file)
            else:
                os.mkdir('temp/pbpjsons')
                with open('temp/pbpjsons/' + file_name + '_' + str(game_id) + '.json', 'w') as file:
                    json.dump(json_data, file)

            # Convert JSON files to CSV files
            pd_data = transform_espn_ncaaf_data(json_data)
            dfs.append(pd_data)
        except:
            print('Error')

    df = pd.concat(dfs)
    df.set_index('id', inplace=True)
    df.to_csv('temp/play-by-play_' + str(start_date) + '_to_' + str(end_date) + '.csv')


if __name__ == '__main__':
    main()
