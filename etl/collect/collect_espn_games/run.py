import pandas as pd
import datetime
import logging
import argparse
import os
import json
from format_input_date import date_list_generation
from create_box_score_api_urls import create_urls
from retrieve_game_data import retrieve_espn_game_data
from json_to_csv import transform_espn_ncaaf_game_data

parser = argparse. \
    ArgumentParser(description='Input game dates (oldest to newest) to pull NCAA Football game records')
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
    file_name = 'urls_'
    date_urls = create_urls(date_prefix, date_suffix, dates)

    with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
        for item in date_urls:
            file.write('%s\n' % item)

    # Retrieve json play by play data
    dfs = []
    file_name = 'json_games'
    for i in range(len(date_urls)):
        json_data = retrieve_espn_game_data(date_urls[i])
        beg_substring = 'dates='
        end_substring = '&'
        url_date = (date_urls[i].split(beg_substring)[1]).split(end_substring)[0]
        logging.info('{}: {} scraped and saved'.format(str(i + 1), date_urls[i]))
        if os.path.exists('temp/gamejsons'):
            with open('temp/gamejsons/' + file_name + '_' + str(url_date) + '.json', 'w') as file:
                json.dump(json_data, file)
        else:
            os.mkdir('temp/gamejsons')
            with open('temp/gamejsons/' + file_name + '_' + str(url_date) + '.json', 'w') as file:
                json.dump(json_data, file)
        pd_data = transform_espn_ncaaf_game_data(json_data)
        dfs.append(pd_data)
    df = pd.concat(dfs)
    df.set_index('id', inplace=True)
    df.to_csv('temp/games_' + str(start_date) + '_to_' + str(end_date) + '.csv')


if __name__ == '__main__':
    main()
