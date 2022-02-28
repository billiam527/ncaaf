import pandas as pd
import datetime
from datetime import timedelta
import argparse
import os
import json
from format_input_date import year_list_generation
from create_api_urls import create_urls
from retrieve_calendar_data import retrieve_espncalendardata
from json_to_csv import transform_espn_ncaaf_week_data

parser = argparse. \
    ArgumentParser(description='Input game dates (oldest to newest) to pull NCAA Football calendar weeks')
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
    print(start_date)
    print(end_date)
    dates = year_list_generation(start_date, end_date)
    file_name = 'years_'
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
    date_suffix = ''
    file_name = 'urls_'
    year_urls = create_urls(date_prefix, date_suffix, dates)
    with open('temp/' + file_name + str(start_date) + '_to_' + str(end_date), 'w') as file:
        for item in year_urls:
            file.write('%s\n' % item)

    # Retrieve json play by play data
    dfs = []
    file_name = 'json_calendar'
    for i in range(len(year_urls)):
        json_data = retrieve_espncalendardata(year_urls[i])
        url_split = 'dates='
        url_year = (year_urls[i].split(url_split)[1])
        if os.path.exists('temp/calendarjsons'):
            with open('temp/calendarjsons/' + file_name + '_' + str(url_year) + '.json', 'w') as file:
                json.dump(json_data, file)
        else:
            os.mkdir('temp/calendarjsons')
            with open('temp/calendarjsons/' + file_name + '_' + str(url_year) + '.json', 'w') as file:
                json.dump(json_data, file)
        pd_data = transform_espn_ncaaf_week_data(json_data)
        if pd_data.shape[0] != 0:
            max_date = datetime.datetime.strftime(datetime.datetime.strptime(max(pd_data.week_start_date), '%Y-%m-%d') + \
                                                  timedelta(days=6), '%Y-%m-%d')
            idx = pd.date_range(min(pd_data.week_start_date), max_date)
            date_index_df = pd_data.set_index('week_start_date')
            date_index_df.index = pd.DatetimeIndex(date_index_df.index)
            new_df = date_index_df.reindex(idx, method='ffill').reset_index() \
                .reset_index(drop=True) \
                .rename(columns={'index': 'date'})
            dfs.append(new_df)
        else:
            continue

    df = pd.concat(dfs)

    #df.set_index('id', inplace=True)
    df.to_csv('temp/schedule_' + str(start_date.split('-')[0]) + '_to_' + str(end_date.split('-')[0]) + '.csv', \
              index=False)


if __name__ == '__main__':
    main()
