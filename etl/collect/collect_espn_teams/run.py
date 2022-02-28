import pandas as pd
import datetime
import logging
import os
import json
from retrieve_team_data import retrieve_espn_team_data
from json_to_csv import transform_espn_ncaaf_team_data

today = (datetime.datetime.today()).strftime('%Y-%m-%d')
url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?limit=900'


def main(url=url):

    file_name = 'json_teams'
    json_data = retrieve_espn_team_data(url)
    if os.path.exists('temp/'):
        with open('temp/' + file_name + '.json', 'w') as file:
            json.dump(json_data, file)
    else:
        os.mkdir('temp/')
        with open('temp/' + file_name + '.json', 'w') as file:
            json.dump(json_data, file)
    team_data = transform_espn_ncaaf_team_data(json_data)
    team_data.set_index('id', inplace=True)
    team_data.to_csv('temp/teams' + '.csv')


if __name__ == '__main__':
    main()
