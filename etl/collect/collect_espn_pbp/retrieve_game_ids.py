import urllib.request
import json
from datetime import datetime
import glob
import os


def main():

    end_date = datetime.today()
    start_date = '2019-08-01'
    list_of_files = glob.glob('temp/date_urls*')
    latest_file = max(list_of_files, key=os.path.getctime)
    urls = [line.rstrip('\n') for line in open(latest_file)]
    game_ids = retrieve_espngameids(urls)
    file_name = 'game_ids_'
    if os.path.exists('temp'):
        with open('temp/' + file_name + str(start_date) + '_to_' + end_date.strftime('%Y-%m-%d'), 'w') as file:
            for item in game_ids:
                file.write('%s\n' % item)
    else:
        os.mkdir('temp')
        with open('temp/' + file_name + str(start_date) + '_to_' + end_date.strftime('%Y-%m-%d'), 'w') as file:
            for item in game_ids:
                file.write('%s\n' % item)


def retrieve_espn_game_ids(urls: list
                         ) -> list:

    """
    Retrieve list of game_ids from ESPN api.

    param dateList: (list) list of date strings

    return gameIds: (list) list of id strings
    """

    assert isinstance(urls, list), 'urls list must be in list format'
    for url in urls:
        assert isinstance(url, str), 'dates in date_list must be strings'

    game_ids = []
    for url in urls:  # open and capture URL data
        try:
            with urllib.request.urlopen(url) as jsonObject:
                data = json.loads(jsonObject.read().decode())

                for event in data['events']:  # navigate json to id field
                    game_ids.append(event['id'])

        except urllib.error.HTTPError as e:
            pass
        except urllib.error.URLError as e:
            pass

    return game_ids


if __name__ == "__main__":
    main()
