import urllib.request
import json
import glob
import os


def main():

    list_of_files = glob.glob('temp/gameid_url*')
    latest_file = max(list_of_files, key=os.path.getctime)
    urls = [line.rstrip('\n') for line in open(latest_file)]
    file_name = 'pbp_data'
    for i in range(len(urls)):
        data = retrieve_espnplaybyplaydata(urls[i])
        beg_substring = 'event='
        end_substring = '&'
        gameid = (urls[i].split(beg_substring)[1]).split(end_substring)[0]
        print(str(i + 1) + ':', urls[i], 'scraped and saved')
        if os.path.exists('temp/pbpjsons'):
            with open('temp/pbpjsons/' + file_name + '_' + str(gameid), 'w') as file:
                json.dump(data, file)
        else:
            os.mkdir('temp/pbpjsons')
            with open('temp/jsons/' + file_name + '_' + str(gameid), 'w') as file:
                json.dump(data, file)


def retrieve_espn_play_by_play_data(url: str) -> dict:

    """
    Retrieve dict of specific game id from ESPN api.

    param url: (str) url from espn api.

    return data: (dict) play by play dictionary.
    """

    assert isinstance(url, str), 'urls must be strings'

    try:
        with urllib.request.urlopen(url) as web_data:
            data = json.loads(web_data.read().decode())

    except urllib.error.HTTPError as e:
        print('Error', e)
    except urllib.error.URLError as e:
        print('Error', e)
    except:
        print('Error', url)

    return data


if __name__ == "__main__":
    main()
