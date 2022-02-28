import urllib.request
import json


def retrieve_espn_team_data(url: str) -> dict:

    """
    Retrieve dict of teams from ESPN api.

    param url: (str) url from espn api.

    return data: (dict) play by play dictionary.
    """

    assert isinstance(url, str), 'url must be strings'

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
    None
