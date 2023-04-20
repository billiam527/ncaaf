import urllib.request
import json


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


if __name__ == "__main__":
    None
