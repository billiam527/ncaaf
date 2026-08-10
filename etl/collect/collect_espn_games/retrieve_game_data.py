import urllib.request
import json
import logging
import time

def retrieve_espn_game_data(url: str, max_attempts=3, delay=1):
    """
    Retrieve dict of specific game data from ESPN API with retry logic.

    Args:
        url (str): URL from ESPN API
        max_attempts (int): Maximum number of retry attempts
        delay (int): Initial delay between retries in seconds

    Returns:
        dict: Game data dictionary, or None if request fails

    Raises:
        ValueError: If URL is not a string
    """
    if not isinstance(url, str):
        raise ValueError('URL must be a string')
    
    if not url.strip():
        raise ValueError('URL cannot be empty')

    for attempt in range(max_attempts):
        try:
            # Send urllib's own default User-Agent. A spoofed browser UA gets a
            # blanket 403 from ESPN's edge, which is what silently killed every
            # collection run; the library default is accepted.
            request = urllib.request.Request(url)
            
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.getcode() != 200:
                    logging.warning(f"Unexpected status code {response.getcode()} for URL: {url}")
                    return None
                    
                data = json.loads(response.read().decode('utf-8'))
                
                # Basic validation of response structure
                if not isinstance(data, dict):
                    logging.warning(f"Expected dict response, got {type(data)} for URL: {url}")
                    return None
                    
                return data

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logging.info(f"No data available (404) for URL: {url}")
                return None
            elif e.code == 429:
                if attempt < max_attempts - 1:
                    wait_time = delay * (2 ** attempt)
                    logging.warning(f"Rate limited (429). Retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Rate limited after {max_attempts} attempts for URL: {url}")
                    return None
            else:
                if attempt < max_attempts - 1:
                    wait_time = delay * (2 ** attempt)
                    logging.warning(f"HTTP Error {e.code}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"HTTP Error {e.code} after {max_attempts} attempts for URL: {url}")
                    return None
                    
        except urllib.error.URLError as e:
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logging.warning(f"URL Error: {e.reason}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"URL Error after {max_attempts} attempts for {url}: {e.reason}")
                return None
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for URL {url}: {e}")
            return None
            
        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logging.warning(f"Unexpected error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Unexpected error after {max_attempts} attempts for {url}: {e}")
                return None

    return None

if __name__ == "__main__":
    # Example usage
    test_url = "http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=20240901&limit=100"
    
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        data = retrieve_espn_game_data(test_url)
        if data:
            print(f"Successfully retrieved data with {len(data.get('events', []))} events")
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")