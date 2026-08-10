import urllib.request
import json
import glob
import os
import logging
import time


def retrieve_espn_play_by_play_data(url, max_attempts=3, delay=1):
    """
    Retrieve play-by-play data for a specific game from ESPN API.

    Args:
        url (str): ESPN API URL for the game
        max_attempts (int): Maximum retry attempts
        delay (int): Initial delay between retries

    Returns:
        dict: Play-by-play data dictionary, or None if failed
    """
    if not isinstance(url, str):
        raise TypeError('URL must be a string')

    for attempt in range(max_attempts):
        try:
            # Spoofed browser UAs get a blanket 403 from ESPN; use the default.
            request = urllib.request.Request(url)
            
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.getcode() != 200:
                    logging.warning(f"Unexpected status code {response.getcode()} for URL: {url}")
                    return None
                    
                data = json.loads(response.read().decode('utf-8'))
                
                # Basic validation
                if not isinstance(data, dict):
                    logging.warning(f"Expected dict response, got {type(data)} for URL: {url}")
                    return None
                    
                return data

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logging.info(f"No play-by-play data available (404) for URL: {url}")
                return None
            elif attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logging.warning(f"HTTP Error {e.code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"HTTP Error {e.code} after {max_attempts} attempts for URL: {url}")
                return None
                
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logging.warning(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed after {max_attempts} attempts for URL {url}: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Unexpected error for URL {url}: {e}")
            return None

    return None
