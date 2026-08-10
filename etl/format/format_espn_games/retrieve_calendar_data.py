import json
import urllib.request
import logging
import time

def retrieve_espncalendardata(url, max_attempts=3, delay=1):
    """
    Retrieve calendar data from ESPN API with retry logic.

    Args:
        url (str): ESPN API URL
        max_attempts (int): Maximum retry attempts
        delay (int): Initial delay between retries

    Returns:
        dict: Calendar data dictionary, or None if failed
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
                
                if not isinstance(data, dict):
                    logging.warning(f"Expected dict response, got {type(data)} for URL: {url}")
                    return None
                    
                return data

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logging.info(f"No calendar data available (404) for URL: {url}")
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

if __name__ == '__main__':
    # Test the function
    logging.basicConfig(level=logging.INFO)
    test_url = "http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=2024"
    
    data = retrieve_espncalendardata(test_url)
    if data:
        print("Calendar data retrieval test successful")
    else:
        print("Calendar data retrieval test failed")