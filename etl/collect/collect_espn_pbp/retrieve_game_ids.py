import urllib.request
import json
import logging
import time

def retrieve_espn_game_ids(urls, max_attempts=3, delay=1):
    """
    Retrieve list of game_ids from ESPN API with retry logic.

    Args:
        urls (list): List of URL strings
        max_attempts (int): Maximum retry attempts per URL
        delay (int): Initial delay between retries

    Returns:
        list: List of game ID strings
    """
    if not isinstance(urls, list):
        raise TypeError('urls must be in list format')
    
    for url in urls:
        if not isinstance(url, str):
            raise TypeError('URLs in list must be strings')

    game_ids = []
    failed_urls = 0
    
    for i, url in enumerate(urls, 1):
        if i % 20 == 0:  # Progress update every 20 URLs
            logging.info(f"Processing URL {i}/{len(urls)} for game IDs...")
            
        success = False
        
        for attempt in range(max_attempts):
            try:
                # Spoofed browser UAs get a blanket 403 from ESPN; use the default.
                request = urllib.request.Request(url)
                
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    # Extract game IDs from events
                    events = data.get('events', [])
                    for event in events:
                        game_id = event.get('id')
                        if game_id:
                            game_ids.append(game_id)
                    
                    success = True
                    break
                    
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # No games for this date - not an error
                    success = True
                    break
                elif attempt < max_attempts - 1:
                    wait_time = delay * (2 ** attempt)
                    logging.warning(f"HTTP Error {e.code} for URL {i}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"HTTP Error {e.code} after {max_attempts} attempts for URL {i}")
                    
            except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
                if attempt < max_attempts - 1:
                    wait_time = delay * (2 ** attempt)
                    logging.warning(f"Error for URL {i}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed after {max_attempts} attempts for URL {i}: {e}")
        
        if not success:
            failed_urls += 1

    logging.info(f"Retrieved {len(game_ids)} game IDs. Failed URLs: {failed_urls}")
    return game_ids