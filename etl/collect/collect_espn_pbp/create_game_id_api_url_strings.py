from datetime import datetime
import glob
import os
import logging
from typing import List

def create_urls(prefix: str, suffix: str, data: List[str]) -> List[str]:
    """
    Create a list of string URLs using a prefix, suffix, and data to loop through.

    Args:
        prefix (str): URL prefix string
        suffix (str): URL suffix string
        data (List[str]): List of strings to be inserted between prefix and suffix

    Returns:
        List[str]: List of complete URL strings
    """
    if not isinstance(prefix, str):
        raise TypeError('prefix must be a string')
    if not isinstance(suffix, str):
        raise TypeError('suffix must be a string')
    if not isinstance(data, list):
        raise TypeError('data must be a list')
    
    if not data:
        raise ValueError('data list cannot be empty')
    
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise TypeError(f'data[{i}] must be a string, got {type(item)}')

    return [f"{prefix}{item}{suffix}" for item in data]

def main():
    """
    Read the most recent game IDs file and create URLs for play-by-play scraping.
    """
    try:
        # Find the most recent game_ids file
        list_of_files = glob.glob('temp/game_ids*')
        if not list_of_files:
            raise FileNotFoundError("No game_ids files found in temp directory")
            
        latest_file = max(list_of_files, key=os.path.getctime)
        logging.info(f"Using game IDs file: {latest_file}")
        
        # Read game IDs from file
        with open(latest_file, 'r') as f:
            game_ids = [line.strip() for line in f if line.strip()]
        
        if not game_ids:
            raise ValueError("No valid game IDs found in file")
        
        # Create URLs
        prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event='
        suffix = '&limit=900'
        urls = create_urls(prefix, suffix, game_ids)
        
        # Save URLs to file
        end_date = datetime.today()
        start_date = '2019-08-01'  # This should ideally be extracted from the game_ids file
        filename = f'temp/gameid_urls_{start_date}_to_{end_date.strftime("%Y-%m-%d")}_test'
        
        os.makedirs('temp', exist_ok=True)
        with open(filename, 'w') as file:
            for url in urls:
                file.write(f'{url}\n')
                
        logging.info(f"Created {len(urls)} game ID URLs and saved to {filename}")
        
    except Exception as e:
        logging.error(f"Error creating game ID URLs: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()