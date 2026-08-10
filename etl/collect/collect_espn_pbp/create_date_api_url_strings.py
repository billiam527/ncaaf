# create_date_api_url_strings.py - Improved version
from datetime import datetime
import os
import glob
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

    Raises:
        TypeError: If inputs are not the correct type
        ValueError: If any input is empty or invalid
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
        if not item.strip():
            raise ValueError(f'data[{i}] cannot be empty or whitespace')

    return [f"{prefix}{item}{suffix}" for item in data]

def main():
    """
    Read the most recent date list file and create URLs for scraping.
    Note: This function is designed for standalone use and may not be needed
    when integrated into the main run.py script.
    """
    try:
        # Find the most recent dates file
        list_of_files = glob.glob('temp/dates*')
        if not list_of_files:
            raise FileNotFoundError("No dates files found in temp directory")
            
        latest_file = max(list_of_files, key=os.path.getctime)
        logging.info(f"Using dates file: {latest_file}")
        
        # Read dates from file
        with open(latest_file, 'r') as f:
            dates = [line.strip() for line in f if line.strip()]
        
        if not dates:
            raise ValueError("No valid dates found in file")
        
        # Create URLs
        prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
        suffix = '&limit=900'
        urls = create_urls(prefix, suffix, dates)
        
        # Save URLs to file
        end_date = datetime.today()
        start_date = '2019-08-01'  # This should ideally be extracted from the dates file
        filename = f'temp/date_urls_{start_date}_to_{end_date.strftime("%Y-%m-%d")}_test'
        
        os.makedirs('temp', exist_ok=True)
        with open(filename, 'w') as file:
            for url in urls:
                file.write(f'{url}\n')
                
        logging.info(f"Created {len(urls)} URLs and saved to {filename}")
        
    except Exception as e:
        logging.error(f"Error creating date URLs: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()