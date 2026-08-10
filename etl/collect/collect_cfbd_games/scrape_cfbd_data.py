#!/usr/bin/env python3
"""
CFBD Spreads Scraper - Optimized Version
Pulls all betting spreads for a specified year range using CFBD API
"""

import requests
import pandas as pd
from datetime import datetime
import time
import argparse
import sys
import logging
from tqdm import tqdm

# Setup clean logging
def setup_logging():
    """Setup clean logging - only errors and major progress to file"""
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler('cfbd_collection.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler for minimal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_cfbd_spreads(api_key, year, week=None, season_type='regular'):
    """
    Get betting lines from CFBD API for a specific year/week
    
    Args:
        api_key (str): Your CFBD API key
        year (int): Season year
        week (int): Week number (optional)
        season_type (str): 'regular' or 'postseason'
    
    Returns:
        list: List of games with betting data
    """
    
    url = "https://api.collegefootballdata.com/lines"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'accept': 'application/json'
    }
    
    params = {
        'year': year,
        'seasonType': season_type
    }
    
    if week is not None:
        params['week'] = week
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data if data else []
        elif response.status_code == 429:
            logging.warning(f"Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            return get_cfbd_spreads(api_key, year, week, season_type)
        else:
            logging.error(f"Error {response.status_code} for {year} week {week}: {response.text}")
            return []
            
    except requests.RequestException as e:
        logging.error(f"Request error for {year} week {week}: {e}")
        return []

def parse_cfbd_game(game_data):
    """
    Parse game data from CFBD response into clean format
    
    Args:
        game_data (dict): Raw game data from CFBD
    
    Returns:
        list: List of parsed game records (one per betting provider)
    """
    
    parsed_games = []
    
    # Basic game info
    base_info = {
        'game_id': game_data.get('id'),
        'season': game_data.get('season'),
        'week': game_data.get('week'),
        'season_type': game_data.get('seasonType'),
        'start_date': game_data.get('startDate'),
        'home_team': game_data.get('homeTeam'),
        'away_team': game_data.get('awayTeam'),
        'home_conference': game_data.get('homeConference'),
        'away_conference': game_data.get('awayConference'),
        'home_score': game_data.get('homeScore'),
        'away_score': game_data.get('awayScore')
    }
    
    # Parse betting lines (can have multiple providers)
    lines = game_data.get('lines', [])
    
    if not lines:
        # No betting lines, but still include the game
        base_info.update({
            'provider': None,
            'spread': None,
            'formatted_spread': None,
            'spread_open': None,
            'over_under': None,
            'over_under_open': None,
            'home_moneyline': None,
            'away_moneyline': None
        })
        parsed_games.append(base_info.copy())
    else:
        # Create record for each betting provider
        for line in lines:
            game_record = base_info.copy()
            game_record.update({
                'provider': line.get('provider'),
                'spread': line.get('spread'),
                'formatted_spread': line.get('formattedSpread'),
                'spread_open': line.get('spreadOpen'),
                'over_under': line.get('overUnder'),
                'over_under_open': line.get('overUnderOpen'),
                'home_moneyline': line.get('homeMoneyline'),
                'away_moneyline': line.get('awayMoneyline')
            })
            parsed_games.append(game_record)
    
    return parsed_games

def scrape_cfbd_year_range(api_key, start_year, end_year, include_postseason=True, output_file=None):
    """
    Scrape CFBD betting lines for a range of years
    
    Args:
        api_key (str): Your CFBD API key
        start_year (int): Starting year
        end_year (int): Ending year
        include_postseason (bool): Whether to include bowl games
        output_file (str): Optional CSV output filename
    
    Returns:
        pd.DataFrame: All betting data
    """
    
    all_games = []
    years = list(range(start_year, end_year + 1))
    
    print(f"Collecting CFBD betting lines from {start_year} to {end_year}")
    if include_postseason:
        print("   Including postseason games")
    
    logging.info(f"Scraping CFBD betting lines from {start_year} to {end_year}")
    logging.info(f"Include postseason: {include_postseason}")
    
    # Calculate total operations for progress bar
    total_operations = len(years) * (2 if include_postseason else 1)
    
    with tqdm(total=total_operations, desc="Collecting seasons", unit="season") as pbar:
        for year in years:
            # Regular season
            pbar.set_description(f"Regular {year}")
            regular_season_data = get_cfbd_spreads(api_key, year, season_type='regular')
            
            if regular_season_data:
                for game in regular_season_data:
                    parsed_games = parse_cfbd_game(game)
                    all_games.extend(parsed_games)
                
                logging.info(f"Found {len(regular_season_data)} regular season games for {year}")
            
            pbar.update(1)
            time.sleep(1)  # Rate limiting
            
            # Postseason (bowl games, playoffs)
            if include_postseason:
                pbar.set_description(f"Postseason {year}")
                postseason_data = get_cfbd_spreads(api_key, year, season_type='postseason')
                
                if postseason_data:
                    for game in postseason_data:
                        parsed_games = parse_cfbd_game(game)
                        all_games.extend(parsed_games)
                    
                    logging.info(f"Found {len(postseason_data)} postseason games for {year}")
                
                pbar.update(1)
                time.sleep(1)  # Rate limiting
    
    # Convert to DataFrame
    if all_games:
        print("Processing collected data...")
        df = pd.DataFrame(all_games)
        
        # Clean and sort the data
        df = clean_cfbd_data(df)
        
        print(f"SUCCESS: Collection Complete!")
        print(f"Total records: {len(df):,}")
        print(f"Unique games: {df['game_id'].nunique():,}")
        print(f"Betting providers: {df['provider'].nunique()}")
        
        logging.info(f"Total records collected: {len(df)}")
        logging.info(f"Unique games: {df.groupby(['game_id']).ngroups}")
        logging.info(f"Betting providers: {df['provider'].nunique()}")
        
        # Save to CSV if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved to: {output_file}")
            logging.info(f"Data saved to {output_file}")
        
        return df
    else:
        print("WARNING: No data collected!")
        logging.warning("No data collected!")
        return pd.DataFrame()

def clean_cfbd_data(df):
    """
    Clean and standardize CFBD betting data
    
    Args:
        df (pd.DataFrame): Raw CFBD data
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    
    # Convert start_date to datetime
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Sort by date and game_id
    df = df.sort_values(['start_date', 'game_id', 'provider']).reset_index(drop=True)
    
    # Remove completely duplicate rows
    df = df.drop_duplicates()
    
    return df

def main():
    """Main execution function with argument parsing"""
    
    setup_logging()
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Scrape CFBD betting spreads data'
    )
    
    # Get current year for defaults
    current_year = datetime.now().year
    
    parser.add_argument('--start_year', type=int, default=current_year,
                       help=f'Starting year (default: {current_year})')
    parser.add_argument('--end_year', type=int, default=current_year,
                       help=f'Ending year (default: {current_year})')
    parser.add_argument('--no_postseason', action='store_true',
                       help='Exclude postseason games')
    parser.add_argument('--output', type=str, default='cfbd_spread_data.csv',
                       help='Output CSV filename (default: cfbd_spread_data.csv)')
    
    args = parser.parse_args()
    
    print(f"\nCFBD Spreads Collection")
    print(f"Years: {args.start_year} to {args.end_year}")
    print(f"Include postseason: {not args.no_postseason}")
    print("=" * 50)
    
    # Validate year range
    if args.start_year > args.end_year:
        print("ERROR: Start year cannot be greater than end year")
        logging.error("Start year cannot be greater than end year")
        sys.exit(1)
    
    if args.start_year < 2001:
        print("WARNING: CFBD data may be limited before 2001")
        logging.warning("CFBD data may be limited before 2001")
    
    # Configuration
    API_KEY = "BYASZNhBGbXh1uzoqM4fRbZjnPetC+M4/yKmCMSNXPeq18+uvw5zGFLh5QY5d4Ka"
    
    # Validate API key
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your CFBD API key in the script!")
        print("   Get your key from: https://collegefootballdata.com/key")
        logging.error("CFBD API key not configured")
        sys.exit(1)
    
    try:
        # Scrape the data
        betting_data = scrape_cfbd_year_range(
            api_key=API_KEY,
            start_year=args.start_year,
            end_year=args.end_year,
            include_postseason=not args.no_postseason,
            output_file=args.output
        )
        
        if not betting_data.empty:
            print("\nSUCCESS: CFBD collection completed successfully!")
            logging.info("CFBD collection completed successfully")
        else:
            print("\nWARNING: No data collected - check your parameters")
            logging.warning("No data collected")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nERROR: Collection interrupted by user")
        logging.info("Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Collection failed: {e}")
        logging.error(f"Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()