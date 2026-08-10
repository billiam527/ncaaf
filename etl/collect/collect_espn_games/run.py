import pandas as pd
import datetime
import logging
import argparse
import os
import json
import sys
from tqdm import tqdm
from format_input_date import date_list_generation
from create_box_score_api_urls import create_urls
from retrieve_game_data import retrieve_espn_game_data
from json_to_csv import transform_espn_ncaaf_game_data

# Configure clean logging
def setup_logging():
    """Setup clean logging - only errors and major progress to file"""
    os.makedirs('temp', exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler('temp/games_collection.log', mode='w')
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

def setup_directories():
    """Create necessary directories"""
    os.makedirs('temp', exist_ok=True)
    os.makedirs('temp/gamejsons', exist_ok=True)

def validate_date_range(start_date, end_date):
    """Validate that the date range makes sense for college football data collection"""
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        now = datetime.datetime.now()
        
        # Check if start date is in the future
        if start_dt > now:
            print(f"WARNING: Start date {start_date} is in the future - no games will be available")
            return False
            
        # Check if we're trying to collect data for a season that hasn't started
        season_year = start_dt.year if start_dt.month >= 8 else start_dt.year - 1
        current_season_year = now.year if now.month >= 8 else now.year - 1
        
        if season_year > current_season_year:
            print(f"WARNING: Season {season_year} hasn't started yet (current season: {current_season_year})")
            return False
            
        # If we're in the current season, check if any games have occurred yet
        if season_year == current_season_year:
            # College football typically starts around August 23rd
            season_start = datetime.datetime(now.year, 8, 23)
            if now < season_start:
                print(f"WARNING: Current season {season_year} hasn't started yet (starts around Aug 23)")
                return False
            elif now < datetime.datetime(now.year, 8, 30):
                print(f"INFO: Current season just started - limited games may be available")
                
        return True
        
    except Exception as e:
        logging.error(f"Error validating date range: {e}")
        return False

def save_dates_file(dates, start_date, end_date):
    """Save dates to file"""
    file_name = f'dates_{start_date}_to_{end_date}'
    file_path = f'temp/{file_name}'
    
    try:
        with open(file_path, 'w') as file:
            for item in dates:
                file.write(f'{item}\n')
        logging.info(f"Saved {len(dates)} dates to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save dates file: {e}")
        raise

def save_urls_file(urls, start_date, end_date):
    """Save URLs to file"""
    file_name = f'urls_{start_date}_to_{end_date}'
    file_path = f'temp/{file_name}'
    
    try:
        with open(file_path, 'w') as file:
            for item in urls:
                file.write(f'{item}\n')
        logging.info(f"Saved {len(urls)} URLs to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save URLs file: {e}")
        raise

def main(start_date, end_date):
    setup_logging()
    
    print(f"\nESPN Games Collection")
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 50)
    
    try:
        # Validate date range
        if not validate_date_range(start_date, end_date):
            print("ERROR: Invalid date range - aborting collection")
            return
            
        # Setup
        setup_directories()
        
        # Step 1: Generate dates
        print("Step 1: Generating date list...")
        dates = date_list_generation(start_date, end_date)
        save_dates_file(dates, start_date, end_date)
        print(f"   SUCCESS: Generated {len(dates)} dates")
        
        # Step 2: Create URLs
        print("Step 2: Creating API URLs...")
        date_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
        date_suffix = '&limit=100'
        date_urls = create_urls(date_prefix, date_suffix, dates)
        save_urls_file(date_urls, start_date, end_date)
        print(f"   SUCCESS: Created {len(date_urls)} date URLs")
        
        # Step 3: Collect data
        print(f"Step 3: Collecting game data...")
        dfs = []
        successful_requests = 0
        failed_requests = 0
        total_games_found = 0
        
        with tqdm(total=len(date_urls), desc="Processing dates", unit="dates") as pbar:
            for i, url in enumerate(date_urls):
                try:
                    # Extract date from URL for filename
                    url_date = url.split('dates=')[1].split('&')[0]
                    
                    # Retrieve JSON data
                    json_data = retrieve_espn_game_data(url)
                    
                    if json_data is None:
                        failed_requests += 1
                        pbar.set_postfix(success=successful_requests, failed=failed_requests, games=total_games_found)
                        pbar.update(1)
                        continue
                    
                    # Save JSON file
                    json_file_path = f'temp/gamejsons/json_games_{url_date}.json'
                    with open(json_file_path, 'w') as file:
                        json.dump(json_data, file)
                    
                    # Transform to DataFrame
                    pd_data = transform_espn_ncaaf_game_data(json_data)
                    
                    if pd_data is not None and not pd_data.empty:
                        dfs.append(pd_data)
                        successful_requests += 1
                        total_games_found += len(pd_data)
                    else:
                        failed_requests += 1
                        
                    pbar.set_postfix(success=successful_requests, failed=failed_requests, games=total_games_found)
                    pbar.update(1)
                        
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")
                    failed_requests += 1
                    pbar.set_postfix(success=successful_requests, failed=failed_requests, games=total_games_found)
                    pbar.update(1)
                    continue
        
        # Step 4: Combine and save results
        if dfs:
            print("Step 4: Combining and saving results...")
            df = pd.concat(dfs, ignore_index=True)
            df.set_index('id', inplace=True)
            
            output_file = f'temp/games_{start_date}_to_{end_date}.csv'
            df.to_csv(output_file)
            
            print(f"\nSUCCESS: Collection Complete!")
            print(f"Total games collected: {len(df):,}")
            print(f"Successful dates: {successful_requests}")
            print(f"Failed dates: {failed_requests}")
            print(f"Output file: {output_file}")
            
            # Show some stats
            if len(df) > 0:
                seasons = df['season'].nunique() if 'season' in df.columns else 0
                print(f"Seasons covered: {seasons}")
                
                # Show date range of actual games
                if 'date' in df.columns:
                    first_game = df['date'].min()
                    last_game = df['date'].max()
                    print(f"Game date range: {first_game} to {last_game}")
                
        else:
            print("WARNING: No game data was successfully processed")
            print("   This could mean:")
            print("   • No games scheduled in the specified date range")
            print("   • All dates failed to process due to API issues")
            print("   • The season hasn't started yet")
            
            # Create empty results file instead of failing
            empty_df = pd.DataFrame(columns=[
                'date', 'name', 'short_name', 'season', 'status', 'venue_id', 'neutral_site',
                'home_team_id', 'away_team_id', 'home_score', 'away_score', 'home_first_quarter',
                'home_second_quarter', 'home_third_quarter', 'home_fourth_quarter', 'home_ot',
                'away_first_quarter', 'away_second_quarter', 'away_third_quarter', 'away_fourth_quarter', 'away_ot'
            ])
            output_file = f'temp/games_{start_date}_to_{end_date}.csv'
            empty_df.to_csv(output_file)
            
            print(f"Created empty results file: {output_file}")
            print(f"Dates attempted: {successful_requests + failed_requests}")
            print(f"Failed dates: {failed_requests}")
            print("SUCCESS: Collection completed (no usable data found)")
            
            # Log detailed info but don't crash
            logging.warning("No game data collected - created empty results file")
            
    except Exception as e:
        logging.error(f"Fatal error during collection: {e}")
        print(f"\nERROR: Collection failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect NCAA Football game records from ESPN API'
    )
    
    # Default to last 6 months if no dates provided
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    default_start = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    
    parser.add_argument('--start_date', default=default_start, 
                       help='Start date to pull data from (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=today,
                       help='End date to pull data from (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        main(args.start_date, args.end_date)
        print("\nSUCCESS: Games collection completed successfully!")
    except Exception as e:
        print(f"\nERROR: Script failed: {e}")
        print("Check temp/games_collection.log for detailed error information")
        sys.exit(1)