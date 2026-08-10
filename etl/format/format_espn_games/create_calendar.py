import pandas as pd
import datetime
from datetime import timedelta
import argparse
import os
import json
import logging
import sys
from tqdm import tqdm
from format_input_date import year_list_generation
from create_api_urls import create_urls
from retrieve_calendar_data import retrieve_espncalendardata
from json_to_csv import transform_espn_ncaaf_week_data

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
    file_handler = logging.FileHandler('temp/calendar_creation.log', mode='w')
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
    os.makedirs('temp/calendarjsons', exist_ok=True)

def main(start_date, end_date):
    setup_logging()
    
    print(f"\nCreating Calendar Data")
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 40)
    
    try:
        # Setup
        setup_directories()
        
        # Step 1: Generate years
        print("Step 1: Generating year list...")
        dates = year_list_generation(start_date, end_date)
        
        years_filename = f'temp/years_{start_date}_to_{end_date}'
        with open(years_filename, 'w') as file:
            for item in dates:
                file.write(f'{item}\n')
        
        print(f"   Generated {len(dates)} years")
        logging.info(f"Generated {len(dates)} years from {start_date} to {end_date}")
        
        # Step 2: Create URLs
        print("Step 2: Creating calendar API URLs...")
        date_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
        date_suffix = ''
        year_urls = create_urls(date_prefix, date_suffix, dates)
        
        urls_filename = f'temp/urls_{start_date}_to_{end_date}'
        with open(urls_filename, 'w') as file:
            for item in year_urls:
                file.write(f'{item}\n')
        
        print(f"   Created {len(year_urls)} calendar URLs")
        
        # Step 3: Retrieve calendar data
        print(f"Step 3: Retrieving calendar data...")
        dfs = []
        successful_years = 0
        failed_years = 0
        
        with tqdm(total=len(year_urls), desc="Processing years", unit="years") as pbar:
            for i, url in enumerate(year_urls):
                try:
                    year = url.split('dates=')[1]
                    
                    json_data = retrieve_espncalendardata(url)
                    
                    if json_data is None:
                        failed_years += 1
                        pbar.set_postfix(success=successful_years, failed=failed_years)
                        pbar.update(1)
                        continue
                    
                    # Save JSON file
                    json_filename = f'temp/calendarjsons/json_calendar_{year}.json'
                    with open(json_filename, 'w') as file:
                        json.dump(json_data, file)
                    
                    # Transform to DataFrame
                    pd_data = transform_espn_ncaaf_week_data(json_data)
                    
                    if pd_data is not None and not pd_data.empty:
                        # Create date range for this year's calendar
                        max_date = datetime.datetime.strftime(
                            datetime.datetime.strptime(max(pd_data.week_start_date), '%Y-%m-%d') + timedelta(days=6),
                            '%Y-%m-%d'
                        )
                        
                        idx = pd.date_range(min(pd_data.week_start_date), max_date)
                        date_index_df = pd_data.set_index('week_start_date')
                        date_index_df.index = pd.DatetimeIndex(date_index_df.index)
                        
                        new_df = date_index_df.reindex(idx, method='ffill').reset_index() \
                            .reset_index(drop=True) \
                            .rename(columns={'index': 'date'})
                        
                        dfs.append(new_df)
                        successful_years += 1
                    else:
                        failed_years += 1
                    
                    pbar.set_postfix(success=successful_years, failed=failed_years)
                    pbar.update(1)
                        
                except Exception as e:
                    logging.error(f"Error processing year {year}: {e}")
                    failed_years += 1
                    pbar.set_postfix(success=successful_years, failed=failed_years)
                    pbar.update(1)
                    continue
        
        # Step 4: Combine and save results
        if dfs:
            print("Step 4: Combining and saving calendar data...")
            df = pd.concat(dfs, ignore_index=True)
            
            start_year = start_date.split('-')[0]
            end_year = end_date.split('-')[0]
            output_file = f'temp/schedule_{start_year}_to_{end_year}.csv'
            df.to_csv(output_file, index=False)
            
            print(f"\nCalendar Creation Complete!")
            print(f"Total calendar entries: {len(df):,}")
            print(f"Successful years: {successful_years}")
            print(f"Failed years: {failed_years}")
            print(f"Output file: {output_file}")
            
            logging.info(f"Successfully saved {len(df)} calendar entries to {output_file}")
            logging.info(f"Calendar summary: {successful_years} successful years, {failed_years} failed years")
        else:
            print("WARNING: No calendar data collected - all years failed")
            logging.error("No calendar data collected - all years failed")
            raise Exception("Calendar creation failed completely")
            
    except Exception as e:
        logging.error(f"Fatal error during calendar creation: {e}")
        print(f"\nCalendar creation failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create NCAA Football calendar data from ESPN API'
    )
    
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    default_start = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
    
    parser.add_argument('--start_date', default=default_start,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=today,
                       help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        main(args.start_date, args.end_date)
        print("\nCalendar creation completed successfully!")
    except Exception as e:
        print(f"\nScript failed: {e}")
        print("Check temp/calendar_creation.log for detailed error information")
        sys.exit(1)