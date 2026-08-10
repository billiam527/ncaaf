import pandas as pd
import datetime
import logging
import os
import json
import sys
from tqdm import tqdm
from retrieve_team_data import retrieve_espn_team_data, fbs_team_ind
from json_to_csv import transform_espn_ncaaf_team_data

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
    file_handler = logging.FileHandler('temp/teams_collection.log', mode='w')
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

def main():
    setup_logging()
    
    print("\nESPN Teams Collection")
    print("=" * 40)
    
    try:
        # Setup
        os.makedirs('temp', exist_ok=True)
        
        # Step 1: Retrieve team data from ESPN API
        print("Step 1: Retrieving team data from ESPN API...")
        url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?limit=900'
        
        with tqdm(total=1, desc="Fetching teams", unit="request") as pbar:
            json_data = retrieve_espn_team_data(url)
            pbar.update(1)
        
        if json_data is None:
            print("ERROR: Failed to retrieve team data from ESPN API")
            raise Exception("Failed to retrieve team data from ESPN API")
        
        # Save JSON file
        json_filename = 'temp/json_teams.json'
        with open(json_filename, 'w') as file:
            json.dump(json_data, file)
        
        logging.info(f"Saved raw JSON data to {json_filename}")
        print("   SUCCESS: Retrieved team data from API")
        
        # Step 2: Transform JSON to DataFrame
        print("Step 2: Processing team data...")
        
        with tqdm(total=1, desc="Processing", unit="dataset") as pbar:
            team_data = transform_espn_ncaaf_team_data(json_data)
            pbar.update(1)
        
        if team_data is None or team_data.empty:
            print("ERROR: Failed to transform team data - no valid data found")
            raise Exception("Failed to transform team data - no valid data found")
        
        print(f"   SUCCESS: Processed {len(team_data)} teams")
        logging.info(f"Successfully processed {len(team_data)} teams")
        
        # Step 3: Get FBS team indicators
        print("Step 3: Identifying FBS teams...")
        
        try:
            with tqdm(total=1, desc="Scraping FBS", unit="website") as pbar:
                fbs_df = fbs_team_ind()
                pbar.update(1)
            
            if not fbs_df.empty:
                print(f"   SUCCESS: Identified {len(fbs_df)} FBS teams")
                logging.info(f"Successfully identified {len(fbs_df)} FBS teams")
                
                # Join FBS indicators to team data
                team_data = pd.merge(team_data, fbs_df, how='left', on='slug')
                team_data['fbs_ind'] = team_data['fbs_ind'].fillna(0).astype(int)
                
                fbs_count = team_data['fbs_ind'].sum()
                print(f"   SUCCESS: Matched {fbs_count} teams as FBS")
                logging.info(f"Matched {fbs_count} teams as FBS")
            else:
                print("   WARNING: No FBS data collected - continuing without indicators")
                team_data['fbs_ind'] = 0
            
        except Exception as e:
            print("   WARNING: Failed to collect FBS indicators - continuing without them")
            logging.warning(f"Failed to collect FBS indicators: {e}")
            team_data['fbs_ind'] = 0
        
        # Step 4: Save final CSV
        print("Step 4: Saving results...")
        output_file = 'temp/teams.csv'
        team_data.to_csv(output_file, index=False)
        
        print(f"\nSUCCESS: Collection Complete!")
        print(f"Total teams: {len(team_data)}")
        print(f"FBS teams: {team_data['fbs_ind'].sum()}")
        print(f"Non-FBS teams: {len(team_data) - team_data['fbs_ind'].sum()}")
        print(f"Output file: {output_file}")
        
        logging.info(f"Successfully saved {len(team_data)} teams to {output_file}")
        logging.info("Teams collection completed successfully!")
        
    except Exception as e:
        logging.error(f"Fatal error during teams collection: {e}")
        print(f"\nERROR: Collection failed: {e}")
        raise

if __name__ == '__main__':
    try:
        main()
        print("\nSUCCESS: Teams collection completed successfully!")
    except Exception as e:
        print(f"\nERROR: Script failed: {e}")
        print("Check temp/teams_collection.log for detailed error information")
        sys.exit(1)