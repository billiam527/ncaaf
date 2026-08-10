import os
import json
import pandas as pd
import datetime
import argparse
import logging
import sys
import time
from tqdm import tqdm
from date_generation import date_list_generation
from create_date_api_url_strings import create_urls as dt_create_urls
from retrieve_game_ids import retrieve_espn_game_ids
from create_game_id_api_url_strings import create_urls as gid_create_urls
from retrieve_pbp_data import retrieve_espn_play_by_play_data
from json_to_csv import transform_espn_ncaaf_data
# Import your existing functions
from edit_pbp_file import add_binary_play_stats, generate_cum_stats, calc_new_features
import numpy as np

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
    file_handler = logging.FileHandler('temp/pbp_processing.log', mode='w')
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

def find_games_file():
    """Find the games.csv file in various possible locations"""
    possible_paths = [
        '../format_espn_games/temp/games.csv',
        '../../format_espn_games/temp/games.csv', 
        '../games.csv',
        'games.csv',
        'temp/games.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"Found games file at: {path}")
            return path
    
    logging.warning("Games file not found in expected locations")
    return None

def validate_input_file(file_path):
    """Validate that the input file exists and has required columns"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Check if file is readable
    try:
        df_check = pd.read_csv(file_path, nrows=1)
        logging.info(f"Input file validated: {file_path}")
        logging.info(f"Columns found: {list(df_check.columns)}")
        return True
    except Exception as e:
        raise Exception(f"Error reading input file {file_path}: {e}")

def main(input_file_path, games_file_path, output_file_path):
    """
    Main processing function for PBP data enhancement (without EPA)
    
    Args:
        input_file_path (str): Path to input PBP CSV file
        games_file_path (str): Path to games CSV file
        output_file_path (str): Path for output CSV file
    """
    
    setup_logging()
    
    print(f"\nPBP Data Enhancement (No EPA)")
    print(f"Input: {os.path.basename(input_file_path)}")
    print(f"Output: {os.path.basename(output_file_path)}")
    print("=" * 50)
    
    try:
        # Validate input file
        print("Validating input file...")
        validate_input_file(input_file_path)
        print("   SUCCESS: Input file validated")
        
        # Load PBP data
        print("Loading play-by-play data...")
        with tqdm(total=1, desc="Loading PBP", unit="file") as pbar:
            df = pd.read_csv(input_file_path)
            pbar.update(1)
        
        print(f"   SUCCESS: Loaded {len(df):,} plays")
        logging.info(f"Loaded {len(df)} plays from PBP file")
        
        # Load games data if available
        games = None
        if games_file_path and os.path.exists(games_file_path):
            try:
                print("Loading games data...")
                with tqdm(total=1, desc="Loading games", unit="file") as pbar:
                    games = pd.read_csv(games_file_path)
                    pbar.update(1)
                
                print(f"   SUCCESS: Loaded {len(games):,} games")
                logging.info(f"Loaded {len(games)} games from games file")
            except Exception as e:
                print("   WARNING: Failed to load games file - continuing without")
                logging.warning(f"Failed to load games file: {e}")
        else:
            print("No games file found - continuing without games data")
            logging.warning("Games file not found, continuing without games data...")
        
        # Data cleaning
        print("Cleaning data...")
        initial_count = len(df)
        df = df.loc[(df['stat_yardage'] < 100) & (df['stat_yardage'] > -100)]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"   Filtered out {filtered_count:,} plays with extreme yardage")
            logging.info(f"Filtered out {filtered_count} plays with extreme yardage values")
        else:
            print("   SUCCESS: No extreme values found")
        
        if df.empty:
            print("ERROR: No data remaining after filtering")
            raise Exception("DataFrame is empty after filtering")
        
        # Basic processing steps (without EPA)
        processing_steps = [
            ("Binary play statistics", add_binary_play_stats),
            ("Cumulative statistics", generate_cum_stats),
            ("Derived features", calc_new_features)
        ]
        
        for step_name, step_func in processing_steps:
            print(f"{step_name}...")
            with tqdm(total=1, desc=step_name.split()[-1], unit="dataset") as pbar:
                df = step_func(df)
                pbar.update(1)
            print("   SUCCESS: Completed")
        
        # Default win probability values (simple placeholder)
        print("Adding default probability values...")
        df['home_win'] = 0.5
        df['tie'] = 0.0  
        df['away_win'] = 0.5
        df['garbage_time_ind'] = 0
        print("   SUCCESS: Default values added")
        logging.info("Added default probability values")
        
        # Prepare output
        print("Preparing output...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set index if 'id' column exists
        if 'id' in df.columns:
            df.set_index('id', inplace=True)
            logging.info("Set 'id' column as index")
        
        # Save to CSV
        with tqdm(total=1, desc="Saving", unit="file") as pbar:
            df.to_csv(output_file_path)
            pbar.update(1)
        
        print(f"\nSUCCESS: Enhancement Complete!")
        print(f"Enhanced plays: {len(df):,}")
        print(f"Features added: Binary stats, Cumulative stats (NO EPA)")
        print(f"Output file: {output_file_path}")
        
        # Show column count
        new_features = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['cum_', 'success', 'explosive', 'yards_per', 'win'])]
        print(f"New feature columns: {len(new_features)}")
        
        logging.info(f"Successfully saved {len(df)} enhanced plays to {output_file_path}")
        logging.info("PBP data processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Fatal error during PBP processing: {e}")
        print(f"\nERROR: Enhancement failed: {e}")
        raise

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Process and enhance play-by-play CSV data with basic statistics (no EPA)'
    )
    
    parser.add_argument('--input_file_path', 
                       default='temp/pbp.csv',
                       help='Path to input play-by-play CSV file')
    
    parser.add_argument('--output_file_path',
                       default='temp/pbp_edit.csv', 
                       help='Path for output enhanced CSV file')
    
    parser.add_argument('--games_file_path',
                       default=None,
                       help='Path to games CSV file (optional, will auto-detect if not provided)')
    
    args = parser.parse_args()
    
    # Auto-detect games file if not provided
    if args.games_file_path is None:
        args.games_file_path = find_games_file()
    
    try:
        main(
            input_file_path=args.input_file_path,
            games_file_path=args.games_file_path,
            output_file_path=args.output_file_path
        )
        print("\nSUCCESS: PBP enhancement completed successfully!")
        
    except Exception as e:
        print(f"\nERROR: Script failed: {e}")
        print("Check temp/pbp_processing.log for detailed error information")
        logging.error(f"Script failed: {e}")
        sys.exit(1)