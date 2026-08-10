import pandas as pd
import argparse
import logging
import sys
from tqdm import tqdm

# Configure clean logging
def setup_logging():
    """Setup clean logging - only errors and major progress to file"""
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler('temp/merge.log', mode='w')
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

def read_in_data(file1: str, file2: str):
    """
    Read CSV files into DataFrames with error handling.

    Args:
        file1 (str): File path of first CSV file
        file2 (str): File path of second CSV file

    Returns:
        tuple: (df1, df2) pandas DataFrames
    """
    try:
        print("Reading data files...")
        
        with tqdm(total=2, desc="Loading files", unit="files") as pbar:
            df1 = pd.read_csv(file1)
            pbar.set_postfix(file=f"{len(df1)} rows")
            pbar.update(1)
            
            df2 = pd.read_csv(file2)
            pbar.set_postfix(file=f"{len(df2)} rows")
            pbar.update(1)
        
        print(f"   File 1: {len(df1):,} rows, {len(df1.columns)} columns")
        print(f"   File 2: {len(df2):,} rows, {len(df2.columns)} columns")
        
        logging.info(f"File 1 loaded: {len(df1)} rows, {len(df1.columns)} columns")
        logging.info(f"File 2 loaded: {len(df2)} rows, {len(df2.columns)} columns")
        
        return df1, df2
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Error reading files: {e}")
        logging.error(f"Error reading files: {e}")
        raise

def merge(df1: pd.DataFrame, df2: pd.DataFrame, on: str, how: str, fillna):
    """
    Merge two DataFrames with data cleaning and validation.

    Args:
        df1 (pd.DataFrame): First DataFrame to merge
        df2 (pd.DataFrame): Second DataFrame to merge  
        on (str): Column to merge on
        how (str): How to merge ('inner', 'outer', 'left', 'right')
        fillna: Value to fill missing values with

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        print("Merging DataFrames...")
        
        # Clean the merge column in df1 (remove timestamp)
        if on in df1.columns:
            print(f"   Cleaning '{on}' column...")
            df1[on] = df1[on].astype(str).str.split('T').str[0]
            logging.info(f"Cleaned {on} column in df1")
        
        # Validate merge column exists in both DataFrames
        if on not in df1.columns:
            raise ValueError(f"Column '{on}' not found in first DataFrame")
        if on not in df2.columns:
            raise ValueError(f"Column '{on}' not found in second DataFrame")
        
        # Perform merge with progress indication
        with tqdm(total=1, desc=f"Merging on '{on}'", unit="merge") as pbar:
            merged_df = pd.merge(df1, df2, on=on, how=how)
            pbar.update(1)
        
        # Fill missing values
        if fillna is not None:
            print(f"   Filling missing values with: {fillna}")
            merged_df = merged_df.fillna(fillna)
            logging.info(f"Filled missing values with: {fillna}")
        
        print(f"   Merge completed: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
        logging.info(f"Merge completed: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df
        
    except Exception as e:
        print(f"ERROR: Error during merge: {e}")
        logging.error(f"Error during merge: {e}")
        raise

def create_score_differentials_and_total(df: pd.DataFrame):
    """
    Create additional calculated columns for game analysis.

    Args:
        df (pd.DataFrame): DataFrame with game data

    Returns:
        pd.DataFrame: DataFrame with additional calculated columns
    """
    try:
        print("Creating calculated columns...")
        calculations_made = 0
        
        # Game totals and differentials
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df['home_score_differential'] = df['home_score'] - df['away_score']
            df['total'] = df['home_score'] + df['away_score']
            calculations_made += 2
        
        # First quarter
        if all(col in df.columns for col in ['home_first_quarter', 'away_first_quarter']):
            df['home_first_quarter_score_differential'] = df['home_first_quarter'] - df['away_first_quarter']
            df['first_quarter_total'] = df['home_first_quarter'] + df['away_first_quarter']
            calculations_made += 2
        
        # First half
        if all(col in df.columns for col in ['home_first_quarter', 'home_second_quarter', 'away_first_quarter', 'away_second_quarter']):
            df['home_first_half_score_differential'] = (df['home_first_quarter'] + df['home_second_quarter']) - \
                                                     (df['away_first_quarter'] + df['away_second_quarter'])
            df['first_half_total'] = df['home_first_quarter'] + df['home_second_quarter'] + \
                                   df['away_first_quarter'] + df['away_second_quarter']
            calculations_made += 2
        
        print(f"   Created {calculations_made} calculated columns")
        logging.info("Successfully created additional calculated columns")
        return df
        
    except Exception as e:
        print(f"ERROR: Error creating calculated columns: {e}")
        logging.error(f"Error creating calculated columns: {e}")
        raise

if __name__ == '__main__':
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description='Merge two CSV files with optional data transformations'
    )
    
    parser.add_argument('--file1', required=True, help='First CSV file to merge (file path)')
    parser.add_argument('--file2', required=True, help='Second CSV file to merge (file path)')
    parser.add_argument('--on', required=True, help='Column to merge on')
    parser.add_argument('--how', default='left', help='How to merge (inner, outer, left, right)')
    parser.add_argument('--fillna', default=None, help='Value to fill missing data with')
    
    args = parser.parse_args()
    
    print(f"\nDataFrame Merge Operation")
    print(f"Merge column: {args.on}")
    print(f"Merge type: {args.how}")
    print("=" * 40)
    
    try:
        # Read data
        df1, df2 = read_in_data(args.file1, args.file2)
        
        # Merge data
        merged_df = merge(df1, df2, args.on, args.how, args.fillna)
        
        # Create additional columns
        final_df = create_score_differentials_and_total(merged_df)
        
        # Save result
        print("Saving merged data...")
        output_file = 'temp/new_games.csv'
        final_df.to_csv(output_file, index=False)
        
        print(f"\nSUCCESS: Merge Complete!")
        print(f"Final result: {len(final_df):,} rows, {len(final_df.columns)} columns")
        print(f"Output file: {output_file}")
        
        logging.info(f"Successfully saved merged data to {output_file}")
        
    except Exception as e:
        print(f"\nERROR: Script failed: {e}")
        print("Check temp/merge.log for detailed error information")
        logging.error(f"Script failed: {e}")
        sys.exit(1)