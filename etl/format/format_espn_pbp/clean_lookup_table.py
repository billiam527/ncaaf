# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 21:27:42 2025

@author: wfish
"""

#!/usr/bin/env python3
"""
Clean Expected Points Lookup Table Script

This script takes the output from the anomaly detection/smoothing process and creates
a clean expected points lookup table by:
1. Removing all anomaly detection columns
2. Renaming expected_points_smoothed to expected_points  
3. Saving as expected_points_lookup_table.csv

Usage:
    python clean_ep_table.py --input_file expected_points_lookup_table_flagged.csv
"""

import pandas as pd
import argparse
import os

def clean_expected_points_table(input_file: str, output_file: str = None):
    """
    Clean the expected points table by removing anomaly columns and renaming smoothed column
    
    Args:
        input_file: Path to the flagged/smoothed expected points file
        output_file: Path for the clean output file (default: expected_points_lookup_table.csv)
    """
    
    if output_file is None:
        output_file = "expected_points_lookup_table.csv"
    
    try:
        print(f"Loading data from: {input_file}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load the data
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Define columns to remove
        columns_to_remove = [
            'expected_points',  # Remove original, keep smoothed version
            'yards_anomaly',
            'yards_anomaly_severity', 
            'yards_anomaly_ep_change',
            'down_anomaly',
            'down_anomaly_severity',
            'down_anomaly_ep_change',
            'distance_anomaly',
            'distance_anomaly_severity', 
            'distance_anomaly_ep_change',
            'was_smoothed',
            'any_anomaly',
            'anomaly_count',
            'overall_severity',
            'critical_anomaly'
        ]
        
        # Remove columns that exist in the dataframe
        columns_found = [col for col in columns_to_remove if col in df.columns]
        columns_missing = [col for col in columns_to_remove if col not in df.columns]
        
        if columns_found:
            df = df.drop(columns=columns_found)
            print(f"   Removed {len(columns_found)} anomaly detection columns")
        
        if columns_missing:
            print(f"   Note: {len(columns_missing)} expected columns not found: {columns_missing}")
        
        # Rename expected_points_smoothed to expected_points
        if 'expected_points_smoothed' in df.columns:
            df = df.rename(columns={'expected_points_smoothed': 'expected_points'})
            print("   Renamed 'expected_points_smoothed' to 'expected_points'")
        else:
            raise ValueError("Expected column 'expected_points_smoothed' not found in input file")
        
        # Verify we have the essential columns
        required_columns = ['half_seconds_remaining', 'down', 'distance', 'yards_to_goal', 'expected_points']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns after cleaning: {missing_required}")
        
        # Save the cleaned file
        print(f"Saving clean lookup table to: {output_file}")
        df.to_csv(output_file, index=False)
        
        # Summary
        print(f"\nSUCCESS: Clean expected points lookup table created")
        print(f"   Output file: {output_file}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Essential columns preserved: {required_columns}")
        
        # Show EP statistics
        ep_stats = df['expected_points'].describe()
        print(f"\nExpected Points Statistics:")
        print(f"   Range: {ep_stats['min']:.3f} to {ep_stats['max']:.3f}")
        print(f"   Mean: {ep_stats['mean']:.3f}")
        print(f"   Standard deviation: {ep_stats['std']:.3f}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Clean expected points lookup table')
    
    parser.add_argument('--input_file', '-i',
                       default='expected_points_lookup_table_flagged.csv',
                       help='Input CSV file with anomaly detection results')
    
    parser.add_argument('--output_file', '-o',
                       default='expected_points_lookup_table.csv', 
                       help='Output clean CSV file')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Process the file
    result = clean_expected_points_table(args.input_file, args.output_file)
    
    if result is not None:
        print("\nCleaning completed successfully!")
        return 0
    else:
        print("\nCleaning failed!")
        return 1

if __name__ == '__main__':
    exit(main())