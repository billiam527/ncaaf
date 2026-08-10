# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 14:37:21 2025

@author: wfish
"""

import pandas as pd
import numpy as np
import pickle
from itertools import product
import time

def create_time_buckets(half_seconds_remaining):
    """Bucket time remaining into larger groups"""
    if half_seconds_remaining <= 300:
        # Last 5 minutes: 30-second intervals
        return int(half_seconds_remaining // 30) * 30
    else:
        # Rest of half: 60-second intervals
        return int(half_seconds_remaining // 60) * 60

def generate_ep_lookup_table(model_file='ep_model_5yr.pkl', scoring_file='scoring_array_5yr.npy'):
    """
    Generate a complete lookup table for all possible game situations
    
    Returns:
        DataFrame with columns: down, distance, yards_to_end_zone, time_bucket, expected_points
    """
    
    print("GENERATING EXPECTED POINTS LOOKUP TABLE")
    print("=" * 50)
    
    # Load trained model
    print("Loading model...")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        scoring_array = np.load(scoring_file)
        print(f"✓ Model loaded: {model_file}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Define all possible feature combinations
    downs = [1, 2, 3, 4]
    distances = list(range(1, 26))  # 1-25 yards (capped)
    yards_to_endzone = list(range(1, 100))  # 1-99 yards
    
    # Create time buckets
    time_buckets = []
    # Last 5 minutes: 0, 30, 60, ..., 300
    time_buckets.extend(range(0, 301, 30))
    # Rest of half: 360, 420, ..., 1800  
    time_buckets.extend(range(360, 1801, 60))
    
    print(f"Feature ranges:")
    print(f"  Downs: {len(downs)} values")
    print(f"  Distances: {len(distances)} values")
    print(f"  Yards to endzone: {len(yards_to_endzone)} values")
    print(f"  Time buckets: {len(time_buckets)} values")
    
    total_combinations = len(downs) * len(distances) * len(yards_to_endzone) * len(time_buckets)
    print(f"  Total combinations: {total_combinations:,}")
    
    # Generate all combinations
    print(f"\nGenerating all feature combinations...")
    all_combinations = list(product(downs, distances, yards_to_endzone, time_buckets))
    print(f"✓ Generated {len(all_combinations):,} combinations")
    
    # Filter out invalid combinations (distance > yards_to_endzone)
    print("Filtering invalid combinations...")
    valid_combinations = []
    for down, distance, yards_to_endzone, time_bucket in all_combinations:
        if distance <= yards_to_endzone:  # Can't need more yards than distance to endzone
            valid_combinations.append((down, distance, yards_to_endzone, time_bucket))
    
    print(f"✓ Valid combinations: {len(valid_combinations):,}")
    invalid_filtered = len(all_combinations) - len(valid_combinations)
    print(f"  Filtered out {invalid_filtered:,} invalid combinations")
    
    # Predict Expected Points for all valid combinations
    print(f"\nCalculating Expected Points...")
    start_time = time.time()
    
    # Convert to array for batch prediction
    X = np.array(valid_combinations)
    
    # Predict probabilities for all combinations at once
    print("  Running model predictions...")
    probabilities = model.predict_proba(X)
    
    # Calculate expected points
    expected_points = np.dot(probabilities, scoring_array)
    
    prediction_time = time.time() - start_time
    print(f"✓ Predictions completed in {prediction_time:.1f} seconds")
    print(f"  Rate: {len(valid_combinations)/prediction_time:,.0f} predictions/second")
    
    # Create lookup table DataFrame
    print("Creating lookup table...")
    lookup_table = pd.DataFrame({
        'down': X[:, 0],
        'distance': X[:, 1], 
        'yards_to_end_zone': X[:, 2],
        'time_bucket': X[:, 3],
        'expected_points': expected_points
    })
    
    # Add some useful derived columns
    lookup_table['situation'] = (lookup_table['down'].astype(str) + 
                                 ' & ' + 
                                 lookup_table['distance'].astype(str))
    
    # Sort for easier inspection
    lookup_table = lookup_table.sort_values(['down', 'distance', 'yards_to_end_zone', 'time_bucket'])
    
    # Display sample results
    print(f"\nLOOKUP TABLE SUMMARY:")
    print(f"  Total entries: {len(lookup_table):,}")
    print(f"  EP range: {lookup_table['expected_points'].min():.3f} to {lookup_table['expected_points'].max():.3f}")
    print(f"  EP mean: {lookup_table['expected_points'].mean():.3f}")
    
    # Show some sample situations
    print(f"\nSAMPLE EXPECTED POINTS:")
    sample_situations = [
        (1, 10, 75, 900),   # 1st & 10 at own 25, mid-quarter
        (1, 10, 25, 900),   # 1st & 10 in red zone
        (3, 8, 45, 300),    # 3rd & 8 at midfield, late in half
        (4, 2, 35, 120),    # 4th & 2 in FG range, crunch time
        (1, 10, 10, 60)     # 1st & 10 at goal line, final minute
    ]
    
    for down, distance, yards, time_bucket in sample_situations:
        # Find closest time bucket
        actual_time_bucket = create_time_buckets(time_bucket)
        
        row = lookup_table[
            (lookup_table['down'] == down) & 
            (lookup_table['distance'] == distance) & 
            (lookup_table['yards_to_end_zone'] == yards) & 
            (lookup_table['time_bucket'] == actual_time_bucket)
        ]
        
        if len(row) > 0:
            ep = row['expected_points'].iloc[0]
            situation = row['situation'].iloc[0]
            print(f"  {situation} at {100-yards} yard line, {time_bucket}s: EP = {ep:5.2f}")
        else:
            print(f"  {down} & {distance} at {100-yards} yard line: Not found")
    
    return lookup_table

def save_lookup_table(lookup_table, filename='ep_lookup_table.csv'):
    """Save lookup table to CSV for fast loading"""
    print(f"\nSaving lookup table...")
    lookup_table.to_csv(filename, index=False)
    
    file_size = len(lookup_table) * len(lookup_table.columns) * 8 / (1024*1024)  # Rough MB estimate
    print(f"✓ Saved to {filename}")
    print(f"  File size: ~{file_size:.1f} MB")
    print(f"  Rows: {len(lookup_table):,}")
    
    return filename

def load_lookup_table(filename='ep_lookup_table.csv'):
    """Load pre-computed lookup table"""
    return pd.read_csv(filename)

def lookup_expected_points(lookup_table, down, distance, yards_to_end_zone, half_seconds_remaining):
    """
    Fast lookup of Expected Points for a given situation
    
    Args:
        lookup_table: Pre-computed EP lookup table
        down: Down (1-4)
        distance: Distance to first down (1-25)
        yards_to_end_zone: Yards to end zone (1-99)
        half_seconds_remaining: Seconds left in half (0-1800)
    
    Returns:
        expected_points: EP for this situation
    """
    
    # Apply same constraints as training
    down = max(1, min(4, down))
    distance = max(1, min(25, distance))
    yards_to_end_zone = max(1, min(99, yards_to_end_zone))
    half_seconds_remaining = max(0, min(1800, half_seconds_remaining))
    
    # Convert to time bucket
    time_bucket = create_time_buckets(half_seconds_remaining)
    
    # Lookup
    row = lookup_table[
        (lookup_table['down'] == down) & 
        (lookup_table['distance'] == distance) & 
        (lookup_table['yards_to_end_zone'] == yards_to_end_zone) & 
        (lookup_table['time_bucket'] == time_bucket)
    ]
    
    if len(row) > 0:
        return row['expected_points'].iloc[0]
    else:
        # Fallback: return 0 if exact situation not found
        return 0.0

# Example usage
if __name__ == '__main__':
    # Generate the lookup table
    lookup_table = generate_ep_lookup_table()
    
    if lookup_table is not None:
        # Save it
        filename = save_lookup_table(lookup_table)
        
        # Test the lookup function
        print(f"\nTESTING LOOKUP FUNCTION:")
        test_ep = lookup_expected_points(lookup_table, 1, 10, 75, 900)
        print(f"1st & 10 at own 25, mid-quarter: EP = {test_ep:.3f}")
        
        print(f"\nLookup table generation complete!")
        print(f"Use: lookup_expected_points(table, down, distance, yards_to_endzone, seconds)")