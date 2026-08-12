#!/usr/bin/env python3
"""
College Football Play-by-Play Enhancement Script
Features: Progress bars, 5-year rolling training window
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import argparse
import sys
from typing import Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two values, returning default if denominator is 0 or NaN"""
    try:
        result = numerator / denominator
        return result.fillna(default) if hasattr(result, 'fillna') else (default if pd.isna(result) else result)
    except (ZeroDivisionError, TypeError):
        return default

def add_binary_play_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary play statistics for the dataframe
    
    Args:
        df: Play-by-play DataFrame
        
    Returns:
        DataFrame with added binary statistics
    """
    logging.info("Adding binary play statistics...")
    
    # Define play type lists
    offensive_plays = [
        3,   # Pass incompletion
        4,   # Pass completion
        5,   # Rush
        6,   # Pass interception
        7,   # Sack
        8,   # Penalty
        9,   # Fumble recovery (own)
        24,  # Pass reception
        26,  # Pass interception return
        29,  # Fumble recovery (opp)
        36,  # Interception return (TD)
        51,  # Pass
        63,  # Interception
        67,  # Pass TD
        68   # Rush TD
    ]
    
    special_teams_plays = [
        12,  # Kickoff return
        17,  # Punt return
        18,  # Field Goal attempt
        32,  # Kickoff Return
        34,  # Punt Return
        36,  # Interception return (TD) - also in offensive
        37,  # Fumble Return TD
        38,  # Punt Return TD
        39,  # Kickoff Return TD
        40,  # Field Goal good
        41,  # Field Goal missed
        52,  # Punt
        53,  # Kickoff
        56,  # Two Point Conversion Good
        57,  # Two Point Conversion Failed
        59,  # Safety
        60,  # Field Goal Missed
        65   # End of half/game
    ]
    
    rushing_plays = [
        5,   # Rush
        68   # Rush TD
    ]
    
    passing_plays = [
        3,   # Pass incompletion
        4,   # Pass completion
        6,   # Pass interception
        7,   # Sack
        24,  # Pass reception
        26,  # Pass interception return
        36,  # Interception return (TD)
        51,  # Pass
        63,  # Interception
        67   # Pass TD
    ]
    
    # Initialize all binary columns
    binary_columns = [
        'offensive_play', 'special_teams_play', 'rushing_play', 'passing_play',
        'successful_play', 'rushing_successful_play', 'passing_successful_play',
        'explosive_play_eligible', 'explosive_play', 'rushing_explosive_play_eligible',
        'rushing_explosive_play', 'passing_explosive_play_eligible', 
        'passing_explosive_play', 'turnover_on_downs', 'turnover'
    ]
    
    for col in binary_columns:
        df[col] = 0
    
    # Progress bar for play type assignment
    with tqdm(total=len(binary_columns), desc="Assigning play types", unit="type") as pbar:
        # Offensive plays
        df.loc[df['play_type_id'].isin(offensive_plays), 'offensive_play'] = 1
        pbar.update(1)
        
        # Special teams plays
        df.loc[df['play_type_id'].isin(special_teams_plays), 'special_teams_play'] = 1
        pbar.update(1)
        
        # Rushing plays
        df.loc[df['play_type_id'].isin(rushing_plays), 'rushing_play'] = 1
        # Add fumbles on running plays
        fumble_mask = (df['play_type_id'].isin([9, 29])) & (df['play_text'].str.contains('run ', na=False))
        df.loc[fumble_mask, 'rushing_play'] = 1
        pbar.update(1)
        
        # Passing plays
        df.loc[df['play_type_id'].isin(passing_plays), 'passing_play'] = 1
        # Add fumbles on passing plays
        fumble_pass_mask = (df['play_type_id'].isin([9, 29])) & (df['play_text'].str.contains('pass ', na=False))
        df.loc[fumble_pass_mask, 'passing_play'] = 1
        pbar.update(1)
        
        # Play success calculations
        success_conditions = [
            # 1st down: >= 50% of distance
            (df['down'] == 1) & (df['distance'] > 0) & ((df['stat_yardage'] / df['distance']) >= 0.5),
            # 2nd down: >= 75% of distance  
            (df['down'] == 2) & (df['distance'] > 0) & ((df['stat_yardage'] / df['distance']) >= 0.75),
            # 3rd/4th down: >= 100% of distance
            (df['down'] >= 3) & (df['distance'] > 0) & ((df['stat_yardage'] / df['distance']) >= 1.0)
        ]
        
        for condition in success_conditions:
            df.loc[df['play_type_id'].isin(offensive_plays) & condition, 'successful_play'] = 1
        pbar.update(1)
        
        # Initialize yardage columns
        yardage_columns = ['offensive_yards', 'rushing_yards', 'passing_yards']
        for col in yardage_columns:
            df[col] = 0.0
        
        # Offensive yards
        df.loc[df['play_type_id'].isin(offensive_plays), 'offensive_yards'] = df['stat_yardage'].fillna(0)
        pbar.update(1)
        
        # Rushing yards
        df.loc[df['play_type_id'].isin(rushing_plays), 'rushing_yards'] = df['stat_yardage'].fillna(0)
        df.loc[fumble_mask, 'rushing_yards'] = df['stat_yardage'].fillna(0)
        pbar.update(1)
        
        # Passing yards
        df.loc[df['play_type_id'].isin(passing_plays), 'passing_yards'] = df['stat_yardage'].fillna(0)
        df.loc[fumble_pass_mask, 'passing_yards'] = df['stat_yardage'].fillna(0)
        pbar.update(1)
        
        # Successful play breakdowns
        df.loc[(df['rushing_play'] == 1) & (df['successful_play'] == 1), 'rushing_successful_play'] = 1
        df.loc[(df['passing_play'] == 1) & (df['successful_play'] == 1), 'passing_successful_play'] = 1
        pbar.update(1)
        
        # Explosive plays (25+ yards when >= 25 yards to endzone)
        explosive_eligible = (df['play_type_id'].isin(offensive_plays)) & (df['yards_to_end_zone'] >= 25)
        df.loc[explosive_eligible, 'explosive_play_eligible'] = 1
        df.loc[explosive_eligible & (df['offensive_yards'] >= 25), 'explosive_play'] = 1
        pbar.update(1)
        
        # Explosive rushing plays
        rush_explosive_eligible = (df['rushing_play'] == 1) & (df['yards_to_end_zone'] >= 25)
        df.loc[rush_explosive_eligible, 'rushing_explosive_play_eligible'] = 1
        df.loc[rush_explosive_eligible & (df['rushing_yards'] >= 25), 'rushing_explosive_play'] = 1
        pbar.update(1)
        
        # Explosive passing plays
        pass_explosive_eligible = (df['passing_play'] == 1) & (df['yards_to_end_zone'] >= 25)
        df.loc[pass_explosive_eligible, 'passing_explosive_play_eligible'] = 1
        df.loc[pass_explosive_eligible & (df['passing_yards'] >= 25), 'passing_explosive_play'] = 1
        pbar.update(1)
        
        # Turnover on downs
        # when play is fourth down AND does not equal field goal make,
        # turnover play (int or fumble), punt (treated as first down for opponent), 
        # safety, or a timeout and stat_yardage < distance
        not_play_types = [21, # timeout
                          59, # field goal make
                          8,  # penalty
                          29, # fumble recovery (opp)
                          26, # interception
                          36, # interception
                          63, # interception
                          20, # safety
                          52  # punt
                          ]
        df.loc[(df['down'] == 4) & \
            (~df['play_type_id'].isin(not_play_types)) & \
            (df['stat_yardage'] < df['distance']),\
            'turnover_on_downs'] = 1
        pbar.update(1)
        
        # Turnover (non touchdown)
        turnovers = [6, 26, 29, 60, 63]
        df.loc[df['play_type_id'].isin(turnovers), 'turnover'] = 1
        df.loc[df['turnover_on_downs'] == 1, 'turnover'] = 1
        pbar.update(1)
    
    logging.info(f"Added binary statistics for {len(df):,} plays")
    return df

# Add this function after the add_binary_play_stats function:

def clean_special_teams_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean special teams play data that has incorrect down/distance values
    
    Args:
        df: Play-by-play DataFrame
        
    Returns:
        DataFrame with cleaned special teams data
    """
    logging.info("Cleaning special teams play data...")
    
    try:
        # Define special teams play types that should have standard values
        kickoff_plays = [12, 53]  # Kickoff
        punt_plays = [52]     # Punt
        fg_plays = [18, 40, 41, 60]  # Field goal attempts/makes/misses
        
        # Fix kickoffs - should be 1st & 10 at receiving team's 25 (65 yards to endzone after touchback)
        kickoff_mask = df['play_type_id'].isin(kickoff_plays)
        if kickoff_mask.sum() > 0:
            print(f"   Fixing {kickoff_mask.sum()} kickoff plays")
            df.loc[kickoff_mask, 'down'] = 1
            df.loc[kickoff_mask, 'distance'] = 10
            df.loc[kickoff_mask, 'yards_to_end_zone'] = 75
            
        # Create a mask for all kickoff plays (play_type_id 12 and 53)
        kickoff_mask = df['play_type_id'].isin([12, 53])
        
        # For all kickoffs, switch team_id to the next play's team_id (receiving team)
        if kickoff_mask.any():
            kickoff_indices = df[kickoff_mask].index.tolist()
            
            for idx in kickoff_indices:
                try:
                    # Find the position of current index in the DataFrame
                    current_pos = df.index.get_loc(idx)
                    # Get next row's team_id if it exists
                    if current_pos + 1 < len(df):
                        next_idx = df.index[current_pos + 1]
                        df.loc[idx, 'team_id'] = df.loc[next_idx, 'team_id']
                    # If no next play exists, keep the current team_id (fallback)
                except KeyError:
                    # If there's an issue finding the index, keep original team_id
                    pass
            
        # Handling punts add to pbp_edit
        # If "return" in play_text count EPA for the returning team (assign team_id to return team)
        # If no return take the dif between 4th down EP and next -EP of the next possession
        # Create a mask for punt plays (play_type_id 52)
        punt_mask = df['play_type_id'] == 52

        # Create a mask for punts with returns
        punt_return_mask = punt_mask & df['play_text'].str.contains('return', case=False, na=False) & ~df['play_text'].str.contains('no return', case=False, na=False)

        # For punt returns, switch team_id to the next play's team_id.
        # shift(-1) is positional, so this works whatever the index looks like.
        # The previous version did df.loc[idx + 1] with idx taken from the index
        # and guarded by `idx + 1 < len(df)` - that treats an index LABEL as a
        # position. Any non-contiguous index (which concatenated season files
        # produce) raised KeyError on the label, and because the whole block sits
        # in one try/except that aborted punt attribution and the field-goal
        # handling below for the entire file.
        if punt_return_mask.any():
            next_team = df['team_id'].shift(-1)
            # last row has no following play; keep its own team_id
            next_team.iloc[-1] = df['team_id'].iloc[-1]
            df.loc[punt_return_mask, 'team_id'] = next_team[punt_return_mask]
            
        # Fix field goals - these should keep the actual down/distance since they're kicking team plays
        # But we need to make sure they're not inheriting weird values
        fg_mask = df['play_type_id'].isin(fg_plays)
        if fg_mask.sum() > 0:
            print(f"   Field goals: {fg_mask.sum()} plays (keeping existing down/distance)")
            
        logging.info("Special teams data cleaned successfully")
        
    except Exception as e:
        logging.error(f"Error cleaning special teams data: {e}")
    
    return df

def add_expected_points(df: pd.DataFrame, ep_lookup_path: str = None) -> pd.DataFrame:
    """
    Add expected points column using lookup table joined by game situation
    Uses time buckets: 60 (≤60), 120 (61-120), 180 (121-180), 240 (181-240), 300 (241-300), 1500 (301+)
    
    Args:
        df: Play-by-play DataFrame
        ep_lookup_path: Path to expected points lookup table CSV
        
    Returns:
        DataFrame with expected_points column added
    """
    logging.info("Adding expected points from lookup table...")
    
    try:
        # Find the expected points lookup file
        if ep_lookup_path is None:
            possible_paths = [
                'expected_points_lookup_table.csv'
            ]
            
            ep_lookup_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    ep_lookup_path = path
                    break
            
            if ep_lookup_path is None:
                logging.warning("Expected points lookup file not found - setting default values")
                print("   Warning: Expected points lookup file not found - using defaults")
                df['expected_points'] = 0.0
                return df
        
        # Load the expected points lookup table
        logging.info(f"Loading expected points lookup from: {ep_lookup_path}")
        print(f"   Loading lookup table: {os.path.basename(ep_lookup_path)}")
        ep_lookup = pd.read_csv(ep_lookup_path)
        
        # Verify required columns exist in lookup table
        required_cols = ['half_seconds_remaining', 'down', 'distance', 'yards_to_goal', 'expected_points']
        missing_cols = [col for col in required_cols if col not in ep_lookup.columns]
        if missing_cols:
            logging.error(f"Missing columns in lookup table: {missing_cols}")
            print(f"   Error: Missing columns in lookup table: {missing_cols}")
            df['expected_points'] = 0.0
            return df
        
        print(f"   Loaded {len(ep_lookup):,} expected points lookup records")
        logging.info(f"Loaded {len(ep_lookup):,} expected points lookup records")
        
        # Calculate half_seconds_remaining for PBP data
        df['half_seconds_remaining'] = 0  # Default
        
        if 'period' in df.columns:
            # The clock arrives from ESPN in a column simply called 'clock',
            # holding strings like "14:50". The two names checked before it
            # ('clock_minutes'/'clock_seconds' and 'clock_display_value') do
            # not exist in this data, so every run fell through to the period
            # approximation below and half_seconds_remaining took one of two
            # values, 1200 or 600. That column is a feature of the expected
            # points lookup, so the EP model - and therefore every EPA-derived
            # statistic - was being built without real time remaining.
            def parse_clock_display(clock_str):
                try:
                    if pd.isna(clock_str) or clock_str == '':
                        return np.nan
                    parts = str(clock_str).strip().split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    if len(parts) == 3:            # occasionally h:mm:ss
                        return int(parts[1]) * 60 + int(parts[2])
                    return np.nan
                except (ValueError, TypeError):
                    return np.nan

            clock_col = next((c for c in ('clock', 'clock_display_value',
                                          'clock_text')
                              if c in df.columns), None)

            if 'clock_minutes' in df.columns and 'clock_seconds' in df.columns:
                # Calculate seconds remaining in current half
                df['clock_total_seconds'] = (df['clock_minutes'].fillna(0) * 60 +
                                            df['clock_seconds'].fillna(0))

                # For periods 1 and 3 (start of each half), add 15*60 = 900 seconds for the next quarter
                df['half_seconds_remaining'] = np.where(
                    df['period'].isin([1, 3]),
                    df['clock_total_seconds'] + 900,  # Current quarter + next quarter
                    df['clock_total_seconds']  # Just current quarter for periods 2, 4
                )
            elif clock_col is not None:
                parsed = df[clock_col].map(parse_clock_display)
                # Overtime periods have no meaningful half clock; treat them as
                # the end of regulation rather than letting them go negative.
                df['clock_total_seconds'] = parsed.fillna(0)
                df['half_seconds_remaining'] = np.where(
                    df['period'].isin([1, 3]),
                    df['clock_total_seconds'] + 900,
                    df['clock_total_seconds']
                )
                parsed_share = parsed.notna().mean()
                print(f"   Parsed clock from '{clock_col}' on "
                      f"{parsed_share:.1%} of plays")
                if parsed_share < 0.5:
                    logging.warning(
                        f"clock column '{clock_col}' parsed on only "
                        f"{parsed_share:.1%} of plays")
            else:
                # Use period to estimate time (rough approximation)
                logging.warning("no clock column found; half_seconds_remaining "
                                "falls back to a two-value period estimate")
                df['half_seconds_remaining'] = np.where(
                    df['period'].isin([1, 3]), 1200,  # Early in half
                    600  # Later in half
                )

        elif 'seconds_left_half' in df.columns:
            # Use existing column if available
            df['half_seconds_remaining'] = df['seconds_left_half'].fillna(1500)
        else:
            # Default to mid-game value
            df['half_seconds_remaining'] = 1500
        
        # Ensure required columns exist in PBP data with defaults
        if 'yards_to_goal' not in df.columns:
            if 'yards_to_end_zone' in df.columns:
                df['yards_to_goal'] = df['yards_to_end_zone']
            elif 'yardline_100' in df.columns:
                df['yards_to_goal'] = df['yardline_100']
            else:
                df['yards_to_goal'] = 50  # Default to midfield
        
        # Fill missing values with reasonable defaults for joining
        df['down'] = df['down'].fillna(1).astype(int)
        df['distance'] = df['distance'].fillna(10).astype(int)
        df['yards_to_goal'] = df['yards_to_goal'].fillna(50).astype(int)
        df['half_seconds_remaining'] = df['half_seconds_remaining'].fillna(1500).astype(int)
        
        # Define time bucket mapping function
        def map_to_time_bucket(seconds):
            """Map seconds to the appropriate time bucket"""
            if seconds <= 60:
                return 60
            elif seconds <= 120:
                return 120
            elif seconds <= 180:
                return 180
            elif seconds <= 240:
                return 240
            elif seconds <= 300:
                return 300
            else:
                return 1500
        
        # Get unique values from lookup table for other dimensions
        unique_times = sorted(ep_lookup['half_seconds_remaining'].unique())
        unique_downs = sorted(ep_lookup['down'].unique())
        unique_distances = sorted(ep_lookup['distance'].unique())
        unique_yards_to_goal = sorted(ep_lookup['yards_to_goal'].unique())
        
        print(f"   Lookup table structure:")
        print(f"     Time buckets: {unique_times}")
        print(f"     Downs: {unique_downs}")
        print(f"     Distance range: {min(unique_distances)}-{max(unique_distances)}")
        print(f"     Yards to goal range: {min(unique_yards_to_goal)}-{max(unique_yards_to_goal)}")
        
        # Distance binning - find closest distance in lookup table
        def find_closest_distance(dist):
            # Ensure distance is within reasonable bounds
            capped_dist = max(1, min(dist, max(unique_distances)))
            return min(unique_distances, key=lambda x: abs(x - capped_dist))
        
        # Yards to goal binning - find closest yards in lookup table
        def find_closest_yards_to_goal(yards):
            # Ensure yards is within reasonable bounds
            capped_yards = max(1, min(yards, max(unique_yards_to_goal)))
            return min(unique_yards_to_goal, key=lambda x: abs(x - capped_yards))
        
        with tqdm(total=5, desc="Processing situation variables", unit="step") as pbar:
            # Map to time buckets
            df['time_bucket'] = df['half_seconds_remaining'].apply(map_to_time_bucket)
            pbar.update(1)
            
            # Ensure downs are within lookup table range
            df['down_mapped'] = df['down'].clip(lower=min(unique_downs), upper=max(unique_downs))
            pbar.update(1)
            
            # Map distance and yards to closest available values
            df['distance_mapped'] = df['distance'].apply(find_closest_distance)
            pbar.update(1)
            
            df['yards_mapped'] = df['yards_to_goal'].apply(find_closest_yards_to_goal)
            pbar.update(1)
            
            # Create lookup dictionary for fast joining
            ep_lookup['lookup_key'] = (ep_lookup['half_seconds_remaining'].astype(str) + '_' +
                                     ep_lookup['down'].astype(str) + '_' +
                                     ep_lookup['distance'].astype(str) + '_' +
                                     ep_lookup['yards_to_goal'].astype(str))
            
            ep_dict = dict(zip(ep_lookup['lookup_key'], ep_lookup['expected_points']))
            
            # Create keys for PBP data
            df['lookup_key'] = (df['time_bucket'].astype(str) + '_' +
                              df['down_mapped'].astype(str) + '_' +
                              df['distance_mapped'].astype(str) + '_' +
                              df['yards_mapped'].astype(str))
            
            # Map expected points
            df['expected_points'] = df['lookup_key'].map(ep_dict).fillna(0.0)
            pbar.update(1)
        
        # Calculate match statistics
        matched_plays = (df['expected_points'] != 0.0).sum()
        total_plays = len(df)
        match_rate = matched_plays / total_plays if total_plays > 0 else 0
        avg_ep = df['expected_points'].mean()
        
        print(f"   SUCCESS: Expected points mapping complete")
        print(f"   Matched plays: {matched_plays:,} / {total_plays:,} ({match_rate:.1%})")
        print(f"   Average expected points: {avg_ep:.3f}")
        
        if match_rate < 0.5:  # Less than 50% match rate might indicate an issue
            print(f"   WARNING: Low match rate. Sample mismatched situations:")
            unmatched_sample = df[df['expected_points'] == 0.0][['time_bucket', 'down_mapped', 'distance_mapped', 'yards_mapped', 'lookup_key']].head(3)
            for _, row in unmatched_sample.iterrows():
                print(f"     Time:{row['time_bucket']} Down:{row['down_mapped']} Dist:{row['distance_mapped']} Yards:{row['yards_mapped']}")
        
        logging.info(f"Expected points mapping complete:")
        logging.info(f"  Matched plays: {matched_plays:,} / {total_plays:,} ({match_rate:.1%})")
        logging.info(f"  Average expected points: {avg_ep:.3f}")
        
        # Clean up temporary columns
        cleanup_cols = ['lookup_key', 'time_bucket', 'down_mapped', 'distance_mapped', 'yards_mapped']
        if 'clock_total_seconds' in df.columns:
            cleanup_cols.append('clock_total_seconds')
        
        df = df.drop(columns=cleanup_cols, errors='ignore')
        
        return df
        
    except Exception as e:
        logging.error(f"Error adding expected points: {e}")
        print(f"   Error adding expected points: {e}")
        import traceback
        traceback.print_exc()
        df['expected_points'] = 0.0
        return df
    
def calculate_epa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Expected Points Added (EPA) for each play
    
    EPA Formula:
    - For scoring plays: EPA = actual_points_scored - expected_points
    - For non-scoring plays: EPA = next_play_expected_points - current_expected_points
    
    Args:
        df: Play-by-play DataFrame with expected_points column
        
    Returns:
        DataFrame with EPA column added
    """
    logging.info("Calculating Expected Points Added (EPA)...")
    
    try:
        # Ensure we have the expected_points column
        if 'expected_points' not in df.columns:
            logging.warning("Expected points column not found - EPA will be 0")
            df['epa'] = 0.0
            return df
        
        # Initialize EPA column
        df['epa'] = 0.0
        df['actual_points'] = 0.0  # Points scored on this play
        
        # Define scoring play types and their point values
        scoring_plays = {
            # Touchdowns (6 points)
            32: 7, # Kickoff return TD
            67: 7,  # Pass TD
            68: 7,  # Rush TD  
            36: -7,  # Interception return TD
            37: -7,  # Blocked Punt Return TD
            38: 7,  # Punt Return TD
            39: -7,  # Fumble Return TD
            
            # Field Goals (3 points)
            40: 3,  # Field Goal good
            18: 3,  # Field Goal attempt (assuming good - may need refinement)
            59: 3,  # Field Goal good
            
            # Two Point Conversions (2 points)
            57: 2,  # Defensive Two Point Conversion Good
            
            # Safety (2 points for opposing team)
            20: 2,  # Safety
        }
        
        # Identify scoring plays and assign point values
        print("   Identifying scoring plays...")
        scoring_mask = df['play_type_id'].isin(scoring_plays.keys())
        
        with tqdm(total=4, desc="Calculating EPA", unit="step") as pbar:
            # Step 1: Assign actual points for scoring plays
            for play_type_id, points in scoring_plays.items():
                mask = df['play_type_id'] == play_type_id
                df.loc[mask, 'actual_points'] = points
            
            # Handle special cases
            # Field goal misses (0 points)
            fg_miss_mask = df['play_type_id'].isin([41, 60])  # FG missed
            df.loc[fg_miss_mask, 'actual_points'] = 0.0
            
            # Two point conversion failed (0 points)
            two_pt_fail_mask = df['play_type_id'] == 57
            df.loc[two_pt_fail_mask, 'actual_points'] = 0.0
            
            pbar.update(1)
            
            # Step 2: Handle safety scoring (opponent gets 2 points)
            # For safeties, the defending team gets 2 points, so EPA calculation is different
            safety_mask = df['play_type_id'] == 59
            if safety_mask.sum() > 0:
                # Safety gives 2 points to the opposing team
                df.loc[safety_mask, 'actual_points'] = -2  # Negative because it's points for opponent
            
            pbar.update(1)
            
            # Step 3: Calculate EPA for scoring plays
            # EPA = actual_points - expected_points
            df.loc[scoring_mask | safety_mask, 'epa'] = (
                df.loc[scoring_mask | safety_mask, 'actual_points'] - 
                df.loc[scoring_mask | safety_mask, 'expected_points']
            )
            
            pbar.update(1)
            
            # Step 4: Calculate EPA for Turnovers            
            turnover_mask = df['turnover'] == 1
            
            if turnover_mask.any():
                current_ep = df.loc[turnover_mask, 'expected_points']
                
                # Get the actual next row for each turnover, not just index + 1
                next_ep_values = []
                turnover_indices = df[turnover_mask].index.tolist()
                
                for idx in turnover_indices:
                    # Find the position of current index in the DataFrame
                    try:
                        current_pos = df.index.get_loc(idx)
                        # Get next row if it exists
                        if current_pos + 1 < len(df):
                            next_idx = df.index[current_pos + 1]
                            next_ep_values.append(df.loc[next_idx, 'expected_points'])
                        else:
                            next_ep_values.append(0)
                    except KeyError:
                        next_ep_values.append(0)
                
                # Calculate turnover EPA
                df.loc[turnover_mask, 'epa'] = -pd.Series(next_ep_values, index=turnover_indices) - current_ep
            pbar.update(1)
            
            # Step 5: Calculate EPA for non-scoring plays
            # EPA = next_play_EP - current_play_EP
            # Need to group by game and possession to get the next play's EP
            
            non_scoring_mask = ~(scoring_mask | safety_mask | turnover_mask)
            
            if 'game_id' in df.columns:
                # Sort by game and play sequence
                if 'sequence_number' in df.columns:
                    df_sorted = df.sort_values(['game_id', 'sequence_number'])
                elif 'id' in df.columns:
                    df_sorted = df.sort_values(['game_id', 'id'])
                else:
                    df_sorted = df.sort_values(['game_id', df.index])
                
                # Calculate next play's expected points within each game
                df_sorted['next_expected_points'] = df_sorted.groupby('game_id')['expected_points'].shift(-1)
                
                # For non-scoring plays: EPA = next_EP - current_EP
                non_scoring_epa = df_sorted['next_expected_points'].fillna(0) - df_sorted['expected_points']
                
                # Only apply to non-scoring plays
                df.loc[non_scoring_mask, 'epa'] = non_scoring_epa.loc[non_scoring_mask].fillna(0)
                
                # Handle end of game/half scenarios (when next_expected_points is NaN)
                end_of_possession_mask = df_sorted['next_expected_points'].isna() & non_scoring_mask
                df.loc[end_of_possession_mask, 'epa'] = -df.loc[end_of_possession_mask, 'expected_points']
                
            else:
                # Fallback: simple calculation without game context
                logging.warning("No game_id column found - using simplified EPA calculation")
                df['next_expected_points'] = df['expected_points'].shift(-1).fillna(0)
                df.loc[non_scoring_mask, 'epa'] = (
                    df.loc[non_scoring_mask, 'next_expected_points'] - 
                    df.loc[non_scoring_mask, 'expected_points']
                )
                df = df.drop(columns=['next_expected_points'], errors='ignore')
            
            pbar.update(1)
        
        # End of game / end of half
        df.loc[df['play_type_id'].isin([65, 66]), 'epa'] = 0
        df.loc[df['play_type_id'].isin([65, 66]), 'expected_points'] = 0
        
        # Two point trys in OT
        df.loc[df['play_type_id'].isin([15, 16]), 'epa'] = 0
        df.loc[df['play_type_id'].isin([15, 16]), 'expected_points'] = 0
        
        # Calculate statistics
        total_plays = len(df)
        scoring_plays_count = scoring_mask.sum()
        safety_plays_count = safety_mask.sum()
        avg_epa = df['epa'].mean()
        
        # Points distribution
        total_points_scored = df['actual_points'].sum()
        
        print(f"   SUCCESS: EPA calculation complete")
        print(f"   Total plays: {total_plays:,}")
        print(f"   Scoring plays: {scoring_plays_count:,}")
        print(f"   Safeties: {safety_plays_count:,}")
        print(f"   Total points scored: {total_points_scored:.0f}")
        print(f"   Average EPA: {avg_epa:.4f}")
        
        # Show EPA distribution by play type
        if scoring_plays_count > 0:
            scoring_epa_avg = df.loc[scoring_mask, 'epa'].mean()
            non_scoring_epa_avg = df.loc[non_scoring_mask, 'epa'].mean()
            print(f"   Average EPA (scoring plays): {scoring_epa_avg:.4f}")
            print(f"   Average EPA (non-scoring plays): {non_scoring_epa_avg:.4f}")
        
        logging.info(f"EPA calculation complete:")
        logging.info(f"  Total plays: {total_plays:,}, Scoring plays: {scoring_plays_count:,}")
        logging.info(f"  Average EPA: {avg_epa:.4f}")
        
        # Clean up temporary columns
        if 'next_expected_points' in df.columns:
            df = df.drop(columns=['next_expected_points'], errors='ignore')
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating EPA: {e}")
        print(f"   Error calculating EPA: {e}")
        import traceback
        traceback.print_exc()
        df['epa'] = 0.0
        df['actual_points'] = 0.0
        return df


def add_epa_to_cumulative_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EPA to the cumulative statistics calculations
    This should be called after generate_cum_stats() to include EPA in cumulative metrics
    
    Args:
        df: DataFrame with EPA column
        
    Returns:
        DataFrame with EPA added to cumulative stats
    """
    logging.info("Adding EPA to cumulative statistics...")
    
    try:
        if 'epa' not in df.columns:
            logging.warning("EPA column not found - skipping cumulative EPA stats")
            return df
        
        if 'game_id' not in df.columns or 'team_id' not in df.columns:
            logging.warning("Missing game_id or team_id columns for cumulative EPA stats")
            return df
        
        groupby_cols = ['game_id', 'team_id']
        
        # Add cumulative EPA
        df['cum_epa'] = df.groupby(groupby_cols)['epa'].cumsum().fillna(0)
        
        # Add cumulative EPA for different play types
        if 'rushing_play' in df.columns and 'passing_play' in df.columns:
            # Rushing EPA
            df['rush_epa'] = df['epa'] * df['rushing_play']
            df['cum_rush_epa'] = df.groupby(groupby_cols)['rush_epa'].cumsum().fillna(0)
            
            # Passing EPA  
            df['pass_epa'] = df['epa'] * df['passing_play']
            df['cum_pass_epa'] = df.groupby(groupby_cols)['pass_epa'].cumsum().fillna(0)
            
            # Clean up temporary columns
            df = df.drop(columns=['rush_epa', 'pass_epa'], errors='ignore')
        
        logging.info("EPA added to cumulative statistics successfully")
        
    except Exception as e:
        logging.error(f"Error adding EPA to cumulative stats: {e}")
    
    return df

def find_games_file():
    """Find the games.csv file in various possible locations, prioritizing collect folder"""
    possible_paths = [
        # First priority: collect folder (raw data before processing)
        '../../collect/collect_espn_games/temp/games.csv',
        '../../../collect/collect_espn_games/temp/games.csv',
        '../../collect/collect_espn_games/temp/games_2024.csv',
        '../../../collect/collect_espn_games/temp/games_2024.csv',
        '../../collect/collect_espn_games/games.csv',
        '../../../collect/collect_espn_games/games.csv',
        
        # Check for multi-year files in collect
        '../../collect/collect_espn_games/temp/games_2020-2024.csv',
        '../../collect/collect_espn_games/temp/games_multi_year.csv',
        '../../../collect/collect_espn_games/temp/games_2020-2024.csv',
        '../../../collect/collect_espn_games/temp/games_multi_year.csv',
        
        # Second priority: format folder (processed but not summarized)
        '../format_espn_games/temp/games.csv',
        '../../format_espn_games/temp/games.csv',
        '../format/format_espn_games/temp/games.csv',
        '../../format/format_espn_games/temp/games.csv',
        
        # Third priority: local and current directory
        'games.csv',
        'temp/games.csv',
        '../games.csv',
        
        # Last resort: summarize folder (current step)
        '../summarize/temp/games.csv',
        'summarize/temp/games.csv',
        '../../etl/summarize/temp/games.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it's a substantial file (not empty)
            try:
                df = pd.read_csv(path, nrows=5)  # Just check first few rows
                if len(df) > 0:
                    logging.info(f"Found games file at: {path}")
                    
                    # Load full file to check years available
                    full_df = pd.read_csv(path)
                    if 'season' in full_df.columns:
                        years = sorted(full_df['season'].unique())
                        logging.info(f"  Contains {len(years)} seasons: {years}")
                        print(f"   Found {len(years)} seasons of data: {years}")
                    else:
                        logging.info(f"  Contains {len(full_df):,} games (no season column)")
                        print(f"   Found {len(full_df):,} games")
                    
                    return path
            except Exception as e:
                logging.warning(f"Could not read {path}: {e}")
                continue
    
    logging.warning("Games file not found in any expected locations")
    print("   Warning: No games file found for EPA model training")
    return None

def add_basic_win_prob(df: pd.DataFrame,
                       include_garbage_time: bool = False) -> pd.DataFrame:
    """Add basic win probability calculations"""
    logging.info("Adding basic win probability...")
    
    try:
        if 'home_score' in df.columns and 'away_score' in df.columns:
            score_diff = df['home_score'].fillna(0) - df['away_score'].fillna(0)
            # Simple logistic function for win probability
            home_win_prob = 1 / (1 + np.exp(-score_diff / 10))
            df['home_win'] = home_win_prob
            df['away_win'] = 1 - home_win_prob
            df['tie'] = 0.05
            
            # Normalize probabilities
            total_prob = df['home_win'] + df['away_win'] + df['tie']
            df['home_win'] = df['home_win'] / total_prob
            df['away_win'] = df['away_win'] / total_prob
            df['tie'] = df['tie'] / total_prob
        else:
            df['home_win'] = 0.5
            df['away_win'] = 0.5
            df['tie'] = 0.0
            
        # Garbage time indicator.
        #
        # This previously keyed off the win probabilities above, which are a
        # logistic on score differential alone with no time component. After
        # the tie-normalisation divides by 1.05, "win prob > 0.9" resolves to a
        # lead of more than 28.4 points, so only period 3 and 4 plays in a
        # 29-point game were ever caught and 5.7% of the file remained at
        # second-half leads of 22 to 28.
        #
        # The thresholds below are the standard college-football ones, which
        # loosen as the clock runs down: a 24-point lead is a live game in the
        # first quarter and a finished one in the fourth.
        df['garbage_time_ind'] = 0
        if 'period' in df.columns and {'home_score', 'away_score'} <= set(df.columns):
            lead = (df['home_score'].fillna(0) - df['away_score'].fillna(0)).abs()
            period = pd.to_numeric(df['period'], errors='coerce')
            thresholds = {1: 43, 2: 37, 3: 27, 4: 22}
            garbage_condition = pd.Series(False, index=df.index)
            for qtr, margin in thresholds.items():
                garbage_condition |= (period == qtr) & (lead > margin)
            # overtime is never garbage time
            garbage_condition &= period <= 4
            df.loc[garbage_condition, 'garbage_time_ind'] = 1
            logging.info(f"garbage time flagged on "
                         f"{garbage_condition.mean():.2%} of plays")

    except Exception as e:
        logging.warning(f"Win probability calculation failed: {e}")
        df['home_win'] = 0.5
        df['away_win'] = 0.5
        df['tie'] = 0.0
        df['garbage_time_ind'] = 0
    
    return df

def generate_cum_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cumulative statistics by game and team
    
    Args:
        df: Play-by-play DataFrame
        
    Returns:
        DataFrame with cumulative statistics
    """
    logging.info("Generating cumulative statistics...")
    
    try:
        # Ensure we have required grouping columns
        if 'game_id' not in df.columns or 'team_id' not in df.columns:
            logging.warning("Missing game_id or team_id columns for cumulative stats")
            return df
        
        groupby_cols = ['game_id', 'team_id']
        
        # Cumulative play counts
        cum_columns = {
            'offensive_play': 'cum_plays',
            'rushing_play': 'cum_rushes', 
            'passing_play': 'cum_passes',
            'successful_play': 'cum_successful_plays',
            'rushing_successful_play': 'cum_successful_rush_plays',
            'passing_successful_play': 'cum_successful_pass_plays',
            'explosive_play_eligible': 'cum_eligible_explosive_plays',
            'rushing_explosive_play_eligible': 'cum_rushing_eligible_explosive_plays',
            'passing_explosive_play_eligible': 'cum_passing_eligible_explosive_plays',
            'explosive_play': 'cum_explosive_plays',
            'rushing_explosive_play': 'cum_rushing_explosive_plays',
            'passing_explosive_play': 'cum_passing_explosive_plays'
        }
        
        with tqdm(total=len(cum_columns) + 6, desc="Calculating cumulative stats", unit="stat") as pbar:
            for source_col, cum_col in cum_columns.items():
                if source_col in df.columns:
                    df[cum_col] = df.groupby(groupby_cols)[source_col].cumsum().fillna(0).astype(int)
                else:
                    df[cum_col] = 0
                pbar.update(1)
            
            # Cumulative yards
            yard_columns = {
                'offensive_yards': 'cum_yards',
                'rushing_yards': 'cum_rush_yards',
                'passing_yards': 'cum_pass_yards'
            }
            
            for source_col, cum_col in yard_columns.items():
                if source_col in df.columns:
                    df[cum_col] = df.groupby(groupby_cols)[source_col].cumsum().fillna(0)
                else:
                    df[cum_col] = 0.0
                pbar.update(1)
            
            # Cumulative EPA
            epa_columns = {
                'points_added': 'cum_epa',
                'epa_per_rush': 'cum_rush_epa', 
                'epa_per_pass': 'cum_pass_epa'
            }
            
            for source_col, cum_col in epa_columns.items():
                if source_col in df.columns:
                    df[cum_col] = df.groupby(groupby_cols)[source_col].cumsum().fillna(0)
                else:
                    df[cum_col] = 0.0
                pbar.update(1)
        
        logging.info("Cumulative statistics generated successfully")
        
    except Exception as e:
        logging.error(f"Error generating cumulative stats: {e}")
    
    return df

def calc_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features from cumulative statistics
    
    Args:
        df: DataFrame with cumulative statistics
        
    Returns:
        DataFrame with calculated features
    """
    logging.info("Calculating derived features...")
    
    try:
        feature_calculations = [
            ("Yards per play", lambda: safe_divide(df.get('cum_yards', 0), df.get('cum_plays', 1)).round(2)),
            ("Rush yards per play", lambda: safe_divide(df.get('cum_rush_yards', 0), df.get('cum_rushes', 1)).round(2)),
            ("Pass yards per play", lambda: safe_divide(df.get('cum_pass_yards', 0), df.get('cum_passes', 1)).round(2)),
            ("Play success rate", lambda: safe_divide(df.get('cum_successful_plays', 0), df.get('cum_plays', 1)).round(2)),
            ("Rush success rate", lambda: safe_divide(df.get('cum_successful_rush_plays', 0), df.get('cum_rushes', 1)).round(2)),
            ("Pass success rate", lambda: safe_divide(df.get('cum_successful_pass_plays', 0), df.get('cum_passes', 1)).round(2)),
            ("Explosive play rate", lambda: safe_divide(df.get('cum_explosive_plays', 0), df.get('cum_eligible_explosive_plays', 1)).round(2)),
            ("Explosive rush rate", lambda: safe_divide(df.get('cum_rushing_explosive_plays', 0), df.get('cum_rushing_eligible_explosive_plays', 1)).round(2)),
            ("Explosive pass rate", lambda: safe_divide(df.get('cum_passing_explosive_plays', 0), df.get('cum_passing_eligible_explosive_plays', 1)).round(2)),
            ("EPA per play", lambda: safe_divide(df.get('cum_epa', 0), df.get('cum_plays', 1)).round(2)),
            ("EPA per rush", lambda: safe_divide(df.get('cum_rush_epa', 0), df.get('cum_rushes', 1)).round(2)),
            ("EPA per pass", lambda: safe_divide(df.get('cum_pass_epa', 0), df.get('cum_passes', 1)).round(2))
        ]
        
        feature_names = ['yards_per_play', 'rush_yards_per_play', 'pass_yards_per_play',
                        'play_success', 'rush_success', 'pass_success',
                        'explosive_play_rate', 'explosive_rush_rate', 'explosive_pass_rate',
                        'epa_per_play', 'epa_per_rush', 'epa_per_pass']
        
        with tqdm(total=len(feature_calculations), desc="Calculating derived features", unit="feature") as pbar:
            for i, (desc, calc_func) in enumerate(feature_calculations):
                df[feature_names[i]] = calc_func()
                pbar.update(1)
        
        logging.info("Derived features calculated successfully")
        
    except Exception as e:
        logging.error(f"Error calculating derived features: {e}")
    
    return df

def process_pbp_file(input_file: str, 
                     output_file: str, 
                     include_epa: bool = True, 
                     include_win_prob: bool = True, 
                     ep_lookup_path: str = None,
                     include_garbage_time: bool = False,
                     include_OT_plays: bool = False
                     ):    
    """
    Main function to process a PBP file with enhancements
    """
    
    try:
        # Load data
        print("Loading play-by-play data...")
        logging.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        print(f"   Loaded {initial_rows:,} plays")
        logging.info(f"Loaded {initial_rows:,} plays")
        
        # Load games data for EPA model
        games = None
        if include_epa:
            games_file = find_games_file()
            if games_file:
                try:
                    print("Loading games data...")
                    logging.info(f"Loading games data from {games_file}")
                    games = pd.read_csv(games_file)
                    print(f"   Loaded {len(games):,} games")
                    logging.info(f"Loaded {len(games):,} games for EPA model")
                except Exception as e:
                    print(f"   Warning: Failed to load games file: {e}")
                    logging.warning(f"Failed to load games file: {e}")
                    logging.warning("EPA model will run without games data (less accurate)")
            else:
                print("   Warning: Games file not found - EPA model will be less accurate")
                logging.warning("Games file not found - EPA model will be less accurate")
        
        # Basic data cleaning
        print("Cleaning data...")
        logging.info("Cleaning data...")
        df = df.loc[(df['stat_yardage'] < 100) & (df['stat_yardage'] > -100)]
        df = df.loc[df['period'] > 0]
        filtered_rows = initial_rows - len(df)
        if filtered_rows > 0:
            print(f"   Filtered out {filtered_rows:,} plays with extreme yardage")
            logging.info(f"Filtered out {filtered_rows:,} plays with extreme yardage")
        
        if df.empty:
            raise ValueError("No data remaining after filtering")
            
        # remove any OT plays for analysis (if desired)
        if include_OT_plays == 0:
            df = df.loc[df['period'] <= 4]
            
        # NEW: Clean special teams data
        print("Cleaning special teams data...")
        df = clean_special_teams_data(df)
        
        # Add binary play statistics
        print("Adding play type statistics...")
        df = add_binary_play_stats(df)
        
        # Add expected points from lookup table
        print("Adding expected points...")
        df = add_expected_points(df, ep_lookup_path)
        
        # Calculate Expected Points Added (EPA)
        print("Calculating Expected Points Added (EPA)...")
        df = calculate_epa(df)
        
        # Add win probability
        if include_win_prob:
            print("Adding win probability...")
            df = add_basic_win_prob(df)
        
        # Generate cumulative statistics
        print("Generating cumulative statistics...")
        df = generate_cum_stats(df)
        
        # Add EPA to cumulative statistics
        print("Adding EPA to cumulative statistics...")
        df = add_epa_to_cumulative_stats(df)
        
        # Calculate derived features
        print("Calculating derived features...")
        df = calc_new_features(df)
        
        # Keep or remove garbage time
        if include_garbage_time == 0:
            df = df.loc[df['garbage_time_ind'] != 1]
        
        # Set index if 'id' column exists
        if 'id' in df.columns:
            df.set_index('id', inplace=True)
            logging.info("Set 'id' column as index")
        
        # Save output
        print("Saving enhanced data...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file)
        
        print(f"SUCCESS: Processed {len(df):,} plays")
        print(f"   Output saved to {output_file}")
        logging.info(f"Successfully processed {len(df):,} plays")
        logging.info(f"Output saved to {output_file}")
        new_features = [col for col in df.columns if any(keyword in col.lower() 
               for keyword in ['cum_', 'epa', 'success', 'explosive', 'yards_per', 'win', 'expected_points'])]

        # Add EPA-specific summary
        if 'epa' in df.columns:
            epa_stats = df['epa'].describe()
            print(f"   EPA stats: mean={epa_stats['mean']:.4f}, std={epa_stats['std']:.4f}")
        
        print(f"   Added {len(new_features)} new feature columns")
        logging.info(f"Added {len(new_features)} new feature columns")
        
        return df
        
    except Exception as e:
        print(f"ERROR: {e}")
        logging.error(f"Error processing PBP file: {e}")
        raise

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Process and enhance play-by-play CSV data')
    
    parser.add_argument('--input_file', '-i', required=True,
                       help='Path to input play-by-play CSV file')
    
    parser.add_argument('--output_file', '-o', required=True,
                       help='Path for output enhanced CSV file')
    
    parser.add_argument('--include_epa', action='store_true', default=True,
                       help='Include EPA calculations')
    
    parser.add_argument('--include_win_prob', action='store_true', default=True,
                       help='Include win probability calculations')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    parser.add_argument('--ep_lookup_path', 
                        help='Path to expected points lookup CSV file')
    
    parser.add_argument('--include_garbage_time', default=0,
                        help='1 for yes, 0 for no')
    
    parser.add_argument('--include_OT_plays', default=0,
                        help='1 for yes, 0 for no')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Debug: Print what we received
    print(f"DEBUG: Received arguments:")
    print(f"  input_file: {args.input_file}")
    print(f"  output_file: {args.output_file}")
    print(f"  include_epa: {args.include_epa}")
    print(f"  include_win_prob: {args.include_win_prob}")
    print(f"  include_garbage_time: {args.include_garbage_time}")
    print(f"  include_OT_plays: {args.include_OT_plays}")
    
    try:
        print(f"Processing: {os.path.basename(args.input_file)}")
        print(f"Output: {os.path.basename(args.output_file)}")
        
        # Verify input file exists
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Call the processing function
        result_df = process_pbp_file(
            input_file=args.input_file,
            output_file=args.output_file,
            include_epa=args.include_epa,
            include_win_prob=args.include_win_prob,
            ep_lookup_path=args.ep_lookup_path,
            include_garbage_time=args.include_garbage_time,
            include_OT_plays=args.include_OT_plays
        )
        
        print("SUCCESS: PBP enhancement completed!")
        print(f"Processed {len(result_df):,} plays")
        
        # Verify output file was created
        if os.path.exists(args.output_file):
            file_size = os.path.getsize(args.output_file)
            print(f"Output file created: {args.output_file} ({file_size:,} bytes)")
        else:
            print(f"WARNING: Output file not found at {args.output_file}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()