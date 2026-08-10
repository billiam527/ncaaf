# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 09:36:54 2025

@author: wfish
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_summarize_cfb_data(filepath, aggregation='median'):
    """
    Load college football data and create dynamic summaries by season and team.
    
    Parameters:
    filepath (str): Path to the CSV file
    aggregation (str): Type of aggregation ('median', 'mean', or 'both')
    
    Returns:
    tuple: (season_summary, team_summary, team_season_summary)
    """
    
    # Load the data
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} game records")
    print(f"Columns found: {df.shape[1]}")
    print(f"Aggregation method: {aggregation}\n")
    
    # Convert date to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Calculate win/loss if score columns exist
    if 'team_score' in df.columns and 'opponent_score' in df.columns:
        df['won'] = (df['team_score'] > df['opponent_score']).astype(int)
        df['lost'] = (df['team_score'] < df['opponent_score']).astype(int)
        df['tied'] = (df['team_score'] == df['opponent_score']).astype(int)
    
    # Identify numeric columns for aggregation (excluding IDs and categorical data)
    exclude_cols = ['game_id', 'team_id', 'opponent_id', 'team_name', 'opponent_name', 
                   'date', 'season', 'is_home']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    # Define aggregation functions based on parameter
    if aggregation == 'median':
        agg_func = 'median'
        agg_label = 'median'
    elif aggregation == 'mean':
        agg_func = 'mean'
        agg_label = 'avg'
    else:  # both
        agg_func = ['median', 'mean']
        agg_label = 'both'
    
    # 1. Season Summary
    print("Creating season summary...")
    season_agg_dict = {col: agg_func for col in numeric_cols}
    
    # Add count metrics
    if 'game_id' in df.columns:
        season_agg_dict['game_id'] = 'nunique'
    if 'team_name' in df.columns:
        season_agg_dict['team_name'] = 'nunique'
    
    season_summary = df.groupby('season').agg(season_agg_dict).round(3)
    
    # Rename columns for clarity
    if aggregation in ['median', 'mean']:
        new_cols = []
        for col in season_summary.columns:
            if col == 'game_id':
                new_cols.append('total_games')
            elif col == 'team_name':
                new_cols.append('teams_count')
            else:
                new_cols.append(f'{agg_label}_{col}')
        season_summary.columns = new_cols
    else:
        # Handle multi-level columns when using both aggregations
        season_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in season_summary.columns.values]
    
    # 2. Team Summary (across all seasons)
    print("Creating team summary...")
    team_agg_dict = {}
    
    # Add season range info
    if 'season' in df.columns:
        team_agg_dict['season'] = ['min', 'max', 'nunique']
    
    # Add game counts
    if 'game_id' in df.columns:
        team_agg_dict['game_id'] = 'count'
    
    # Add win/loss if available
    if 'won' in df.columns:
        team_agg_dict['won'] = 'sum'
        team_agg_dict['lost'] = 'sum'
    
    # Add numeric columns
    for col in numeric_cols:
        if col not in ['won', 'lost', 'tied']:
            team_agg_dict[col] = agg_func
    
    team_summary = df.groupby('team_name').agg(team_agg_dict).round(3)
    
    # Flatten and rename columns
    if aggregation in ['median', 'mean']:
        flat_cols = []
        for col in team_summary.columns:
            if isinstance(col, tuple):
                if col[1] == 'min':
                    flat_cols.append('first_season')
                elif col[1] == 'max':
                    flat_cols.append('last_season')
                elif col[1] == 'nunique':
                    flat_cols.append('seasons_played')
                else:
                    flat_cols.append(col[0] if col[1] == '' else f'{col[0]}_{col[1]}')
            else:
                if col == 'game_id':
                    flat_cols.append('total_games')
                elif col in ['won', 'lost']:
                    flat_cols.append(f'total_{col}')
                else:
                    flat_cols.append(f'{agg_label}_{col}')
        team_summary.columns = flat_cols
    else:
        # Handle multi-level columns
        team_summary.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col 
                               for col in team_summary.columns.values]
    
    # Calculate win percentage if wins/losses are available
    if 'total_won' in team_summary.columns or 'won' in team_summary.columns:
        wins_col = 'total_won' if 'total_won' in team_summary.columns else 'won'
        losses_col = 'total_lost' if 'total_lost' in team_summary.columns else 'lost'
        if losses_col in team_summary.columns:
            team_summary['win_pct'] = (team_summary[wins_col] / 
                                      (team_summary[wins_col] + team_summary[losses_col])).round(3)
    
    # 3. Team-Season Summary
    print("Creating team-season summary...")
    team_season_agg_dict = {}
    
    # Add game count
    if 'game_id' in df.columns:
        team_season_agg_dict['game_id'] = 'count'
    
    # Add wins/losses
    if 'won' in df.columns:
        team_season_agg_dict['won'] = 'sum'
        team_season_agg_dict['lost'] = 'sum'
    
    # Add numeric columns
    for col in numeric_cols:
        if col not in ['won', 'lost', 'tied']:
            team_season_agg_dict[col] = agg_func
    
    team_season_summary = df.groupby(['season', 'team_name']).agg(team_season_agg_dict).round(3)
    
    # Rename columns
    if aggregation in ['median', 'mean']:
        new_cols = []
        for col in team_season_summary.columns:
            if col == 'game_id':
                new_cols.append('games')
            elif col in ['won', 'lost']:
                new_cols.append(col)
            else:
                new_cols.append(f'{agg_label}_{col}')
        team_season_summary.columns = new_cols
    else:
        # Handle multi-level columns
        team_season_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                      for col in team_season_summary.columns.values]
    
    # Calculate win percentage and point differential if available
    if 'won' in team_season_summary.columns and 'lost' in team_season_summary.columns:
        team_season_summary['win_pct'] = (team_season_summary['won'] / 
                                         (team_season_summary['won'] + team_season_summary['lost'])).round(3)
    
    # Calculate point differential if score columns exist
    for suffix in ['', '_median', '_mean']:
        pts_for = f'{agg_label}_team_score{suffix}' if agg_label != 'both' else f'team_score{suffix}'
        pts_against = f'{agg_label}_opponent_score{suffix}' if agg_label != 'both' else f'opponent_score{suffix}'
        
        if pts_for in team_season_summary.columns and pts_against in team_season_summary.columns:
            diff_col = f'point_diff{suffix}' if suffix else 'point_diff'
            team_season_summary[diff_col] = (team_season_summary[pts_for] - 
                                            team_season_summary[pts_against]).round(1)
    
    return season_summary, team_summary, team_season_summary

def save_summaries(season_summary, team_summary, team_season_summary, output_dir='cfb_summaries'):
    """
    Save summary dataframes to CSV files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save season summary
    season_file = output_path / 'season_summary.csv'
    season_summary.to_csv(season_file)
    print(f"\nSaved season summary to {season_file}")
    print(f"  Shape: {season_summary.shape}")
    
    # Save team summary
    team_file = output_path / 'team_summary.csv'
    team_summary.to_csv(team_file)
    print(f"Saved team summary to {team_file}")
    print(f"  Shape: {team_summary.shape}")
    
    # Save team-season summary
    team_season_file = output_path / 'team_season_summary.csv'
    team_season_summary.to_csv(team_season_file)
    print(f"Saved team-season summary to {team_season_file}")
    print(f"  Shape: {team_season_summary.shape}")
    
    return output_path

def print_sample_summaries(season_summary, team_summary, team_season_summary, n=5):
    """
    Print sample rows from each summary for verification.
    """
    print("\n" + "="*60)
    print("SAMPLE SUMMARIES")
    print("="*60)
    
    print(f"\n--- Season Summary (first {n} seasons) ---")
    print(season_summary.head(n))
    
    print(f"\n--- Team Summary (top {n} teams by games played) ---")
    games_col = 'total_games' if 'total_games' in team_summary.columns else 'game_id'
    if games_col in team_summary.columns:
        print(team_summary.nlargest(n, games_col))
    else:
        print(team_summary.head(n))
    
    print(f"\n--- Team-Season Summary (sample {n} team-seasons) ---")
    print(team_season_summary.head(n))

def analyze_columns(df):
    """
    Analyze and report on the columns in the dataframe.
    """
    print("\n" + "="*60)
    print("COLUMN ANALYSIS")
    print("="*60)
    
    # Categorize columns
    offensive_cols = [col for col in df.columns if 'offensive' in col.lower()]
    defensive_cols = [col for col in df.columns if 'defensive' in col.lower()]
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Offensive metrics: {len(offensive_cols)}")
    print(f"Defensive metrics: {len(defensive_cols)}")
    print(f"Score-related columns: {len(score_cols)}")
    
    print("\nNumeric columns that will be aggregated:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['game_id', 'team_id', 'opponent_id', 'season', 'is_home']
    aggregated_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for i, col in enumerate(aggregated_cols, 1):
        print(f"  {i:2d}. {col}")
    
    return aggregated_cols

# Main execution
if __name__ == "__main__":
    # File path
    filepath = 'temp/cfb_game_team_summary.csv'
    
    try:
        # Load the data to analyze columns
        print("Analyzing data structure...")
        df = pd.read_csv(filepath)
        aggregated_cols = analyze_columns(df)
        
        # Create summaries using median (can change to 'mean' or 'both')
        season_summary, team_summary, team_season_summary = load_and_summarize_cfb_data(
            filepath, 
            aggregation='median'  # Change to 'mean' or 'both' as needed
        )
        
        # Save summaries
        output_path = save_summaries(season_summary, team_summary, team_season_summary)
        
        # Print samples
        print_sample_summaries(season_summary, team_summary, team_season_summary)
        
        print(f"\n✅ All summaries saved to '{output_path}' directory")
        print("\nYou can now load these summaries for further analysis:")
        print("  - season_summary.csv: Trends across seasons")
        print("  - team_summary.csv: Overall team performance")
        print("  - team_season_summary.csv: Team performance by season")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{filepath}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check that the CSV file is properly formatted.")