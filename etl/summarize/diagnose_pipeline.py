# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 12:30:21 2025

@author: wfish
"""

#!/usr/bin/env python3
"""
Diagnostic script to check what's happening in the summarize pipeline
"""

import pandas as pd
import numpy as np

def diagnose_pipeline():
    """Check the pipeline data at each stage"""
    
    print("CFB Pipeline Diagnostics")
    print("=" * 50)
    
    # Check input files
    print("\n1. INPUT FILES:")
    
    files_to_check = [
        "temp/pbp.csv",
        "temp/games.csv", 
        "temp/teams.csv"
    ]
    
    for file_path in files_to_check:
        try:
            df = pd.read_csv(file_path)
            print(f"   {file_path}: {len(df):,} rows")
            print(f"      Columns: {list(df.columns)}")
            
            if 'team_id' in df.columns:
                unique_teams = df['team_id'].nunique()
                print(f"      Unique teams: {unique_teams}")
                
            if 'play_success' in df.columns:
                success_stats = df['play_success'].describe()
                print(f"      Play success stats: mean={success_stats['mean']:.3f}, std={success_stats['std']:.3f}")
                
        except Exception as e:
            print(f"   ERROR loading {file_path}: {e}")
    
    # Check team-level variation in key stats
    print("\n2. TEAM-LEVEL VARIATION CHECK:")
    
    try:
        pbp = pd.read_csv("temp/pbp.csv")
        
        if 'team_id' in pbp.columns and 'game_id' in pbp.columns:
            # Calculate team averages for key stats
            stats_to_check = ['play_success', 'rush_success', 'pass_success', 
                            'yards_per_play', 'epa_per_play']
            
            for stat in stats_to_check:
                if stat in pbp.columns:
                    # Calculate team averages
                    team_stats = pbp[pbp['offensive_play'] == 1].groupby('team_id')[stat].mean()
                    
                    print(f"   {stat}:")
                    print(f"      Teams: {len(team_stats)}")
                    print(f"      Range: {team_stats.min():.3f} to {team_stats.max():.3f}")
                    print(f"      Std: {team_stats.std():.3f}")
                    
                    if team_stats.std() < 0.01:
                        print(f"      WARNING: Very low variation - teams might be too similar")
                    
                    # Show a few examples
                    sample_teams = team_stats.head()
                    print(f"      Sample: {dict(sample_teams)}")
                else:
                    print(f"   {stat}: NOT FOUND")
        
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Check game summaries if they exist
    print("\n3. GAME SUMMARIES CHECK:")
    
    game_summary_files = [
        "results/game_by_game_summaries.csv",
        "results/season_summaries.csv"
    ]
    
    for file_path in game_summary_files:
        try:
            df = pd.read_csv(file_path)
            print(f"   {file_path}: {len(df):,} rows")
            
            # Check for team variation
            if 'team_id' in df.columns:
                unique_teams = df['team_id'].nunique()
                print(f"      Unique teams: {unique_teams}")
                
                # Check a few stats for variation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                sample_cols = [col for col in numeric_cols if 'adjusted' in col][:3]
                
                for col in sample_cols:
                    if col in df.columns:
                        col_std = df[col].std()
                        col_range = df[col].max() - df[col].min()
                        print(f"      {col}: std={col_std:.3f}, range={col_range:.3f}")
                        
                        if col_std < 0.01:
                            print(f"        WARNING: No variation in {col}")
                
        except Exception as e:
            print(f"   {file_path}: {e}")
    
    # Check specific team examples
    print("\n4. SPECIFIC TEAM EXAMPLES:")
    
    try:
        pbp = pd.read_csv("temp/pbp.csv")
        
        if 'team_id' in pbp.columns and 'play_success' in pbp.columns:
            # Get a few different teams
            teams = pbp['team_id'].unique()[:5]
            
            for team in teams:
                team_data = pbp[(pbp['team_id'] == team) & (pbp['offensive_play'] == 1)]
                
                if len(team_data) > 0:
                    avg_success = team_data['play_success'].mean()
                    avg_yards = team_data['yards_per_play'].mean() if 'yards_per_play' in team_data.columns else 0
                    
                    print(f"   Team {team}: {len(team_data)} plays, "
                          f"success={avg_success:.3f}, yards/play={avg_yards:.3f}")
                
        # Check if teams are actually different
        team_summary = pbp[pbp['offensive_play'] == 1].groupby('team_id').agg({
            'play_success': 'mean',
            'yards_per_play': 'mean'
        }).round(3)
        
        print(f"\n   First 10 team summaries:")
        print(team_summary.head(10))
        
        # Check for duplicate values
        if len(team_summary) > 1:
            success_duplicates = team_summary['play_success'].duplicated().sum()
            yards_duplicates = team_summary['yards_per_play'].duplicated().sum()
            
            print(f"\n   Duplicate values:")
            print(f"      Play success: {success_duplicates}/{len(team_summary)} teams have identical values")
            print(f"      Yards per play: {yards_duplicates}/{len(team_summary)} teams have identical values")
        
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    diagnose_pipeline()