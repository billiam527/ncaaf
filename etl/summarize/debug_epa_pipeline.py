#!/usr/bin/env python3
"""
Debug EPA values in your pipeline
"""

import pandas as pd
import numpy as np

def debug_epa_in_pipeline():
    """Debug EPA values at each stage of your pipeline"""
    
    print("DEBUG: EPA Values in Pipeline")
    print("=" * 50)
    
    # Step 1: Check raw PBP data
    print("1. Checking raw PBP data...")
    try:
        pbp = pd.read_csv('temp/pbp.csv')  # Your raw PBP file
        
        epa_cols = [col for col in pbp.columns if 'epa' in col.lower()]
        print(f"   EPA columns found: {epa_cols}")
        
        for col in epa_cols:
            if col in pbp.columns:
                values = pbp[col].dropna()
                print(f"   {col}:")
                print(f"      Count: {len(values):,}")
                print(f"      Non-zero: {(values != 0).sum():,}")
                print(f"      Min: {values.min():.4f}")
                print(f"      Max: {values.max():.4f}")
                print(f"      Mean: {values.mean():.4f}")
                print(f"      Std: {values.std():.4f}")
                print(f"      Unique values: {values.nunique()}")
                
                if values.nunique() <= 5:
                    print(f"      Value counts: {values.value_counts().head()}")
                print()
                
    except Exception as e:
        print(f"   ERROR reading raw PBP: {e}")
    
    # Step 2: Check processed/enhanced PBP data
    print("2. Checking enhanced PBP data...")
    try:
        pbp_enhanced = pd.read_csv('temp/pbp_edit.csv')  # Your enhanced PBP file
        
        epa_cols = [col for col in pbp_enhanced.columns if 'epa' in col.lower()]
        print(f"   EPA columns found: {epa_cols}")
        
        for col in epa_cols:
            if col in pbp_enhanced.columns:
                values = pbp_enhanced[col].dropna()
                print(f"   {col}:")
                print(f"      Count: {len(values):,}")
                print(f"      Non-zero: {(values != 0).sum():,}")
                print(f"      Min: {values.min():.4f}")
                print(f"      Max: {values.max():.4f}")
                print(f"      Mean: {values.mean():.4f}")
                print(f"      Std: {values.std():.4f}")
                print()
                
    except Exception as e:
        print(f"   ERROR reading enhanced PBP: {e}")
    
    # Step 3: Check game summaries
    print("3. Checking game summaries...")
    try:
        game_summaries = pd.read_csv('results/game_by_game_summaries.csv')
        
        epa_cols = [col for col in game_summaries.columns if 'epa' in col.lower()]
        print(f"   EPA columns found: {epa_cols}")
        
        for col in epa_cols:
            if col in game_summaries.columns:
                values = game_summaries[col].dropna()
                print(f"   {col}:")
                print(f"      Count: {len(values):,}")
                print(f"      Non-zero: {(values != 0).sum():,}")
                if len(values) > 0:
                    print(f"      Min: {values.min():.4f}")
                    print(f"      Max: {values.max():.4f}")
                    print(f"      Mean: {values.mean():.4f}")
                    print(f"      Std: {values.std():.4f}")
                print()
                
    except Exception as e:
        print(f"   ERROR reading game summaries: {e}")
    
    # Step 4: Show sample of raw data
    print("4. Sample raw EPA data...")
    try:
        pbp = pd.read_csv('temp/pbp.csv', nrows=1000)
        epa_cols = [col for col in pbp.columns if 'epa' in col.lower()]
        
        if epa_cols:
            sample = pbp[['game_id', 'team_id'] + epa_cols].head(10)
            print(sample.to_string())
        else:
            print("   No EPA columns found in sample")
            
    except Exception as e:
        print(f"   ERROR sampling data: {e}")

if __name__ == "__main__":
    debug_epa_in_pipeline()
