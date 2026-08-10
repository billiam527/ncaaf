import pandas as pd
import numpy as np

def summarize_cfb_data(filepath):
    """
    Summarize college football play-by-play data by game_id and team_id.
    Calculates both offensive and defensive statistics.
    """
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Define the stats columns to summarize
    stats_columns = [
        'play_success', 'rush_success', 'pass_success',
        'yards_per_play', 'rush_yards_per_play', 'pass_yards_per_play',
        'explosive_play_rate', 'explosive_rush_rate', 'explosive_pass_rate',
        'epa_per_play', 'epa_per_rush', 'epa_per_pass'
    ]
    
    # Filter to only offensive plays (where offensive_play = 1)
    offensive_plays = df[df['offensive_play'] == 1].copy()
    
    # Calculate offensive stats by game_id and team_id
    offensive_stats = []
    
    for (game_id, team_id), group in offensive_plays.groupby(['game_id', 'team_id']):
        stats_dict = {
            'game_id': game_id,
            'team_id': team_id
        }
        
        # Get the last row for cumulative stats (these are already calculated in the data)
        last_row = group.iloc[-1]
        
        for stat in stats_columns:
            if stat in last_row:
                stats_dict[f'offensive_{stat}'] = last_row[stat]
            else:
                stats_dict[f'offensive_{stat}'] = np.nan
        
        # Add team name and opponent info
        stats_dict['team_name'] = group['home_team_name'].iloc[0] if group['home_team_id'].iloc[0] == team_id else group['away_team_name'].iloc[0]
        stats_dict['opponent_id'] = group['away_team_id'].iloc[0] if group['home_team_id'].iloc[0] == team_id else group['home_team_id'].iloc[0]
        stats_dict['opponent_name'] = group['away_team_name'].iloc[0] if group['home_team_id'].iloc[0] == team_id else group['home_team_name'].iloc[0]
        stats_dict['date'] = group['date'].iloc[0]
        stats_dict['season'] = group['season'].iloc[0]
        
        # Add home/away status and scores
        # First, determine who is technically the "home" team (for score assignment purposes)
        is_designated_home = group['home_team_id'].iloc[0] == team_id
        
        # Check if it's a neutral site
        is_neutral = group['neutral_site'].iloc[0] == 1
        
        ###################################
        # Get the scores from the last play
        stats_dict['home_score'] = group['home_final_score'].iloc[-1]
        stats_dict['away_score'] = group['away_final_score'].iloc[-1]
        
        # Assign team/opponent scores based on original designation (not affected by neutral site)
        stats_dict['team_score'] = stats_dict['home_score'] if is_designated_home else stats_dict['away_score']
        stats_dict['opponent_score'] = stats_dict['away_score'] if is_designated_home else stats_dict['home_score']
        
        # Set is_home to 0 for neutral sites, otherwise use the original logic
        stats_dict['is_home'] = 0 if is_neutral else (1 if is_designated_home else 0)
        offensive_stats.append(stats_dict)
    
    # Create offensive stats dataframe
    offensive_df = pd.DataFrame(offensive_stats)
    
    # Now calculate defensive stats (which are the inverse - opponent's offensive stats)
    defensive_stats = []
    
    for idx, row in offensive_df.iterrows():
        # Find the opponent's offensive stats for this game
        opponent_stats = offensive_df[
            (offensive_df['game_id'] == row['game_id']) & 
            (offensive_df['team_id'] == row['opponent_id'])
        ]
        
        if not opponent_stats.empty:
            opp_row = opponent_stats.iloc[0]
            stats_dict = {
                'game_id': row['game_id'],
                'team_id': row['team_id'],
                'team_name': row['team_name'],
                'opponent_id': row['opponent_id'],
                'opponent_name': row['opponent_name'],
                'date': row['date'],
                'season': row['season'],
                'is_home': row['is_home'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                'team_score': row['team_score'],
                'opponent_score': row['opponent_score']
            }
            
            # Add offensive stats
            for stat in stats_columns:
                stats_dict[f'offensive_{stat}'] = row[f'offensive_{stat}']
            
            # Add defensive stats (opponent's offensive stats)
            for stat in stats_columns:
                stats_dict[f'defensive_{stat}'] = opp_row[f'offensive_{stat}']
            
            defensive_stats.append(stats_dict)
    
    # Create final summary dataframe
    summary_df = pd.DataFrame(defensive_stats)
    
    # Sort by game_id and team_id
    summary_df = summary_df.sort_values(['game_id', 'team_id'])
    
    # Round numeric columns to 3 decimal places for readability
    numeric_cols = [col for col in summary_df.columns if 'offensive_' in col or 'defensive_' in col]
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
    
    return summary_df

def main():
    """
    Main function to run the summary analysis.
    """
    # File path - update this to your actual file path
    filepath = 'temp/merged_cfb_data.csv'
    
    try:
        # Generate summary
        summary_df = summarize_cfb_data(filepath)
        
        # Display basic info
        print(f"Summary created successfully!")
        print(f"Total rows: {len(summary_df)}")
        print(f"Games summarized: {summary_df['game_id'].nunique()}")
        print(f"Teams included: {summary_df['team_id'].nunique()}")
        print("\n")
        
        # Display first few rows
        print("First 5 rows of summary:")
        print(summary_df.head())
        print("\n")
        
        # Display columns
        print("Columns in summary:")
        print(list(summary_df.columns))
        print("\n")
        
        # Save to CSV
        output_file = 'temp/cfb_game_team_summary.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
        
        # Display sample stats for one team
        if len(summary_df) > 0:
            sample_row = summary_df.iloc[0]
            print("\nSample team stats from first row:")
            print(f"Team: {sample_row['team_name']} vs {sample_row['opponent_name']}")
            print(f"Date: {sample_row['date']}")
            
            # Check if score columns exist before trying to display them
            if 'is_home' in summary_df.columns and 'neutral_site' in summary_df.columns:
                if sample_row['neutral_site'] == 1:
                    print(f"Team Status: Neutral Site")
                else:
                    print(f"Team Status: {'Home' if sample_row['is_home'] else 'Away'}")
            elif 'is_home' in summary_df.columns:
                print(f"Team Status: {'Home' if sample_row['is_home'] else 'Away'}")
            if 'team_score' in summary_df.columns and 'opponent_score' in summary_df.columns:
                print(f"Final Score: {sample_row['team_name']} {sample_row['team_score']:.0f} - {sample_row['opponent_score']:.0f} {sample_row['opponent_name']}")
            
            print("\nOffensive Stats:")
            for stat in ['play_success', 'yards_per_play', 'explosive_play_rate', 'epa_per_play']:
                if f'offensive_{stat}' in summary_df.columns:
                    print(f"  {stat}: {sample_row[f'offensive_{stat}']:.3f}")
            print("\nDefensive Stats (opponent allowed):")
            for stat in ['play_success', 'yards_per_play', 'explosive_play_rate', 'epa_per_play']:
                if f'defensive_{stat}' in summary_df.columns:
                    print(f"  {stat}: {sample_row[f'defensive_{stat}']:.3f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data file format and try again.")

if __name__ == "__main__":
    main()