#!/usr/bin/env python3
"""
Simplified College Football Data Loader
Only loads and prepares data, then saves the merged result
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


@dataclass
class AnalyticsConfig:
    """Configuration for football analytics pipeline"""
    # Data filtering
    max_periods: int = 4  # Regular time only (no OT)
    remove_garbage_time: bool = True  # Use the garbage time ind and only use =0
    
    # Sampling
    sample_games: int = 0  # 0 = no sampling, >0 = games per season to sample
    
    # Output
    output_dir: str = "temp"
    round_decimals: int = 2


class CFBDataLoader:
    """Handles loading and initial cleaning of CFB data"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_all_data(self, pbp_file: str, games_file: str, teams_file: str):
        """Load and validate all data files"""
        print("Loading data files...")
        
        pbp = self._load_and_validate_csv(pbp_file, "play-by-play")
        games = self._load_and_validate_csv(games_file, "games")
        teams = self._load_and_validate_csv(teams_file, "teams")
        
        return pbp, games, teams
    
    def _load_and_validate_csv(self, file_path: str, data_type: str) -> pd.DataFrame:
        """Safely load CSV with validation"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{data_type} file not found: {file_path}")
            
            # Suppress DtypeWarning for mixed-type columns
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
                df = pd.read_csv(file_path, low_memory=False)
            
            if df.empty:
                raise ValueError(f"{data_type} file is empty: {file_path}")
                
            print(f"✓ Loaded {data_type}: {len(df):,} rows")
            return df
            
        except Exception as e:
            print(f"✗ Failed to load {data_type} file: {e}")
            raise
    
    def prepare_data(self, pbp: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        """Clean and merge all data"""
        print("\nPreparing data...")
        
        # Clean teams data
        teams_clean = self._clean_teams_data(teams)
        
        # Clean games data
        games_clean = self._clean_games_data(games, teams_clean)
        
        # Apply sampling if requested
        if self.config.sample_games > 0:
            games_clean = self._sample_games(games_clean)
        
        # Clean PBP data
        pbp_clean = self._clean_pbp_data(pbp)
        
        # Merge everything
        merged_data = self._merge_data(games_clean, pbp_clean)
        
        print(f"✓ Final dataset: {len(merged_data):,} plays")
        print(f"✓ Columns available: {len(merged_data.columns)}")
        
        return merged_data
    
    def _sample_games(self, games: pd.DataFrame) -> pd.DataFrame:
        """Sample a subset of games per season for testing"""
        print(f"  - Sampling {self.config.sample_games} games per season...")
        
        if 'season' not in games.columns:
            print("    WARNING: No 'season' column found - sampling random games instead")
            sampled = games.sample(n=min(self.config.sample_games * 3, len(games)), random_state=42)
        else:
            # Sample games per season
            sampled_list = []
            for season in sorted(games['season'].unique()):
                season_games = games[games['season'] == season]
                n_sample = min(self.config.sample_games, len(season_games))
                season_sample = season_games.sample(n=n_sample, random_state=42)
                sampled_list.append(season_sample)
                print(f"    Season {season}: {n_sample} games sampled")
            
            sampled = pd.concat(sampled_list, ignore_index=True)
        
        print(f"    Total sampled games: {len(sampled):,}")
        return sampled
    
    def _clean_teams_data(self, teams: pd.DataFrame) -> pd.DataFrame:
        """Clean teams data - keep ALL teams for name lookup"""
        print("  - Cleaning teams data...")
        
        # Check available columns
        print(f"    Teams columns: {list(teams.columns)}")
        
        # Keep required columns
        required_cols = ['id']
        optional_cols = ['slug', 'name', 'display_name', 'fbs_ind']
        
        available_cols = [col for col in required_cols + optional_cols if col in teams.columns]
        teams_clean = teams[available_cols].copy()
        
        # Log FBS vs FCS counts if available
        if 'fbs_ind' in teams_clean.columns:
            fbs_count = (teams_clean['fbs_ind'] == 1).sum()
            fcs_count = (teams_clean['fbs_ind'] == 0).sum()
            print(f"    Teams: {fbs_count} FBS, {fcs_count} FCS")
        
        # Choose the best name column
        name_col = 'display_name'
        if 'display_name' not in teams_clean.columns:
            if 'name' in teams_clean.columns:
                name_col = 'name'
            elif 'slug' in teams_clean.columns:
                name_col = 'slug'
            else:
                name_col = 'id'
        
        print(f"    Using '{name_col}' for team names")
        
        return teams_clean[['id', name_col]].rename(columns={name_col: 'team_name'})
    
    def _clean_games_data(self, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich games data"""
        print("  - Cleaning games data...")
        
        # Check available columns - now including neutral_site
        required_cols = ['id']
        optional_cols = ['date', 'season', 'home_team_id', 'away_team_id', 
                         'neutral_site', 'away_score', 'home_score']
        
        available_cols = [col for col in required_cols + optional_cols if col in games.columns]
        games_clean = games[available_cols].copy()
        
        # Add team names if we have team IDs
        if 'home_team_id' in games_clean.columns and not teams.empty:
            games_clean = games_clean.merge(
                teams, left_on='home_team_id', right_on='id', suffixes=('', '_home'), how='left'
            )
            if 'team_name' in games_clean.columns:
                games_clean = games_clean.rename(columns={'team_name': 'home_team_name'})
        
        if 'away_team_id' in games_clean.columns and not teams.empty:
            games_clean = games_clean.merge(
                teams, left_on='away_team_id', right_on='id', suffixes=('', '_away'), how='left'
            )
            if 'team_name' in games_clean.columns:
                games_clean = games_clean.rename(columns={'team_name': 'away_team_name'})
        
        # Clean up extra id columns
        cols_to_drop = [col for col in games_clean.columns if col.endswith('_home') or col.endswith('_away')]
        if cols_to_drop:
            games_clean = games_clean.drop(columns=cols_to_drop)
        
        # Convert types
        if 'id' in games_clean.columns:
            games_clean['id'] = pd.to_numeric(games_clean['id'], errors='coerce')
        
        # Handle neutral_site column
        if 'neutral_site' in games_clean.columns:
            # Ensure it's binary (0 or 1)
            games_clean['neutral_site'] = games_clean['neutral_site'].fillna(0).astype(int)
            neutral_count = (games_clean['neutral_site'] == 1).sum()
            print(f"    Neutral site games: {neutral_count:,} of {len(games_clean):,}")
            
        # Clean final score cols
        games_clean = games_clean.rename(columns={
            'home_score': 'home_final_score',
            'away_score': 'away_final_score'
        })
        
        print(f"    Games processed: {len(games_clean):,}")
        return games_clean
    
    def _clean_pbp_data(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Clean play-by-play data"""
        print("  - Cleaning play-by-play data...")
        
        pbp_clean = pbp.copy()
        initial_plays = len(pbp_clean)
        
        # Apply filters if columns exist
        if self.config.remove_garbage_time and 'garbage_time_ind' in pbp_clean.columns:
            pbp_clean = pbp_clean[pbp_clean['garbage_time_ind'] == 0]
            removed = initial_plays - len(pbp_clean)
            print(f"    Removed garbage time: {removed:,} plays")
        
        # Remove overtime if period column exists
        if 'period' in pbp_clean.columns:
            before_ot = len(pbp_clean)
            pbp_clean = pbp_clean[pbp_clean['period'] <= self.config.max_periods]
            ot_removed = before_ot - len(pbp_clean)
            if ot_removed > 0:
                print(f"    Removed overtime: {ot_removed:,} plays")
        
        # Convert types
        if 'game_id' in pbp_clean.columns:
            pbp_clean['game_id'] = pd.to_numeric(pbp_clean['game_id'], errors='coerce')
        
        # Handle play_id
        if 'id' in pbp_clean.columns:
            pbp_clean['play_id'] = pbp_clean['id']
        
        print(f"    Final plays: {len(pbp_clean):,}")
        return pbp_clean
    
    def _merge_data(self, games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
        """Merge games and play-by-play data"""
        print("  - Merging games with play-by-play...")
        
        # Check if we can merge
        if 'id' not in games.columns or 'game_id' not in pbp.columns:
            print("    WARNING: Cannot merge - missing required columns")
            return pbp  # Return PBP data as-is
        
        # If we're sampling, only keep PBP data for sampled games
        if self.config.sample_games > 0:
            sampled_game_ids = set(games['id'].dropna())
            print(f"    Filtering PBP to {len(sampled_game_ids)} sampled games...")
            pbp = pbp[pbp['game_id'].isin(sampled_game_ids)]
            print(f"    Filtered PBP: {len(pbp):,} plays")
        
        # Merge games with PBP
        merged = games.merge(pbp, left_on='id', right_on='game_id', how='right', suffixes=('_game', '_pbp'))
        
        # Clean up duplicate id columns
        if 'id_game' in merged.columns:
            merged = merged.drop(columns=['id_game'])
        if 'id_pbp' in merged.columns:
            merged = merged.rename(columns={'id_pbp': 'id'})
        
        print(f"    Merged: {len(merged):,} plays")
        return merged


def setup_simple_logging():
    """Setup basic logging"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/data_loader.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def save_merged_data(merged_data: pd.DataFrame, output_dir: str, is_sample: bool = False):
    """Save the merged data to CSV"""
    suffix = "_sample" if is_sample else ""
    print(f"\nSaving merged data{' (sample)' if is_sample else ''} to {output_dir}...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save merged data
    output_file = Path(output_dir) / f"merged_cfb_data{suffix}.csv"
    
    try:
        merged_data.to_csv(output_file, index=False)
        print(f"✓ Saved merged data: {len(merged_data):,} rows to {output_file}")
        
        # Show some basic info about the data
        print(f"\nData Summary:")
        print(f"  Total plays: {len(merged_data):,}")
        print(f"  Total columns: {len(merged_data.columns)}")
        
        if 'season' in merged_data.columns:
            seasons = merged_data['season'].dropna().unique()
            print(f"  Seasons: {sorted(seasons)}")
            
            # Show games per season
            if 'game_id' in merged_data.columns:
                games_per_season = merged_data.groupby('season')['game_id'].nunique().sort_index()
                print(f"  Games per season:")
                for season, count in games_per_season.items():
                    print(f"    {season}: {count} games")
        
        if 'game_id' in merged_data.columns:
            games = merged_data['game_id'].nunique()
            print(f"  Total unique games: {games:,}")
        
        if 'team_id' in merged_data.columns:
            teams = merged_data['team_id'].nunique()
            print(f"  Teams involved: {teams}")
        
        # Show neutral site info if available
        if 'neutral_site' in merged_data.columns:
            neutral_games = merged_data[merged_data['neutral_site'] == 1]['game_id'].nunique()
            home_games = merged_data[merged_data['neutral_site'] == 0]['game_id'].nunique()
            print(f"  Neutral site games: {neutral_games:,}")
            print(f"  Home/away games: {home_games:,}")
        
        # Show some example matchups if we have team names
        if 'home_team_name' in merged_data.columns and 'away_team_name' in merged_data.columns:
            print(f"\nExample matchups:")
            sample_games = merged_data.drop_duplicates('game_id')[['home_team_name', 'away_team_name', 'season', 'neutral_site']].head(5)
            for _, game in sample_games.iterrows():
                location = " (Neutral Site)" if game.get('neutral_site', 0) == 1 else ""
                print(f"  {game['away_team_name']} @ {game['home_team_name']} ({game['season']}){location}")
        
        # Show column names for reference
        print(f"\nColumn names saved to: {output_dir}/column_names{suffix}.txt")
        with open(Path(output_dir) / f"column_names{suffix}.txt", 'w') as f:
            for i, col in enumerate(merged_data.columns, 1):
                f.write(f"{i:3d}. {col}\n")
        
    except Exception as e:
        print(f"✗ Failed to save merged data: {e}")
        raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CFB Data Loader - Load and merge CFB data files')
    parser.add_argument('--pbp_file', required=True, help='Path to play-by-play CSV file')
    parser.add_argument('--games_file', required=True, help='Path to games CSV file')
    parser.add_argument('--teams_file', required=True, help='Path to teams CSV file')
    parser.add_argument('--output_dir', default='temp', help='Output directory for merged data')
    parser.add_argument('--max_periods', type=int, default=4, help='Maximum periods to include (no OT)')
    parser.add_argument('--include_garbage_time', action='store_true', help='Include garbage time plays')
    parser.add_argument('--sample', type=int, default=0, help='Sample N games per season for testing (0 = no sampling)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_simple_logging()
    
    # Create configuration
    config = AnalyticsConfig(
        max_periods=args.max_periods,
        remove_garbage_time=not args.include_garbage_time,
        sample_games=args.sample,
        output_dir=args.output_dir
    )
    
    print("=== CFB Data Loader ===")
    print(f"Configuration:")
    print(f"  Max periods: {config.max_periods}")
    print(f"  Remove garbage time: {config.remove_garbage_time}")
    print(f"  Sample games per season: {config.sample_games if config.sample_games > 0 else 'None (full dataset)'}")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    try:
        # Initialize data loader
        loader = CFBDataLoader(config)
        
        # Load data
        pbp, games, teams = loader.load_all_data(args.pbp_file, args.games_file, args.teams_file)
        
        # Prepare and merge data
        merged_data = loader.prepare_data(pbp, games, teams)
        
        # Save results
        save_merged_data(merged_data, args.output_dir, is_sample=config.sample_games > 0)
        
        print("\n=== Data Loading Complete ===")
        
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()