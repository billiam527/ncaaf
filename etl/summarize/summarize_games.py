#!/usr/bin/env python3
"""
College Football Analytics Pipeline - FIXED VERSION
Fixed to handle actual data columns and calculate validation stats properly
"""

import logging
import os
import gc
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def _is_rate_statistic(stat: str) -> bool:
    """True when a statistic is a per-unit rate and must be averaged, not summed.

    Covers success rates, explosive-play rates, and every "_per_" statistic
    (per_play, per_rush, per_pass). The "_per_" test matters: matching only
    "per_play" left epa_per_rush and epa_per_pass falling through to a sum, so
    columns named as per-attempt rates were really game totals that scaled with
    play volume.
    """
    name = stat.lower()
    return 'rate' in name or 'success' in name or '_per_' in name


# How each rate is rebuilt: (numerator column, denominator).
# The denominator is either 'plays' (every play in the group), a play-type flag
# to count, or an eligibility column to sum. These names collide with running
# season-to-date columns of the same name in the play-by-play, which is exactly
# why the rate must be recomputed rather than read.
RATE_COMPONENTS = {
    'yards_per_play':       ('stat_yardage', 'plays'),
    'rush_yards_per_play':  ('stat_yardage', 'rushing_play'),
    'pass_yards_per_play':  ('stat_yardage', 'passing_play'),
    'play_success':         ('successful_play', 'plays'),
    'rush_success':         ('rushing_successful_play', 'rushing_play'),
    'pass_success':         ('passing_successful_play', 'passing_play'),
    'explosive_play_rate':  ('explosive_play', 'explosive_play_eligible'),
    'explosive_rush_rate':  ('rushing_explosive_play',
                             'rushing_explosive_play_eligible'),
    'explosive_pass_rate':  ('passing_explosive_play',
                             'passing_explosive_play_eligible'),
    'epa_per_play':         ('epa', 'plays'),
    'epa_per_rush':         ('epa', 'rushing_play'),
    'epa_per_pass':         ('epa', 'passing_play'),
}


@dataclass
class AnalyticsConfig:
    """Configuration for football analytics pipeline"""
    # Data filtering
    max_periods: int = 4  # Regular time only (no OT)
    fbs_only: bool = True  # Only include FBS teams
    remove_garbage_time: bool = True # Use the garbage time ind and only use =0
    
    # Statistical thresholds
    outlier_zscore_threshold: float = 3.0 # 3 standard deviations
    ridge_alpha: float = 1.0
    
    # Week numbering
    bowl_week_number: int = 17
    championship_week_number: int = 18
    
    # Output
    output_dir: str = "results"
    # Two decimals quantised the model's own inputs. Explosive rush rate
    # averages 0.032, so rounding it to 0.03 is a 16% error and collapsed
    # 3,655 team-seasons onto 13 distinct values - a feature the trees could
    # barely split on. Six keeps every rate's precision without bloating the
    # file, since these are all bounded quantities.
    round_decimals: int = 6


class CFBDataLoader:
    """Handles loading and initial cleaning of CFB data"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_all_data(self, pbp_file: str, games_file: str, teams_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and validate all data files"""
        with tqdm(total=3, desc="Loading data files", unit="file") as pbar:
            pbp = self._load_and_validate_csv(pbp_file, "play-by-play")
            pbar.update(1)
            
            games = self._load_and_validate_csv(games_file, "games")
            pbar.update(1)
            
            teams = self._load_and_validate_csv(teams_file, "teams")
            pbar.update(1)
        
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
                
            self.logger.info(f"Loaded {data_type}: {len(df):,} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load {data_type} file: {e}")
            raise
    
    def prepare_data(self, pbp: pd.DataFrame, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        """Clean and merge all data"""
        with tqdm(total=4, desc="Preparing data", unit="step") as pbar:
            # Clean teams data
            teams_clean = self._clean_teams_data(teams)
            pbar.update(1)
            
            # Clean games data
            games_clean = self._clean_games_data(games, teams_clean)
            pbar.update(1)
            
            # Clean PBP data
            pbp_clean = self._clean_pbp_data(pbp)
            pbar.update(1)
            
            # Merge everything
            merged_data = self._merge_data(games_clean, pbp_clean)
            pbar.update(1)
            
        self.logger.info(f"Final dataset: {len(merged_data):,} plays")
        
        # DEBUG: Print column names to understand the structure
        self.logger.info(f"Merged data columns: {list(merged_data.columns)}")
        
        return merged_data
    
    def _clean_teams_data(self, teams: pd.DataFrame) -> pd.DataFrame:
        """Clean teams data - keep ALL teams for name lookup, filter later for analysis"""
        
        # Check available columns
        self.logger.info(f"Teams columns available: {list(teams.columns)}")
        
        # Keep ALL teams for name lookup purposes
        # We'll filter to FBS later when needed, but we want all team names available
        required_cols = ['id']
        optional_cols = ['slug', 'name', 'display_name', 'fbs_ind']
        
        available_cols = []
        for col in required_cols + optional_cols:
            if col in teams.columns:
                available_cols.append(col)
        
        teams_clean = teams[available_cols].copy()
        
        # Log FBS vs FCS counts for info
        if 'fbs_ind' in teams_clean.columns:
            fbs_count = (teams_clean['fbs_ind'] == 1).sum()
            fcs_count = (teams_clean['fbs_ind'] == 0).sum()
            self.logger.info(f"Teams: {fbs_count} FBS, {fcs_count} FCS (keeping all for name lookup)")
        
        # Choose the best name column
        name_col = None
        if 'display_name' in teams_clean.columns:
            name_col = 'display_name'
        elif 'slug' in teams_clean.columns:
            name_col = 'slug'
        elif 'name' in teams_clean.columns:
            name_col = 'name'
        else:
            name_col = 'id'
        
        self.logger.info(f"Using '{name_col}' column for team names")
        
        # Return ALL teams with names (don't filter by FBS here)
        return teams_clean[['id', name_col]].rename(columns={name_col: 'team_name'})
    
    def _clean_games_data(self, games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich games data"""
        
        # Check available columns
        self.logger.info(f"Games columns: {list(games.columns)}")
        
        # Adapt to available columns
        required_cols = ['id']
        optional_cols = ['date', 'season', 'home_team_id', 'away_team_id']
        
        available_cols = []
        for col in required_cols + optional_cols:
            if col in games.columns:
                available_cols.append(col)
        
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
        
        return games_clean
    
    def _clean_pbp_data(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Clean play-by-play data"""
        
        # Check available columns
        self.logger.info(f"PBP columns: {list(pbp.columns)}")
        
        pbp_clean = pbp.copy()
        
        # Apply filters if columns exist
        if self.config.remove_garbage_time and 'garbage_time_ind' in pbp_clean.columns:
            initial_plays = len(pbp_clean)
            pbp_clean = pbp_clean[pbp_clean['garbage_time_ind'] == 0]
            removed = initial_plays - len(pbp_clean)
            self.logger.info(f"Removed garbage time: {removed:,} plays")
        
        # Remove overtime if period column exists
        if 'period' in pbp_clean.columns:
            pbp_clean = pbp_clean[pbp_clean['period'] <= self.config.max_periods]
        
        # Convert types
        if 'game_id' in pbp_clean.columns:
            pbp_clean['game_id'] = pd.to_numeric(pbp_clean['game_id'], errors='coerce')
        
        # Handle play_id
        if 'id' in pbp_clean.columns:
            pbp_clean['play_id'] = pbp_clean['id']
            # Don't drop 'id' yet, we need it for indexing
        
        return pbp_clean
    
    def _merge_data(self, games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
        """Merge games and play-by-play data"""
        
        # Check if we can merge
        if 'id' not in games.columns or 'game_id' not in pbp.columns:
            self.logger.warning("Cannot merge - missing required columns")
            return pbp  # Return PBP data as-is
        
        # Merge games with PBP
        merged = games.merge(pbp, left_on='id', right_on='game_id', how='right', suffixes=('_game', '_pbp'))
        
        # Clean up duplicate id columns
        if 'id_game' in merged.columns:
            merged = merged.drop(columns=['id_game'])
        if 'id_pbp' in merged.columns:
            merged = merged.rename(columns={'id_pbp': 'id'})
        
        return merged


class GameStatsCalculator:
    """Calculates game-by-game team statistics"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_game_summaries(self, df: pd.DataFrame, statistics: List[str]) -> pd.DataFrame:
        """Calculate game-by-game summaries for all teams - WITH DEBUGGING"""
        
        # Log what we're working with
        self.logger.info(f"Available columns in data: {list(df.columns)}")
        self.logger.info(f"Requested statistics: {statistics}")
        
        # Check which statistics actually exist in the data
        available_stats = []
        missing_stats = []
        
        for stat in statistics:
            if stat in df.columns:
                available_stats.append(stat)
            else:
                missing_stats.append(stat)
        
        if missing_stats:
            self.logger.warning(f"Missing statistics (will be calculated if possible): {missing_stats}")
        
        self.logger.info(f"Available statistics to process: {available_stats}")
        
        # Check required columns
        required_cols = ['home_team_id', 'away_team_id', 'game_id', 'team_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            self.logger.error(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # DEBUG: Log sample data to verify structure
        self.logger.debug(f"DataFrame shape: {df.shape}")
        self.logger.debug(f"Sample team_id values: {df['team_id'].unique()[:5]}")
        self.logger.debug(f"Sample home_team_id values: {df['home_team_id'].unique()[:5]}")
        self.logger.debug(f"Sample game_id values: {df['game_id'].unique()[:5]}")
        
        # Pre-filter the data once
        filtered_df = df.copy()
        
        # Apply offensive play filter if available
        if 'offensive_play' in filtered_df.columns:
            initial_rows = len(filtered_df)
            print(f"CONSOLE DEBUG: Before offensive play filter: {initial_rows} rows")
            print(f"CONSOLE DEBUG: offensive_play column values: {filtered_df['offensive_play'].value_counts()}")
            
            filtered_df = filtered_df[filtered_df['offensive_play'] == 1]
            after_rows = len(filtered_df)
            
            print(f"CONSOLE DEBUG: After offensive play filter: {after_rows} rows")
            self.logger.info(f"Filtered to offensive plays: {after_rows:,} rows (was {initial_rows:,})")
        
        if filtered_df.empty:
            print("CONSOLE ERROR: No offensive plays found after filtering")
            self.logger.error("No offensive plays found after filtering")
            return pd.DataFrame()
        
        print(f"CONSOLE DEBUG: Final filtered data shape: {filtered_df.shape}")
        print(f"CONSOLE DEBUG: Sample of filtered data columns: {list(filtered_df.columns)[:10]}")
        
        # Use the original approach but optimized - avoid complex DataFrame manipulations
        self.logger.info("Calculating team statistics...")
        
        # Create simple team configurations
        team_configs = [
            {'name': 'home_offense', 'team_col': 'home_team_id', 'filter_team': 'home_team_id', 'side': 'offense'},
            {'name': 'away_offense', 'team_col': 'away_team_id', 'filter_team': 'away_team_id', 'side': 'offense'},
            {'name': 'home_defense', 'team_col': 'home_team_id', 'filter_team': 'away_team_id', 'side': 'defense'}, 
            {'name': 'away_defense', 'team_col': 'away_team_id', 'filter_team': 'home_team_id', 'side': 'defense'}
        ]
        
        # Calculate stats for each configuration using simple approach
        all_results = []
        
        print(f"CONSOLE DEBUG: About to process {len(team_configs)} team configurations")
        
        for config in tqdm(team_configs, desc="Processing team configurations"):
            print(f"CONSOLE DEBUG: Starting {config['name']}...")
            config_results = self._calculate_simple_team_stats(filtered_df, config, available_stats)
            print(f"CONSOLE DEBUG: {config['name']} returned {len(config_results) if not config_results.empty else 0} results")
            
            if not config_results.empty:
                all_results.append(config_results)
                self.logger.info(f"{config['name']}: {len(config_results)} games calculated")
                # DEBUG: Log first few rows and columns
                self.logger.debug(f"{config['name']} columns: {list(config_results.columns)}")
                if len(config_results) > 0:
                    self.logger.debug(f"{config['name']} first row keys: {list(config_results.iloc[0].keys()) if hasattr(config_results.iloc[0], 'keys') else 'No keys'}")
            else:
                print(f"CONSOLE WARNING: {config['name']}: No results generated!")
                self.logger.warning(f"{config['name']}: No results generated!")
        
        print(f"CONSOLE DEBUG: Total configurations with results: {len(all_results)}")
        
        if not all_results:
            self.logger.error("No team statistics calculated - all configurations returned empty results")
            return pd.DataFrame()
        
        self.logger.info(f"Successfully calculated stats for {len(all_results)} configurations")
        
        # DEBUG: Check each result before concatenation
        for i, result_df in enumerate(all_results):
            self.logger.debug(f"Result {i}: shape={result_df.shape}, columns={list(result_df.columns)}")
        
        # Simple concatenation
        try:
            combined_df = pd.concat(all_results, ignore_index=True)
            self.logger.info(f"Combined all stats: {len(combined_df)} records")
            self.logger.debug(f"Combined columns: {list(combined_df.columns)}")
        except Exception as e:
            self.logger.error(f"Failed to concatenate results: {e}")
            # Try alternative approach - stack all dictionaries
            all_dicts = []
            for result_df in all_results:
                all_dicts.extend(result_df.to_dict('records'))
            combined_df = pd.DataFrame(all_dicts)
            self.logger.info(f"Alternative approach - Combined all stats: {len(combined_df)} records")
        
        if combined_df.empty:
            self.logger.error("Combined DataFrame is empty after concatenation")
            return pd.DataFrame()
        
        # Check for required columns
        if 'side' not in combined_df.columns:
            self.logger.error("'side' column missing from combined DataFrame")
            self.logger.error(f"Available columns: {list(combined_df.columns)}")
            return pd.DataFrame()
        
        # Separate offense and defense
        try:
            offense_mask = combined_df['side'] == 'offense'
            defense_mask = combined_df['side'] == 'defense'
            
            offense_df = combined_df[offense_mask].copy()
            defense_df = combined_df[defense_mask].copy()
            
            self.logger.info(f"Separated - Offense: {len(offense_df)}, Defense: {len(defense_df)}")
            
            if offense_df.empty:
                self.logger.error("No offense records found after separation")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to separate offense and defense: {e}")
            return pd.DataFrame()
        
        # Simple merge on game_id and team_id
        try:
            if not defense_df.empty:
                merge_cols = ['game_id', 'team_id']
                
                # Check merge columns exist
                missing_merge_offense = [col for col in merge_cols if col not in offense_df.columns]
                missing_merge_defense = [col for col in merge_cols if col not in defense_df.columns]
                
                if missing_merge_offense or missing_merge_defense:
                    self.logger.error(f"Missing merge columns - Offense: {missing_merge_offense}, Defense: {missing_merge_defense}")
                    return pd.DataFrame()
                
                # Defense stat columns already carry a _def suffix from
                # _add_requested_stats. All four side-frames were concatenated
                # above, so defense_df also carries the offense schema (all-NaN
                # on these rows). Appending _def to those produced a second
                # <stat>_def column, and the next loop iteration renamed both to
                # <stat>_def_def - duplicate names that later broke the rolling
                # calculator. Keep the merge keys and the genuine defense
                # columns instead; offense supplies the shared metadata.
                defense_cols = [
                    col for col in defense_df.columns
                    if col in merge_cols or col.endswith('_def')
                ]
                dropped_cols = [c for c in defense_df.columns if c not in defense_cols]
                self.logger.info(
                    f"Defense frame: keeping {len(defense_cols)} columns "
                    f"({defense_cols}), dropping {len(dropped_cols)} offense-schema columns"
                )
                defense_renamed = defense_df[defense_cols].copy()

                # Trim offense symmetrically. offense_df carries the same union
                # of columns, so it holds all-NaN <stat>_def placeholders. Left
                # in place they collide with the real defensive columns on the
                # merge below: pandas keeps offense's NaN copy under the plain
                # name and suffixes the populated one to <stat>_def_dup, which
                # the cleanup step then drops - discarding every defensive value
                # and leaving the opponent adjuster with all-NaN inputs.
                offense_cols = [
                    col for col in offense_df.columns
                    if col in merge_cols or not col.endswith('_def')
                ]
                self.logger.info(
                    f"Offense frame: keeping {len(offense_cols)} columns, "
                    f"dropping {len(offense_df.columns) - len(offense_cols)} empty _def placeholders"
                )
                offense_trimmed = offense_df[offense_cols].copy()

                # Merge
                final_df = offense_trimmed.merge(defense_renamed, on=merge_cols, how='left', suffixes=('', '_dup'))

                # Clean up. With both sides trimmed there should be no collisions
                # left; warn if any appear rather than dropping data silently.
                dup_cols = [col for col in final_df.columns if col.endswith('_dup')]
                if dup_cols:
                    self.logger.warning(
                        f"Unexpected column collisions on offense/defense merge: {dup_cols}"
                    )
                    final_df = final_df.drop(columns=dup_cols)
                
                self.logger.info(f"Merged offense and defense: {len(final_df)} records")
            else:
                self.logger.warning("No defense data - using offense only")
                final_df = offense_df
                
        except Exception as e:
            self.logger.error(f"Failed to merge offense and defense: {e}")
            return pd.DataFrame()
        
        # Remove side columns
        for col in ['side', 'side_def']:
            if col in final_df.columns:
                final_df = final_df.drop(columns=[col])
        
        # Simple index setting
        try:
            index_cols = []
            for col in ['season', 'date', 'game_id', 'team_id']:
                if col in final_df.columns:
                    index_cols.append(col)
            
            if index_cols:
                self.logger.debug(f"Setting index with columns: {index_cols}")
                # Remove duplicates first
                before_dedup = len(final_df)
                final_df = final_df.drop_duplicates(subset=index_cols, keep='first')
                after_dedup = len(final_df)
                if before_dedup != after_dedup:
                    self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
                
                final_df = final_df.set_index(index_cols)
                self.logger.info(f"Successfully set index")
            else:
                self.logger.warning("No suitable columns found for index")
                
        except Exception as e:
            self.logger.error(f"Failed to set index: {e}")
            # Continue without setting index
        
        self.logger.info(f"Final game summaries: {len(final_df):,} team-game records")
        
        if final_df.empty:
            self.logger.error("Final DataFrame is empty!")
            return pd.DataFrame()
            
        return final_df
    
    def _calculate_simple_team_stats(self, df: pd.DataFrame, config: Dict, statistics: List[str]) -> pd.DataFrame:
        """Simple, robust team stats calculation - WITH CONSOLE DEBUGGING"""
        
        print(f"\n=== DEBUGGING {config['name']} ===")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input columns (first 20): {list(df.columns)[:20]}")
        
        # Check required columns exist
        required_cols = ['team_id', 'home_team_id', 'away_team_id', 'game_id']
        print(f"Checking for required columns: {required_cols}")
        
        missing_cols = []
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)
            else:
                print(f"✓ Found column: {col}")
        
        if missing_cols:
            print(f"✗ ERROR: Missing required columns for {config['name']}: {missing_cols}")
            print(f"All available columns: {list(df.columns)}")
            self.logger.error(f"Missing required columns for {config['name']}: {missing_cols}")
            return pd.DataFrame()
        
        print(f"All required columns present for {config['name']}")
        
        # Show sample data
        print(f"Sample team_id values: {df['team_id'].dropna().unique()[:5]}")
        print(f"Sample home_team_id values: {df['home_team_id'].dropna().unique()[:5]}")
        print(f"Sample away_team_id values: {df['away_team_id'].dropna().unique()[:5]}")
        
        # FIX: Convert data types to match
        print("Converting data types to ensure compatibility...")
        df = df.copy()
        
        # Convert team_id to numeric to match home/away team IDs
        df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
        df['home_team_id'] = pd.to_numeric(df['home_team_id'], errors='coerce')
        df['away_team_id'] = pd.to_numeric(df['away_team_id'], errors='coerce')
        
        print(f"After conversion - Sample team_id values: {df['team_id'].dropna().unique()[:5]}")
        print(f"After conversion - Sample home_team_id values: {df['home_team_id'].dropna().unique()[:5]}")
        print(f"After conversion - Sample away_team_id values: {df['away_team_id'].dropna().unique()[:5]}")
        
        # DEFENSIVE: Filter plays where the specified team has the ball
        try:
            print(f"Starting filtering for {config['name']}...")
            print(f"Config details: {config}")
            
            # Sample the data to understand structure
            if len(df) > 0:
                sample_row = df.iloc[0]
                print(f"Sample team_id: {sample_row['team_id']}")
                print(f"Sample home_team_id: {sample_row['home_team_id']}")
                print(f"Sample away_team_id: {sample_row['away_team_id']}")
                print(f"Sample game_id: {sample_row['game_id']}")
            
            if config['side'] == 'offense':
                # For offense: filter where this team has the ball
                if config['name'] == 'home_offense':
                    print("Filtering for home offense: team_id == home_team_id")
                    print(f"Before filter: {len(df)} rows")
                    filter_condition = df['team_id'] == df['home_team_id']
                    print(f"Filter condition created, matches: {filter_condition.sum()} rows")
                elif config['name'] == 'away_offense':
                    print("Filtering for away offense: team_id == away_team_id")
                    print(f"Before filter: {len(df)} rows")
                    filter_condition = df['team_id'] == df['away_team_id']
                    print(f"Filter condition created, matches: {filter_condition.sum()} rows")
                else:
                    print(f"ERROR: Unknown offense configuration: {config['name']}")
                    self.logger.error(f"Unknown offense configuration: {config['name']}")
                    return pd.DataFrame()
            else:  # defense
                # For defense: filter where the opponent has the ball
                if config['name'] == 'home_defense':
                    print("Filtering for home defense: team_id == away_team_id")
                    print(f"Before filter: {len(df)} rows")
                    filter_condition = df['team_id'] == df['away_team_id']  # Away team has ball, home team defending
                    print(f"Filter condition created, matches: {filter_condition.sum()} rows")
                elif config['name'] == 'away_defense':
                    print("Filtering for away defense: team_id == home_team_id")
                    print(f"Before filter: {len(df)} rows")
                    filter_condition = df['team_id'] == df['home_team_id']  # Home team has ball, away team defending
                    print(f"Filter condition created, matches: {filter_condition.sum()} rows")
                else:
                    print(f"ERROR: Unknown defense configuration: {config['name']}")
                    self.logger.error(f"Unknown defense configuration: {config['name']}")
                    return pd.DataFrame()
            
            print(f"Applying filter...")
            filtered_df = df[filter_condition].copy()
            print(f"Filtered DataFrame shape: {filtered_df.shape}")
            
        except KeyError as e:
            print(f"CONSOLE ERROR: KeyError in filtering for {config['name']}: {e}")
            print(f"DataFrame columns: {list(df.columns)}")
            self.logger.error(f"KeyError in filtering for {config['name']}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"CONSOLE ERROR: Error in filtering for {config['name']}: {e}")
            print(f"Error type: {type(e)}")
            self.logger.error(f"Error in filtering for {config['name']}: {e}")
            return pd.DataFrame()
        
        if filtered_df.empty:
            print(f"WARNING: No plays found for {config['name']} after filtering")
            self.logger.warning(f"No plays found for {config['name']} after filtering")
            return pd.DataFrame()
        
        print(f"SUCCESS: Processing {config['name']}: {len(filtered_df)} plays")
        self.logger.debug(f"Processing {config['name']}: {len(filtered_df)} plays")
        
        # Continue with the rest of the method unchanged...
        # Group by game and calculate stats
        game_results = []
        
        try:
            for game_id, game_group in filtered_df.groupby('game_id'):
                # DEFENSIVE: Check that required columns exist in game_group
                if 'home_team_id' not in game_group.columns or 'away_team_id' not in game_group.columns:
                    print(f"ERROR: Missing team columns in game group for {config['name']}")
                    self.logger.error(f"Missing team columns in game group for {config['name']}")
                    continue
                
                # Set the correct team_id for the result
                if config['side'] == 'offense':
                    # For offense, use the team that's on offense
                    result_team_id = game_group[config['team_col']].iloc[0]
                else:
                    # For defense, use the team that's on defense (not the team with the ball)
                    if config['name'] == 'home_defense':
                        result_team_id = game_group['home_team_id'].iloc[0]
                    else:  # away_defense
                        result_team_id = game_group['away_team_id'].iloc[0]
                
                # Basic game info
                result = {
                    'game_id': game_id,
                    'team_id': result_team_id,
                    'side': config['side'],
                    'play_count': len(game_group)
                }
                
                # Add metadata safely
                metadata_cols = ['season', 'date', 'home_team_name', 'away_team_name', 'home_team_id', 'away_team_id']
                for col in metadata_cols:
                    if col in game_group.columns:
                        try:
                            result[col] = game_group[col].iloc[0]
                        except Exception as e:
                            print(f"WARNING: Could not get {col} for {config['name']}: {e}")
                            self.logger.warning(f"Could not get {col} for {config['name']}: {e}")

                # The opponent is whichever side of the matchup this team is not.
                # OpponentAdjuster._adjust_season requires this column and raises
                # without it, which silently killed every season's adjustment and
                # left season_summaries.csv unwritten.
                if 'home_team_id' in result and 'away_team_id' in result:
                    if result['team_id'] == result['home_team_id']:
                        result['opponent_id'] = result['away_team_id']
                    else:
                        result['opponent_id'] = result['home_team_id']

                # NOTE: a call to self._add_basic_stats() used to sit here, but that
                # method was never implemented - every game raised AttributeError and
                # hit the `continue` below, so no game ever reached game_results.
                # Nothing downstream consumes volume stats: the models use only the
                # adjusted_* columns derived from `statistics`, and the rolling
                # calculator takes whatever numeric columns exist.

                # Process requested statistics
                try:
                    self._add_requested_stats(result, game_group, statistics, config['side'])
                except Exception as e:
                    print(f"ERROR: Error in _add_requested_stats for {config['name']}: {e}")
                    self.logger.error(f"Error in _add_requested_stats for {config['name']}: {e}")
                    continue
                
                game_results.append(result)
                
        except Exception as e:
            print(f"CONSOLE ERROR: Error in game processing for {config['name']}: {e}")
            print(f"Error type: {type(e)}")
            self.logger.error(f"Error in game processing for {config['name']}: {e}")
            return pd.DataFrame()
        
        if not game_results:
            print(f"WARNING: No game results generated for {config['name']}")
            self.logger.warning(f"No game results generated for {config['name']}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        try:
            result_df = pd.DataFrame(game_results)
            print(f"SUCCESS: Calculated {len(result_df)} games for {config['name']}")
            self.logger.debug(f"Calculated {len(result_df)} games for {config['name']}")
            return result_df
        except Exception as e:
            print(f"CONSOLE ERROR: Error creating DataFrame for {config['name']}: {e}")
            self.logger.error(f"Error creating DataFrame for {config['name']}: {e}")
            return pd.DataFrame()
    
    def _add_requested_stats(self, result: dict, game_group: pd.DataFrame, statistics: List[str], side: str):
        """Add requested statistics to result dictionary.

        Every rate here is rebuilt from its own numerator and denominator. It
        cannot be read off the column of the same name: the play-by-play
        carries running season-to-date averages under exactly these names
        (yards_per_play, play_success, epa_per_play and the rush/pass
        variants), and averaging a running average weights the first play of a
        game as heavily as the whole rest of it.

        Old Dominion's 2025 passing offence is what surfaced this. Their true
        figure is 8.1 yards per pass play, which is what their own site lists;
        the mean of the running column made it 10.2 and put them 2nd in FBS.
        Against a correct recomputation of the 2025 season the old values
        correlated only 0.70 to 0.84, so roughly a third of the variance in
        every core model feature was an artefact of this.
        """

        suffix = '_def' if side == 'defense' else ''
        play_count = len(game_group)

        for stat in statistics:
            value = self._compute_rate(game_group, stat, play_count)

            if value is None:
                # not a known rate - fall back to reading the column directly,
                # which is still correct for anything that is not an average
                if stat not in game_group.columns:
                    continue
                stat_data = game_group[stat].dropna()
                if len(stat_data) == 0:
                    result[f'{stat}{suffix}'] = 0
                    continue
                if _is_rate_statistic(stat):
                    value = stat_data.mean()
                else:
                    value = (stat_data.mean() if stat_data.max() <= 1.0
                             else stat_data.sum())

            # Defensive EPA is stored as suppression: a defence that gives up
            # negative EPA scores positively. Note this makes the EPA _def
            # columns the opposite polarity to the success and yardage _def
            # columns, which hold what was allowed.
            if 'epa' in stat.lower() and side == 'defense':
                value = -value

            result[f'{stat}{suffix}'] = value

    def _compute_rate(self, g: pd.DataFrame, stat: str, play_count: int):
        """Rebuild one rate from its components, or None if `stat` is not one.

        Returns 0.0 rather than None when the components exist but the
        denominator is empty - a game with no rushing plays has a rush rate of
        zero, not a missing one.
        """
        spec = RATE_COMPONENTS.get(stat)
        if spec is None:
            return None
        num_col, den = spec
        if num_col not in g.columns:
            return None

        if den == 'plays':
            denom = play_count
            num = pd.to_numeric(g[num_col], errors='coerce').sum()
        elif den in ('rushing_play', 'passing_play'):
            if den not in g.columns:
                return None
            mask = pd.to_numeric(g[den], errors='coerce') == 1
            denom = int(mask.sum())
            num = pd.to_numeric(g.loc[mask, num_col], errors='coerce').sum()
        else:
            # explosive rates are over the plays that could have been explosive
            if den not in g.columns:
                return None
            denom = pd.to_numeric(g[den], errors='coerce').sum()
            num = pd.to_numeric(g[num_col], errors='coerce').sum()

        if not denom or denom <= 0:
            return 0.0
        return float(num) / float(denom)
    
    def _calculate_team_game_stats_vectorized(self, game_plays: pd.DataFrame, team_id: int, 
                                            opponent_id: int, side: str, statistics: List[str], 
                                            game_meta: dict) -> dict:
        """Calculate stats for one team in one game using vectorized operations"""
        
        # Start with game metadata
        result = game_meta.copy()
        
        # Add team info
        result.update({
            'team_id': team_id,
            'opponent_id': opponent_id,
            'side': side,
            'play_count': len(game_plays)
        })
        
        # Add team names if available
        if 'home_team_id' in game_plays.columns and 'away_team_id' in game_plays.columns:
            home_id = game_plays['home_team_id'].iloc[0]
            away_id = game_plays['away_team_id'].iloc[0]
            
            if team_id == home_id:
                result['team_name'] = game_plays.get('home_team_name', {}).iloc[0] if 'home_team_name' in game_plays.columns else f"Team_{team_id}"
                result['opponent_name'] = game_plays.get('away_team_name', {}).iloc[0] if 'away_team_name' in game_plays.columns else f"Team_{opponent_id}"
            else:
                result['team_name'] = game_plays.get('away_team_name', {}).iloc[0] if 'away_team_name' in game_plays.columns else f"Team_{team_id}"
                result['opponent_name'] = game_plays.get('home_team_name', {}).iloc[0] if 'home_team_name' in game_plays.columns else f"Team_{opponent_id}"
        
        # VECTORIZED CALCULATIONS
        
        # Basic yards
        if 'stat_yardage' in game_plays.columns:
            total_yards = game_plays['stat_yardage'].sum()
            suffix = '_allowed' if side == 'defense' else ''
            result[f'total_yards{suffix}'] = total_yards
            if len(game_plays) > 0:
                result[f'yards_per_play{suffix}'] = total_yards / len(game_plays)
        
        # Success rate
        if 'successful_play' in game_plays.columns:
            success_count = game_plays['successful_play'].sum()
            suffix = '_allowed' if side == 'defense' else ''
            result[f'play_success{suffix}'] = success_count / len(game_plays) if len(game_plays) > 0 else 0
        
        # EPA
        epa_cols = ['points_added', 'epa', 'expected_points_added']
        for col in epa_cols:
            if col in game_plays.columns:
                epa_values = game_plays[col].fillna(0)
                total_epa = epa_values.sum()
                if side == 'defense':
                    total_epa = -total_epa
                suffix = '_defense' if side == 'defense' else ''
                result[f'epa_per_play{suffix}'] = total_epa / len(game_plays) if len(game_plays) > 0 else 0
                break
        
        # Process other statistics
        for stat in statistics:
            if stat in game_plays.columns:
                stat_data = game_plays[stat].dropna()
                if len(stat_data) > 0:
                    # Determine aggregation method (see _is_rate_statistic)
                    if _is_rate_statistic(stat):
                        stat_value = stat_data.mean()
                    else:
                        stat_value = stat_data.mean() if stat_data.max() <= 1.0 else stat_data.sum()
                    
                    # Apply defense perspective for EPA
                    if 'epa' in stat.lower() and side == 'defense':
                        stat_value = -stat_value
                    
                    suffix = '_def' if side == 'defense' else ''
                    result[f'{stat}{suffix}'] = stat_value
        
        return result
    
    def _calculate_team_game_stats_vectorized(self, game_plays: pd.DataFrame, team_id: int, 
                                        opponent_id: int, side: str, statistics: List[str]) -> dict:
        """Calculate stats for one team in one game using vectorized operations"""
        
        # Basic info
        result = {
            'game_id': game_plays['game_id'].iloc[0],
            'team_id': team_id,
            'opponent_id': opponent_id,
            'side': side,
            'play_count': len(game_plays)
        }
        
        # Add metadata columns if available
        for col in ['season', 'date', 'home_team_name', 'away_team_name']:
            if col in game_plays.columns:
                result[col] = game_plays[col].iloc[0]
        
        # VECTORIZED CALCULATIONS - much faster than apply()
        
        # Basic yards and plays
        if 'stat_yardage' in game_plays.columns:
            total_yards = game_plays['stat_yardage'].sum()
            result['total_yards' + ('_allowed' if side == 'defense' else '')] = total_yards
            if len(game_plays) > 0:
                result['yards_per_play' + ('_allowed' if side == 'defense' else '')] = total_yards / len(game_plays)
        
        # Rushing stats
        if 'rushing_play' in game_plays.columns:
            rush_mask = game_plays['rushing_play'] == 1
            rush_plays = rush_mask.sum()
            result['total_rushing_plays' + ('_allowed' if side == 'defense' else '')] = rush_plays
            
            if 'stat_yardage' in game_plays.columns:
                rush_yards = game_plays.loc[rush_mask, 'stat_yardage'].sum()
                result['total_rushing_yards' + ('_allowed' if side == 'defense' else '')] = rush_yards
                if rush_plays > 0:
                    result['rush_yards_per_play' + ('_allowed' if side == 'defense' else '')] = rush_yards / rush_plays
        
        # Passing stats
        if 'passing_play' in game_plays.columns:
            pass_mask = game_plays['passing_play'] == 1
            pass_plays = pass_mask.sum()
            result['total_passing_plays' + ('_allowed' if side == 'defense' else '')] = pass_plays
            
            if 'stat_yardage' in game_plays.columns:
                pass_yards = game_plays.loc[pass_mask, 'stat_yardage'].sum()
                result['total_passing_yards' + ('_allowed' if side == 'defense' else '')] = pass_yards
                if pass_plays > 0:
                    result['pass_yards_per_play' + ('_allowed' if side == 'defense' else '')] = pass_yards / pass_plays
        
        # Success rates
        if 'successful_play' in game_plays.columns:
            success_count = game_plays['successful_play'].sum()
            result['total_successful_plays' + ('_allowed' if side == 'defense' else '')] = success_count
            result['play_success' + ('_allowed' if side == 'defense' else '')] = success_count / len(game_plays)
        
        # EPA calculations
        epa_cols = ['points_added', 'epa', 'expected_points_added']
        for col in epa_cols:
            if col in game_plays.columns:
                epa_values = game_plays[col].fillna(0)
                total_epa = epa_values.sum()
                if side == 'defense':
                    total_epa = -total_epa  # Defense perspective
                result['total_epa' + ('_defense' if side == 'defense' else '')] = total_epa
                result['epa_per_play' + ('_defense' if side == 'defense' else '')] = total_epa / len(game_plays)
                break
        
        # Process requested statistics efficiently
        for stat in statistics:
            if stat in game_plays.columns:
                stat_data = game_plays[stat].dropna()
                if len(stat_data) > 0:
                    # Use appropriate aggregation (see _is_rate_statistic)
                    if _is_rate_statistic(stat):
                        stat_value = stat_data.mean()
                        if 'epa' in stat.lower() and side == 'defense':
                            stat_value = -stat_value
                    else:
                        stat_value = stat_data.mean() if stat_data.max() <= 1.0 else stat_data.sum()
                    
                    result[stat + ('_def' if side == 'defense' else '')] = stat_value
        
        return result
    
    def _get_team_configs(self, df: pd.DataFrame) -> List[Dict]:
        """Determine team configurations based on available columns - INCLUDING BOTH OFFENSE AND DEFENSE"""
        
        configs = []
        
        # Check what team identification columns we have
        has_home_away = 'home_team_id' in df.columns and 'away_team_id' in df.columns
        has_team_id = 'team_id' in df.columns
        
        if has_home_away:
            # Traditional home/away setup - BOTH offense and defense for each team
            configs = [
                # Home team's offensive performance (when home team has the ball)
                {'name': 'home_offense', 'team_col': 'home_team_id', 'team_name_col': 'home_team_name', 
                 'opp_col': 'away_team_id', 'opp_name_col': 'away_team_name', 'filter_team': 'home_team_id',
                 'side': 'offense'},
                # Home team's defensive performance (when away team has the ball)
                {'name': 'home_defense', 'team_col': 'home_team_id', 'team_name_col': 'home_team_name', 
                 'opp_col': 'away_team_id', 'opp_name_col': 'away_team_name', 'filter_team': 'away_team_id',
                 'side': 'defense'},
                # Away team's offensive performance (when away team has the ball)
                {'name': 'away_offense', 'team_col': 'away_team_id', 'team_name_col': 'away_team_name',
                 'opp_col': 'home_team_id', 'opp_name_col': 'home_team_name', 'filter_team': 'away_team_id',
                 'side': 'offense'},
                # Away team's defensive performance (when home team has the ball)
                {'name': 'away_defense', 'team_col': 'away_team_id', 'team_name_col': 'away_team_name',
                 'opp_col': 'home_team_id', 'opp_name_col': 'home_team_name', 'filter_team': 'home_team_id',
                 'side': 'defense'}
            ]
        elif has_team_id:
            # Simple team_id setup (need to derive opponent somehow)
            configs = [
                {'name': 'team_offense', 'team_col': 'team_id', 'team_name_col': 'team_id', 
                 'opp_col': 'team_id', 'opp_name_col': 'team_id', 'filter_team': 'team_id',
                 'side': 'offense'}
            ]
        
        # Filter configs to only those with available columns
        valid_configs = []
        for config in configs:
            required_cols = [config['team_col'], config['filter_team']]
            if all(col in df.columns for col in required_cols):
                # Use fallback names if name columns don't exist
                if config['team_name_col'] not in df.columns:
                    config['team_name_col'] = config['team_col']
                if config['opp_name_col'] not in df.columns:
                    config['opp_name_col'] = config['opp_col']
                valid_configs.append(config)
        
        self.logger.info(f"Found {len(valid_configs)} valid team configurations (offense + defense)")
        return valid_configs
    
    def _calculate_team_stats(self, df: pd.DataFrame, config: Dict, statistics: List[str]) -> pd.DataFrame:
        """Calculate stats for a specific team configuration - FIXED EPA and added defense"""
        
        # Filter plays for this team configuration
        filter_condition = df['team_id'] == df[config['filter_team']]
        
        # Add offensive play filter if available
        if 'offensive_play' in df.columns:
            filter_condition = filter_condition & (df['offensive_play'] == 1)
        
        filtered_df = df[filter_condition]
        
        if filtered_df.empty:
            self.logger.warning(f"No plays found for {config['name']}")
            return pd.DataFrame()
        
        self.logger.debug(f"Processing {config['name']}: {len(filtered_df)} plays")
        
        def calculate_game_stats(game_group):
            """Calculate comprehensive statistics for a single game"""
            
            # Basic game info (same for all plays in the game)
            result = {
                config['team_col']: game_group[config['team_col']].iloc[0],
                'play_count': len(game_group),
                'side': config.get('side', 'offense')  # Track if this is offense or defense
            }
            
            # Add team name if column exists
            if config['team_name_col'] in game_group.columns:
                result[config['team_name_col']] = game_group[config['team_name_col']].iloc[0]
            
            # Add opponent info if columns exist
            if config['opp_col'] in game_group.columns:
                result[config['opp_col']] = game_group[config['opp_col']].iloc[0]
            if config['opp_name_col'] in game_group.columns:
                result[config['opp_name_col']] = game_group[config['opp_name_col']].iloc[0]
            
            # ====================================
            # SCORES AND BASIC METRICS
            # ====================================
            
            # Final scores (take from last play of game)
            if 'home_score' in game_group.columns and 'away_score' in game_group.columns:
                final_home_score = game_group['home_score'].iloc[-1]
                final_away_score = game_group['away_score'].iloc[-1]
                result['final_home_score'] = final_home_score
                result['final_away_score'] = final_away_score
                
                # Team-specific score
                if 'home_team_id' in game_group.columns:
                    if game_group[config['team_col']].iloc[0] == game_group['home_team_id'].iloc[0]:
                        result['team_score'] = final_home_score
                        result['opponent_score'] = final_away_score
                    else:
                        result['team_score'] = final_away_score
                        result['opponent_score'] = final_home_score
                
                # Score differential and win indicator
                if 'team_score' in result and 'opponent_score' in result:
                    result['score_differential'] = result['team_score'] - result['opponent_score']
                    result['win'] = 1 if result['score_differential'] > 0 else 0
            
            # ====================================
            # CALCULATE STATISTICS FROM RAW DATA
            # ====================================
            
            # Determine the perspective: for defense, we're measuring what the opponent did
            is_defense = config.get('side') == 'defense'
            
            # Total yards calculation
            yards_columns = ['offensive_yards', 'stat_yardage', 'yards_gained']
            for col in yards_columns:
                if col in game_group.columns:
                    total_yards = game_group[col].sum()
                    if is_defense:
                        result['total_yards_allowed'] = total_yards
                    else:
                        result['total_yards'] = total_yards
                    break
            
            # Rushing yards
            if 'rushing_yards' in game_group.columns:
                rush_yards = game_group['rushing_yards'].sum()
            elif 'stat_yardage' in game_group.columns and 'rushing_play' in game_group.columns:
                rush_mask = game_group['rushing_play'] == 1
                rush_yards = game_group.loc[rush_mask, 'stat_yardage'].sum()
            else:
                rush_yards = 0
            
            if is_defense:
                result['total_rushing_yards_allowed'] = rush_yards
            else:
                result['total_rushing_yards'] = rush_yards
            
            # Passing yards
            if 'passing_yards' in game_group.columns:
                pass_yards = game_group['passing_yards'].sum()
            elif 'stat_yardage' in game_group.columns and 'passing_play' in game_group.columns:
                pass_mask = game_group['passing_play'] == 1
                pass_yards = game_group.loc[pass_mask, 'stat_yardage'].sum()
            else:
                pass_yards = 0
            
            if is_defense:
                result['total_passing_yards_allowed'] = pass_yards
            else:
                result['total_passing_yards'] = pass_yards
            
            # Play counts
            if 'rushing_play' in game_group.columns:
                rush_plays = game_group['rushing_play'].sum()
                if is_defense:
                    result['total_rushing_plays_allowed'] = rush_plays
                else:
                    result['total_rushing_plays'] = rush_plays
            
            if 'passing_play' in game_group.columns:
                pass_plays = game_group['passing_play'].sum()
                if is_defense:
                    result['total_passing_plays_allowed'] = pass_plays
                else:
                    result['total_passing_plays'] = pass_plays
            
            # Success calculations
            if 'successful_play' in game_group.columns:
                successful_plays = game_group['successful_play'].sum()
                if is_defense:
                    result['total_successful_plays_allowed'] = successful_plays
                else:
                    result['total_successful_plays'] = successful_plays
            
            if 'rushing_successful_play' in game_group.columns:
                rush_success = game_group['rushing_successful_play'].sum()
            elif 'successful_play' in game_group.columns and 'rushing_play' in game_group.columns:
                rush_success_mask = (game_group['rushing_play'] == 1) & (game_group['successful_play'] == 1)
                rush_success = rush_success_mask.sum()
            else:
                rush_success = 0
            
            if is_defense:
                result['total_successful_rushes_allowed'] = rush_success
            else:
                result['total_successful_rushes'] = rush_success
            
            if 'passing_successful_play' in game_group.columns:
                pass_success = game_group['passing_successful_play'].sum()
            elif 'successful_play' in game_group.columns and 'passing_play' in game_group.columns:
                pass_success_mask = (game_group['passing_play'] == 1) & (game_group['successful_play'] == 1)
                pass_success = pass_success_mask.sum()
            else:
                pass_success = 0
            
            if is_defense:
                result['total_successful_passes_allowed'] = pass_success
            else:
                result['total_successful_passes'] = pass_success
            
            # Explosive plays
            if 'explosive_play' in game_group.columns:
                explosive_plays = game_group['explosive_play'].sum()
                if is_defense:
                    result['total_explosive_plays_allowed'] = explosive_plays
                else:
                    result['total_explosive_plays'] = explosive_plays
            
            if 'rushing_explosive_play' in game_group.columns:
                rush_explosive = game_group['rushing_explosive_play'].sum()
            elif 'explosive_play' in game_group.columns and 'rushing_play' in game_group.columns:
                rush_explosive_mask = (game_group['rushing_play'] == 1) & (game_group['explosive_play'] == 1)
                rush_explosive = rush_explosive_mask.sum()
            else:
                rush_explosive = 0
            
            if is_defense:
                result['total_explosive_rushes_allowed'] = rush_explosive
            else:
                result['total_explosive_rushes'] = rush_explosive
            
            if 'passing_explosive_play' in game_group.columns:
                pass_explosive = game_group['passing_explosive_play'].sum()
            elif 'explosive_play' in game_group.columns and 'passing_play' in game_group.columns:
                pass_explosive_mask = (game_group['passing_play'] == 1) & (game_group['explosive_play'] == 1)
                pass_explosive = pass_explosive_mask.sum()
            else:
                pass_explosive = 0
            
            if is_defense:
                result['total_explosive_passes_allowed'] = pass_explosive
            else:
                result['total_explosive_passes'] = pass_explosive
            
            # ====================================
            # EPA CALCULATIONS - FIXED!
            # ====================================
            
            # Find EPA column and calculate properly
            epa_total = 0
            epa_columns = ['points_added', 'epa', 'expected_points_added']
            
            for col in epa_columns:
                if col in game_group.columns:
                    # CRITICAL FIX: For defense, EPA should be NEGATIVE of what opponent gained
                    epa_values = game_group[col].fillna(0)
                    if is_defense:
                        # Defense: negative of opponent's EPA (good defense = negative opponent EPA)
                        epa_total = -epa_values.sum()
                        result['total_epa_defense'] = epa_total
                    else:
                        # Offense: positive EPA
                        epa_total = epa_values.sum()
                        result['total_epa'] = epa_total
                    break
            
            # Rush EPA
            if 'epa_per_rush' in game_group.columns and 'rushing_play' in game_group.columns:
                rush_mask = game_group['rushing_play'] == 1
                rush_epa_values = game_group.loc[rush_mask, 'epa_per_rush'].fillna(0)
                if is_defense:
                    result['total_rushing_epa_defense'] = -rush_epa_values.sum()
                else:
                    result['total_rushing_epa'] = rush_epa_values.sum()
            
            # Pass EPA
            if 'epa_per_pass' in game_group.columns and 'passing_play' in game_group.columns:
                pass_mask = game_group['passing_play'] == 1
                pass_epa_values = game_group.loc[pass_mask, 'epa_per_pass'].fillna(0)
                if is_defense:
                    result['total_passing_epa_defense'] = -pass_epa_values.sum()
                else:
                    result['total_passing_epa'] = pass_epa_values.sum()
            
            # ====================================
            # PROCESS REQUESTED STATISTICS
            # ====================================
            
            for stat in statistics:
                if stat not in game_group.columns:
                    continue
                
                stat_data = game_group[stat].dropna()
                
                if len(stat_data) == 0:
                    if is_defense:
                        result[f'{stat}_def'] = np.nan
                    else:
                        result[stat] = np.nan
                    continue
                
                # Determine how to calculate this statistic
                stat_lower = stat.lower()
                
                if 'success' in stat_lower or 'rate' in stat_lower or stat_lower.endswith('_rate'):
                    # Success rates and rates: mean of 0/1 indicators or pre-calculated rates
                    stat_value = stat_data.mean()
                elif 'per_play' in stat_lower or 'epa' in stat_lower:
                    # Per-play stats and EPA: take mean
                    # CRITICAL FIX: For defense EPA, negate the values
                    if 'epa' in stat_lower and is_defense:
                        stat_value = -stat_data.mean()
                    else:
                        stat_value = stat_data.mean()
                elif 'yards' in stat_lower and 'per' in stat_lower:
                    # Yards per play/rush/pass: take mean
                    stat_value = stat_data.mean()
                else:
                    # Default: mean for rates/percentages, sum for counting stats
                    if stat_data.dtype in ['int64', 'float64']:
                        if stat_data.max() <= 1.0 and stat_data.min() >= 0.0:
                            stat_value = stat_data.mean()  # Looks like a rate
                        else:
                            stat_value = stat_data.sum()   # Looks like a count
                    else:
                        stat_value = stat_data.mean()
                
                # Store with appropriate suffix
                if is_defense:
                    result[f'{stat}_def'] = stat_value
                else:
                    result[stat] = stat_value
            
            # ====================================
            # VALIDATION CALCULATIONS
            # ====================================
            
            # Calculate yards per play if we have the components
            yards_key = 'total_yards_allowed' if is_defense else 'total_yards'
            if result.get(yards_key, 0) > 0 and result['play_count'] > 0:
                ypp = result[yards_key] / result['play_count']
                if is_defense:
                    result['calculated_yards_per_play_def'] = ypp
                else:
                    result['calculated_yards_per_play'] = ypp
            
            # Calculate success rate if we have the components
            success_key = 'total_successful_plays_allowed' if is_defense else 'total_successful_plays'
            if result.get(success_key, 0) >= 0 and result['play_count'] > 0:
                success_rate = result[success_key] / result['play_count']
                if is_defense:
                    result['calculated_success_rate_def'] = success_rate
                else:
                    result['calculated_success_rate'] = success_rate
            
            # Calculate EPA per play if we have the components
            epa_key = 'total_epa_defense' if is_defense else 'total_epa'
            if result.get(epa_key) is not None and result['play_count'] > 0:
                epa_pp = result[epa_key] / result['play_count']
                if is_defense:
                    result['calculated_epa_per_play_def'] = epa_pp
                else:
                    result['calculated_epa_per_play'] = epa_pp
            
            return pd.Series(result)
        
        # Group by game and calculate stats - FIXED: Only one try block
        try:
            # Use available grouping columns
            group_cols = ['game_id']
            if 'date' in filtered_df.columns:
                group_cols.append('date')
            if 'season' in filtered_df.columns:
                group_cols.append('season')
            
            game_stats = filtered_df.groupby(group_cols).apply(calculate_game_stats).reset_index()
            
            # Check if we got valid results
            if game_stats.empty:
                self.logger.warning(f"No game stats calculated for {config['name']}")
                return pd.DataFrame()
            
            # Rename team columns to standard names
            rename_dict = {config['team_col']: 'team_id'}
            if config['team_name_col'] in game_stats.columns:
                rename_dict[config['team_name_col']] = 'team_name'
            if config['opp_col'] in game_stats.columns:
                rename_dict[config['opp_col']] = 'opponent_id'
            if config['opp_name_col'] in game_stats.columns:
                rename_dict[config['opp_name_col']] = 'opponent_name'
            
            game_stats = game_stats.rename(columns=rename_dict)
            
            # Set proper index
            index_cols = group_cols
            if 'season' not in index_cols and 'season' in game_stats.columns:
                index_cols = ['season'] + index_cols
            
            game_stats = game_stats.set_index(index_cols)
            
            self.logger.debug(f"Calculated {len(game_stats)} games for {config['name']}")
            return game_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating stats for {config['name']}: {e}")
            return pd.DataFrame()
    
    def _combine_team_stats(self, team_stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine offense and defense stats into complete team summaries"""
        
        # Separate offense and defense stats
        offense_dfs = []
        defense_dfs = []
        
        for config_name, stats_df in team_stats.items():
            if stats_df.empty:
                continue
                
            if 'offense' in config_name:
                offense_dfs.append(stats_df)
                self.logger.info(f"Added {len(stats_df)} offensive games from {config_name}")
            elif 'defense' in config_name:
                defense_dfs.append(stats_df)
                self.logger.info(f"Added {len(stats_df)} defensive games from {config_name}")
        
        # Combine offense stats
        if offense_dfs:
            offense_combined = pd.concat(offense_dfs).reset_index()
            
            # Debug: Check what columns we actually have
            self.logger.info(f"Offense columns after concat: {list(offense_combined.columns)}")
            
            # Find the actual team identification columns
            possible_team_cols = ['team_id', 'home_team_id', 'away_team_id']
            possible_game_cols = ['game_id', 'id']
            
            team_col = None
            game_col = None
            
            for col in possible_team_cols:
                if col in offense_combined.columns:
                    team_col = col
                    break
            
            for col in possible_game_cols:
                if col in offense_combined.columns:
                    game_col = col
                    break
            
            if team_col is None or game_col is None:
                self.logger.error(f"Cannot find team_id or game_id columns. Available: {list(offense_combined.columns)}")
                # Try to continue without deduplication
            else:
                # Remove duplicates based on available columns
                dedup_cols = [game_col, team_col]
                self.logger.info(f"Deduplicating offense stats using columns: {dedup_cols}")
                before_count = len(offense_combined)
                offense_combined = offense_combined.drop_duplicates(subset=dedup_cols, keep='first')
                after_count = len(offense_combined)
                if before_count != after_count:
                    self.logger.info(f"Removed {before_count - after_count} duplicate offense records")
        else:
            self.logger.error("No offensive stats found")
            return pd.DataFrame()
        
        # Combine defense stats  
        if defense_dfs:
            defense_combined = pd.concat(defense_dfs).reset_index()
            
            # Debug: Check what columns we actually have
            self.logger.info(f"Defense columns after concat: {list(defense_combined.columns)}")
            
            # Find the actual team identification columns (same logic as offense)
            team_col = None
            game_col = None
            
            for col in possible_team_cols:
                if col in defense_combined.columns:
                    team_col = col
                    break
            
            for col in possible_game_cols:
                if col in defense_combined.columns:
                    game_col = col
                    break
            
            if team_col is not None and game_col is not None:
                dedup_cols = [game_col, team_col]
                self.logger.info(f"Deduplicating defense stats using columns: {dedup_cols}")
                before_count = len(defense_combined)
                defense_combined = defense_combined.drop_duplicates(subset=dedup_cols, keep='first')
                after_count = len(defense_combined)
                if before_count != after_count:
                    self.logger.info(f"Removed {before_count - after_count} duplicate defense records")
        else:
            self.logger.warning("No defensive stats found - will only have offensive stats")
            defense_combined = pd.DataFrame()
        
        # Merge offense and defense on available columns
        if not defense_combined.empty:
            # Find common columns for merging
            merge_cols = []
            possible_merge_cols = ['season', 'date', 'game_id', 'id', 'team_id', 'home_team_id', 'away_team_id']
            
            for col in possible_merge_cols:
                if col in offense_combined.columns and col in defense_combined.columns:
                    merge_cols.append(col)
            
            self.logger.info(f"Merging offense and defense on columns: {merge_cols}")
            
            if merge_cols:
                final_df = offense_combined.merge(
                    defense_combined, 
                    on=merge_cols, 
                    how='left',
                    suffixes=('', '_def_dup')
                )
                
                # Clean up duplicate columns from the merge
                dup_cols = [col for col in final_df.columns if col.endswith('_def_dup')]
                if dup_cols:
                    final_df = final_df.drop(columns=dup_cols)
                
                self.logger.info(f"Merged offense and defense: {len(final_df)} complete team-game records")
            else:
                self.logger.warning("Cannot merge offense and defense - no common columns found")
                self.logger.warning(f"Offense columns: {list(offense_combined.columns)}")
                self.logger.warning(f"Defense columns: {list(defense_combined.columns)}")
                final_df = offense_combined
        else:
            final_df = offense_combined
        
        # Clean up side column if it exists
        if 'side' in final_df.columns:
            final_df = final_df.drop(columns=['side'])
        
        # Create proper index from available columns
        index_cols = []
        possible_index_cols = ['season', 'date', 'game_id', 'id', 'team_id', 'team_name', 
                              'opponent_id', 'opponent_name', 'home_team_id', 'away_team_id']
        
        for col in possible_index_cols:
            if col in final_df.columns:
                index_cols.append(col)
        
        # Limit index columns to avoid too many levels
        if len(index_cols) > 6:
            index_cols = index_cols[:6]
        
        if index_cols:
            try:
                final_df = final_df.set_index(index_cols)
                self.logger.info(f"Set index using columns: {index_cols}")
            except Exception as e:
                self.logger.warning(f"Could not set index with {index_cols}: {e}")
                # Continue without setting index
        
        self.logger.info(f"Final combined stats: {len(final_df)} unique team-game records")
        return final_df

class OpponentAdjuster:
    """Handles opponent adjustments using Ridge regression"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _fit_ridge_model(self, df: pd.DataFrame, stat: str, team_ids: np.ndarray) -> List[float]:
        """Fit Ridge regression for proper opponent adjustment"""
        
        # Debug info
        self.logger.info(f"Ridge regression for {stat}")
        self.logger.info(f"  Data: {len(df)} games, {df['team_id'].nunique()} teams")
        
        # Check for missing values
        if df[stat].isna().all():
            raise ValueError(f"All values are NaN for {stat}")
        
        # Remove rows with NaN values
        valid_df = df.dropna(subset=[stat, 'team_id', 'opponent_id'])
        
        if len(valid_df) < 10:
            raise ValueError(f"Not enough valid data for {stat}: {len(valid_df)} games (need at least 10)")
        
        # Check if we have variation in the target
        if valid_df[stat].std() < 0.001:
            raise ValueError(f"No variation in {stat}: std={valid_df[stat].std():.6f}")
        
        # Verify we have opponent_id column
        if 'opponent_id' not in valid_df.columns:
            raise ValueError(f"opponent_id column missing - cannot do opponent adjustment")
        
        # Create dummy variables for teams and opponents
        self.logger.info(f"  Creating dummy variables...")
        
        # Get all unique teams and opponents
        all_team_ids = sorted(set(valid_df['team_id'].unique()) | set(valid_df['opponent_id'].unique()))
        
        if len(all_team_ids) < 4:
            raise ValueError(f"Not enough teams for opponent adjustment: {len(all_team_ids)} teams")
        
        # Create design matrix
        n_teams = len(all_team_ids)
        n_games = len(valid_df)
        
        # Create team and opponent matrices
        X = np.zeros((n_games, 2 * n_teams))
        
        team_id_to_idx = {tid: idx for idx, tid in enumerate(all_team_ids)}
        
        for i, (_, row) in enumerate(valid_df.iterrows()):
            team_id = row['team_id']
            opp_id = row['opponent_id']
            
            if team_id not in team_id_to_idx:
                raise ValueError(f"Team ID {team_id} not found in mapping")
            if opp_id not in team_id_to_idx:
                raise ValueError(f"Opponent ID {opp_id} not found in mapping")
            
            team_idx = team_id_to_idx[team_id]
            opp_idx = team_id_to_idx[opp_id]
            
            # Team effect (positive coefficient = good offense)
            X[i, team_idx] = 1
            # Opponent effect (positive coefficient = bad defense, easier to score against)
            X[i, n_teams + opp_idx] = 1
        
        y = valid_df[stat].values
        
        # Check that X matrix is properly formed
        if X.shape[0] != len(y):
            raise ValueError(f"Design matrix shape mismatch: X={X.shape}, y={len(y)}")
        
        if np.all(X == 0):
            raise ValueError("Design matrix is all zeros")
        
        # Fit Ridge regression
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        ridge.fit(X, y)
        
        # Check model quality
        r2_score = ridge.score(X, y)
        if r2_score < 0:
            raise ValueError(f"Ridge regression failed: R² = {r2_score:.3f} (negative)")
        
        # Extract coefficients
        team_effects = ridge.coef_[:n_teams]  # Team offensive effects
        opp_effects = ridge.coef_[n_teams:]   # Team defensive effects (how much they allow)
        intercept = ridge.intercept_
        
        self.logger.info(f"  Ridge fitted successfully:")
        self.logger.info(f"    R² = {r2_score:.3f}")
        self.logger.info(f"    Team effects range: {team_effects.min():.3f} to {team_effects.max():.3f}")
        self.logger.info(f"    Opp effects range: {opp_effects.min():.3f} to {opp_effects.max():.3f}")
        self.logger.info(f"    Intercept: {intercept:.3f}")
        
        # Calculate opponent-adjusted ratings for requested teams
        result = []
        
        for team_id in team_ids:
            if team_id not in team_id_to_idx:
                raise ValueError(f"Requested team {team_id} not found in season data")
            
            team_idx = team_id_to_idx[team_id]
            
            # Opponent-adjusted rating = intercept + team_effect + average_opponent_difficulty
            # We use the average opponent effect as baseline (league average defense)
            avg_opp_effect = opp_effects.mean()
            adjusted_rating = intercept + team_effects[team_idx] + avg_opp_effect
            
            result.append(adjusted_rating)
        
        # Verify results have variation
        result_array = np.array(result)
        if result_array.std() < 0.001:
            raise ValueError(f"Opponent adjustment failed - no variation in results: std={result_array.std():.6f}")
        
        self.logger.info(f"  Final adjusted ratings:")
        self.logger.info(f"    Range: {result_array.min():.3f} to {result_array.max():.3f}")
        self.logger.info(f"    Std: {result_array.std():.3f}")
        
        return result
    
    def _adjust_season(self, season_data: pd.DataFrame, stat_columns: List[Tuple[str, str]], season: int) -> pd.DataFrame:
        """Adjust statistics for a single season"""
        season_df = season_data.reset_index()
        
        # Check if we have enough data
        if len(season_df) < 20:
            raise ValueError(f"Not enough data for season {season}: {len(season_df)} games (need at least 20)")
        
        # Ensure we have opponent information
        if 'opponent_id' not in season_df.columns:
            raise ValueError(f"No opponent_id column for season {season} - cannot do opponent adjustment")
        
        # Check that we have actual opponent variation
        unique_opponents = season_df['opponent_id'].nunique()
        if unique_opponents < 4:
            raise ValueError(f"Not enough opponent variation for season {season}: {unique_opponents} unique opponents")
        
        # Get unique teams that actually played games
        team_ids = sorted(season_df['team_id'].unique())
        
        if len(team_ids) < 4:
            raise ValueError(f"Not enough teams for season {season}: {len(team_ids)} teams")
        
        # Initialize result dictionary
        adjusted_stats = {
            'season': [season] * len(team_ids),
            'team_id': team_ids
        }
        
        # Process each statistic. stat_columns holds (source, output) pairs so
        # offense and defense are adjusted separately and land in distinctly
        # named columns.
        for source_col, output_col in stat_columns:
            if source_col not in season_df.columns:
                raise ValueError(f"Statistic {source_col} not found in season {season} data")

            self.logger.info(f"Processing {source_col} -> {output_col} for season {season}...")
            adjusted_values = self._fit_ridge_model(season_df, source_col, team_ids)
            adjusted_stats[output_col] = adjusted_values
        
        # Create result DataFrame
        result_df = pd.DataFrame(adjusted_stats)
        
        # Verify final result
        if result_df.empty:
            raise ValueError(f"Empty result DataFrame for season {season}")
        
        # Check that adjusted stats have variation
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.startswith('adjusted_') and result_df[col].std() < 0.001:
                raise ValueError(f"No variation in {col} for season {season}: std={result_df[col].std():.6f}")
        
        return result_df
    
    def adjust_for_opponents(self, game_summaries: pd.DataFrame, statistics: List[str]) -> pd.DataFrame:
        """Apply opponent adjustments to game statistics"""
        
        if game_summaries.empty:
            raise ValueError("No game summaries to adjust")
        
        # Every statistic has an offensive and a defensive side: the offensive
        # column is unsuffixed (or _off), the defensive one carries _def. BOTH
        # must be adjusted - the models consume adjusted_<stat>_off *and*
        # adjusted_<stat>_def. This previously took whichever variant matched
        # first and stopped, so only offense was ever adjusted and the output
        # was named without a side suffix, leaving every model feature missing.
        # Entries are (source_column, output_column).
        stat_columns = []
        for stat in statistics:
            off_col = next(
                (c for c in (f"{stat}_off", stat) if c in game_summaries.columns),
                None
            )
            def_col = f"{stat}_def" if f"{stat}_def" in game_summaries.columns else None

            if off_col:
                stat_columns.append((off_col, f"adjusted_{stat}_off"))
            else:
                self.logger.warning(f"No offensive column for {stat} - skipping that side")

            if def_col:
                stat_columns.append((def_col, f"adjusted_{stat}_def"))
            else:
                self.logger.warning(f"No defensive column for {stat} - skipping that side")
        
        if not stat_columns:
            self.logger.warning("No statistics found for opponent adjustment - returning empty DataFrame")
            return pd.DataFrame()
        
        self.logger.info(f"Adjusting {len(stat_columns)} statistics: {stat_columns}")
        
        adjusted_results = []
        
        # Process each season separately
        df_reset = game_summaries.reset_index()
        if 'season' not in df_reset.columns:
            self.logger.warning("No season column found - processing all data as one season")
            df_reset['season'] = 2024
        
        seasons = sorted(df_reset['season'].unique())
        self.logger.info(f"Processing {len(seasons)} seasons: {seasons}")
        
        for season in tqdm(seasons, desc="Adjusting seasons"):
            season_data = df_reset[df_reset['season'] == season].copy()
            
            if season_data.empty:
                self.logger.warning(f"No data for season {season}")
                continue
            
            try:
                self.logger.info(f"Processing season {season}: {len(season_data)} games")
                season_adjusted = self._adjust_season(season_data, stat_columns, season)
                adjusted_results.append(season_adjusted)
            except Exception as e:
                self.logger.error(f"Failed to adjust season {season}: {e}")
                continue
        
        if not adjusted_results:
            self.logger.warning("No adjusted stats generated")
            return pd.DataFrame()
        
        result_df = pd.concat(adjusted_results, ignore_index=True)
        
        # Final verification
        if result_df.empty:
            self.logger.warning("Final result DataFrame is empty")
            return pd.DataFrame()
        
        self.logger.info(f"Opponent adjustments complete: {len(result_df):,} team-seasons")
        return result_df


class RollingStatsCalculator:
    """Calculates rolling statistics and trends"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_rolling_stats(self, game_summaries: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages for all statistics"""
        
        if game_summaries.empty:
            return pd.DataFrame()
        
        df = game_summaries.reset_index()
        
        # Get numeric columns (exclude ID and name columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['season', 'game_id', 'team_id', 'opponent_id']
        stat_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not stat_cols:
            self.logger.warning("No numeric statistics found for rolling calculation")
            return pd.DataFrame()
        
        rolling_results = []
        
        # Process each season separately
        if 'season' in df.columns:
            seasons = sorted(df['season'].unique())
        else:
            seasons = [2024]  # Default
            df['season'] = 2024
        
        for season in tqdm(seasons, desc="Calculating rolling stats"):
            season_data = df[df['season'] == season].copy()
            season_rolling = self._calculate_season_rolling(season_data, stat_cols)
            if not season_rolling.empty:
                rolling_results.append(season_rolling)
        
        if not rolling_results:
            return pd.DataFrame()
            
        result_df = pd.concat(rolling_results)
        self.logger.info(f"Rolling statistics complete: {len(result_df):,} game records")
        return result_df
    
    def _calculate_season_rolling(self, season_data: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
        """Calculate rolling stats for a single season"""
        
        if 'team_id' not in season_data.columns:
            return pd.DataFrame()
        
        # Sort by date within each team
        sort_cols = ['team_id']
        if 'date' in season_data.columns:
            sort_cols.append('date')
        if 'game_id' in season_data.columns:
            sort_cols.append('game_id')
        
        season_data = season_data.sort_values(sort_cols)
        
        # Calculate rolling means for each team
        rolling_data = season_data.copy()
        
        for col in stat_cols:
            if col in rolling_data.columns:
                rolling_data[f'{col}_rolling_avg'] = (
                    rolling_data.groupby('team_id')[col]
                    .transform(lambda x: x.expanding().mean().shift(1))
                )
        
        # Keep only essential columns
        keep_cols = ['season', 'team_id']
        if 'date' in rolling_data.columns:
            keep_cols.append('date')
        if 'game_id' in rolling_data.columns:
            keep_cols.append('game_id')
        if 'team_name' in rolling_data.columns:
            keep_cols.append('team_name')
        if 'opponent_id' in rolling_data.columns:
            keep_cols.append('opponent_id')
        if 'opponent_name' in rolling_data.columns:
            keep_cols.append('opponent_name')
        
        # Add rolling average columns
        rolling_cols = [f'{col}_rolling_avg' for col in stat_cols if f'{col}_rolling_avg' in rolling_data.columns]
        keep_cols.extend(rolling_cols)
        
        return rolling_data[keep_cols]


class CFBAnalyticsPipeline:
    """Main analytics pipeline orchestrator"""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self._setup_logging()
        
        # Initialize components
        self.data_loader = CFBDataLoader(self.config)
        self.stats_calculator = GameStatsCalculator(self.config)
        self.opponent_adjuster = OpponentAdjuster(self.config)
        self.rolling_calculator = RollingStatsCalculator(self.config)
    
    def _setup_logging(self):
        """Configure logging with console error output"""
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Clear any existing handlers to avoid conflicts
        logging.getLogger().handlers = []
        
        # Setup file logging (detailed)
        file_handler = logging.FileHandler("logs/summarize.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Setup console logging (errors and warnings only, but make sure they show)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Show warnings and errors
        console_formatter = logging.Formatter('CONSOLE %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Configure our specific logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Test logging
        self.logger.error("TEST: Console error logging enabled")
        self.logger.warning("TEST: Console warning logging enabled")
        
        # Force flush
        sys.stdout.flush()
        sys.stderr.flush()
    
    def run_full_pipeline(self, pbp_file: str, games_file: str, teams_file: str, 
                         statistics: List[str]) -> Dict[str, pd.DataFrame]:
        """Run the complete analytics pipeline"""
        self.logger.info("Starting CFB Analytics Pipeline")
        
        try:
            # Load and prepare data
            pbp, games, teams = self.data_loader.load_all_data(pbp_file, games_file, teams_file)
            merged_data = self.data_loader.prepare_data(pbp, games, teams)
            self.logger.info(f"Merged data shape: {merged_data.shape}")
            
            # Calculate game-by-game summaries
            with tqdm(total=1, desc="Calculating game summaries") as pbar:
                game_summaries = self.stats_calculator.calculate_game_summaries(merged_data, statistics)
                pbar.update(1)
            
            if game_summaries.empty:
                self.logger.error("No game summaries generated")
                return {}
            
            # Calculate opponent adjustments
            with tqdm(total=1, desc="Calculating adjustments") as pbar:
                adjusted_stats = self.opponent_adjuster.adjust_for_opponents(game_summaries, statistics)
                pbar.update(1)
            
            # Calculate rolling statistics
            with tqdm(total=1, desc="Calculating rolling stats") as pbar:
                rolling_stats = self.rolling_calculator.calculate_rolling_stats(game_summaries)
                pbar.update(1)
            
            # Prepare results with correct naming
            results = {
                'game_summaries': game_summaries,
                'rolling_stats': rolling_stats
            }
            
            if not adjusted_stats.empty:
                results['adjusted_stats'] = adjusted_stats
            
            self.logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def save_results(self, results: Dict[str, pd.DataFrame], output_dir: str):
        """Save pipeline results to CSV files with correct naming"""
        self.logger.info(f"Saving results to {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Map internal names to expected output file names
        file_name_mapping = {
            'game_summaries': 'game_by_game_summaries.csv',
            'rolling_stats': 'rolling_summaries.csv', 
            'adjusted_stats': 'season_summaries.csv'
        }
        
        # Save each result with correct naming
        for internal_name, df in results.items():
            if df.empty:
                self.logger.warning(f"Skipping empty result: {internal_name}")
                continue
            
            # Get the correct output filename
            output_filename = file_name_mapping.get(internal_name, f"{internal_name}.csv")
            output_file = Path(output_dir) / output_filename
            
            try:
                # Round numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_rounded = df.copy()
                df_rounded[numeric_cols] = df_rounded[numeric_cols].round(self.config.round_decimals)
                
                # Save to CSV
                df_rounded.to_csv(output_file)
                print(f"Saved {internal_name}: {len(df_rounded):,} rows to {output_file}")
                self.logger.info(f"Saved {internal_name}: {len(df_rounded)} rows to {output_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to save {internal_name}: {e}")
        
        self.logger.info("Results saved successfully")


def main():
    """Main entry point with command line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='College Football Analytics Pipeline')
    parser.add_argument('--pbp_file', required=True, help='Path to play-by-play CSV file')
    parser.add_argument('--games_file', required=True, help='Path to games CSV file')
    parser.add_argument('--teams_file', required=True, help='Path to teams CSV file')
    parser.add_argument('--statistics', nargs='+', required=True, help='List of statistics to calculate')
    parser.add_argument('--output_dir', default='results', help='Output directory for results')
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge regression alpha parameter')
    parser.add_argument('--max_periods', type=int, default=4, help='Maximum periods to include (no OT)')
    parser.add_argument('--no_fbs_filter', action='store_true', help='Include non-FBS teams')
    parser.add_argument('--include_garbage_time', action='store_true', help='Include garbage time plays')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalyticsConfig(
        max_periods=args.max_periods,
        fbs_only=not args.no_fbs_filter,
        remove_garbage_time=not args.include_garbage_time,
        ridge_alpha=args.alpha,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    pipeline = CFBAnalyticsPipeline(config)
    results = pipeline.run_full_pipeline(
        args.pbp_file, 
        args.games_file, 
        args.teams_file, 
        args.statistics
    )
    
    # Save results
    if results:
        pipeline.save_results(results, args.output_dir)
    else:
        print("No results to save")


if __name__ == "__main__":
    main()