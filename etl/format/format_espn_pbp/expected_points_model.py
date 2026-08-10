# -*- coding: utf-8 -*-
"""
Expected Points Model for Football Analysis
Improved version with better structure, error handling, and documentation

@author: wfish
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

# Suppress pandas warnings that clutter progress bars
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', message='.*mixed types.*')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExpectedPointsModel:
    """
    A class to build and train an Expected Points model for football games.
    """
    
    def __init__(self, model_params: Optional[dict] = None):
        """
        Initialize the Expected Points Model.
        
        Args:
            model_params: Dictionary of XGBoost parameters
        """
        self.model_params = model_params or {
            'enable_categorical': True,
            'random_state': 42,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
        self.model = None
        self.key_df = None
        
    def import_data(self, pbp_dir: str, games_dir: str, 
                   seasons: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Import play-by-play and games data from CSV files organized by season.
        
        Args:
            pbp_dir: Directory containing play-by-play CSV files
            games_dir: Directory containing games CSV files
            seasons: List of seasons to include (e.g., ['2020', '2021']). If None, includes all available seasons.
            
        Returns:
            Tuple of (pbp_dataframe, games_dataframe)
        """
        try:
            pbp_dir = Path(pbp_dir)
            games_dir = Path(games_dir)
            
            if not pbp_dir.exists():
                raise FileNotFoundError(f"Play-by-play directory not found: {pbp_dir}")
            if not games_dir.exists():
                raise FileNotFoundError(f"Games directory not found: {games_dir}")
            
            # Get all CSV files in directories
            pbp_files = list(pbp_dir.glob("*.csv"))
            games_files = list(games_dir.glob("*.csv"))
            
            if not pbp_files:
                raise FileNotFoundError(f"No CSV files found in {pbp_dir}")
            if not games_files:
                raise FileNotFoundError(f"No CSV files found in {games_dir}")
            
            # Filter by seasons if specified
            if seasons:
                season_filter = lambda files: [f for f in files if any(season in f.name for season in seasons)]
                pbp_files = season_filter(pbp_files)
                games_files = season_filter(games_files)
            
            logger.info(f"Found {len(pbp_files)} play-by-play files and {len(games_files)} games files")
            
            # Load and combine play-by-play data
            pbp_dfs = []
            for file in tqdm(sorted(pbp_files), desc="Loading play-by-play files", ncols=80):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df = pd.read_csv(file)
                    # Add season identifier from filename
                    season_match = self._extract_season_from_filename(file.name)
                    if season_match:
                        df['season'] = season_match
                    pbp_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error loading {file.name}: {e}")
                    continue
            
            # Load and combine games data
            games_dfs = []
            for file in tqdm(sorted(games_files), desc="Loading games files", ncols=80):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df = pd.read_csv(file)
                    # Add season identifier from filename
                    season_match = self._extract_season_from_filename(file.name)
                    if season_match:
                        df['season'] = season_match
                    games_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error loading {file.name}: {e}")
                    continue
            
            if not pbp_dfs:
                raise ValueError("No play-by-play data could be loaded")
            if not games_dfs:
                raise ValueError("No games data could be loaded")
            
            # Combine all dataframes
            pbp_combined = pd.concat(pbp_dfs, ignore_index=True)
            games_combined = pd.concat(games_dfs, ignore_index=True)
            
            logger.info(f"Successfully imported {len(pbp_combined)} plays and {len(games_combined)} games")
            logger.info(f"Data covers seasons: {sorted(pbp_combined['season'].unique()) if 'season' in pbp_combined.columns else 'Unknown'}")
            
            return pbp_combined, games_combined
            
        except FileNotFoundError as e:
            logger.error(f"Directory not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            raise
    
    def _extract_season_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract season information from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Season string or None if not found
        """
        import re
        
        # Look for patterns like "2020-08-01_to_2021-02-01"
        # Extract the ending year as the season
        pattern = r'(\d{4})-\d{2}-\d{2}_to_(\d{4})-\d{2}-\d{2}'
        match = re.search(pattern, filename)
        if match:
            return match.group(2)  # Return the ending year
        
        # Fallback: look for any 4-digit year
        year_pattern = r'(\d{4})'
        matches = re.findall(year_pattern, filename)
        if matches:
            return matches[-1]  # Return the last year found
        
        return None
    
    def clean_pbp_data(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare play-by-play data for modeling.
        
        Args:
            pbp: Raw play-by-play dataframe
            
        Returns:
            Cleaned play-by-play dataframe
        """
        logger.info("Cleaning play-by-play data")
        
        # Remove kickoffs from training data
        pbp = pbp.loc[~pbp['play_type_id'].isin([12, 53])]
        
        # Select relevant columns
        required_cols = ['game_id', 'id', 'home_score', 'away_score', 
                        'team_id', 'period', 'clock', 'down', 
                        'distance', 'yards_to_end_zone']
        
        missing_cols = set(required_cols) - set(pbp.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        pbp_clean = pbp[required_cols].copy()
        
        # Add half indicator
        pbp_clean['half'] = np.where(pbp_clean['period'] <= 2, 1, 2)
        
        # Calculate score differences within each half
        pbp_clean = pbp_clean.sort_values(['game_id', 'id'])
        pbp_clean['home_score_diff'] = pbp_clean.groupby(['game_id', 'half'])['home_score'].diff()
        pbp_clean['away_score_diff'] = pbp_clean.groupby(['game_id', 'half'])['away_score'].diff()
        
        # Process clock information
        pbp_clean = self._process_clock_data(pbp_clean)
        
        # Clean data quality issues
        pbp_clean = self._clean_data_quality(pbp_clean)
        
        logger.info(f"Cleaned data: {len(pbp_clean)} plays remaining")
        return pbp_clean
    
    def _process_clock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process clock information to calculate time remaining."""
        # Handle missing or malformed clock data
        df['clock'] = df['clock'].fillna('15:00')
        
        # Split clock into minutes and seconds
        clock_split = df['clock'].str.split(':', expand=True)
        df['minutes'] = pd.to_numeric(clock_split[0], errors='coerce').fillna(0).astype(int)
        df['seconds'] = pd.to_numeric(clock_split[1], errors='coerce').fillna(0).astype(int)
        
        # Filter to regulation periods only
        df = df[df['period'].isin([1, 2, 3, 4])].copy()
        
        # Calculate half seconds remaining
        df['half_seconds_remaining'] = df['minutes'] * 60 + df['seconds']
        
        # Add quarter time for first and third quarters
        mask = df['period'].isin([1, 3])
        df.loc[mask, 'half_seconds_remaining'] = df.loc[mask, 'half_seconds_remaining'] + 900
        
        return df
    
    def _clean_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data quality issues in the dataset."""
        # Cap distance at reasonable maximum
        df.loc[df['distance'] > 25, 'distance'] = 25
        
        # Remove invalid downs
        df = df[df['down'] != -1].copy()
        
        # Fix distance for goal line situations
        mask = (df['down'] != 0) & (df['distance'] == 0)
        df.loc[mask, 'distance'] = df.loc[mask, 'yards_to_end_zone']
        
        # Filter to valid downs and distances
        df = df[
            (df['down'] >= 1) & (df['down'] <= 4) &
            (df['distance'] >= 1) & (df['distance'] <= 99)
        ].copy()
        
        return df
    
    def create_scoring_outcomes(self, pbp_clean: pd.DataFrame, games: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create scoring outcome labels for both home and away teams.
        
        Args:
            pbp_clean: Cleaned play-by-play data
            games: Games data with team IDs
            
        Returns:
            Tuple of (combined_model_data, scoring_key)
        """
        logger.info("Creating scoring outcome labels")
        
        # Prepare games data
        games_subset = games[['id', 'home_team_id', 'away_team_id']].copy()
        games_subset.rename(columns={'id': 'game_id'}, inplace=True)
        
        # Process home and away data
        home_data = self._create_team_scoring_data(pbp_clean, games_subset, 'home')
        away_data = self._create_team_scoring_data(pbp_clean, games_subset, 'away')
        
        # Combine datasets
        combined_data = pd.concat([home_data, away_data], ignore_index=True)
        
        # Create scoring key
        scoring_key = self._create_scoring_key(combined_data)
        
        logger.info(f"Created {len(combined_data)} training examples")
        return combined_data, scoring_key
    
    def _create_team_scoring_data(self, pbp_clean: pd.DataFrame, games: pd.DataFrame, team_type: str) -> pd.DataFrame:
        """Create scoring data for a specific team type (home/away)."""
        data = pbp_clean.copy()
        
        # Merge with games data
        data = pd.merge(data, games, on='game_id')
        
        # Define scoring patterns based on team type
        if team_type == 'home':
            own_score_col = 'home_score_diff'
            opp_score_col = 'away_score_diff'
            team_filter = data['home_team_id'] == data['team_id']
        else:
            own_score_col = 'away_score_diff'
            opp_score_col = 'home_score_diff'
            team_filter = data['away_team_id'] == data['team_id']
        
        # Assign scoring outcomes
        data = self._assign_scoring_outcomes(data, own_score_col, opp_score_col)
        
        # Mark end of halves
        data = self._mark_end_of_halves(data)
        
        # Forward fill missing outcomes
        data['next_play'] = data.groupby('game_id')['next_play'].bfill()
        
        # Select relevant columns and filter to team
        model_cols = ['game_id', 'team_id', 'down', 'distance', 
                     'yards_to_end_zone', 'half_seconds_remaining', 'next_play']
        
        return data.loc[team_filter, model_cols].copy()
    
    def _assign_scoring_outcomes(self, data: pd.DataFrame, own_score: str, opp_score: str) -> pd.DataFrame:
        """Assign scoring outcome labels based on score changes."""
        data['next_play'] = np.nan
        
        # Own team scoring
        data.loc[data[own_score].isin([6, 7, 8]), 'next_play'] = 'touchdown'
        data.loc[data[own_score] == 3, 'next_play'] = 'field_goal'
        data.loc[data[own_score] == 2, 'next_play'] = 'safety'
        
        # Opponent scoring
        data.loc[data[opp_score].isin([6, 7, 8]), 'next_play'] = 'opponent_touchdown'
        data.loc[data[opp_score] == 3, 'next_play'] = 'opponent_field_goal'
        data.loc[data[opp_score] == 2, 'next_play'] = 'opponent_safety'
        
        return data
    
    def _mark_end_of_halves(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mark the end of halves as no-score plays."""
        end_of_half = data.groupby(['game_id', 'half']).tail(1)
        data.loc[data['id'].isin(end_of_half['id']), 'next_play'] = 'no_score'
        return data
    
    def _create_scoring_key(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a key mapping scoring outcomes to numerical codes."""
        data['next_play'] = data['next_play'].astype('category')
        data['next_play_cat'] = data['next_play'].cat.codes
        
        key_df = (data[['next_play', 'next_play_cat']]
                 .drop_duplicates()
                 .sort_values('next_play_cat')
                 .reset_index(drop=True))
        
        return key_df
    
    def train_model(self, model_data: pd.DataFrame, test_size: float = 0.2, 
                   validate: bool = True) -> dict:
        """
        Train the Expected Points model.
        
        Args:
            model_data: Prepared training data
            test_size: Fraction of data to use for testing
            validate: Whether to perform model validation
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training Expected Points model")
        
        # Prepare features and target
        feature_cols = ['down', 'distance', 'yards_to_end_zone', 'half_seconds_remaining']
        X = model_data[feature_cols]
        y = model_data['next_play_cat']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        results = {}
        
        if validate:
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            self.model = XGBClassifier(**self.model_params)
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results['accuracy'] = accuracy
            results['classification_report'] = classification_report(y_test, y_pred)
            results['feature_importance'] = dict(zip(feature_cols, self.model.feature_importances_))
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            
        else:
            # Train on full dataset
            self.model = XGBClassifier(**self.model_params)
            self.model.fit(X, y)
            
            results['feature_importance'] = dict(zip(feature_cols, self.model.feature_importances_))
        
        logger.info("Model training completed")
        return results
    
    def save_model(self, model_path: str = 'ep_model.pkl', key_path: str = 'scoring_key.csv'):
        """
        Save the trained model and scoring key.
        
        Args:
            model_path: Path to save the model
            key_path: Path to save the scoring key
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if self.key_df is not None:
            logger.info(f"Saving scoring key to {key_path}")
            self.key_df.to_csv(key_path, index=False)
    
    def predict_expected_points(self, down: int, distance: int, 
                              yards_to_end_zone: int, half_seconds_remaining: int = 900) -> dict:
        """
        Predict expected points for a given situation.
        
        Args:
            down: Down (1-4)
            distance: Distance to first down
            yards_to_end_zone: Yards to end zone
            half_seconds_remaining: Seconds remaining in half (default: 15 minutes)
            
        Returns:
            Dictionary with predicted probabilities for each outcome
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X = pd.DataFrame({
            'down': [down],
            'distance': [distance],
            'yards_to_end_zone': [yards_to_end_zone],
            'half_seconds_remaining': [half_seconds_remaining]
        })
        
        probabilities = self.model.predict_proba(X)[0]
        
        if self.key_df is not None:
            outcomes = self.key_df['next_play'].tolist()
            return dict(zip(outcomes, probabilities))
        else:
            return dict(enumerate(probabilities))
    
    def calculate_expected_points(self, probabilities: dict) -> float:
        """
        Calculate expected points from outcome probabilities.
        
        Args:
            probabilities: Dictionary with outcome probabilities
            
        Returns:
            Expected points value
        """
        # Point values for each outcome
        point_values = {
            'touchdown': 7,
            'field_goal': 3,
            'safety': 2,
            'opponent_touchdown': -7,
            'opponent_field_goal': -3,
            'opponent_safety': -2,
            'no_score': 0
        }
        
        expected_points = 0
        for outcome, probability in probabilities.items():
            if outcome in point_values:
                expected_points += probability * point_values[outcome]
        
        return expected_points
    
    def evaluate_model_performance(self, model_data: pd.DataFrame, results: dict) -> dict:
        """
        Comprehensive model performance evaluation.
        
        Args:
            model_data: Training data used
            results: Training results from train_model
            
        Returns:
            Dictionary with detailed performance metrics
        """
        performance = {}
        
        # Basic statistics
        performance['total_plays'] = len(model_data)
        performance['unique_games'] = model_data['game_id'].nunique() if 'game_id' in model_data.columns else 'Unknown'
        
        if 'season' in model_data.columns:
            performance['seasons'] = sorted(model_data['season'].unique())
            performance['seasons_count'] = len(performance['seasons'])
        
        # Outcome distribution
        outcome_dist = model_data['next_play'].value_counts(normalize=True).to_dict()
        performance['outcome_distribution'] = outcome_dist
        
        # Feature statistics
        feature_cols = ['down', 'distance', 'yards_to_end_zone', 'half_seconds_remaining']
        feature_stats = {}
        for col in feature_cols:
            if col in model_data.columns:
                feature_stats[col] = {
                    'mean': model_data[col].mean(),
                    'std': model_data[col].std(),
                    'min': model_data[col].min(),
                    'max': model_data[col].max()
                }
        performance['feature_statistics'] = feature_stats
        
        # Model performance
        if 'accuracy' in results:
            performance['accuracy'] = results['accuracy']
        if 'feature_importance' in results:
            performance['feature_importance'] = results['feature_importance']
        
        return performance
    
    def create_comprehensive_lookup_table(self, include_time_scenarios: bool = True) -> pd.DataFrame:
        """
        Create a comprehensive lookup table for all possible game situations.
        
        Args:
            include_time_scenarios: Whether to include time-based scenarios
            
        Returns:
            DataFrame with expected points for all combinations
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info("Creating comprehensive expected points lookup table")
        
        # Define ranges for each variable
        downs = [1, 2, 3, 4]
        distances = list(range(1, 26))  # 1 to 25 yards (we cap distance at 25 in cleaning)
        yards_to_goal = list(range(1, 100))  # 1 to 99 yards from goal line
        
        # Define time scenarios
        if include_time_scenarios:
            # First 1500 seconds (25 minutes) of each half treated the same
            # Last 300 seconds (5 minutes) broken down by minute
            time_scenarios = [
                1500,  # Early/middle of half (25+ minutes remaining)
                300,   # 5:00 remaining
                240,   # 4:00 remaining  
                180,   # 3:00 remaining
                120,   # 2:00 remaining
                60     # 1:00 remaining
            ]
        else:
            time_scenarios = [900]  # Default: 15 minutes
        
        # Calculate total combinations
        total_combinations = len(downs) * len(distances) * len(yards_to_goal) * len(time_scenarios)
        logger.info(f"Generating {total_combinations:,} scenario predictions")
        
        # Create all combinations
        scenarios = []
        
        # Use nested loops with progress bar
        with tqdm(total=total_combinations, desc="Generating lookup table", ncols=80) as pbar:
            for time_remaining in time_scenarios:
                for down in downs:
                    for distance in distances:
                        for ytg in yards_to_goal:
                            # Skip impossible situations
                            if distance > ytg and ytg < 10:
                                # If we're close to goal line, distance to first down 
                                # can't be more than yards to goal
                                pbar.update(1)
                                continue
                            
                            # Get prediction for this scenario
                            try:
                                probs = self.predict_expected_points(
                                    down=down,
                                    distance=distance,
                                    yards_to_end_zone=ytg,
                                    half_seconds_remaining=time_remaining
                                )
                                
                                ep_value = self.calculate_expected_points(probs)
                                most_likely = max(probs.items(), key=lambda x: x[1])
                                
                                # Create time label for readability
                                if time_remaining >= 1500:
                                    time_label = "Early/Mid Half (25+ min)"
                                elif time_remaining == 300:
                                    time_label = "5:00 remaining"
                                elif time_remaining == 240:
                                    time_label = "4:00 remaining"
                                elif time_remaining == 180:
                                    time_label = "3:00 remaining"
                                elif time_remaining == 120:
                                    time_label = "2:00 remaining"
                                elif time_remaining == 60:
                                    time_label = "1:00 remaining"
                                else:
                                    time_label = f"{time_remaining}s remaining"
                                
                                scenario = {
                                    'half_seconds_remaining': time_remaining,
                                    'time_label': time_label,
                                    'down': down,
                                    'distance': distance,
                                    'yards_to_goal': ytg,
                                    'expected_points': round(ep_value, 4),
                                    'most_likely_outcome': most_likely[0],
                                    'most_likely_prob': round(most_likely[1], 4),
                                    'td_prob': round(probs.get('touchdown', 0), 4),
                                    'fg_prob': round(probs.get('field_goal', 0), 4),
                                    'safety_prob': round(probs.get('safety', 0), 4),
                                    'opp_td_prob': round(probs.get('opponent_touchdown', 0), 4),
                                    'opp_fg_prob': round(probs.get('opponent_field_goal', 0), 4),
                                    'opp_safety_prob': round(probs.get('opponent_safety', 0), 4),
                                    'no_score_prob': round(probs.get('no_score', 0), 4)
                                }
                                
                                scenarios.append(scenario)
                                
                            except Exception as e:
                                logger.warning(f"Error predicting for {down}&{distance} at {ytg} with {time_remaining}s: {e}")
                            
                            pbar.update(1)
        
        lookup_df = pd.DataFrame(scenarios)
        logger.info(f"Created lookup table with {len(lookup_df):,} scenarios")
        
        return lookup_df
    
    def analyze_lookup_table(self, lookup_df: pd.DataFrame) -> dict:
        """
        Analyze the comprehensive lookup table for insights.
        
        Args:
            lookup_df: The comprehensive lookup table
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_scenarios'] = len(lookup_df)
        analysis['ep_stats'] = {
            'min': lookup_df['expected_points'].min(),
            'max': lookup_df['expected_points'].max(),
            'mean': lookup_df['expected_points'].mean(),
            'median': lookup_df['expected_points'].median(),
            'std': lookup_df['expected_points'].std()
        }
        
        # Best and worst scenarios
        best_scenario = lookup_df.loc[lookup_df['expected_points'].idxmax()]
        worst_scenario = lookup_df.loc[lookup_df['expected_points'].idxmin()]
        
        analysis['best_scenario'] = {
            'down': int(best_scenario['down']),
            'distance': int(best_scenario['distance']),
            'yards_to_goal': int(best_scenario['yards_to_goal']),
            'expected_points': best_scenario['expected_points']
        }
        
        analysis['worst_scenario'] = {
            'down': int(worst_scenario['down']),
            'distance': int(worst_scenario['distance']),
            'yards_to_goal': int(worst_scenario['yards_to_goal']),
            'expected_points': worst_scenario['expected_points']
        }
        
        # Expected points by field position ranges
        field_position_ranges = [
            ('Red Zone (1-20)', (1, 20)),
            ('Scoring Territory (21-40)', (21, 40)),
            ('Midfield (41-60)', (41, 60)),
            ('Own Territory (61-80)', (61, 80)),
            ('Deep Own Territory (81-99)', (81, 99))
        ]
        
        analysis['ep_by_field_position'] = {}
        for label, (start, end) in field_position_ranges:
            subset = lookup_df[
                (lookup_df['yards_to_goal'] >= start) & 
                (lookup_df['yards_to_goal'] <= end)
            ]
            if len(subset) > 0:
                analysis['ep_by_field_position'][label] = {
                    'mean_ep': subset['expected_points'].mean(),
                    'scenarios': len(subset)
                }
        
        # Expected points by down
        analysis['ep_by_down'] = {}
        for down in [1, 2, 3, 4]:
            subset = lookup_df[lookup_df['down'] == down]
            analysis['ep_by_down'][f'{down}'] = {
                'mean_ep': subset['expected_points'].mean(),
                'scenarios': len(subset)
            }
        
        # Most common outcomes by field position
        analysis['common_outcomes_by_position'] = {}
        for label, (start, end) in field_position_ranges:
            subset = lookup_df[
                (lookup_df['yards_to_goal'] >= start) & 
                (lookup_df['yards_to_goal'] <= end)
            ]
            if len(subset) > 0:
                most_common = subset['most_likely_outcome'].value_counts().head(3)
                analysis['common_outcomes_by_position'][label] = most_common.to_dict()
        
        return analysis
    
    def test_prediction_scenarios(self) -> pd.DataFrame:
        """
        Test the model on various game scenarios.
        
        Returns:
            DataFrame with scenario predictions
        """
        examples = [
            {'down': 1, 'distance': 10, 'yardline_100': 1, 'scenario': '1st & 10 at goal line'},
            {'down': 1, 'distance': 10, 'yardline_100': 25, 'scenario': '1st & 10 in red zone'},
            {'down': 1, 'distance': 10, 'yardline_100': 50, 'scenario': '1st & 10 at midfield'},
            {'down': 1, 'distance': 10, 'yardline_100': 75, 'scenario': '1st & 10 in own territory'},   
            {'down': 1, 'distance': 10, 'yardline_100': 89, 'scenario': '1st & 10 near own goal'},
            {'down': 2, 'distance': 10, 'yardline_100': 25, 'scenario': '2nd & 10 in red zone'},
            {'down': 3, 'distance': 10, 'yardline_100': 25, 'scenario': '3rd & 10 in red zone'},
            {'down': 4, 'distance': 10, 'yardline_100': 25, 'scenario': '4th & 10 in red zone'},
            {'down': 2, 'distance': 5, 'yardline_100': 50, 'scenario': '2nd & 5 at midfield'},
            {'down': 2, 'distance': 10, 'yardline_100': 50, 'scenario': '2nd & 10 at midfield'},
            {'down': 2, 'distance': 15, 'yardline_100': 50, 'scenario': '2nd & 15 at midfield'},
            {'down': 2, 'distance': 20, 'yardline_100': 50, 'scenario': '2nd & 20 at midfield'}
        ]
        
        results = []
        
        for example in examples:
            # Get probabilities
            probs = self.predict_expected_points(
                down=example['down'],
                distance=example['distance'],
                yards_to_end_zone=example['yardline_100'],
                half_seconds_remaining=900  # 15 minutes
            )
            
            # Calculate expected points
            ep_value = self.calculate_expected_points(probs)
            
            # Get most likely outcome
            most_likely = max(probs.items(), key=lambda x: x[1])
            
            result = {
                'scenario': example['scenario'],
                'down': example['down'],
                'distance': example['distance'],
                'yardline_100': example['yardline_100'],
                'expected_points': round(ep_value, 3),
                'most_likely_outcome': most_likely[0],
                'most_likely_prob': round(most_likely[1], 3),
                'td_prob': round(probs.get('touchdown', 0), 3),
                'fg_prob': round(probs.get('field_goal', 0), 3),
                'no_score_prob': round(probs.get('no_score', 0), 3)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
        """
        Test the model on various game scenarios.
        
        Returns:
            DataFrame with scenario predictions
        """
        examples = [
            {'down': 1, 'distance': 10, 'yardline_100': 1, 'scenario': '1st & 10 at goal line'},
            {'down': 1, 'distance': 10, 'yardline_100': 25, 'scenario': '1st & 10 in red zone'},
            {'down': 1, 'distance': 10, 'yardline_100': 50, 'scenario': '1st & 10 at midfield'},
            {'down': 1, 'distance': 10, 'yardline_100': 75, 'scenario': '1st & 10 in own territory'},   
            {'down': 1, 'distance': 10, 'yardline_100': 89, 'scenario': '1st & 10 near own goal'},
            {'down': 2, 'distance': 10, 'yardline_100': 25, 'scenario': '2nd & 10 in red zone'},
            {'down': 3, 'distance': 10, 'yardline_100': 25, 'scenario': '3rd & 10 in red zone'},
            {'down': 4, 'distance': 10, 'yardline_100': 25, 'scenario': '4th & 10 in red zone'},
            {'down': 2, 'distance': 5, 'yardline_100': 50, 'scenario': '2nd & 5 at midfield'},
            {'down': 2, 'distance': 10, 'yardline_100': 50, 'scenario': '2nd & 10 at midfield'},
            {'down': 2, 'distance': 15, 'yardline_100': 50, 'scenario': '2nd & 15 at midfield'},
            {'down': 2, 'distance': 20, 'yardline_100': 50, 'scenario': '2nd & 20 at midfield'}
        ]
        
        results = []
        
        for example in examples:
            # Get probabilities
            probs = self.predict_expected_points(
                down=example['down'],
                distance=example['distance'],
                yards_to_end_zone=example['yardline_100'],
                half_seconds_remaining=900  # 15 minutes
            )
            
            # Calculate expected points
            ep_value = self.calculate_expected_points(probs)
            
            # Get most likely outcome
            most_likely = max(probs.items(), key=lambda x: x[1])
            
            result = {
                'scenario': example['scenario'],
                'down': example['down'],
                'distance': example['distance'],
                'yardline_100': example['yardline_100'],
                'expected_points': round(ep_value, 3),
                'most_likely_outcome': most_likely[0],
                'most_likely_prob': round(most_likely[1], 3),
                'td_prob': round(probs.get('touchdown', 0), 3),
                'fg_prob': round(probs.get('field_goal', 0), 3),
                'no_score_prob': round(probs.get('no_score', 0), 3)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)


def main():
    """Main function to run the Expected Points model pipeline."""
    # Updated file paths for your directory structure
    pbp_dir = '../../../etl/collect/collect_espn_pbp/temp/'
    games_dir = '../../../etl/collect/collect_espn_games/temp/'
    
    # Check if directories exist
    if not Path(pbp_dir).exists() or not Path(games_dir).exists():
        logger.error("Input directories not found. Please check directory paths.")
        return
    
    try:
        # Initialize model
        ep_model = ExpectedPointsModel()
        
        # Load and process data from multiple seasons
        # You can specify specific seasons like: seasons=['2020', '2021', '2022']
        # Or leave as None to include all available seasons
        pbp, games = ep_model.import_data(pbp_dir, games_dir, seasons=None)
        
        pbp_clean = ep_model.clean_pbp_data(pbp)
        model_data, key_df = ep_model.create_scoring_outcomes(pbp_clean, games)
        
        # Store key for later use
        ep_model.key_df = key_df
        
        # Train model with validation
        results = ep_model.train_model(model_data, validate=True)
        
        # Comprehensive performance evaluation
        performance = ep_model.evaluate_model_performance(model_data, results)
        
        # Print detailed results
        print("="*60)
        print("EXPECTED POINTS MODEL - PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\nDATASET OVERVIEW:")
        print(f"  Total plays: {performance['total_plays']:,}")
        print(f"  Unique games: {performance['unique_games']:,}")
        if 'seasons' in performance:
            print(f"  Seasons: {performance['seasons'][0]}-{performance['seasons'][-1]} ({performance['seasons_count']} seasons)")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        
        print(f"\nFEATURE IMPORTANCE:")
        for feature, importance in results['feature_importance'].items():
            print(f"  {feature}: {importance:.3f}")
        
        print(f"\nOUTCOME DISTRIBUTION:")
        for outcome, pct in sorted(performance['outcome_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {outcome}: {pct:.1%}")
        
        print(f"\nFEATURE STATISTICS:")
        for feature, stats in performance['feature_statistics'].items():
            print(f"  {feature}:")
            print(f"    Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
            print(f"    Range: {stats['min']:.0f} - {stats['max']:.0f}")
        
        # Create comprehensive lookup table
        print("\n" + "="*60)
        print("CREATING COMPREHENSIVE LOOKUP TABLE WITH TIME SCENARIOS")
        print("="*60)
        
        lookup_table = ep_model.create_comprehensive_lookup_table(include_time_scenarios=True)
        
        # Analyze the lookup table
        lookup_analysis = ep_model.analyze_lookup_table(lookup_table)
        
        print(f"\nLOOKUP TABLE SUMMARY:")
        print(f"  Total scenarios: {lookup_analysis['total_scenarios']:,}")
        if 'time_scenarios' in lookup_analysis:
            print(f"  Time scenarios: {lookup_analysis['time_scenarios_count']}")
            print(f"  Time periods: {', '.join(lookup_analysis['time_scenarios'])}")
        print(f"  Expected Points Range: {lookup_analysis['ep_stats']['min']:.3f} to {lookup_analysis['ep_stats']['max']:.3f}")
        print(f"  Mean EP: {lookup_analysis['ep_stats']['mean']:.3f}")
        print(f"  Median EP: {lookup_analysis['ep_stats']['median']:.3f}")
        
        print(f"\nBEST SCENARIO:")
        best = lookup_analysis['best_scenario']
        print(f"  {best['down']}&{best['distance']} at {best['yards_to_goal']}-yard line")
        if 'time_label' in best:
            print(f"  Time: {best['time_label']}")
        print(f"  Expected Points: {best['expected_points']:.3f}")
        
        print(f"\nWORST SCENARIO:")
        worst = lookup_analysis['worst_scenario']
        print(f"  {worst['down']}&{worst['distance']} at {worst['yards_to_goal']}-yard line")
        if 'time_label' in worst:
            print(f"  Time: {worst['time_label']}")
        print(f"  Expected Points: {worst['expected_points']:.3f}")
        
        # Show time impact if available
        if 'time_impact' in lookup_analysis and 'by_time' in lookup_analysis['time_impact']:
            print(f"\nTIME IMPACT ANALYSIS ({lookup_analysis['time_impact']['sample_situation']}):")
            for time_label, data in lookup_analysis['time_impact']['by_time'].items():
                print(f"  {time_label}: {data['expected_points']:.3f} EP")
        
        print(f"\nEXPECTED POINTS BY FIELD POSITION (Early/Mid Half):")
        for position, stats in lookup_analysis['ep_by_field_position'].items():
            print(f"  {position}: {stats['mean_ep']:.3f} EP ({stats['scenarios']:,} scenarios)")
        
        print(f"\nEXPECTED POINTS BY DOWN (Early/Mid Half):")
        for down, stats in lookup_analysis['ep_by_down'].items():
            print(f"  {down} down: {stats['mean_ep']:.3f} EP ({stats['scenarios']:,} scenarios)")
        
        # Test prediction scenarios (original examples)
        print("\n" + "="*60)
        print("EXAMPLE SCENARIO PREDICTIONS")
        print("="*60)
        
        scenario_results = ep_model.test_prediction_scenarios()
        
        print(f"\n{'Scenario':<25} {'EP':<6} {'Most Likely':<20} {'Prob':<6} {'TD%':<6} {'FG%':<6} {'No Score%':<9}")
        print("-" * 85)
        
        for _, row in scenario_results.iterrows():
            print(f"{row['scenario']:<25} {row['expected_points']:<6.2f} "
                  f"{row['most_likely_outcome']:<20} {row['most_likely_prob']:<6.1%} "
                  f"{row['td_prob']:<6.1%} {row['fg_prob']:<6.1%} {row['no_score_prob']:<9.1%}")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        
        # Find highest and lowest EP scenarios
        max_ep_row = scenario_results.loc[scenario_results['expected_points'].idxmax()]
        min_ep_row = scenario_results.loc[scenario_results['expected_points'].idxmin()]
        
        print(f"  Highest EP: {max_ep_row['scenario']} ({max_ep_row['expected_points']:.2f} points)")
        print(f"  Lowest EP:  {min_ep_row['scenario']} ({min_ep_row['expected_points']:.2f} points)")
        
        # Down comparison at same field position
        red_zone_scenarios = scenario_results[scenario_results['yardline_100'] == 25]
        if len(red_zone_scenarios) > 1:
            print(f"  Red zone EP by down:")
            for _, row in red_zone_scenarios.iterrows():
                if 'red zone' in row['scenario']:
                    print(f"    {row['down']}th down: {row['expected_points']:.2f} points")
        
        # Distance impact at midfield
        midfield_scenarios = scenario_results[
            (scenario_results['yardline_100'] == 50) & (scenario_results['down'] == 2)
        ]
        if len(midfield_scenarios) > 1:
            print(f"  Distance impact (2nd down, midfield):")
            for _, row in midfield_scenarios.iterrows():
                print(f"    {row['distance']} yards: {row['expected_points']:.2f} points")
        
        # Save model and all outputs
        ep_model.save_model()
        
        # Save scenario results
        scenario_results.to_csv('scenario_predictions.csv', index=False)
        
        # Save comprehensive lookup table
        lookup_table.to_csv('expected_points_lookup_table.csv', index=False)
        
        print(f"\nFILES SAVED:")
        print(f"  Model: 'ep_model.pkl'")
        print(f"  Scoring key: 'scoring_key.csv'")
        print(f"  Example scenarios: 'scenario_predictions.csv'")
        print(f"  Complete lookup table: 'expected_points_lookup_table.csv' ({len(lookup_table):,} scenarios)")
        
        logger.info("Expected Points model pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise


if __name__ == '__main__':
    main()