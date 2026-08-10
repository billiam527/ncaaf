import pandas as pd
import logging

def safe_get(data, *keys, default=None):
    """
    Safely navigate nested dictionaries and lists.
    
    Args:
        data: The data structure to navigate
        *keys: Keys/indices to navigate through
        default: Default value if navigation fails
        
    Returns:
        The value at the specified path, or default if not found
    """
    try:
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
                data = data[key]
            else:
                return default
        return data
    except (KeyError, IndexError, TypeError):
        return default

def transform_espn_ncaaf_data(json_data):
    """
    Transform ESPN NCAAF play-by-play data from JSON to DataFrame.

    Args:
        json_data (dict): Data from ESPN's NCAAF API at the game level

    Returns:
        pd.DataFrame: Play-by-play data in relational table form, or None if no valid data
    """
    if json_data is None:
        logging.warning("No JSON data provided")
        return None
        
    if not isinstance(json_data, dict):
        logging.error(f"Expected dict, got {type(json_data)}")
        return None

    # Initialize data containers
    play_data = {
        'id': [], 'game_id': [], 'modified': [], 'home_score': [], 'away_score': [],
        'drive_id': [], 'team_id': [], 'play_type_id': [], 'play_type_text': [],
        'play_text': [], 'period': [], 'clock': [], 'scoring_play': [],
        'down': [], 'distance': [], 'yard_line': [], 'yards_to_end_zone': [], 'stat_yardage': []
    }

    plays_processed = 0
    plays_skipped = 0

    try:
        # Get game ID
        game_id = safe_get(json_data, 'header', 'competitions', 0, 'id')
        if not game_id:
            logging.warning("No game ID found in JSON data")
            return None

        # Get drives
        drives = safe_get(json_data, 'drives', 'previous', default=[])
        if not drives:
            logging.info("No drives found in JSON data")
            return None

        for drive in drives:
            drive_id = safe_get(drive, 'id', default='unknown')
            plays = safe_get(drive, 'plays', default=[])
            
            for play in plays:
                try:
                    # Required fields
                    play_id = safe_get(play, 'id')
                    if not play_id:
                        plays_skipped += 1
                        continue

                    # Basic play information
                    play_data['id'].append(play_id)
                    play_data['game_id'].append(game_id)
                    play_data['drive_id'].append(drive_id)
                    play_data['modified'].append(safe_get(play, 'modified', default=''))
                    
                    # Scores
                    play_data['home_score'].append(safe_get(play, 'homeScore', default=0))
                    play_data['away_score'].append(safe_get(play, 'awayScore', default=0))
                    
                    # Play type
                    play_data['play_type_id'].append(safe_get(play, 'type', 'id', default=999))
                    play_data['play_type_text'].append(safe_get(play, 'type', 'text', default='unknown'))
                    
                    # Play details
                    play_data['play_text'].append(safe_get(play, 'text', default=''))
                    play_data['period'].append(safe_get(play, 'period', 'number', default=0))
                    play_data['clock'].append(safe_get(play, 'clock', 'displayValue', default='00:00'))
                    play_data['scoring_play'].append(safe_get(play, 'scoringPlay', default=False))
                    play_data['stat_yardage'].append(safe_get(play, 'statYardage', default=0))
                    
                    # Start position information
                    play_data['down'].append(safe_get(play, 'start', 'down', default=0))
                    play_data['distance'].append(safe_get(play, 'start', 'distance', default=0))
                    play_data['yard_line'].append(safe_get(play, 'start', 'yardLine', default=0))
                    play_data['yards_to_end_zone'].append(safe_get(play, 'start', 'yardsToEndzone', default=0))
                    play_data['team_id'].append(safe_get(play, 'start', 'team', 'id', default='unknown'))

                    plays_processed += 1

                except Exception as e:
                    logging.warning(f"Error processing play {safe_get(play, 'id', default='unknown')}: {e}")
                    plays_skipped += 1
                    continue

    except Exception as e:
        logging.error(f"Error processing drives data: {e}")
        return None

    if plays_processed == 0:
        logging.warning("No plays successfully processed")
        return None

    logging.info(f"Processed {plays_processed} plays, skipped {plays_skipped}")

    # Create DataFrame
    try:
        df = pd.DataFrame(play_data)

        # Convert data types
        numeric_columns = ['home_score', 'away_score', 'play_type_id', 'period', 
                          'down', 'distance', 'yard_line', 'yards_to_end_zone', 'stat_yardage']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Convert boolean columns
        df['scoring_play'] = df['scoring_play'].astype(bool)

        logging.info(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        return None

if __name__ == '__main__':
    # Example usage with sample data structure
    sample_data = {
        'header': {
            'competitions': [{'id': '401520282'}]
        },
        'drives': {
            'previous': [
                {
                    'id': '4015202821',
                    'plays': [
                        {
                            'id': '40152028211',
                            'type': {'id': '67', 'text': 'Rush'},
                            'text': 'Rush for 5 yards',
                            'homeScore': 7,
                            'awayScore': 0,
                            'period': {'number': 1},
                            'clock': {'displayValue': '14:23'},
                            'scoringPlay': False,
                            'modified': '2023-09-02T17:30:00Z',
                            'start': {
                                'down': 1,
                                'distance': 10,
                                'yardLine': 25,
                                'yardsToEndzone': 75,
                                'team': {'id': '52'}
                            },
                            'statYardage': 5
                        }
                    ]
                }
            ]
        }
    }
    
    df = transform_espn_ncaaf_data(sample_data)
    if df is not None:
        print(f"Sample transformation successful: {len(df)} plays processed")
        print(df.head())
    else:
        print("Sample transformation failed")