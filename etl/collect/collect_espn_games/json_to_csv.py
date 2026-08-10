import pandas as pd
import logging

def safe_get(data, *keys, default=999):
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
            if isinstance(data, (dict, list)) and key in data:
                data = data[key]
            elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
                data = data[key]
            else:
                return default
        return data
    except (KeyError, IndexError, TypeError):
        return default

def calculate_overtime_points(total_score, q1, q2, q3, q4):
    """Calculate overtime points safely"""
    try:
        regulation_total = sum([q1, q2, q3, q4])
        return max(0, int(total_score) - regulation_total)
    except (ValueError, TypeError):
        return 999

def transform_espn_ncaaf_game_data(json_data):
    """
    Transform ESPN NCAAF game data from JSON to DataFrame.

    Args:
        json_data (dict): Data pulled from ESPN's NCAAF API at the game level

    Returns:
        pd.DataFrame: Data in relational table form, or None if no valid data
    """
    if json_data is None:
        logging.warning("No JSON data provided")
        return None
        
    if not isinstance(json_data, dict):
        logging.error(f"Expected dict, got {type(json_data)}")
        return None
        
    events = json_data.get('events', [])
    if not events:
        logging.info("No events found in JSON data")
        return None

    # Initialize data containers
    game_data = {
        'id': [], 'date': [], 'name': [], 'short_name': [], 'season': [], 'status': [],
        'venue_id': [], 'neutral_site': [], 'home_team_id': [], 'away_team_id': [],
        'home_score': [], 'away_score': [],
        'home_first_quarter': [], 'home_second_quarter': [], 'home_third_quarter': [], 
        'home_fourth_quarter': [], 'home_ot': [],
        'away_first_quarter': [], 'away_second_quarter': [], 'away_third_quarter': [], 
        'away_fourth_quarter': [], 'away_ot': []
    }

    games_processed = 0
    games_skipped = 0

    for game in events:
        try:
            # Skip games without required ID
            game_id = safe_get(game, 'id', default=None)
            if not game_id:
                games_skipped += 1
                continue

            # Basic game information
            game_data['id'].append(game_id)
            game_data['date'].append(safe_get(game, 'date', default=''))
            game_data['name'].append(safe_get(game, 'name', default=''))
            game_data['short_name'].append(safe_get(game, 'shortName', default=''))
            game_data['season'].append(safe_get(game, 'season', 'year', default=0))
            game_data['status'].append(safe_get(game, 'status', 'type', 'name', default=''))

            # Competition data
            competition = safe_get(game, 'competitions', 0, default={})
            game_data['venue_id'].append(safe_get(competition, 'venue', 'id', default=999))
            game_data['neutral_site'].append(safe_get(competition, 'neutralSite', default=False))

            # Team information
            competitors = safe_get(competition, 'competitors', default=[])
            if len(competitors) >= 2:
                game_data['home_team_id'].append(safe_get(competitors, 0, 'id', default=''))
                game_data['away_team_id'].append(safe_get(competitors, 1, 'id', default=''))
                game_data['home_score'].append(safe_get(competitors, 0, 'score', default=0))
                game_data['away_score'].append(safe_get(competitors, 1, 'score', default=0))

                # Quarter scores for home team
                home_linescores = safe_get(competitors, 0, 'linescores', default=[])
                home_q1 = safe_get(home_linescores, 0, 'value', default=999)
                home_q2 = safe_get(home_linescores, 1, 'value', default=999)
                home_q3 = safe_get(home_linescores, 2, 'value', default=999)
                home_q4 = safe_get(home_linescores, 3, 'value', default=999)

                game_data['home_first_quarter'].append(home_q1)
                game_data['home_second_quarter'].append(home_q2)
                game_data['home_third_quarter'].append(home_q3)
                game_data['home_fourth_quarter'].append(home_q4)

                # Calculate home OT points
                home_total = safe_get(competitors, 0, 'score', default=0)
                home_ot = calculate_overtime_points(home_total, home_q1, home_q2, home_q3, home_q4)
                game_data['home_ot'].append(home_ot)

                # Quarter scores for away team
                away_linescores = safe_get(competitors, 1, 'linescores', default=[])
                away_q1 = safe_get(away_linescores, 0, 'value', default=999)
                away_q2 = safe_get(away_linescores, 1, 'value', default=999)
                away_q3 = safe_get(away_linescores, 2, 'value', default=999)
                away_q4 = safe_get(away_linescores, 3, 'value', default=999)

                game_data['away_first_quarter'].append(away_q1)
                game_data['away_second_quarter'].append(away_q2)
                game_data['away_third_quarter'].append(away_q3)
                game_data['away_fourth_quarter'].append(away_q4)

                # Calculate away OT points
                away_total = safe_get(competitors, 1, 'score', default=0)
                away_ot = calculate_overtime_points(away_total, away_q1, away_q2, away_q3, away_q4)
                game_data['away_ot'].append(away_ot)

            else:
                # Handle missing competitor data
                for key in ['home_team_id', 'away_team_id', 'home_score', 'away_score',
                           'home_first_quarter', 'home_second_quarter', 'home_third_quarter',
                           'home_fourth_quarter', 'home_ot', 'away_first_quarter',
                           'away_second_quarter', 'away_third_quarter', 'away_fourth_quarter',
                           'away_ot']:
                    default_val = '' if 'team_id' in key else 999
                    game_data[key].append(default_val)

            games_processed += 1

        except Exception as e:
            logging.warning(f"Error processing game {safe_get(game, 'shortName', default='Unknown')}: {e}")
            games_skipped += 1
            continue

    if games_processed == 0:
        logging.warning("No games successfully processed")
        return None

    logging.info(f"Processed {games_processed} games, skipped {games_skipped}")

    # Create DataFrame
    try:
        df = pd.DataFrame(game_data)

        # Convert numeric columns
        numeric_columns = [
            'season', 'home_score', 'away_score', 'home_first_quarter', 'home_second_quarter',
            'home_third_quarter', 'home_fourth_quarter', 'home_ot', 'away_first_quarter',
            'away_second_quarter', 'away_third_quarter', 'away_fourth_quarter', 'away_ot'
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(999).astype('int32')

        # Convert boolean columns
        df['neutral_site'] = df['neutral_site'].astype(bool)

        logging.info(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        return None

if __name__ == '__main__':
    # Example usage
    sample_data = {
        'events': [
            {
                'id': '401520282',
                'date': '2023-09-02T17:00Z',
                'name': 'Example vs Test',
                'shortName': 'EX vs TST',
                'season': {'year': 2023},
                'status': {'type': {'name': 'Final'}},
                'competitions': [
                    {
                        'venue': {'id': '3794'},
                        'neutralSite': False,
                        'competitors': [
                            {
                                'id': '52',
                                'score': '31',
                                'linescores': [
                                    {'value': 7}, {'value': 14}, {'value': 3}, {'value': 7}
                                ]
                            },
                            {
                                'id': '103',
                                'score': '17',
                                'linescores': [
                                    {'value': 0}, {'value': 10}, {'value': 0}, {'value': 7}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    df = transform_espn_ncaaf_game_data(sample_data)
    if df is not None:
        print(f"Sample transformation successful: {len(df)} games processed")
        print(df.head())
    else:
        print("Sample transformation failed")