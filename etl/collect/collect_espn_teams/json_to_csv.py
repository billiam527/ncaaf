import pandas as pd
import logging

def safe_get(data, *keys, default=None):
    """
    Safely navigate nested dictionaries and lists.
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

def transform_espn_ncaaf_team_data(json_data):
    """
    Transform ESPN NCAAF team data from JSON to DataFrame.

    Args:
        json_data (dict): Data from ESPN's NCAAF teams API

    Returns:
        pd.DataFrame: Team data in relational table form, or None if no valid data
    """
    if json_data is None:
        logging.warning("No JSON data provided")
        return None
        
    if not isinstance(json_data, dict):
        logging.error(f"Expected dict, got {type(json_data)}")
        return None

    # Initialize data containers
    team_data = {
        'id': [], 'slug': [], 'location': [], 'name': [], 'nickname': [],
        'abbreviation': [], 'display_name': [], 'short_display_name': []
    }

    teams_processed = 0
    teams_skipped = 0

    try:
        sports = safe_get(json_data, 'sports', default=[])
        if not sports:
            logging.warning("No sports found in JSON data")
            return None

        for sport in sports:
            leagues = safe_get(sport, 'leagues', default=[])
            for league in leagues:
                teams = safe_get(league, 'teams', default=[])
                
                for team_entry in teams:
                    try:
                        team = safe_get(team_entry, 'team', default={})
                        if not team:
                            teams_skipped += 1
                            continue

                        # Required fields
                        team_id = safe_get(team, 'id')
                        if not team_id:
                            teams_skipped += 1
                            continue

                        # Extract team data with fallbacks
                        team_data['id'].append(team_id)
                        team_data['slug'].append(safe_get(team, 'slug', default=''))
                        team_data['location'].append(safe_get(team, 'location', default=''))
                        
                        # Name with fallback to displayName
                        name = safe_get(team, 'name', default=None)
                        if name is None:
                            name = safe_get(team, 'displayName', default='')
                        team_data['name'].append(name)
                        
                        team_data['nickname'].append(safe_get(team, 'nickname', default=''))
                        team_data['abbreviation'].append(safe_get(team, 'abbreviation', default=''))
                        team_data['display_name'].append(safe_get(team, 'displayName', default=''))
                        team_data['short_display_name'].append(safe_get(team, 'shortDisplayName', default=''))

                        teams_processed += 1

                    except Exception as e:
                        logging.warning(f"Error processing team {safe_get(team, 'displayName', default='unknown')}: {e}")
                        teams_skipped += 1
                        continue

    except Exception as e:
        logging.error(f"Error processing team data: {e}")
        return None

    if teams_processed == 0:
        logging.warning("No teams successfully processed")
        return None

    logging.info(f"Processed {teams_processed} teams, skipped {teams_skipped}")

    # Create DataFrame
    try:
        df = pd.DataFrame(team_data)

        # Convert data types
        df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int32')  # Nullable integer
        
        # Remove any rows where ID conversion failed
        df = df.dropna(subset=['id'])

        logging.info(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        return None

if __name__ == '__main__':
    # Test with sample data
    sample_data = {
        'sports': [
            {
                'leagues': [
                    {
                        'teams': [
                            {
                                'team': {
                                    'id': '52',
                                    'slug': 'miami-fl',
                                    'location': 'Miami',
                                    'name': 'Hurricanes',
                                    'nickname': 'Hurricanes',
                                    'abbreviation': 'MIA',
                                    'displayName': 'Miami Hurricanes',
                                    'shortDisplayName': 'Miami'
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    df = transform_espn_ncaaf_team_data(sample_data)
    if df is not None:
        print(f"Sample transformation successful: {len(df)} teams processed")
        print(df.head())
    else:
        print("Sample transformation failed")