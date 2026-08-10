import pandas as pd
import logging

def safe_get(data, *keys, default=None):
    """Safely navigate nested dictionaries and lists."""
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

def transform_espn_ncaaf_week_data(json_data):
    """
    Transform ESPN NCAAF calendar data from JSON to DataFrame.

    Args:
        json_data (dict): Data from ESPN's NCAAF calendar API

    Returns:
        pd.DataFrame: Calendar data in relational table form, or None if no valid data
    """
    if json_data is None:
        logging.warning("No JSON data provided")
        return None
        
    if not isinstance(json_data, dict):
        logging.error(f"Expected dict, got {type(json_data)}")
        return None

    # Initialize data containers
    calendar_data = {
        'week': [],
        'week_start_date': []
    }

    weeks_processed = 0

    try:
        entries = safe_get(json_data, 'leagues', 0, 'calendar', 0, 'entries', default=[])
        
        if not entries:
            logging.info("No calendar entries found in JSON data")
            return pd.DataFrame(calendar_data)

        for entry in entries:
            try:
                week_label = safe_get(entry, 'label')
                start_date = safe_get(entry, 'startDate')
                
                if week_label and start_date:
                    # Extract just the date part (remove time)
                    start_date = start_date.split('T')[0]
                    
                    calendar_data['week'].append(week_label)
                    calendar_data['week_start_date'].append(start_date)
                    weeks_processed += 1

            except Exception as e:
                logging.warning(f"Error processing calendar entry: {e}")
                continue

    except Exception as e:
        logging.error(f"Error processing calendar data: {e}")
        return None

    logging.info(f"Processed {weeks_processed} calendar weeks")

    # Create DataFrame
    try:
        df = pd.DataFrame(calendar_data)
        logging.info(f"Successfully created DataFrame with {len(df)} rows")
        return df

    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        return None

if __name__ == '__main__':
    # Test with sample data
    sample_data = {
        'leagues': [
            {
                'calendar': [
                    {
                        'entries': [
                            {
                                'label': 'Week 1',
                                'startDate': '2024-08-24T00:00:00Z'
                            },
                            {
                                'label': 'Week 2', 
                                'startDate': '2024-08-31T00:00:00Z'
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    df = transform_espn_ncaaf_week_data(sample_data)
    if df is not None:
        print(f"Sample transformation successful: {len(df)} weeks processed")
        print(df.head())
    else:
        print("Sample transformation failed")