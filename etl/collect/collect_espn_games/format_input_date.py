from datetime import datetime, timedelta
from typing import List

def date_list_generation(start_date: str, end_date: str) -> List[str]:
    """
    Generate a list of dates in YYYYMMDD format.

    Args:
        start_date (str): Starting date in YYYY-MM-DD format
        end_date (str): Ending date in YYYY-MM-DD format

    Returns:
        List[str]: Inclusive list of date strings in YYYYMMDD format

    Raises:
        TypeError: If dates are not strings
        ValueError: If date format is invalid or end_date is before start_date
    """
    if not isinstance(start_date, str):
        raise TypeError('start_date must be a string')
    if not isinstance(end_date, str):
        raise TypeError('end_date must be a string')

    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f'Invalid date format. Expected YYYY-MM-DD: {e}')

    if end_dt < start_dt:
        raise ValueError('end_date cannot be before start_date')

    delta = end_dt - start_dt
    date_list = []
    
    for i in range(delta.days + 1):
        current_date = start_dt + timedelta(days=i)
        formatted_date = current_date.strftime('%Y%m%d')
        date_list.append(formatted_date)

    return date_list


if __name__ == '__main__':
    # Test the functions
    try:
        # Test date generation
        dates = date_list_generation('2024-01-01', '2024-01-03')
        print(f"Generated dates: {dates}")
        
        # Test URL creation
        prefix = "https://api.example.com/data/"
        suffix = "/games"
        urls = create_urls(prefix, suffix, dates)
        print(f"Generated URLs: {urls[:2]}...")  # Show first 2
        
    except Exception as e:
        print(f"Test failed: {e}")