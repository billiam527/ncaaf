from datetime import datetime, timedelta
import os
import logging
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

def main():
    """
    Function to generate list of date strings and save to file.
    Note: This standalone function uses hardcoded dates and is mainly for testing.
    """
    try:
        end_date = datetime.today()
        start_date = '2019-08-01'
        dates = date_list_generation(start_date, end_date.strftime('%Y-%m-%d'))
        
        os.makedirs('temp', exist_ok=True)
        filename = f'temp/dates_{start_date}_to_{end_date.strftime("%Y-%m-%d")}_test'
        
        with open(filename, 'w') as file:
            for item in dates:
                file.write(f'{item}\n')
                
        print(f"Generated {len(dates)} dates and saved to {filename}")
        
    except Exception as e:
        print(f"Error in date generation: {e}")

if __name__ == "__main__":
    main()