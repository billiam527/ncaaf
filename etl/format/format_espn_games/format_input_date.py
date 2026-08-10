from typing import List

def year_list_generation(start_date: str, end_date: str) -> List[str]:
    """
    Generate a list of years from start_date to end_date.

    Args:
        start_date (str): Starting date in YYYY-MM-DD format
        end_date (str): Ending date in YYYY-MM-DD format

    Returns:
        List[str]: Inclusive list of year strings between start and end date

    Raises:
        TypeError: If dates are not strings
        ValueError: If date format is invalid
    """
    if not isinstance(start_date, str):
        raise TypeError('start_date must be a string')
    if not isinstance(end_date, str):
        raise TypeError('end_date must be a string')

    try:
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
    except (ValueError, IndexError) as e:
        raise ValueError(f'Invalid date format. Expected YYYY-MM-DD: {e}')

    if end_year < start_year:
        raise ValueError('end_date year cannot be before start_date year')

    years_list = list(range(start_year, end_year + 1))
    return [str(year) for year in years_list]

if __name__ == "__main__":
    # Test the function
    try:
        years = year_list_generation('2020-01-01', '2024-12-31')
        print(f"Generated years: {years}")
    except Exception as e:
        print(f"Test failed: {e}")