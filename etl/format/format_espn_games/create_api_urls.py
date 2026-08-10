from typing import List

def create_urls(prefix: str, suffix: str, data: List[str]) -> List[str]:
    """
    Create a list of string URLs using a prefix, suffix, and data to loop through.

    Args:
        prefix (str): URL prefix string
        suffix (str): URL suffix string to come after looped data
        data (List[str]): List of strings to be inserted between prefix and suffix

    Returns:
        List[str]: List of complete URL strings

    Raises:
        TypeError: If inputs are not the correct type
        ValueError: If any input is empty or invalid
    """
    if not isinstance(prefix, str):
        raise TypeError('prefix must be a string')
    if not isinstance(suffix, str):
        raise TypeError('suffix must be a string')
    if not isinstance(data, list):
        raise TypeError('data must be a list')
    
    if not data:
        raise ValueError('data list cannot be empty')
    
    for i, item in enumerate(data):
        if not isinstance(item, str):
            raise TypeError(f'data[{i}] must be a string, got {type(item)}')
        if not item.strip():
            raise ValueError(f'data[{i}] cannot be empty or whitespace')

    return [f"{prefix}{item}{suffix}" for item in data]

if __name__ == '__main__':
    # Test the function
    try:
        test_prefix = "https://api.example.com/data/"
        test_suffix = "/info"
        test_data = ["2020", "2021", "2022"]
        
        urls = create_urls(test_prefix, test_suffix, test_data)
        print(f"Generated URLs: {urls}")
    except Exception as e:
        print(f"Test failed: {e}")