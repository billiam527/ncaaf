def create_urls(prefix, suffix, data):

    """
    Create a list of string URLs using a prefix, suffix, and any data that needs to be looped through.

    param prefix: (str) string for URLs
    param suffix: (str) string to come after looped data
    param data: (list) list of string to be looped through and added to prefix and suffix

    return URLs: (list) list of URL strings
    """

    assert isinstance(prefix, str), 'prefix must be a string'
    assert isinstance(suffix, str), 'suffix must be a string'
    assert isinstance(data, list), 'data must be a list'
    for string in data:
        assert isinstance(string, str), 'each value in data must be a string'

    urls = [prefix + j + suffix for j in data]

    return urls


if __name__ == '__main__':
    None
