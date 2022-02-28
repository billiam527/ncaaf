from datetime import datetime
import os
import glob


def main():

    """
    Read the local date-list file and use the dates to create urls to scrape.
    """

    end_date = datetime.today()
    start_date = '2019-08-01'
    list_of_files = glob.glob('temp/dates*')
    latest_file = max(list_of_files, key=os.path.getctime)
    dates = [line.rstrip('\n') for line in open(latest_file)]
    prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
    suffix = '&limit=900'
    urls = create_urls(prefix, suffix, dates)
    file_name = 'date_urls_'
    if os.path.exists('temp'):
        with open('temp/' + file_name + str(start_date) + '_to_' + end_date.strftime('%Y-%m-%d'), 'w') as file:
            for item in urls:
                file.write('%s\n' % item)
    else:
        os.mkdir('temp')
        with open('temp/' + file_name + str(start_date) + '_to_' + end_date.strftime('%Y-%m-%d'), 'w') as file:
            for item in urls:
                file.write('%s\n' % item)


def create_urls(prefix, suffix, data):
    
    """
    Create a list of string URLs using a prefix, suffix, and any data that needs
    to be looped through.
    
    param prefix: (str) string for URLs
    param suffix: (str) string to come after looped data
    param data: (list) list of string to be looped through and added to prefix
        and suffix
        
    return URLs: (list) list of URL strings
    """
    
    assert isinstance(prefix, str), 'prefix must be a string'
    assert isinstance(suffix, str), 'suffix must be a string'
    assert isinstance(data, list), 'data must be a list'
    for string in data:
        assert isinstance(string, str), 'each value in data must be a string'
        
    urls = [prefix + j + suffix for j in data]
    
    return urls


if __name__ == "__main__":
    main()
