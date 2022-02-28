from datetime import datetime, timedelta
import os


def main():
    
    """
    Function to generate list of strings and save to bucket.
    """

    end_date = datetime.today()
    start_date = '2019-08-01'
    dates = date_list_generation(start_date, end_date.strftime('%Y-%m-%d'))
    file_name = 'dates_'
    if os.path.exists('temp'):
        with open('temp/' + file_name + str(start_date) + '_to_' + end_date.strftime('%Y-%m-%d'), 'w') as file:
            for item in dates:
                file.write('%s\n' % item)
    else:
        os.mkdir('temp')
        with open('temp/' + file_name + str(start_date) + '_to_' + end_date.strftime('%Y-%m-%d'), 'w') as file:
            for item in dates:
                file.write('%s\n' % item)


def date_list_generation(start_date: str,
                         end_date: str) -> list:

    """
    Generate a list of dates.
    
    param start_date: (string) starting date in yyyy-mm-dd format
    param end_date: (string) ending date in yyyy-mm-dd format
    
    return date_list: (list) inclusive list of strings between the start and end date
    """
    
    assert isinstance(start_date, str), 'start_date must be a string'
    assert isinstance(end_date, str), 'end_date must be a string'
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')  # start date
    end_date = datetime.strptime(end_date, '%Y-%m-%d')  # end date
    
    delta = end_date - start_date  # as timedelta
    
    date_list = []
    for i in range(delta.days + 1):  # loop through range of timedelta + 1
        day = datetime.strftime(start_date + timedelta(days=i), '%Y%m%d')
        date_list.append(day)
    
    return date_list


if __name__ == "__main__":
    main()
