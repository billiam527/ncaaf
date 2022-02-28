from datetime import datetime, timedelta


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

    date_list = [i.replace('-', '') for i in date_list]

    return date_list


if __name__ == "__main__":
    None
