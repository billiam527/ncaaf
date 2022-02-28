

def year_list_generation(start_date: str,
                         end_date: str) -> list:

    """
    Generate a list of years.

    param start_date: (string) starting date in yyyy-mm-dd format
    param end_date: (string) ending date in yyyy-mm-dd format

    return date_list: (list) inclusive list of strings between the start and end date
    """

    assert isinstance(start_date, str), 'start_date must be a string'
    assert isinstance(end_date, str), 'end_date must be a string'

    start_date = start_date.split('-')[0]
    end_date = end_date.split('-')[0]

    years_list = list(range(int(start_date), int(end_date)))
    years_list.append(int(end_date))
    years_list = [str(i) for i in years_list]

    return years_list


if __name__ == "__main__":
    None
