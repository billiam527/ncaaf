import pandas as pd


def transform_espn_ncaaf_week_data(json_data):

    """
    param json_data: (json) data pulled from espn's ncaaf api at the game level

    return df: (pd.DataFrame) data in the relational table form
    """

    # calendar data
    week = []
    week_start_date = []

    try:
        for i in json_data['leagues'][0]['calendar'][0]['entries']:
            week.append(i['label'])
            start_date = i['startDate'].split('T')[0]
            week_start_date.append(start_date)
    except:
        None

    columns = list(zip(week,
                       week_start_date))
    col_names = ['week', 'week_start_date']
    df = pd.DataFrame(columns, columns=col_names)

    return df


if __name__ == '__main__':
    None
