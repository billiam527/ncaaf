import pandas as pd


def transform_espn_ncaaf_game_data(json_data):

    """
    param json_data: (json) data pulled from espn's ncaaf api at the game level

    return df: (pd.DataFrame) data in the relational table form
    """

    # game data
    id = []
    date = []
    name = []
    short_name = []
    season = []
    status = []
    venue_id = []
    neutral_site = []

    # team info
    home_team_id = []
    away_team_id = []

    # scoring
    home_score = []
    away_score = []
    home_first_quarter = []
    home_second_quarter = []
    home_third_quarter = []
    home_fourth_quarter = []
    home_ot = []
    away_first_quarter = []
    away_second_quarter = []
    away_third_quarter = []
    away_fourth_quarter = []
    away_ot = []

    if json_data is not None:
        for game in json_data['events']:
            id.append(game['id'])
            date.append(game['date'])
            name.append(game['name'])
            short_name.append(game['shortName'])
            season.append(game['season']['year'])
            status.append(game['status']['type']['name'])
            competition = game['competitions'][0]
            try:
                venue_id.append(competition['venue']['id'])
            except KeyError as e:
                # print('Error:', e, 'venue_id', game['shortName'])
                venue_id.append(999)
            neutral_site.append(competition['neutralSite'])
            home_team_id.append(competition['competitors'][0]['id'])
            away_team_id.append(competition['competitors'][1]['id'])
            home_score.append(competition['competitors'][0]['score'])
            away_score.append(competition['competitors'][1]['score'])
            try:
                home_1q = competition['competitors'][0]['linescores'][0]['value']
                home_first_quarter.append(home_1q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'home_1q', game['shortName'])
                home_first_quarter.append(999)
            try:
                home_2q = competition['competitors'][0]['linescores'][1]['value']
                home_second_quarter.append(home_2q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'home_2q', game['shortName'])
                home_second_quarter.append(999)
            try:
                home_3q = competition['competitors'][0]['linescores'][2]['value']
                home_third_quarter.append(home_3q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'home_3q', game['shortName'])
                home_third_quarter.append(999)
            try:
                home_4q = competition['competitors'][0]['linescores'][3]['value']
                home_fourth_quarter.append(home_4q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'home_4q', game['shortName'])
                home_fourth_quarter.append(999)
            try:
                home_ot_calc = int(competition['competitors'][0]['score']) - (home_1q + home_2q + home_3q + home_4q)
                home_ot.append(home_ot_calc)
            except UnboundLocalError as e:
                # print('Error:', e, 'cannot distinguish OT point')
                home_ot.append(999)
            try:
                away_1q = competition['competitors'][1]['linescores'][0]['value']
                away_first_quarter.append(away_1q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'away_1q', game['shortName'])
                away_first_quarter.append(999)
            try:
                away_2q = competition['competitors'][1]['linescores'][1]['value']
                away_second_quarter.append(away_2q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'away_2q', game['shortName'])
                away_second_quarter.append(999)
            try:
                away_3q = competition['competitors'][1]['linescores'][2]['value']
                away_third_quarter.append(away_3q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'away_3q', game['shortName'])
                away_third_quarter.append(999)
            try:
                away_4q = competition['competitors'][1]['linescores'][3]['value']
                away_fourth_quarter.append(away_4q)
            except (KeyError, IndexError) as e:
                # print('Error:', e, 'away_4q', game['shortName'])
                away_fourth_quarter.append(999)
            try:
                away_ot_calc = int(competition['competitors'][1]['score']) - (away_1q + away_2q + away_3q + away_4q)
                away_ot.append(away_ot_calc)
            except UnboundLocalError as e:
                # print('Error:', e, 'cannot distinguish OT point')
                away_ot.append(999)

        columns = list(zip(id,
                           date,
                           name,
                           short_name,
                           season,
                           status,
                           venue_id,
                           neutral_site,
                           home_team_id,
                           away_team_id,
                           home_score,
                           away_score,
                           home_first_quarter,
                           home_second_quarter,
                           home_third_quarter,
                           home_fourth_quarter,
                           home_ot,
                           away_first_quarter,
                           away_second_quarter,
                           away_third_quarter,
                           away_fourth_quarter,
                           away_ot))

        col_names = ['id',
                     'date',
                     'name',
                     'short_name',
                     'season',
                     'status',
                     'venue_id',
                     'neutral_site',
                     'home_team_id',
                     'away_team_id',
                     'home_score',
                     'away_score',
                     'home_first_quarter',
                     'home_second_quarter',
                     'home_third_quarter',
                     'home_fourth_quarter',
                     'home_ot',
                     'away_first_quarter',
                     'away_second_quarter',
                     'away_third_quarter',
                     'away_fourth_quarter',
                     'away_ot']

        df = pd.DataFrame(columns, columns=col_names)

        df[['home_score', 'away_score',
            'home_first_quarter', 'home_second_quarter', 'home_third_quarter', 'home_fourth_quarter',
            'home_ot',
            'away_first_quarter', 'away_second_quarter', 'away_third_quarter', 'away_fourth_quarter',
            'away_ot']] = \
            df[['home_score', 'away_score',
                'home_first_quarter', 'home_second_quarter', 'home_third_quarter', 'home_fourth_quarter',
                'home_ot',
                'away_first_quarter', 'away_second_quarter', 'away_third_quarter', 'away_fourth_quarter',
                'away_ot']].astype('int32')

        return df


if __name__ == '__main__':
    None
