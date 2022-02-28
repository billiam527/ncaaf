import pandas as pd
import json
import glob


def main():

    dfs = []
    iteration = 0
    for file in glob.glob('temp/pbpjsons/*'):
        with open(file) as json_data:
            data = json.load(json_data)
        pd_data = transform_espn_ncaaf_data(data)
        iteration = iteration + 1
        print(str(iteration) + ":" + file)
        dfs.append(pd_data)
    df = pd.concat(dfs)
    file = glob.glob('temp/dates*')
    date = (file[0].split('_'))[1]
    year = date.split('-')[0]
    df.to_csv('temp/play-by-play_' + str(year) + '.csv')


def transform_espn_ncaaf_data(json_data):

    """
    param json_data: (json) data pulled from espn's ncaaf api at the game level

    return df: (pd.DataFrame) data in the relational table form
    """

    # game data
    game_id = []

    # drive data
    drive_id = []

    # play data
    id = []
    play_type_id = []
    play_type_text = []
    play_text = []
    away_score = []
    home_score = []
    period = []
    clock = []
    scoring_play = []
    modified = []
    down = []
    distance = []
    yard_line = []
    yards_to_end_zone = []
    team_id = []
    stat_yardage = []

    try:
        for drive in json_data['drives']['previous']:
            plays = drive['plays']
            for play in range(len(plays)):

                #drive data
                drive_id.append(drive['id'])

                #play data
                id.append(plays[play]['id'])
                try:
                    play_type_id.append(plays[play]['type']['id'])
                    play_type_text.append(plays[play]['type']['text'])
                except KeyError:
                    play_type_id.append(999)
                    play_type_text.append('na')
                play_text.append(plays[play]['text'])
                away_score.append(plays[play]['awayScore'])
                home_score.append(plays[play]['homeScore'])
                period.append(plays[play]['period']['number'])
                clock.append(plays[play]['clock']['displayValue'])
                scoring_play.append(plays[play]['scoringPlay'])
                modified.append(plays[play]['modified'])
                down.append(plays[play]['start']['down'])
                distance.append(plays[play]['start']['distance'])
                yard_line.append(plays[play]['start']['yardLine'])
                yards_to_end_zone.append(plays[play]['start']['yardsToEndzone'])
                try:
                    team_id.append(plays[play]['start']['team']['id'])
                except KeyError:
                    team_id.append(999)
                stat_yardage.append(plays[play]['statYardage'])
                game_id.append(json_data['header']['competitions'][0]['id'])

    except Exception as e:
        if e == 'drives':
            print('No drives', e)

    columns = list(zip(id,
                       game_id,
                       modified,
                       home_score,
                       away_score,
                       drive_id,
                       team_id,
                       play_type_id,
                       play_type_text,
                       play_text,
                       period,
                       clock,
                       scoring_play,
                       down,
                       distance,
                       yard_line,
                       yards_to_end_zone,
                       stat_yardage))

    col_names = ['id',
                 'game_id',
                 'modified',
                 'home_score',
                 'away_score',
                 'drive_id',
                 'team_id',
                 'play_type_id',
                 'play_type_text',
                 'play_text',
                 'period',
                 'clock',
                 'scoring_play',
                 'down',
                 'distance',
                 'yard_line',
                 'yards_to_end_zone',
                 'stat_yardage']

    df = pd.DataFrame(columns, columns=col_names)

    return df


if __name__ == '__main__':
    main()
