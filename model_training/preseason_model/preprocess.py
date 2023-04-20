import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump


def read_data(games_df_file_loc: str,
              season_summary_df_file_loc: str,
              experiment_info_txt_file_loc: str):

    with open(experiment_info_txt_file_loc) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'train_season_range' in line:
            year_range = data
            start_year = int(year_range.split(' - ')[0])
            end_year = int(year_range.split(' - ')[1])
            continue
        elif 'data_features' in line:
            features = data.split(',')
            continue
        elif 'test_size' in line:
            test = float(data)
            continue
        elif 'test_year' in line:
            test = int(data)

    return pd.read_csv(games_df_file_loc), \
        pd.read_csv(season_summary_df_file_loc), \
        start_year, \
        end_year, \
        features, \
        test


def edit_files(games_df: pd.DataFrame,
               season_summary_df: pd.DataFrame,
               features: list,
               start_year: int,
               end_year: int
               ):

    # edit games
    games_w_cols = games_df[['id', 'date', 'season', 'short_name', 'status',
                             'away_team_id', 'home_team_id',
                             'home_score_differential']]

    games_w_years_edit = games_w_cols.loc[(games_w_cols['season'] >= int(start_year)) &
                                          (games_w_cols['season'] <= int(end_year))]

    games_w_status_edit = games_w_years_edit.loc[games_w_years_edit['status'] == 'STATUS_FINAL']

    # some smaller schools might not have a full history w ESPN, give them each a year column even if its NA for now
    # goes from...
    # team_id, szn, features
    # 1, 2010, 123
    # 1, 2011, 123
    # 1, 2013, 123
    # to...
    # team_id, szn, features
    # 1, 2010, 123
    # 1, 2011, 123
    # 1, 2012, NA
    # 1, 2013, 123

    # season summaries column edits
    season_summaries = season_summary_df[['team_id', 'season'] + features]

    years = list(range(start_year, end_year))
    teams = []
    new_years = []
    for team in list(season_summaries['team_id'].unique()):
        for year in years:
            teams.append(team)
            new_years.append(int(year))

    teams_and_szns = pd.DataFrame(zip(teams, new_years), columns=['team_id', 'season'])

    season_summaries_add_years_edit = pd.merge(season_summaries, teams_and_szns, how='right',
                                               left_on=['team_id', 'season'],
                                               right_on=['team_id', 'season']
                                               )

    ss_edit = season_summaries_add_years_edit

    # Fill cols for prev years

    # Shift all columns
    # FY is the summarized data from the previous year
    # FY-1 is the summarized data from two years ago, FY-2 from 3 years ago
    # Example
    # 2018
    # 2019
    # 2020 FY would be summarized 2019 data, FY-1 2018, FY-2 2017
    # 2021 FY would be 2020 summarized data, FY-1 2019, FY-2 2018
    for i in [1, 2, 3]:
        for feature in features:
            if i == 1:
                ss_edit[feature + '_FY'] = season_summaries_add_years_edit.groupby('team_id')[feature].shift(i)
            else:
                ss_edit[feature + '_FY-{}'.format(i - 1)] = season_summaries_add_years_edit.groupby('team_id')[feature].shift(i)

    ss_edit.drop(features, axis=1, inplace=True)

    return games_w_status_edit, season_summaries, season_summaries_add_years_edit, ss_edit


def merge_games_and_stats(games_df: pd.DataFrame,
                          stats_df: pd.DataFrame):

    away_df = pd.merge(games_df, stats_df, left_on=['away_team_id', 'season'], right_on=['team_id', 'season'])
    home_df = pd.merge(away_df, stats_df,
                       left_on=['home_team_id', 'season'],
                       right_on=['team_id', 'season'],
                       suffixes=('_away', '_home'))

    final_cols = []
    for col in list(home_df):
        if col in ['team_id_away', 'team_id_home', 'away_team_id', 'home_team_id',
                   'status', 'season', 'id', 'date', 'short_name']:
            continue
        else:
            final_cols.append(col)

    return home_df.dropna(), home_df, home_df[final_cols].dropna(), final_cols


def split_data(data,
               test,
               y_col):
    """

    :param data: (pd.DataFrame) dataframe to be split into train and test dfs
    :param test_size: (float) percentage of dataset to allocate to test
    :param y_col: (str) name of column containing y data
    :return: train_data, test_data
    """

    seasons = list(data.season.unique())
    # if test is a float that means that the input mustve been a percentage to holdout as test
    if isinstance(test, float) is True:

        train_dfs = []
        test_dfs = []
        for season in seasons:
            season_data = data.loc[data['season'] == season]
            X = season_data.drop(y_col, axis=1)
            y = season_data[y_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test)
            train_dfs.append(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1))
            test_dfs.append(pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1))

        return pd.concat(train_dfs).drop(['season', 'status'], axis=1), \
            pd.concat(test_dfs).drop(['season', 'status'], axis=1)

    # if test is an int that means that the input mustve been a year to holdout as test
    elif isinstance(test, int) is True:

        return data.loc[data['season'] != test].drop(['season', 'status'], axis=1), \
            data.loc[data['season'] == test].drop(['season', 'status'], axis=1)


def final_edits(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                features: list,
                y_col: str):

    identifier_cols = ['id', 'date', 'short_name', 'away_team_id', 'home_team_id']
    scaler = StandardScaler()  # or: MinMaxScaler(feature_range=(0,1))

    train = train_df[features]
    train_id_df = train_df[identifier_cols]
    test = test_df[features]
    test_id_df = test_df[identifier_cols]

    train_X = train.drop(y_col, axis=1)
    train_y = train[y_col]

    train_X_scaled = scaler.fit_transform(train_X)

    test_X = test.drop(y_col, axis=1)
    test_y = test[y_col]

    test_X_scaled = scaler.transform(test_X)

    # save the scaler
    dump(scaler, open('temp/scaler.pkl', 'wb'))

    features.remove(y_col)

    return pd.DataFrame(train_X_scaled, columns=features), \
        pd.DataFrame(test_X_scaled, columns=features), \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df


if __name__ == '__main__':

    # All files ending with .txt
    experiment_info_txt_file_loc = glob.glob('temp/preseason_experiment*')[0]

    games_df, \
        season_summaries_df, \
        start_year, \
        end_year, \
        features, \
        test = read_data(games_df_file_loc='temp/games.csv',
                         season_summary_df_file_loc='temp/season_summaries.csv',
                         experiment_info_txt_file_loc=experiment_info_txt_file_loc)

    season_summaries_df.to_csv('temp/season_summaries_raw.csv')

    games_edit, \
        season_summaries_w_features, \
        season_summaries_add_years_edit, \
        ss_edit = edit_files(games_df=games_df,
                             season_summary_df=season_summaries_df,
                             features=features,
                             start_year=start_year,
                             end_year=end_year)

    games_edit.to_csv('temp/games_edit.csv')
    season_summaries_w_features.to_csv('temp/season_summaries_features_edit.csv')
    season_summaries_add_years_edit.to_csv('temp/season_summaries_add_years_edit.csv')
    ss_edit.to_csv('temp/season_summaries_edit.csv')

    merged_df, \
        merged_df_raw, \
        merged_df_final, \
        final_cols = merge_games_and_stats(games_df=games_edit,
                                           stats_df=ss_edit)

    merged_df.to_csv('temp/merged.csv')
    merged_df_raw.to_csv('temp/merged_raw.csv')
    merged_df_final.to_csv('temp/merged_final.csv')

    train_df, \
        test_df = split_data(data=merged_df,
                             test=test,
                             y_col='home_score_differential')

    train_df.to_csv('temp/train_df.csv')
    test_df.to_csv('temp/test_df.csv')

    train_X_scaled, \
        test_X_scaled, \
        train_y, \
        test_y,\
        train_id_df,\
        test_id_df = final_edits(train_df=train_df,
                                 test_df=test_df,
                                 features=final_cols,
                                 y_col='home_score_differential')

    train_X_scaled.to_csv('temp/train_X_scaled.csv')
    test_X_scaled.to_csv('temp/test_X_scaled.csv')
    pd.DataFrame(train_y, columns=['home_score_differential']).to_csv('temp/train_y.csv')
    pd.DataFrame(test_y, columns=['home_score_differential']).to_csv('temp/test_y.csv')
    train_id_df.to_csv('temp/train_id_cols.csv')
    test_id_df.to_csv('temp/test_id_cols.csv')
