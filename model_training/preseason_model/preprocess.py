import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
            test_size = float(data)
            continue
        elif 'seasons_back' in line:
            seasons_back = int(data)
            continue
        elif 'weights' in line:
            weights = []
            for i in data.split(','):
                weights.append(float(i))
            continue

    return pd.read_csv(games_df_file_loc), \
           pd.read_csv(season_summary_df_file_loc), \
           start_year, \
           end_year, \
           features, \
           test_size, \
           seasons_back, \
           weights


def edit_files(games_df: pd.DataFrame,
               season_summary_df: pd.DataFrame,
               features: list,
               start_year: int,
               end_year: int,
               seasons_back: int,
               weights: list
               ):

    # edit games
    games_w_cols = games_df[['id', 'date', 'season', 'short_name', 'status',
        'away_team_id', 'home_team_id',
        'home_score_differential']]

    games_w_years_edit = games_w_cols.loc[(games_w_cols['season'] >= int(start_year)) &
                                          (games_w_cols['season'] <= int(end_year))]

    games_w_status_edit = games_w_years_edit.loc[games_w_years_edit['status'] == 'STATUS_FINAL']

    # season summaries edits
    season_summaries = season_summary_df[['team_id', 'season'] + features]

    #weights.reverse()

    #for col in features:
    #    season_summaries['weighted_ma_' + col] = season_summaries.groupby(['team_id'])[col] \
    #        .rolling(window=int(seasons_back), center=False) \
    #        .apply(lambda x: np.sum(weights * x)).values

    #    season_summaries['weighted_ma_' + col + '_shifted'] = season_summaries.groupby('team_id')['weighted_ma_' + col].shift(1)

    #keep_cols = ['team_id', 'season']
    #for col in features:
    #    keep_cols.append('weighted_ma_' + col + '_shifted')

    #season_summaries_edit = season_summaries[keep_cols]
    #for col in list(season_summaries_edit):
    #    season_summaries_edit.rename({col: col.replace('weighted_ma_', '').replace('_shifted', '')}
    #                                 , axis=1
    #                                 , inplace=True)

    #return games_w_status_edit, season_summaries, season_summaries_edit


def merge_games_and_stats(games_df: pd.DataFrame,
                          stats_df: pd.DataFrame):

    away_df = pd.merge(games_df, stats_df, left_on=['away_team_id', 'season'], right_on=['team_id', 'season'])
    home_df = pd.merge(away_df, stats_df,
                       left_on=['home_team_id', 'season'], right_on=['team_id', 'season'],
                       suffixes=('_away', '_home'))

    final_cols = []
    for col in list(home_df):
        if 'team_id_' in col:
            continue
        elif 'status' in col:
            continue
        else:
            final_cols.append(col)

    return home_df[final_cols].dropna()


def split_data(data: pd.DataFrame,
               test_size: float,
               y_col: str):

    """

    :param data: (pd.DataFrame) dataframe to be split into train and test dfs
    :param test_size: (float) percentage of dataset to allocate to test
    :param y_col: (str) name of column containing y data
    :return: train_data, test_data
    """

    seasons = list(data.season.unique())

    train_dfs = []
    test_dfs = []
    for season in seasons:
        season_data = data.loc[data['season'] == season]
        X = season_data.drop(y_col, axis=1)
        y = season_data[y_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        train_dfs.append(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1))
        test_dfs.append(pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1))

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    return train_df, test_df


def final_edits(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                features: list,
                y_col: str):

    scaler = StandardScaler()  # or: MinMaxScaler(feature_range=(0,1))
    train = train_df[features + [y_col]]
    test = test_df[features + [y_col]]

    train_X = train_df.drop(y_col, axis=1)
    train_y = train_df[y_col]

    train_X_scaled = scaler.fit_transform(train_X)

    test_X = test_df.drop(y_col, axis=1)
    test_y = test_df[y_col]

    test_X_scaled = scaler.transform(test_X)

    return train_X_scaled, test_X_scaled, train_y, test_y


if __name__ == '__main__':

    # All files ending with .txt
    experiment_info_txt_file_loc = glob.glob('temp/preseason_experiment*')[0]

    games_df, \
    season_summaries_df, \
    start_year, \
    end_year, \
    features, \
    test_size, \
    seasons_back, \
    weights = read_data(games_df_file_loc='temp/games.csv',
        season_summary_df_file_loc='temp/season_summaries.csv',
        experiment_info_txt_file_loc=experiment_info_txt_file_loc)

    games_w_years_edit, season_summaries, season_summaries_edit = edit_files(games_df=games_df,
        season_summary_df=season_summaries_df,
        features=features,
        start_year=start_year,
        end_year=end_year,
        seasons_back=seasons_back,
        weights=weights)

    games_w_years_edit.to_csv('temp/games_w_years_edit.csv')
    season_summaries.to_csv('temp/season_summaries_edit.csv')
    season_summaries_edit.to_csv('temp/season_summaries_edit_editted.csv')

    merged_df = merge_games_and_stats(games_df=games_w_years_edit,
        stats_df=season_summaries_edit)

    merged_df.to_csv('temp/merged.csv')

    train_df, test_df = split_data(data=merged_df, test_size=test_size, y_col='home_score_differential')

    train_df.to_csv('temp/train_df.csv')
    test_df.to_csv('temp/test_df.csv')