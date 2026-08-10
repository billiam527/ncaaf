# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:46:12 2024

@author: wfish
"""

import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
import argparse


def read_data(games_df_file_loc: str,
              season_summary_df_file_loc: str,
              # pbp_df_file_loc: str,
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
        elif 'fbs_only_ind' in line:
            fbs_only_ind = bool(data)
        elif 'training_algorithm' in line:
            algo = data

    return pd.read_csv(games_df_file_loc), \
        pd.read_csv(season_summary_df_file_loc), \
        start_year, \
        end_year, \
        features, \
        test, \
        fbs_only_ind, \
        algo


def edit_files(games_df: pd.DataFrame,
               season_summary_df: pd.DataFrame,
               features: list,
               start_year: int,
               end_year: int
              ):
    
    stats_df = season_summary_df.reset_index()[['season', 'team_id'] + features]
    
    away_df = pd.merge(games_df, stats_df, left_on=['away_team_id', 'season'], right_on=['team_id', 'season'])
    home_df = pd.merge(away_df, stats_df,
                       left_on=['home_team_id', 'season'],
                       right_on=['team_id', 'season'],
                       suffixes=('_away', '_home'))
    
    keep_cols = []
    for i in features:
        for j in list(home_df):
            if i in j:
                keep_cols.append(j)
    
    other_cols = ['season', 'id', 'date', 'short_name',
                  'away_team_id', 'home_team_id', 
                  'status', 'home_score_differential']
    return home_df[other_cols + keep_cols]


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

        return pd.concat(train_dfs).drop('season', axis=1), \
            pd.concat(test_dfs).drop('season', axis=1)

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

    train = train_df[[y_col] + features]
    train_id_df = train_df[identifier_cols]
    test = test_df[[y_col] + features]
    test_id_df = test_df[identifier_cols]

    train_X = train.drop(y_col, axis=1)
    train_y = train[y_col]

    train_X_scaled = scaler.fit_transform(train_X)

    test_X = test.drop(y_col, axis=1)
    test_y = test[y_col]

    test_X_scaled = scaler.transform(test_X)

    # save the scaler
    dump(scaler, open('temp/scaler.pkl', 'wb'))

    return pd.DataFrame(train_X_scaled, columns=features), \
        pd.DataFrame(test_X_scaled, columns=features), \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df


#def parse_args():

#    parser = argparse.ArgumentParser(description='Input path for season summary csv file')
#    parser.add_argument('--season_summary_data')
#    args = parser.parse_args()

#    return args.season_summary_data


if __name__ == '__main__':

    # All files ending with .txt
    experiment_info_txt_file_loc = glob.glob('temp/in_season_experiment*')[0]
    #season_summary_data = parse_args()

    games_df, \
        season_summaries_df, \
        start_year, \
        end_year, \
        features, \
        test,\
        fbs_only_ind, \
        algo = read_data(games_df_file_loc='temp/games.csv',
            season_summary_df_file_loc='temp/season_summaries.csv',
            experiment_info_txt_file_loc=experiment_info_txt_file_loc)

    season_summaries_df.to_csv('temp/season_summaries_raw.csv')

    train_df = edit_files(games_df=games_df,
                          season_summary_df=season_summaries_df,
                          features=features,
                          start_year=start_year,
                          end_year=end_year)

    train_df.to_csv('temp/raw_train.csv')

    train_df, \
        test_df = split_data(data=train_df,
                             test=test,
                             y_col='home_score_differential')

    train_df.to_csv('temp/train_df.csv')
    test_df.to_csv('temp/test_df.csv')
    
    final_cols = []
    for i in features:
        final_cols.append(i + '_home')
        final_cols.append(i + '_away')

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
