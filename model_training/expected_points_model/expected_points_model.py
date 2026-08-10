# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 23:02:23 2024

@author: wfish
"""

import pandas as pd
from xgboost import XGBClassifier
import pickle

def import_data(pbp_file, games_file):
    
    pbp = pd.read_csv(pbp_file)
    games = pd.read_csv(games_file)
    
    return pbp, games

def edit_pbp_file(pbp):
    
    # edit pbp_df
    pbp_smaller = pbp[['game_id', 'id', 'home_score', 'away_score', 
                       'team_id', 'period', 'clock', 'down', 
                       'distance', 'yards_to_end_zone']]
    pbp_smaller.loc[pbp_smaller['period'] <= 2, 'half'] = 1
    pbp_smaller.loc[pbp_smaller['period'] > 2, 'half'] = 2
    pbp_smaller['home_score_dif'] = pbp_smaller.groupby(['game_id', 'half'])['home_score'].diff()
    pbp_smaller['away_score_dif'] = pbp_smaller.groupby(['game_id', 'half'])['away_score'].diff()
    
    # clock adjustments
    pbp_smaller['minutes'] = pbp_smaller['clock'].str.split(':').str[0]
    pbp_smaller['seconds'] = pbp_smaller['clock'].str.split(':').str[1]
    pbp_smaller = pbp_smaller.loc[pbp_smaller['period'].isin([1,2,3,4])]
    pbp_smaller['half_seconds_remaining'] = \
        pbp_smaller['minutes'].astype(int).multiply(60) + pbp_smaller['seconds'].astype(int)
    pbp_smaller.loc[(pbp_smaller['period'] == 1) | 
                    (pbp_smaller['period'] == 3), 
                    'half_seconds_remaining'] = \
        pbp_smaller['half_seconds_remaining'].astype(int) + 900
    
    # some cleanup
    pbp_smaller.loc[pbp_smaller['distance'] > 25, 'distance'] = 25
    pbp_smaller = pbp_smaller.loc[pbp_smaller['down'] != -1]
    pbp_smaller.loc[(pbp_smaller['down'] != 0) & (pbp_smaller['distance'] == 0),
                    'distance'] = pbp_smaller['yards_to_end_zone']
    pbp_smaller = pbp_smaller.loc[(pbp_smaller['down'] >= 1) & 
                                  (pbp_smaller['down'] <= 4)]
    pbp_smaller = pbp_smaller.loc[(pbp_smaller['distance'] >= 1) & 
                                  (pbp_smaller['distance'] <= 99)]
    
    return pbp_smaller


def edit_home_and_away_data(pbp_smaller,
                            games):
    
    
    cols = ['away_team_id', 'home_team_id', 'game_id', 'team_id', 
        'down', 'distance', 'yards_to_end_zone', 'next_play', 
        'next_play_cat', 'half_seconds_remaining']
    
    # home data
    home_data = pbp_smaller
    
    # home scoring types
    home_data.loc[home_data['home_score_dif'] == 7, 'next_play'] = 'touchdown'
    home_data.loc[home_data['home_score_dif'] == 8, 'next_play'] = 'touchdown'
    home_data.loc[home_data['home_score_dif'] == 6, 'next_play'] = 'touchdown'
    home_data.loc[home_data['home_score_dif'] == 3, 'next_play'] = 'field_goal'
    home_data.loc[home_data['home_score_dif'] == 2, 'next_play'] = 'safety'
    home_data.loc[home_data['away_score_dif'] == 7, 'next_play'] = 'opponent_touchdown'
    home_data.loc[home_data['away_score_dif'] == 8, 'next_play'] = 'opponent_touchdown'
    home_data.loc[home_data['away_score_dif'] == 6, 'next_play'] = 'opponent_touchdown'
    home_data.loc[home_data['away_score_dif'] == 3, 'next_play'] = 'opponent_field_goal'
    home_data.loc[home_data['away_score_dif'] == 2, 'next_play'] = 'opponent_safety'
    
    # mark the end of halfs to mark as no_scores
    eoh = home_data.groupby(['game_id', 'half']).last()
    eoh['eoh'] = 1
    eoh = eoh.reset_index()[['id', 'eoh']]
    home_data = pd.merge(home_data, eoh, how='left', on='id')
    home_data.loc[home_data['eoh'] == 1, 'next_play'] = 'no_score'
    home_data['next_play'] = home_data['next_play'].bfill()
    
    # converting type of columns to 'category'
    home_data['next_play'] = home_data['next_play'].astype('category')
    # Assigning numerical values and storing in another column
    home_data['next_play_cat'] = home_data['next_play'].cat.codes
    # Next play cats
    key_df = pd.DataFrame(home_data[['next_play', 'next_play_cat']].value_counts()).reset_index().drop('count', axis=1)
    key_df = key_df.sort_values('next_play_cat')
    
    # bring in games to get home and away team ids
    games = games[['id', 'home_team_id', 'away_team_id']]
    games['game_id'] = games['id']
    games.drop('id', axis=1, inplace=True)
    home_data = pd.merge(games, home_data, on='game_id').drop('eoh', axis=1)
    
    # the home team
    home_model_data = home_data[cols]
    home_model_data = home_model_data.loc[home_model_data['home_team_id'] == home_model_data['team_id']]
    
    # away data
    away_data = pbp_smaller
    
    # away scoring types
    away_data.loc[away_data['away_score_dif'] == 7, 'next_play'] = 'touchdown'
    away_data.loc[away_data['away_score_dif'] == 8, 'next_play'] = 'touchdown'
    away_data.loc[away_data['away_score_dif'] == 6, 'next_play'] = 'touchdown'
    away_data.loc[away_data['away_score_dif'] == 3, 'next_play'] = 'field_goal'
    away_data.loc[away_data['away_score_dif'] == 2, 'next_play'] = 'safety'
    away_data.loc[away_data['home_score_dif'] == 7, 'next_play'] = 'opponent_touchdown'
    away_data.loc[away_data['home_score_dif'] == 8, 'next_play'] = 'opponent_touchdown'
    away_data.loc[away_data['home_score_dif'] == 6, 'next_play'] = 'opponent_touchdown'
    away_data.loc[away_data['home_score_dif'] == 3, 'next_play'] = 'opponent_field_goal'
    away_data.loc[away_data['home_score_dif'] == 2, 'next_play'] = 'opponent_safety'
    
    # mark the end of halfs to mark as no_scores
    eoh = away_data.groupby(['game_id', 'half']).last()
    eoh['eoh'] = 1
    eoh = eoh.reset_index()[['id', 'eoh']]
    away_data = pd.merge(away_data, eoh, how='left', on='id')
    away_data.loc[away_data['eoh'] == 1, 'next_play'] = 'no_score'
    away_data['next_play'] = away_data['next_play'].bfill()
    
    # converting type of columns to 'category'
    away_data['next_play'] = away_data['next_play'].astype('category')
    # Assigning numerical values and storing in another column
    away_data['next_play_cat'] = away_data['next_play'].cat.codes
    
    # bring in games to get home and away team ids
    away_data = pd.merge(games, away_data, on='game_id').drop('eoh', axis=1)
    
    # the home team
    away_model_data = away_data[cols]
    away_model_data = away_model_data.loc[away_model_data['away_team_id'] \
                                          == away_model_data['team_id']]
    
    return pd.concat([away_model_data, home_model_data]), key_df


def expected_points_model(model_data, key_df):
    
    x = model_data[['down', 'distance', 'yards_to_end_zone', 'half_seconds_remaining']]
    y = model_data['next_play_cat']
    
    model = XGBClassifier(enable_categorical=True)
    model.fit(x, y)
    filename = 'ep_model.pkl'
    pickle.dump(model, open(filename, "wb"))
    key_df.to_csv('scoring_key.csv')
    
        
if __name__ == '__main__':
    pbp_file = '../../etl/summarize/temp/pbp.csv'
    games_file = '../../etl/summarize/temp/games.csv'
    pbp, games = import_data(pbp_file, games_file)
    pbp_smaller = edit_pbp_file(pbp)
    model_data, key_df = edit_home_and_away_data(pbp_smaller, games)
    expected_points_model(model_data, key_df)
    