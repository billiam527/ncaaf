# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:23:10 2024

@author: wfish
"""

import pandas as pd
from xgboost import XGBClassifier
import pickle

def import_data(pbp_file, games_file):
    
    pbp = pd.read_csv(pbp_file)
    games = pd.read_csv(games_file)
    
    return pbp, games

def simple_win_prob_model(pbp, games):
    
    ### Very simple model
    ### Does not account that home team is probably better
    ### Does not account for spread
    
    # remove non play plays
    remove = ['End Period',
        'Kickoff Return (Offense)',
        'Timeout',
        'Kickoff Return Touchdown',
        'Kickoff',
        '2pt Conversion',
        'Defensive 2pt Conversion',
        'End of Half',
        'End of Game',
        'Coin Toss'
        'Offensive 1pt Safety'
        'End of Regulation'
        ]
    
    pbp = pbp.loc[~pbp['play_type_text'].isin(remove)]
    
    # add home_team_ids and away_team_ids
    games = games[['id', 'home_team_id', 'away_team_id', 'season']]
    games['game_id'] = games['id']
    games.drop('id', axis=1, inplace=True)
    pbp = pd.merge(games, pbp, on='game_id')
    
    cols = ['game_id', 'home_team_id', 
            'away_team_id', 'team_id', 
            'home_score', 'away_score', 
            'period', 'half_seconds_remaining', 
            'down', 'distance', 'yards_to_end_zone']
    
    pbp_edit = pbp[cols]
    final_score = pbp_edit.groupby('game_id').last()
    
    # home win
    final_score.loc[final_score['home_score'] - final_score['away_score'] > 0, 'home_win'] = 0
    # away win
    final_score.loc[final_score['home_score'] - final_score['away_score'] < 0, 'home_win'] = 2
    # tie (OT)
    final_score.loc[final_score['home_score'] - final_score['away_score'] == 0, 'home_win'] = 1
    
    winner = final_score.reset_index()[['game_id', 'home_win']]
    
    df = pd.merge(pbp_edit, winner, on='game_id').drop('game_id', axis=1)
    
    df.loc[df['team_id'] == df['home_team_id'], 'possession'] = 1 #home ball
    df.loc[df['team_id'] != df['home_team_id'], 'possession'] = 0 #away ball
    df['seconds_remaining'] = df['half_seconds_remaining']
    df.loc[df['period'] <= 2, 'seconds_remaining'] = (df['half_seconds_remaining'] * 60) + 30*60
    df = df.loc[df['period'] <= 4]
    df['home_score_dif'] = df['home_score'] - df['away_score']
    df.drop(['half_seconds_remaining', 'home_team_id', 'away_team_id',
             'home_score', 'away_score', 'period'], axis=1, inplace=True)
    
    x = df[['possession', 'down', 'distance', 
            'yards_to_end_zone', 'home_score_dif', 'seconds_remaining']]
    y = df['home_win']
    
    model = XGBClassifier()
    model.fit(x, y)
    
    filename = 'wp_model.pkl'
    pickle.dump(model, open(filename, "wb"))
    

if __name__ == '__main__':
    
    pbp_file = '../../etl/summarize/temp/pbp.csv'
    games_file = '../../etl/summarize/temp/games.csv'
    pbp, games = import_data(pbp_file, games_file)
    simple_win_prob_model(pbp, games)
    