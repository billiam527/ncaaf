# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:20:23 2024

@author: wfish
"""

import pandas as pd
import os

def round_off_rating(number):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""

    return round(number * 2) / 2

# import predictions game by game
files = os.listdir(os.curdir + '/prediction_file')
files_that_contain_week = [i for i in files if 'Week' in i]

# import odds_table
odds_table = pd.read_csv('odds_table.csv', index_col=0)

for i in files_that_contain_week:
    pred_file = pd.read_csv('prediction_file/' + i + '/' + i + '.csv', 
                            index_col=0)#.dropna().reset_index(drop=True)
    pred_file['home_spread'] = -round_off_rating(pred_file['blended_prediction'])
    for j in pred_file['short_name'].unique():
        game = pred_file.loc[pred_file['short_name'] == j]
        new_game = pd.merge(game, odds_table, left_on='home_spread', right_on='spread')
        new_game.to_csv('prediction_file/' + i + '/' + j.replace(' ', '_') + '.csv')