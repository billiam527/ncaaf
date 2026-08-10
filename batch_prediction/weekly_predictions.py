# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 00:21:06 2024

@author: wfish
"""

import pandas as pd
import shutil
import os

full_df = pd.read_csv('prediction_file/new_predictions.csv', index_col=0)
weeks = list(full_df['week'].unique())
for i in weeks:
    try:
        shutil.rmtree('prediction_file/' + i)    
        os.mkdir('prediction_file/' + i)
        full_df.loc[full_df['week'] == i].to_csv('prediction_file/' + i \
                                                 + '/' + i + '.csv')
    except FileNotFoundError:   
        os.mkdir('prediction_file/' + i)
        full_df.loc[full_df['week'] == i].to_csv('prediction_file/' + i \
                                                 + '/' + i + '.csv')