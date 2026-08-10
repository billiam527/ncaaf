# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:46:15 2024

@author: wfish
"""

import pandas as pd
import numpy as np

def merge_model_and_preds(pred_file,
                          model_file):
    
    preds = pd.read_csv(pred_file, index_col=0)
    model = pd.read_csv(model_file, index_col=0)
    # predictions.csv carries week as a bare integer when it comes from the
    # scheduled-games file, and as a "Week N" label when it comes from the
    # historical games table. blended_model.csv always uses the label, so
    # normalise before joining. `'Week' in i` assumed a string and raised
    # TypeError on the integer form.
    weeks = []
    for i in preds['week']:
        text = str(i).strip()
        weeks.append(text if text.lower().startswith('week') else f'Week {text}')
    preds['week'] = weeks
    model['pre_szn_coefs'] = model['pre_szn_coefs'].round(4)
    model['in_szn_coefs'] = model['in_szn_coefs'].round(4)

    # Left join, not inner: a week with no learned coefficients (a bowl label,
    # or a week absent from the training history) previously vanished from the
    # output entirely rather than falling back to something sensible.
    df = pd.merge(preds, model, on='week', how='left')

    unmatched = df['pre_szn_coefs'].isnull()
    if unmatched.any():
        weeks_missing = sorted(df.loc[unmatched, 'week'].unique())
        print(f"WARNING: no blend weights for {int(unmatched.sum())} game(s) in "
              f"{weeks_missing}; falling back to the preseason model for those.")
        df.loc[unmatched, ['pre_szn_coefs', 'in_szn_coefs', 'intercepts']] = [1.0, 0.0, 0.0]

    return df

def implement_model(df):
    
    # we have a pre season pred but" not in seas"on one
    df.loc[(df['in_season_model_preds'].isnull()) &
           (df['preseason_model_preds'].notnull()), 'blended_prediction'] = df['preseason_model_preds']
    # in season no pre season (unlikely)
    df.loc[(df['preseason_model_preds'].isnull()) &
           (df['in_season_model_preds'].notnull()), 'blended_prediction'] = df['in_season_model_preds']
    # neither prediction
    df.loc[(df['preseason_model_preds'].isnull()) &
           (df['in_season_model_preds'].isnull()), 'blended_prediction'] = np.nan
    # we have both
    df.loc[(df['preseason_model_preds'].notnull()) &
           (df['in_season_model_preds'].notnull()), 'blended_prediction'] = \
        (df['pre_szn_coefs'] * df['preseason_model_preds']) + \
        (df['in_szn_coefs'] * df['in_season_model_preds']) + df['intercepts']
        
    #df['blended_prediction'] = (df['pre_szn_coefs'] * df['preseason_model_preds']) + \
    #    (df['in_szn_coefs'] * df['in_season_model_preds']) + df['intercepts']
        
    return df

if __name__ == '__main__':
    df = merge_model_and_preds(pred_file='prediction_file/predictions.csv',
                               model_file='../model_training/model_blender/blended_model.csv')

    df = implement_model(df)
    df.to_csv('prediction_file/new_predictions.csv')