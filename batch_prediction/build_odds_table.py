# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:33:29 2024

@author: wfish
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings("ignore")

# round number to nearest 0.5
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

# normalize any data
def NormalizeData(data):
    """Normalize data to sum up to 1"""
    
    return (data)/(data).sum()

## import data
def edit_data(df,
              home_score_col,
              away_score_col,
              home_spread_col):
    
    # add in some useful columns
    df['Home_Score_Dif'] = df[home_score_col] - df[away_score_col]
    df['Dif_Check_Col'] = -df['Home_Score_Dif']
    df['Spread_minus_Check_Col'] = df[home_spread_col] - df['Dif_Check_Col']
    
    return df
    

def ordered_model(df):
    
    # run a orderedmodel
    df2 = df[['homeSpread', 'Dif_Check_Col']]
    df2 = df2.dropna()
    mod_prob = OrderedModel(df2['Dif_Check_Col'],
                        df2['homeSpread'],
                        distr='probit')

    res_prob = mod_prob.fit(method='bfgs')
    results_as_html = res_prob.summary().tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    spread_range = np.arange(-50, 50+0.5, 0.5)
    
    # edit the results
    results = []
    probs = []
    spreads = []
    for i in spread_range:
        for j, k in zip(df[1:].reset_index()['index'], res_prob.predict(i)[0]):
            spreads.append(i)
            results.append(j.split('/')[0])
            probs.append(float(k))
        
    result_probs = pd.DataFrame([spreads, results, probs]).T
    result_probs.columns = ['spread', 'values', 'probabilities']
    result_probs['spread'] = result_probs['spread'].astype(float)
    result_probs['values'] = result_probs['values'].astype(float)
    
    faves = result_probs.loc[result_probs['spread'] < 0]
    dogs = result_probs.loc[result_probs['spread'] > 0]
    
    new_dfs = []
    for i, j in zip(list(faves['spread'].unique()), list(dogs.sort_values('spread', ascending=False)['spread'].unique())):
        # favorites
    
        neg = result_probs[result_probs['spread'] == i]
        plus = result_probs[result_probs['spread'] == j]
    
        plus['values'] = plus['values']*-1
        plus['spread'] = plus['spread']*-1
        plus = plus.sort_values('values')
    
        dif = list(neg['spread'])[0] + (list(neg['spread'])[0] - list(neg['values'])[0])
        max_num = neg['values'].max()
    
        to_add = plus.loc[(plus['values'] <= dif) & (plus['values'] > max_num)]
    
        new_df = pd.concat([neg, to_add])
        new_dfs.append(new_df)
        
        # dawgs
        plus = result_probs[result_probs['spread'] == j]
        neg = result_probs[result_probs['spread'] == i]
    
        neg['values'] = neg['values']*-1
        neg['spread'] = neg['spread']*-1
        neg = neg.sort_values('values')
    
        # add up to 74
        num = plus['values'].max()
        new_plus = pd.concat([plus, neg.loc[neg['values'] > num]])
        new_plus
    
        dif = list(new_plus['spread'])[0] - (new_plus['values'].max() - list(new_plus['spread'])[0])
    
        new_dfs.append(new_plus.loc[new_plus['values'] >= dif])
        
    result_probs = pd.concat(new_dfs).sort_values(['spread', 'values'])
    
    # 1 - find any weird outliers where the negative spread/value combo doesnt match negative to positive
    # from +30 spreads, copy probabilitie from -30 and lower
    spreads_less_than_30 = result_probs[result_probs['spread'] <= -30]['spread']*-1
    values_less_than_30 = result_probs[result_probs['spread'] <= -30]['values']*-1
    
    under_neg_30 = pd.DataFrame()
    under_neg_30['spread'] = spreads_less_than_30
    under_neg_30['values'] =  values_less_than_30
    result_probs.reset_index(drop=True, inplace=True)
    under_neg_30['probabilities'] = result_probs['probabilities']
    
    # 2 - extend the table from +63 to +74 to match the (-) side... take probabilities from that side of the table
    under_pos_30 = result_probs[result_probs['spread'] < 30]
    
    result_probs = pd.concat([under_pos_30, under_neg_30.iloc[::-1]])
    
    zero = result_probs.loc[result_probs['spread'] == -0.5]
    zero['spread'] = 0
    result_probs = pd.concat([zero, result_probs])
    
    round_decimal = 4
    dfs = []
    for i in spread_range:
        # fill in other push values
        spread_df = result_probs.loc[result_probs['spread'] == i]
        lst = list(spread_df['values'])
        try:
            missing_values = sorted(set(range(int(lst[0]), int(lst[-1]))) - set(lst))
        
            missing_vals_df = pd.DataFrame()
            missing_vals_df['spread'] = [i] * len(missing_values)
            missing_vals_df['values'] = missing_values
            missing_vals_df['probabilities'] = [np.NaN] * len(missing_values)
    
            spread_df = pd.concat([spread_df, missing_vals_df]).sort_values('values')
            fills = 0.5 * (spread_df.fillna(method='ffill', limit=1) + spread_df.fillna(method='bfill', limit=1))
            spread_df = spread_df.combine_first(fills)
    
            # find the midpoint of ea spread in terms of cum win (push) percentage
            spread_df['probs_cumsum'] = spread_df['probabilities'].cumsum()
            if i.is_integer():
                spread_df.loc[(spread_df['spread'] == spread_df['values']), 'midpoint'] = 1
            else:
                spread_df.loc[(spread_df['values'] == i+0.5), 'midpoint'] = 1
                spread_df.loc[(spread_df['values'] == i-0.5), 'midpoint'] = 1
    
            midpoint_idx = spread_df.reset_index().index[spread_df['midpoint'] == 1][0]
            above_mp = len(spread_df)-midpoint_idx
            below_mp = midpoint_idx
    
            if above_mp <= below_mp:
                num = above_mp
            else:
                num = below_mp
    
            spread_df = spread_df.loc[(spread_df['values'] >= (i-num)) &
                                      (spread_df['values'] <= (i+num))]
    
            spread_df['norm_probs'] = list(NormalizeData(spread_df['probabilities']))
            push_perc = spread_df.loc[spread_df['midpoint'] == 1]['norm_probs'].iloc[0]
            rounded_push_perc = round(push_perc, round_decimal)
            norm_number = round((1-rounded_push_perc)/2, round_decimal)
            midpoint_idx = spread_df.index[spread_df['midpoint'] == 1][0]
            less_than = spread_df[spread_df.index < midpoint_idx]['norm_probs']
            more_than = spread_df[spread_df.index > midpoint_idx]['norm_probs']
    
            nn_probs = []
            for j, k in zip(spread_df['values'], spread_df['norm_probs']):
                if j < i:
                    nn_probs.append(round((k/(less_than.sum()/norm_number)), round_decimal))
                elif j == i:
                    nn_probs.append(round(k, round_decimal))
                else:
                    nn_probs.append(round((k/(more_than.sum()/norm_number)), round_decimal))
            spread_df['new_norm_probs'] = nn_probs  
            #######################
    
            dfs.append(spread_df)
        except:
            print(i)
        
    result_probs = pd.concat(dfs).fillna(0)
    
    result_probs_condensed = result_probs[['spread', 'values', 'new_norm_probs']]
    result_probs_condensed.rename({'new_norm_probs': 'push_probs'}, axis=1, inplace=True)
    result_probs_condensed['win_probs'] = result_probs_condensed.groupby('spread')['push_probs'].cumsum()
    result_probs_condensed['win_probs'] = result_probs_condensed.groupby('spread')['win_probs'].shift(1).fillna(0)
    result_probs_condensed['loss_probs'] = 1 - result_probs_condensed['win_probs'] - result_probs_condensed['push_probs']
    probs_df = result_probs_condensed[['spread', 'values', 'loss_probs', 'push_probs', 'win_probs']]
    
    tweeners = []
    zeros = []
    spreads = []
    for i, j in zip(probs_df['spread'], probs_df['values']):
        spreads.append(i)
        tweeners.append(j + 0.5)
        zeros.append(np.nan)
    
    new_df = pd.DataFrame()    
    new_df['values'] = tweeners
    new_df['spread'] = spreads
    new_df['loss_probs'] = zeros
    new_df['push_probs'] = zeros
    new_df['win_probs'] = zeros
    
    probs_df = pd.concat([new_df, probs_df]).sort_values(['spread', 'values'])
    probs_df = probs_df.reset_index(drop=True).interpolate()[:-1]
    
    probs_cols = ['loss_probs', 'push_probs', 'win_probs']
    for i in probs_cols:
        if i != 'push_probs':
            probs_df.loc[probs_df['values'] % 1 != 0, i] = probs_df[i] + (probs_df['push_probs']/2)
        else:
            probs_df.loc[probs_df['values'] % 1 != 0, 'push_probs'] = 0
        
    for i in probs_cols:    
        probs_df.loc[probs_df[i] < 0, i] = 0
        
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    #### if i is < 1 then
    probs_df.loc[probs_df['total'] < 1, 'loss_probs'] = probs_df['loss_probs'] + ((1-probs_df['total'])/2)
    probs_df.loc[probs_df['total'] < 1, 'win_probs'] = probs_df['win_probs'] + ((1-probs_df['total'])/2)
    
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    probs_df['loss_probs'] = probs_df['loss_probs'].round(4)
    probs_df['win_probs'] = probs_df['win_probs'].round(4)
    probs_df['push_probs'] = probs_df['push_probs'].round(4)
    
    probs_df.loc[(probs_df['total'] > 1) & 
             (probs_df['loss_probs'] == 0) & 
             (probs_df['push_probs'] == 0), 'win_probs'] = 1

    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    probs_df.loc[(probs_df['total'] > 1) & 
                 (probs_df['win_probs'] == 0) & 
                 (probs_df['push_probs'] == 0), 'loss_probs'] = 1
    
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    probs_df.loc[(probs_df['total'] > 1) & 
                 (probs_df['loss_probs'] == 0) & 
                 (probs_df['push_probs'] > 0), 'win_probs'] = probs_df['win_probs']/(probs_df['win_probs'] + probs_df['push_probs'])
    
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    probs_df.loc[(probs_df['total'] > 1) & 
                 (probs_df['loss_probs'] == 0) & 
                 (probs_df['push_probs'] > 0), 'push_probs'] = probs_df['push_probs']/(probs_df['win_probs'] + probs_df['push_probs'])
    
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    probs_df.loc[(probs_df['total'] > 1) & 
                 (probs_df['win_probs'] == 0) & 
                 (probs_df['push_probs'] > 0), 'loss_probs'] = probs_df['loss_probs']/(probs_df['loss_probs'] + probs_df['push_probs'])
    
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    probs_df.loc[(probs_df['total'] > 1) & 
                 (probs_df['win_probs'] == 0) & 
                 (probs_df['push_probs'] > 0), 'push_probs'] = probs_df['push_probs']/(probs_df['loss_probs'] + probs_df['push_probs'])
    
    probs_df['total'] = probs_df['loss_probs'] + probs_df['push_probs'] + probs_df['win_probs']
    
    for i in list(probs_df['spread'].unique()):
        val_count_above = 0
        val_count_below = 0
        for j in probs_df.loc[probs_df['spread'] == i]['values']:
            if j < i:
                val_count_above = val_count_above + 1
            if j > i:
                val_count_below = val_count_below + 1
                
    #### fix the last value in each spread ####
    new_dfs = []
    for i in probs_df['spread'].unique():
        new_dfs.append(probs_df[probs_df['spread'] == i].iloc[:-1])
        
    probs_df = pd.concat(new_dfs)
    
    probs_df.loc[probs_df['values'] == 0, 'loss_probs'] = 0
    probs_df.loc[probs_df['values'] == 0, 'push_probs'] = 0
    probs_df.loc[probs_df['values'] == 0, 'win_probs'] = 0
    
    probs_df = probs_df[probs_df['spread'] != 0]
    probs_df = probs_df[probs_df['spread'] != 0.5]
    
    zeros = probs_df[probs_df['spread'] == -0.5]
    zero_pt_fives = probs_df[probs_df['spread'] == -0.5]
    
    zeros.loc[zeros['spread'] == -0.5, 'spread'] = 0
    zero_pt_fives.loc[zero_pt_fives['spread'] == -0.5, 'spread'] = 0.5
    
    probs_df = pd.concat([pd.concat([probs_df, zeros]), zero_pt_fives]).sort_values(['spread', 'values'])
    
    probs_df = probs_df.reset_index(drop=True)
    
    # win probs that equal 1
    spreads = []
    values = []
    loss_probs = []
    win_probs = []
    push_probs =  []
    
    for i, j, k, l, m in zip(probs_df['values'], 
                          probs_df['win_probs'], probs_df['push_probs'], probs_df['loss_probs'],
                          probs_df['spread']):
        # 0.5ers
        if j == 1.0 and int(i) != i:
            spreads.append(m)
            values.append(i)
            loss_probs.append(.0001)
            win_probs.append(.9999)
            push_probs.append(0)
        # 0.0ers
        if j == 1.0 and int(i) == i:
            spreads.append(m)
            values.append(i)
            loss_probs.append(0)
            win_probs.append(.9999)
            push_probs.append(0.0001)
            
    new_probs = pd.DataFrame()
    new_probs['values'] = values
    new_probs['spread'] = spreads
    new_probs['loss_probs'] = loss_probs
    new_probs['push_probs'] = push_probs
    new_probs['win_probs'] = win_probs
    new_probs['total'] = 1
    
    probs_df = pd.concat([probs_df.loc[probs_df['win_probs'] != 1], new_probs])
    
    # loss probs that equal 1
    spreads = []
    values = []
    loss_probs = []
    win_probs = []
    push_probs =  []
    
    for i, j, k, l, m in zip(probs_df['values'], 
                          probs_df['win_probs'], probs_df['push_probs'], probs_df['loss_probs'],
                          probs_df['spread']):
        # 0.5ers
        if k == 1.0 and int(i) != i:
            spreads.append(m)
            values.append(i)
            loss_probs.append(.0001)
            win_probs.append(.9999)
            push_probs.append(0)
        # 0.0ers
        if k == 1.0 and int(i) == i:
            spreads.append(m)
            values.append(i)
            loss_probs.append(0)
            win_probs.append(.9999)
            push_probs.append(0.0001)
            
    new_probs = pd.DataFrame()
    new_probs['values'] = values
    new_probs['spread'] = spreads
    new_probs['loss_probs'] = loss_probs
    new_probs['push_probs'] = push_probs
    new_probs['win_probs'] = win_probs
    new_probs['total'] = 1
    
    probs_df = pd.concat([probs_df.loc[probs_df['loss_probs'] != 1], new_probs])
    
    probs_df = probs_df.reset_index(drop=True)
    probs_df.loc[probs_df['win_probs'] <= 0.5, 'cover_odds'] = (100/probs_df['win_probs']+(probs_df['push_probs']/2))-100
    probs_df.loc[probs_df['win_probs'] > 0.5, 'cover_odds'] = (((probs_df['win_probs']+(probs_df['push_probs']/2))*100)\
                                                               /(1-(probs_df['win_probs']+(probs_df['push_probs']/2))))*-1
    probs_df.loc[probs_df['loss_probs'] <= 0.5, 'no_cover_odds'] = (100/(probs_df['loss_probs']+(probs_df['push_probs']/2)))-100
    probs_df.loc[probs_df['loss_probs'] > 0.5, 'no_cover_odds'] = (((probs_df['loss_probs']+(probs_df['push_probs']/2))*100)\
                                                                   /(1-(probs_df['loss_probs']+(probs_df['push_probs']/2))))*-1
        
    probs_df['cover_odds'] = probs_df['cover_odds'].round(0)
    probs_df['no_cover_odds'] = probs_df['no_cover_odds'].round(0)

    probs_df.loc[probs_df['cover_odds'] == np.inf, 'cover_odds'] = probs_df['no_cover_odds']*-1
    probs_df.loc[probs_df['spread'] == probs_df['values'], 'cover_odds'] = 100
    probs_df.loc[probs_df['spread'] == probs_df['values'], 'no_cover_odds'] = 100
    
    probs_df = probs_df.sort_values(['spread', 'values']).drop('total', axis=1)
    
    return probs_df


if __name__ == '__main__':
    df = pd.read_csv('../etl/collect/collect_cfbd_games/temp/cfbd_spread_data.csv')
    df = edit_data(df,
                   'homeScore',
                   'awayScore',
                   'homeSpread')
    odds_table = ordered_model(df)
    odds_table.to_csv('prediction_file/odds_table.csv')