# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from numpy import asarray
from numpy import exp
from xgboost import XGBRegressor
import csv

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

def NormalizeData(data):
    
    """Normalize data to sum up to 1"""
    
    return (data)/(data).sum()

def call_in_data(spread_data, predictions):
    
    return pd.read_csv(spread_data, index_col=0), pd.read_csv(predictions, index_col=0)

def edit_data(spread_data, home_score_col, away_score_col, spread_col):
    
    spread_data['Home_Score_Dif'] = spread_data[home_score_col] - spread_data[away_score_col]
    spread_data['Dif_Check_Col'] = -spread_data['Home_Score_Dif']
    spread_data['Spread_minus_Check_Col'] = spread_data[spread_col] - spread_data['Dif_Check_Col']
    
    return spread_data

def create_spread_fitter(spread_data, spread_col_name, predictions, y_pred_col_name):

    import matplotlib.pyplot as plt
    
    probs = {}
    
    spread_range = np.arange(min(spread_data[spread_col_name]), max(spread_data[spread_col_name])+0.5, 0.5)
    cols = [spread_col_name, 'Dif_Check_Col', 'Spread_minus_Check_Col']
    spreads = []
    max_liklihood = []
    median_values = []
    num_of_samples = []
    for i in spread_range:
        spread_pred = i
        new_spread_range = np.arange(i-3, i+3.5, 0.5)
        difs = spread_data[spread_data[spread_col_name].isin(new_spread_range)][cols]

        # remove any outliers by finding any games that are a 3 sigma event
        difs_from_spreads_median = difs['Spread_minus_Check_Col'].median()
        difs_from_spreads_std = round(difs['Spread_minus_Check_Col'].std(), 1)
        difs_max = difs['Spread_minus_Check_Col'].max()
        difs_min = difs['Spread_minus_Check_Col'].min()
        pos_3_stds = round(difs_from_spreads_median + (difs_from_spreads_std*3), 1)
        neg_3_stds = round(difs_from_spreads_median - (difs_from_spreads_std*3), 1)
        no_outliers = difs.loc[(difs['Spread_minus_Check_Col'] < pos_3_stds) & 
                               (difs['Spread_minus_Check_Col'] > neg_3_stds)]

        # remove some more by finding those that have never occurred before
        # start by fitting density
        if len(no_outliers) != 0:
            kde = KernelDensity(bandwidth=1, kernel='linear')
            sample = np.array(no_outliers['Dif_Check_Col']).reshape((len(no_outliers), 1))

            kde.fit(sample)

            # sample probabilities for a range of outcomes
            values = asarray([value for value in np.arange(no_outliers['Dif_Check_Col'].min(), 
                                                           no_outliers['Dif_Check_Col'].max()+0.5, 
                                                           0.5)])
            probabilities = kde.score_samples(values.reshape((len(values), 1)))
            probabilities = exp(probabilities)

            # find all the times a spread has "zero" chance of happening
            cumcount = []
            count = 0
            for i in probabilities:
                if i == 0:
                    count = count + 1
                elif i != 0:
                    count = 0
                cumcount.append(count)

            probs = pd.DataFrame()
            probs['values'] = values
            probs['probs'] = NormalizeData(probabilities)
            probs['zero_count'] = cumcount

            # remove any sequence where there are 3 or more straight 0 probability numbers
            favored = probs.loc[probs['values'] <= 0]
            highest = favored['zero_count'].max()
            if highest >= 3:
                cutoff = list(favored.loc[favored['zero_count'] == highest]['values'])[-1] + 0.5
                favored = favored.loc[favored['values'] >= cutoff]

            # for underdogs
            underdogs = probs.loc[probs['values'] > 0]
            cutoff = list(underdogs.loc[underdogs['zero_count'] >= 3]['values'])
            if cutoff:
                underdogs = underdogs.loc[underdogs['values'] <= cutoff[0] - 1.5]

            #Concat the resulting dfs so that each game is on its own row
            new_df = pd.concat([favored, underdogs])[['values', 'probs']]

            #new_df['normalized_probs'] = NormalizeData(new_df['probs'])
            new_min, new_max = new_df['values'].min(), new_df['values'].max()+0.5, 

            kde = KernelDensity(bandwidth=1, kernel='linear')
            new_samples = []
            for i in list(no_outliers['Dif_Check_Col']):
                if i < new_min or i > new_max:
                    pass
                else:
                    new_samples.append(i)

            sample = np.array(new_samples).reshape((len(new_samples), 1))

            kde.fit(sample)

            # sample probabilities for a range of outcomes
            try:
                values = asarray([value for value in np.arange(new_min, new_max+0.5, 0.5)])
            except ValueError as ve:
                continue
            
            probabilities = kde.score_samples(values.reshape((len(values), 1)))
            probabilities = exp(probabilities)

            new_probs = pd.DataFrame([values, probabilities]).T
            new_probs.columns = ['values', 'probabilities']
            new_probs['normalized_probs'] = NormalizeData(new_probs['probabilities'])
            new_probs.loc[new_probs['normalized_probs'] == new_probs['normalized_probs'].max(), 'max_liklihood'] = 1
            new_spread_option_1 = list(new_probs.loc[new_probs['max_liklihood'] == 1]['values'])[0]
            new_probs['normalized_probs'] = new_probs['normalized_probs'].fillna(0)
            new_probs['cum_probs'] = new_probs['normalized_probs'].cumsum()
            idx = new_probs[new_probs['cum_probs'] > 0.5].index[0]
            new_spread = new_probs['values'].iloc[idx]
            new_spread_option_2 = new_probs['values'].iloc[idx]
            new_probs.loc[new_probs['values'] == new_spread, 'fiftieth_percentile'] = 1
            new_probs.loc[new_probs['values'] != new_spread, 'fiftieth_percentile'] = 0

            if len(values) <= 60:
                pass
            else:
                spreads.append(spread_pred)
                num_of_samples.append(len(values))
                max_liklihood.append(new_spread_option_1)
                median_values.append(new_spread_option_2)
                
    df = pd.DataFrame([spreads, num_of_samples, max_liklihood, median_values]).T
    df.columns = ['spread', 'samples', 'max_l', 'median']
    
    df['max_l_dif'] = abs(df['spread'] - df['max_l'])
    df['med_dif'] = abs(df['spread'] - df['median'])
    
    for idx in range(1, len(df['median'])):
        if df['median'][idx - 1] <= df['median'][idx]:
            pass
        else:
            df.loc[idx]['median'] = df['median'][idx - 1]
    
    # make sure spread tables go in sequential order
    min_val = df['spread'].min()
    max_val = df['spread'].max()
    
    # create a quick model to fill in blank values
    spreads_df = pd.DataFrame(np.arange(min_val, max_val+0.5, 0.5))
    spreads_df.columns = ['spread']
    full_spread_df = pd.merge(df, spreads_df, how='right')
    
    full_spread_df[['spread', 'median']]
    for i in full_spread_df['spread']:
        if i%3 == 0:
            full_spread_df.loc[full_spread_df['spread'] == i, 'divisible_by_3'] = 1
            full_spread_df.loc[full_spread_df['spread'] == i+0.5, 'divisible_by_3'] = 1
            full_spread_df.loc[full_spread_df['spread'] == i-0.5, 'divisible_by_3'] = 1
        if i%7 == 0:
            full_spread_df.loc[full_spread_df['spread'] == i, 'divisible_by_7'] = 1
            full_spread_df.loc[full_spread_df['spread'] == i+0.5, 'divisible_by_7'] = 1
            full_spread_df.loc[full_spread_df['spread'] == i-0.5, 'divisible_by_7'] = 1
        if i%6 == 0:
            full_spread_df.loc[full_spread_df['spread'] == i, 'divisible_by_6'] = 1
            full_spread_df.loc[full_spread_df['spread'] == i+0.5, 'divisible_by_6'] = 1
            full_spread_df.loc[full_spread_df['spread'] == i-0.5, 'divisible_by_6'] = 1
    
    x_1 = full_spread_df[['spread', 'divisible_by_3', 'divisible_by_6', 'divisible_by_7']].fillna(0)
    x_1.to_csv('test2.csv')
    x = np.array(pd.merge(full_spread_df[['spread', 'median']], x_1).dropna()[['spread', 
        'divisible_by_3', 
        'divisible_by_6', 
        'divisible_by_7']])
    y = list(full_spread_df[['spread', 'median']].dropna()['median'])
    
    # use an xgboost model to solve for missing items
    model = XGBRegressor()
    model.fit(x, y)
    
    full_df = pd.merge(full_spread_df[['spread', 'median']], x_1)
    x_to_pred = full_df[full_df.isnull().any(axis=1)].drop('median', axis=1)
    preds = []
    for i in model.predict(x_to_pred):
        preds.append(round_off_rating(i))
    x_to_pred['median'] = preds
    
    filled_df = pd.merge(x_to_pred, df, how ='outer').sort_values('spread')#[['spread', 'median']].reset_index().drop('index', axis=1)
    for idx in range(1, len(filled_df['median'])):
        if filled_df['median'][idx - 1] <= filled_df['median'][idx]:
            pass
        else:
            filled_df.loc[idx]['median'] = filled_df['median'][idx - 1]
            
    for i in filled_df['spread']:
        if i%3 == 0:
            filled_df.loc[filled_df['spread'] == i, 'divisible_by_3'] = 1
            filled_df.loc[filled_df['spread'] == i+0.5, 'divisible_by_3'] = 1
            filled_df.loc[filled_df['spread'] == i-0.5, 'divisible_by_3'] = 1
        if i%7 == 0:
            filled_df.loc[filled_df['spread'] == i, 'divisible_by_7'] = 1
            filled_df.loc[filled_df['spread'] == i+0.5, 'divisible_by_7'] = 1
            filled_df.loc[filled_df['spread'] == i-0.5, 'divisible_by_7'] = 1
        if i%6 == 0:
            filled_df.loc[filled_df['spread'] == i, 'divisible_by_6'] = 1
            filled_df.loc[filled_df['spread'] == i+0.5, 'divisible_by_6'] = 1
            filled_df.loc[filled_df['spread'] == i-0.5, 'divisible_by_6'] = 1
            
    filled_df.loc[filled_df['divisible_by_3'] == 1, 'key_num'] = 1
    filled_df.loc[filled_df['divisible_by_6'] == 1, 'key_num'] = 1
    filled_df.loc[filled_df['divisible_by_7'] == 1, 'key_num'] = 1
    
    filled_df.loc[filled_df['key_num'] == 1, 'new_median'] = filled_df['spread']
    filled_df['new_median'] = filled_df['new_median'].fillna(filled_df['median'])
    
    return filled_df
    
    medians = dict()
    lasts = []
    n = 1
    while True:
        new_list = []
        lt_count = 0
        if n == 1:
            for i, j in zip(filled_df['new_median'], filled_df['new_median'].shift(1)):
                if i > j:
                    new_list.append(j)
                    lt_count = lt_count + 1
                else:
                    new_list.append(i)
        else:
            for i, j in zip(medians[n-1], medians[n-1][1:]):
                if i > j:
                    new_list.append(j)
                    lt_count = lt_count + 1
                else:
                    new_list.append(i)
        medians[n] = new_list
        n += 1
        #if n != 1:
        #    lasts.append(medians[n-1][-1])
        if lt_count == 0:
            break
    
    num = len(filled_df['new_median']) - len(medians[list(medians.keys())[-1]][1:])
    lasts = list(filled_df['new_median'])[-4:]
    new_meds = medians[list(medians.keys())[-1]][1:] + lasts

    filled_df['median'] = new_meds
    
    #return filled_df[['spread', 'median']]
    
    #max_table_number = spread_adjustment_table['spread'].max()
    #min_table_number = spread_adjustment_table['spread'].min()
    #spread_list = np.arange(max_table_number+0.5, abs(min_table_number)+0.5, 0.5)
    
    #new_medians = []
    #for i in spread_list:
    #    new_medians.append(-list(spread_adjustment_table.loc[spread_adjustment_table['spread'] == -i]['median'])[0])
    #extra_spreads = pd.DataFrame([spread_list, new_medians]).T
    #extra_spreads.columns = ['spread', 'median']
    #final_spread_adjustment_table = pd.concat([spread_adjustment_table, extra_spreads], axis=0)
    
    #return final_spread_adjustment_table
    
if __name__ == "__main__":
    
    spread_data = '/home/bill/ncaaf/etl/collect/collect_cfbd_games/cfbd_spread_data.csv'
    predictions = "/home/bill/ncaaf/batch_prediction/prediction_file/predictions.csv"
    spread_data, predictions = call_in_data(spread_data, predictions)
    edit_data(spread_data, 'homeScore', 'awayScore', 'homeSpread')
    df = create_spread_fitter(spread_data, 'homeSpread', predictions, 'home_score_differential')
    
    df.to_csv('test.csv')