import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import pickle
import os
from datetime import datetime
import shutil
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error


def import_data(train_x_loc,
                train_y_loc,
                test_x_loc,
                test_y_loc,
                train_id_df,
                test_id_df,
                experiment_info_txt_file_loc):

    train_X = pd.read_csv(train_x_loc, index_col=0)
    test_X = pd.read_csv(test_x_loc, index_col=0)
    train_y = pd.read_csv(train_y_loc, index_col=0)
    test_y = pd.read_csv(test_y_loc, index_col=0)
    train_id_df = pd.read_csv(train_id_df, index_col=0)
    test_id_df = pd.read_csv(test_id_df, index_col=0)

    with open(experiment_info_txt_file_loc) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'training_algorithm' in line:
            algo = data

    return algo, \
        train_X, \
        test_X, \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df


def train_model(algo,
                train_X,
                train_y):

    if algo == 'linear_regression':
        model = LinearRegression().fit(train_X, train_y)
    #elif algo == 'random_for'

    return model


def output_model_results(algo,
                         experiment_info_txt_file_loc,
                         model,
                         test_X,
                         test_y,
                         train_y):

    # create directory
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    directory = algo + '_' + dt_string
    os.mkdir(directory)

    # save pertinent files
    shutil.copy2(experiment_info_txt_file_loc, directory)
    shutil.copy2('temp/scaler.pkl', directory)
    shutil.copy2('temp/train_X_scaled.csv', directory)
    shutil.copy2('temp/season_summaries_add_years_edit.csv', directory)
    filename = 'model.pkl'
    pickle.dump(model, open(directory + '/' + filename, 'wb'))

    # run and save model results
    preds = model.predict(test_X)

    test_y = test_y.values
    test_id_df['result'] = test_y
    test_id_df['prediction'] = preds

    test_id_df.to_csv(directory + '/results.csv')

    # evaluate model results
    r2 = r2_score(list(test_y), list(preds))  # r2
    adj_r2 = 1 - (1 - r2) * (len(test_y) - 1) / (len(test_y) - len(list(train_X)) - 1)  # adjusted r2
    mse = mean_squared_error(test_y, preds)  # mse
    rmse = math.sqrt(mean_squared_error(test_y, preds))  # rmse
    mae = mean_absolute_error(test_y, preds)  # mae

    # save these results to a text file for interpretation
    with open(experiment_info_txt_file_loc) as f:
        lines = f.readlines()

    for line in lines:
        if 'test_size' in line:
            test = line
            continue
        elif 'test_year' in line:
            test = line

    with open(directory + '/model_results.txt', 'w') as f:
        f.write('test description: ' + test + '\n')
        f.write('sample size: ' + str(len(test_y)) + '\n')
        f.write('r2: ' + str(r2) + '\n')
        f.write('Adjusted r2: ' + str(adj_r2) + '\n')
        f.write('mse: ' + str(mse) + '\n')
        f.write('rmse: ' + str(rmse) + '\n')
        f.write('mae: ' + str(mae) + '\n')


if __name__ == '__main__':

    experiment_info_txt_file_loc = glob.glob('temp/preseason_experiment*')[0]

    algo, \
        train_X, \
        test_X, \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df = import_data(train_x_loc='temp/train_X_scaled.csv',
                                 train_y_loc='temp/train_y.csv',
                                 test_x_loc='temp/test_X_scaled.csv',
                                 test_y_loc='temp/test_y.csv',
                                 train_id_df='temp/train_id_cols.csv',
                                 test_id_df='temp/test_id_cols.csv',
                                 experiment_info_txt_file_loc=experiment_info_txt_file_loc)

    model = train_model(algo=algo, train_X=train_X, train_y=train_y)

    output_model_results(algo=algo,
                         experiment_info_txt_file_loc=experiment_info_txt_file_loc,
                         model=model,
                         test_X=test_X,
                         test_y=test_y,
                         train_y=train_y)
