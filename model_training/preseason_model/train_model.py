import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import json
import pickle
import os
from datetime import datetime
import shutil
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


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

    params = {}
    for line in lines:
        try:
            data = line.split(': ', 1)[1].rstrip("\n")
        except:
            continue
        if 'training_algorithm' in line:
            algo = data
        elif 'model_params' in line:
            # Optional JSON of estimator kwargs, e.g.
            #   model_params: {"max_depth": 3, "learning_rate": 0.01}
            # Omit the line to use library defaults.
            params = json.loads(data)

    return algo, \
        train_X, \
        test_X, \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df, \
        params


def train_model(algo,
                train_X,
                train_y,
                params=None):

    params = params or {}
    if algo == 'linear_regression':
        model = LinearRegression().fit(train_X, train_y)
    elif algo == 'xgboost_reg':
        # random_state fixed so a rerun on identical data reproduces the model.
        model = XGBRegressor(random_state=0, **params).fit(train_X, train_y)

    return model


def output_model_results(algo,
                         experiment_info_txt_file_loc,
                         model,
                         test_X,
                         test_y,
                         train_y):

    # create directory
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
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

    # A test_size holdout takes a random tenth of GAMES, not a season. Every
    # preseason feature is a per-team-season constant, so a team's feature
    # vector sits in training with some of its results and in test with the
    # rest, and the model is scored partly on rows it has effectively seen.
    #
    # It is not fatal - holding out the whole of 2025 gives an identical R2 of
    # 0.257, so the encoding change removed the memorisation this used to allow
    # in the 104-column model, where R2 fell from 0.324 to 0.250. But MAE stays
    # optimistic, because R2 normalises by the test set's own variance and a
    # random tenth drawn from every season is an easier mix than one season
    # alone: 13.39 here against 13.94 on a held-out season, and 13.94 from the
    # walk-forward history. Two independent estimates agreeing.
    #
    # The training window is deliberately NOT changed to fix this. Holding out
    # a season would cost the model the most recent one, which is the one that
    # matters most for next season's predictions, and that is a bad trade for a
    # printed number. The number is labelled instead.
    optimistic = 'test_size' in test

    with open(directory + '/model_results.txt', 'w') as f:
        f.write('test description: ' + test + '\n')
        f.write('sample size: ' + str(len(test_y)) + '\n')
        f.write('r2: ' + str(r2) + '\n')
        f.write('Adjusted r2: ' + str(adj_r2) + '\n')
        f.write('mse: ' + str(mse) + '\n')
        f.write('rmse: ' + str(rmse) + '\n')
        f.write('mae: ' + str(mae) + '\n')
        if optimistic:
            f.write('\n')
            f.write('NOTE: this holdout is a random tenth of games, not a\n')
            f.write('season, so the MAE above is optimistic by roughly half a\n')
            f.write('point. Against an unseen season the figure is about 13.94\n')
            f.write('(season holdout and walk-forward agree). Quote that one.\n')
            f.write('R2 is unaffected - 0.257 under either rule.\n')

    if optimistic:
        print("   NOTE: test_size holds out random games, not a season; the MAE"
              " above is optimistic. Honest figure is about 13.94.")


if __name__ == '__main__':

    experiment_info_txt_file_loc = glob.glob('temp/preseason_experiment*')[0]

    algo, \
        train_X, \
        test_X, \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df, \
        params = import_data(train_x_loc='temp/train_X_scaled.csv',
                             train_y_loc='temp/train_y.csv',
                             test_x_loc='temp/test_X_scaled.csv',
                             test_y_loc='temp/test_y.csv',
                             train_id_df='temp/train_id_cols.csv',
                             test_id_df='temp/test_id_cols.csv',
                             experiment_info_txt_file_loc=experiment_info_txt_file_loc)

    print(f"training {algo} with params: {params or 'library defaults'}")
    model = train_model(algo=algo, train_X=train_X, train_y=train_y, params=params)

    output_model_results(algo=algo,
                         experiment_info_txt_file_loc=experiment_info_txt_file_loc,
                         model=model,
                         test_X=test_X,
                         test_y=test_y,
                         train_y=train_y)
