import pandas as pd
import argparse
import pickle
import glob


# load pickled scaler and model
def load_relevant_files(file_location):

    # get the experiment file
    experiment_file = []
    for i in glob.glob("{}/*".format(file_location)):
        if 'preseason' in i.split(file_location)[1]:
            experiment_file.append(i)

    with open(experiment_file[0]) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'data_features' in line:
            features = data.split(',')

    # find the season summary file
    for i in glob.glob("{}/*".format(file_location)):
        if 'season_sum' in i.split(file_location)[1]:
            season_summary_file = i

    season_summary_file_location = season_summary_file

    model = pickle.load(open(file_location + '/model.pkl', 'rb'))
    scaler = pickle.load(open(file_location + '/scaler.pkl', 'rb'))
    season_summary_df = pd.read_csv(season_summary_file_location, index_col=0)

    return features, model, scaler, season_summary_df


# prepare prediction file
def echo_features(file_location):
    experiment_file = []
    for i in glob.glob("{}/*".format(file_location)):
        if 'preseason' in i.split(file_location)[1]:
            experiment_file.append(i)

    with open(experiment_file[0]) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'data_features' in line:
            features = data.split(',')

    print('\nThe model selected needs the following features:')
    for feature in features:
        print(feature)
    print('\n')


# merge the games that we want to predict with the last three years of sum stats
def merge_games_and_stats(games_df: pd.DataFrame,
                          stats_df: pd.DataFrame):

    games_df = games_df[['id', 'date', 'season', 'short_name', 'status',
                         'away_team_id', 'home_team_id']]
    away_df = pd.merge(games_df, stats_df, left_on=['away_team_id', 'season'], right_on=['team_id', 'season'])
    home_df = pd.merge(away_df, stats_df,
                       left_on=['home_team_id', 'season'],
                       right_on=['team_id', 'season'],
                       suffixes=('_away', '_home'))

    final_cols = []
    col_in_list = ['team_id_away', 'team_id_home', 'away_team_id', 'home_team_id',
                   'status', 'season', 'id', 'date', 'short_name']
    for col in list(home_df):
        if col in col_in_list:
            continue
        else:
            final_cols.append(col)

    return home_df[final_cols].dropna(), home_df.dropna()[col_in_list]


# Predict games
def predict_games(scaler,
                  model,
                  predict_file) -> pd.DataFrame:

    scaled_predict_data = scaler.transform(predict_file)
    predictions = model.predict(scaled_predict_data)

    predict_file['predictions'] = predictions

    return predict_file, predictions


# Take in selected model from bash file selection
def parse_args():

    parser = argparse.ArgumentParser(description='Point to correct files.')
    parser.add_argument('--model_file', help='Select model file to use.')
    parser.add_argument('--predict_dir', help='Select dir where predict file is.')
    parser.add_argument('--predict_file', help='Select file to predict.')

    args = parser.parse_args()
    return args.model_file, args.predict_dir, args.predict_file


if __name__ == '__main__':

    model_file, predict_dir, predict_file = parse_args()
    features, model, scaler, season_summary_df = load_relevant_files(model_file)
    games_df = pd.read_csv(predict_dir + predict_file, index_col=0)
    final_file_to_predict, games_df = merge_games_and_stats(games_df, season_summary_df)
    final_file_to_predict.to_csv(predict_dir + 'features_file.csv')
    predicted_df, predictions = predict_games(scaler=scaler,
                                              model=model,
                                              predict_file=final_file_to_predict)
    games_df['home_score_differential'] = predictions
    games_df.to_csv(predict_dir + 'predictions.csv')
