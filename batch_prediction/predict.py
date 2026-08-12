import pandas as pd
import numpy as np
import argparse
import pickle
import glob
import os
import shutil


def estimate_neutral_site_adjustment(history_loc='../analysis/backtest_expanding_preds.csv',
                                     default=-3.0):
    """Points to add to a neutral-site prediction.

    Neither model has a neutral_site feature, so both apply their learned home
    advantage to games where nobody is home. Measured on walk-forward history,
    the residual on neutral games is about -2.9 points, and the home-field
    advantage implied by the raw data (true-home mean minus neutral mean) is
    +3.3 - two independent estimates of the same quantity.

    Because the effect is an additive shift on a binary flag, correcting here is
    equivalent to giving the models the feature, without retraining them and
    regenerating the blend weights. It is shrunk toward the raw home-field
    estimate because the residual sample is small (n=140).
    """
    try:
        h = pd.read_csv(history_loc)
        h = h[h['week_num'] < 90]
        games = pd.read_csv('../etl/summarize/temp/games.csv', low_memory=False)
        h = h.merge(games[['id', 'neutral_site']].drop_duplicates('id'),
                    on='id', how='left')
        neutral = h[h['neutral_site'] == 1]
        neutral = neutral.dropna(subset=['in_season_model_preds',
                                         'home_score_differential'])
        if len(neutral) < 40:
            return default
        resid = (neutral['home_score_differential']
                 - neutral['in_season_model_preds']).mean()
        # shrink halfway toward the better-powered raw home-field estimate
        return float(np.clip((resid + default) / 2.0, -6.0, 0.0))
    except Exception as exc:
        print(f"  could not estimate neutral-site adjustment ({exc}); "
              f"using {default:+.1f}")
        return default


def apply_neutral_site_adjustment(preds_df, source_games, adjustment, columns):
    """Remove the home-field component from neutral-site games."""
    # The flag may already be present if the caller merged the scheduled games;
    # merging it again would collide into neutral_site_x / neutral_site_y.
    if 'neutral_site' not in preds_df.columns:
        if 'neutral_site' not in source_games.columns:
            print("  no neutral_site column available; skipping the adjustment")
            return preds_df
        flags = source_games[['id', 'neutral_site']].drop_duplicates('id')
        preds_df = preds_df.merge(flags, on='id', how='left')

    mask = preds_df['neutral_site'].fillna(0).astype(bool)

    if mask.any():
        for col in columns:
            if col in preds_df.columns:
                preds_df.loc[mask, col] = preds_df.loc[mask, col] + adjustment
        print(f"  neutral-site adjustment {adjustment:+.2f} applied to "
              f"{int(mask.sum())} game(s)")
    else:
        print("  no neutral-site games in this slate")
    return preds_df


def is_season_live(games_df: pd.DataFrame, season: int) -> bool:
    """True once `season` has at least one completed game.

    In-season features describe how teams have actually performed so far, so
    before the first kickoff there is nothing to build them from and only the
    preseason model can run. Checking for a completed game rather than a date
    avoids hard-coding when the season starts.
    """
    if games_df is None or games_df.empty:
        return False
    if 'season' not in games_df.columns or 'status' not in games_df.columns:
        return False

    played = games_df[(games_df['season'] == season) &
                      (games_df['status'] == 'STATUS_FINAL')]
    return not played.empty


# load pickled scaler and model
def load_relevant_files(preseason_model_file, post_season_model_file):
    
    #### go to the preseason file folder
    # grab the experiment file
    # grab the data
    ### go to the in season file folder
    # ditto
    # go grab the season summaries file
    # go grab the rolling summaries file
    

    # get the experiment file
    experiment_file = []
    for i in glob.glob("{}/*".format(preseason_model_file)):
        if 'preseason' in i.split(preseason_model_file)[1]:
            experiment_file.append(i)

    # preseason model file   
    with open(experiment_file[0]) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'data_features' in line:
            features = data.split(',')
            
    preseason_features = features
    preseason_model = pickle.load(open(preseason_model_file + '/model.pkl', 'rb'))
    preseason_scaler = pickle.load(open(preseason_model_file + '/scaler.pkl', 'rb'))
            
    # get the experiment file
    experiment_file = []
    for i in glob.glob("{}/*".format(in_season_model_file)):
        if 'in_season' in i.split(in_season_model_file)[1]:
            experiment_file.append(i)

    # IN SEASON
    with open(experiment_file[0]) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'data_features' in line:
            features = data.split(',')
            
    in_season_features = features
    in_season_model = pickle.load(open(in_season_model_file + '/model.pkl', 'rb'))
    in_season_scaler = pickle.load(open(in_season_model_file + '/scaler.pkl', 'rb'))
    
    # summarize/temp/ is the pipeline's input staging area; the summaries are
    # its output and are written to summarize/results/.
    other_data_location = '../etl/summarize/results/'
    season_summary_df = pd.read_csv(other_data_location + 'season_summaries.csv')
    rolling_summary_df = pd.read_csv(other_data_location + 'rolling_summaries.csv')

    return preseason_features, preseason_model, preseason_scaler, \
        in_season_features, in_season_model, in_season_scaler, \
        season_summary_df, rolling_summary_df


# prepare prediction file
def echo_features(file_location, file_type):
    experiment_file = []
    for i in glob.glob("{}/*".format(file_location)):
        if file_type in i.split(file_location)[1]:
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


def edit_files(season_summary_df: pd.DataFrame,
               features: list,
               start_year: int,
               end_year: int
               ):

    # some smaller schools might not have a full history w ESPN, give them each a year column even if its NA for now
    # goes from...
    # team_id, szn, features
    # 1, 2010, 123
    # 1, 2011, 123
    # 1, 2013, 123
    # to...
    # team_id, szn, features
    # 1, 2010, 123
    # 1, 2011, 123
    # 1, 2012, NA
    # 1, 2013, 123

    # season summaries column edits
    season_summaries = season_summary_df[['team_id', 'season'] + features]

    # +1 would get us to 2022, +2 will get us to the next year we want to predict
    years = list(range(start_year, end_year+2))
    teams = []
    new_years = []
    for team in list(season_summaries['team_id'].unique()):
        for year in years:
            teams.append(team)
            new_years.append(int(year))

    teams_and_szns = pd.DataFrame(zip(teams, new_years), columns=['team_id', 'season'])

    season_summaries_add_years_edit = pd.merge(season_summaries, teams_and_szns, how='right',
                                               left_on=['team_id', 'season'],
                                               right_on=['team_id', 'season']
                                               )

    ss_edit = season_summaries_add_years_edit

    # Fill cols for prev years

    # Shift all columns
    # FY is the summarized data from the previous year
    # FY-1 is the summarized data from two years ago, FY-2 from 3 years ago
    # Example
    # 2018
    # 2019
    # 2020 FY would be summarized 2019 data, FY-1 2018, FY-2 2017
    # 2021 FY would be 2020 summarized data, FY-1 2019, FY-2 2018
    for i in [1, 2, 3]:
        for feature in features:
            if i == 1:
                ss_edit[feature + '_FY'] = season_summaries_add_years_edit.groupby('team_id')[feature].shift(i)
            else:
                ss_edit[feature + '_FY-{}'.format(i - 1)] = season_summaries_add_years_edit.groupby('team_id')[feature].shift(i)

    ss_edit.drop(features, axis=1, inplace=True)
    ss_edit = ss_edit.loc[ss_edit['season'] == end_year + 1]

    return ss_edit

RETURNING_FILE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'etl', 'summarize', 'results', 'returning_production.csv'))

# Kept in step with model_training/preseason_model/preprocess.py. If the two
# lists drift apart the scaler and the prediction matrix stop matching, which
# surfaces as a width mismatch rather than anything informative.
RETURNING_FEATURES = ['ret_QB_starter', 'ret_RB_starter', 'ret_WR_starter',
                      'ret_TE_starter', 'ret_defense', 'ret_good', 'ret_bad']


def add_returning_production(stats_df: pd.DataFrame,
                             path: str = RETURNING_FILE) -> pd.DataFrame:
    """Join returning production onto per-season stats, without lagging.

    The lagged stats are shifted because a statistic describes the season it
    was recorded in. Returning production is already aligned to the season being
    predicted, so it merges straight onto (team_id, season).
    """
    if not os.path.exists(path):
        print(f"  returning production not found at {path}; skipping")
        return stats_df

    ret = pd.read_csv(path, low_memory=False)
    have = [c for c in RETURNING_FEATURES if c in ret.columns]
    if not have:
        return stats_df

    ret = ret[['team_id', 'season'] + have].drop_duplicates(['team_id', 'season'])
    return stats_df.merge(ret, on=['team_id', 'season'], how='left')


# merge the games that we want to predict with the last three years of sum stats
def merge_games_and_stats(games_df: pd.DataFrame,
                          stats_df: pd.DataFrame):

    games_df = games_df[['id', 'week', 'date', 'season', 'short_name', 'status',
                         'away_team_id', 'home_team_id']]

    away_df = pd.merge(games_df, stats_df, 
                       left_on=['away_team_id', 'season'], 
                       right_on=['team_id', 'season'])
    
    home_df = pd.merge(away_df, stats_df,
                       left_on=['home_team_id', 'season'],
                       right_on=['team_id', 'season'],
                       suffixes=('_away', '_home'))

    final_cols = []
    col_in_list = ['team_id_away', 'team_id_home', 'away_team_id', 'home_team_id',
                   'status', 'season', 'id', 'week', 'date', 'short_name']
    for col in list(home_df):
        if col in col_in_list:
            continue
        else:
            final_cols.append(col)

    return home_df[final_cols].dropna(), home_df.dropna()[col_in_list]


def is_merge_games_and_stats(games_df: pd.DataFrame,
                             stats_df: pd.DataFrame,
                             is_features: list):
    
    # Mirror how the in-season model was trained. See
    # model_training/in_season_model/preprocess.py::edit_files - it merges
    # season-level adjusted stats on team_id + season with no date component.
    # This previously joined on date as well and looked for
    # <feature>_rolling_median_* columns, which nothing in the pipeline has ever
    # produced, so the join could not succeed and the trained model would have
    # been fed a differently-shaped frame even if it had.
    games_df = games_df[['id', 'week', 'date', 'season', 'short_name', 'status',
                     'away_team_id', 'home_team_id']]

    stats_cols = ['season', 'team_id'] + [f for f in is_features if f in stats_df.columns]
    stats_df = stats_df.reset_index(drop=True)[stats_cols]

    away_df = pd.merge(games_df, stats_df,
                       left_on=['away_team_id', 'season'],
                       right_on=['team_id', 'season'])

    home_df = pd.merge(away_df, stats_df,
                       left_on=['home_team_id', 'season'],
                       right_on=['team_id', 'season'],
                       suffixes=('_away', '_home'))

    cols = ['id', 'date', 'week', 'short_name',
            'season', 'status',
            'home_team_id', 'away_team_id']

    feature_cols = []
    for i in is_features:
        feature_cols.append(i + '_home')
        feature_cols.append(i + '_away')

    missing = [c for c in feature_cols if c not in home_df.columns]
    if missing:
        raise KeyError(
            f"in-season merge is missing {len(missing)} model feature(s), "
            f"e.g. {missing[:3]}. season_summaries.csv must carry every column "
            f"listed as data_features in the model's experiment file."
        )

    home_df = home_df[cols + feature_cols]

    return home_df.dropna(), home_df.dropna()[feature_cols]
    
# Predict games
def predict_games(scaler,
                  model,
                  predict_file) -> pd.DataFrame:

    # A scaler is positional: it centres column i against the mean and variance
    # it saw at column i during fit. sklearn raises on a name mismatch when
    # handed a named DataFrame, but only warns on a bare array - so check here
    # and report exactly what is wrong instead of silently mis-scaling.
    expected = list(getattr(scaler, 'feature_names_in_', []))
    if expected and hasattr(predict_file, 'columns'):
        actual = list(predict_file.columns)
        if actual != expected:
            missing = [c for c in expected if c not in actual]
            extra = [c for c in actual if c not in expected]
            if missing or extra:
                raise ValueError(
                    f"Feature mismatch for {type(model).__name__}: "
                    f"{len(missing)} missing {missing[:3]}, "
                    f"{len(extra)} unexpected {extra[:3]}. "
                    f"Expected {len(expected)} features, got {len(actual)}.")
            first = next(i for i, (a, b) in enumerate(zip(actual, expected))
                         if a != b)
            raise ValueError(
                f"Feature columns are correct but ordered wrong, and the "
                f"scaler is positional. First difference at index {first}: "
                f"got '{actual[first]}', expected '{expected[first]}'.")

    scaled_predict_data = scaler.transform(predict_file)
    predictions = model.predict(scaled_predict_data)

    predict_file['predictions'] = predictions

    return predict_file, predictions


# Take in selected model from bash file selection
def parse_args():

    parser = argparse.ArgumentParser(description='Point to correct files.')
    parser.add_argument('--preseason_model_file', help='Select model file to use.')
    parser.add_argument('--in_season_model_file', help='Select model file to use.')
    parser.add_argument('--predict_dir', help='Select dir where predict file is.')
    parser.add_argument('--predict_file', help='Select file to predict.')
    parser.add_argument('--model_blender', help='True or False')

    args = parser.parse_args()
    return args.preseason_model_file, args.in_season_model_file, \
        args.predict_dir, args.predict_file, args.model_blender


if __name__ == '__main__':

    preseason_model_file, in_season_model_file, \
        predict_dir, predict_file, model_blender = parse_args()        
        
    preseason_features, preseason_model, preseason_scaler, \
        in_season_features, in_season_model, in_season_scaler, \
        season_summary_df, rolling_summary_df = load_relevant_files(preseason_model_file, 
                                                                     in_season_model_file)
    
    if model_blender == "False":
        # The games to predict come from the scheduled-games file produced by
        # scrape_scheduled_games.py, not from the historical games table.
        games_df = pd.read_csv(predict_dir + predict_file, index_col=0)
        target_season = int(games_df['season'].mode().iloc[0])
        print(f"Predicting {len(games_df)} games for the {target_season} season")

        # Preseason features come from the three seasons before the target one;
        # edit_files shifts them forward, so it is filtered to end_year + 1.
        ss_edit = edit_files(season_summary_df=season_summary_df,
                                 features=preseason_features,
                                 start_year=(target_season - 1) - 3,
                                 end_year=target_season - 1)
        ss_edit = add_returning_production(ss_edit)

        # preseason model
        final_file_to_predict, games_df = merge_games_and_stats(games_df, ss_edit)
        final_file_to_predict.to_csv(predict_dir + 'features_file.csv')
        predicted_df, predictions = predict_games(scaler=preseason_scaler,
                                                  model=preseason_model,
                                                  predict_file=final_file_to_predict)

        ##########################################################################
        # in season model, is = in_season
        # In-season features describe results so far this season, so they only
        # exist once the season is under way. Before kickoff there is nothing to
        # build them from and the preseason model carries the prediction alone.
        is_cols = ['id', 'date', 'week', 'short_name', 'season',
                   'home_team_id', 'away_team_id', 'in_season_model_preds']

        historical_games_df = pd.read_csv('../etl/summarize/temp/games.csv',
                                          low_memory=False)
        season_live = is_season_live(historical_games_df, target_season)

        if season_live:
            features = in_season_features
            is_stats = season_summary_df.reset_index()
            is_stats = is_stats.loc[is_stats['season'] == target_season]
            is_games_df, is_final_file_to_predict = is_merge_games_and_stats(
                games_df, is_stats, features)
            is_final_file_to_predict.to_csv(predict_dir + 'in_season_features_file.csv')
            try:
                is_predicted_df, is_predictions = predict_games(scaler=in_season_scaler,
                                                                model=in_season_model,
                                                                predict_file=is_final_file_to_predict)
                is_games_df['in_season_model_preds'] = is_predictions
            except ValueError as a:
                print(f"In-season prediction failed: {a}")
                is_games_df['in_season_model_preds'] = np.nan
            is_games_df = is_games_df[is_cols]
        else:
            print(f"Season {target_season} has not started yet "
                  f"(no completed games) - skipping the in-season model; "
                  f"preseason predictions only")
            is_games_df = games_df.copy()
            is_games_df['in_season_model_preds'] = np.nan
            is_games_df = is_games_df[[c for c in is_cols if c in is_games_df.columns]]
        ##########################################################################
        games_df['preseason_model_preds'] = predictions
        preds_df = pd.merge(games_df, is_games_df, on='id', how='left',
                            suffixes=('', '_is'))
        # Join the predictions back onto the full slate being predicted. This
        # read historical games for a hard-coded 2023 season, so the left join
        # matched no ids against the scheduled games and every prediction came
        # back null.
        non_pred_games = pd.read_csv(predict_dir + predict_file, index_col=0)
        full_df = pd.merge(non_pred_games, preds_df, how='left', on='id',
                           suffixes=('', '_pred'))

        # Neither model can see whether a game is at a neutral site, so both
        # apply their learned home advantage to games with no home team.
        neutral_adj = estimate_neutral_site_adjustment()
        full_df = apply_neutral_site_adjustment(
            full_df, non_pred_games, neutral_adj,
            columns=['preseason_model_preds', 'in_season_model_preds'])

        cols = ['id', 'away_team_id', 'home_team_id', 'date', 'week', 'short_name',
                'neutral_site', 'preseason_model_preds', 'in_season_model_preds']
        full_df = full_df[[c for c in cols if c in full_df.columns]]
        full_df.to_csv(predict_dir + 'predictions.csv')
    else:
        # Build the historical per-week files that model_blender.py reads to fit
        # its per-week blend weights.
        #
        # This is delegated to walk_forward so the in-season features for week W
        # are built only from games played before week W. The previous version
        # applied full-season adjusted stats to every week, which leaked results
        # backwards and biased the learned weights towards the in-season model.
        # The season list was also hard-coded to 2016-2023; it now follows
        # whatever completed seasons are present in the data.
        import shutil

        import walk_forward as WF

        completed = pd.read_csv('../etl/summarize/temp/games.csv', low_memory=False)
        completed = completed[completed['home_score_differential'].notna()]
        # A season needs the three prior seasons for preseason features, plus a
        # few seasons of history to train on.
        available = sorted(int(s) for s in completed['season'].dropna().unique())
        years = [y for y in available if (y - 6) >= min(available)]
        print(f"Generating blender training data for seasons: {years}")

        # Each season is predicted by models retrained on seasons before it.
        # The models passed on the command line are deliberately NOT used here:
        # they were trained on these very seasons, so their predictions would be
        # recall rather than prediction (0.98 correlation with the results they
        # were fit on), and the blend weights learned from that data load up on
        # whichever model memorised more. reference_model_dir only supplies the
        # feature list. This is cached; pass force=True to rebuild.
        preds = WF.generate_expanding_predictions(
            years, reference_model_dir=in_season_model_file)

        shutil.rmtree('temp', ignore_errors=True)
        os.makedirs('temp')

        cols = ['id', 'away_team_id', 'home_team_id', 'date', 'week', 'short_name',
                'home_score_differential', 'preseason_model_preds',
                'in_season_model_preds']
        written = 0
        for season, g in preds.groupby('test_season'):
            g = g[[c for c in cols if c in g.columns]]
            for wk in sorted(g['week'].dropna().unique()):
                g.loc[g['week'] == wk].to_csv(f'temp/{wk}_{int(season)}.csv')
                written += 1
            print(f"  {int(season)}: {len(g):>4} games across {g['week'].nunique():>2} weeks")
        print(f"Wrote {written} per-week files to temp/ for model_blender.py")
