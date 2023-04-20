###### NCAAF predictions

##### Folders

#### batch_prediction

### prediction_file

# predict.py

predict.py has four functions, load_relevant_files, echo_features, merge_games_and_stats, and predict_games.

load_relevant_files takes one argument, file_location. The file location is the model_file. load_relevant_files returns features, model, scaler, season_summary_df.
features = the features used in the model
model = the model name / type used
scaler = the type of scaler used
season_summary_df = a summary of the season statistics gathered

After the load_relevant_files function is run. A games_df file is created from the predict.sh predict_dir and predict_file inputs.

merge_games_and_stats is then run and takes the games_df and season_summary_df as arguments. It returns the games_df as well as the final_file_to_predict_df.  The final_file_to_predict_df is built by merging the teams from the upcoming games to their stats.

# predict.sh

This is a shell script used to make batch predictions.  It requires the user to make a number of selections before it will give out a prediction. First it goes into the model training folder and asks a user to select a previously trained model by name. It then asks where the file is that we want to predict. Finally, it runs predict.py.

#### etl

# etl.sh

etl.sh is a shell script that runs all the collection, format, and summarization scripts.

### collect

## collect_espn_games

Pulls the games data from espn. The games data includes team names, scores, etc. More or less the meta data of the games.

# collect_espn_games.sh

Sets scraper in motion. Takes years as arguments. If no years given it scrapes from 2010 on. If one year is given it scrapes just that year. If two years are given it scrapes the range between the years.

It then saves data to s3.

dates are saved to s3://ncaaf-data/espn-games-data/dates/
urls are saved to s3://ncaaf-data/espn-games-data/apis/dates/
games are saved to s3://ncaaf-data/espn-games-data/games/csvs/
jsons are saved to s3://ncaaf-data/espn-games-data/games/jsons

# create_box_score_api_urls.py

Creates url apis using the following format:

date_prefix = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates='
date_suffix = '&limit=900'

# format_input_date.py

Takes a start date and an end date fed in from the collect_espn_games.sh file and creates a list.

# json_to_csv.py

Does all the formatting to turn the json into a usable csv file.

# retrieve_game_data.py

Loops through url list from create_box_score_api_urls.py scrapes in json data.

# run.py

Combines the .py files into one script, runs them, and saves data into a .csv file.

## collect_espn_pbp

Pulls the play by play data from ESPN.

# collect_espn_pbp.sh

Sets scraper in motion. Takes years as arguments. If no years given it scrapes from 2010 on. If one year is given it scrapes just that year. If two years are given it scrapes the range between the years.

It then saves data to s3.

dates are saved to s3://ncaaf-data/espn-pbp-data/dates/
urls are saved to s3://ncaaf-data/espn-pbp-data/apis/dates/
game ids are saved to s3://ncaaf-data/espn-pbp-data/gameids/
game id urls are saved to s3://ncaaf-data/espn-pbp-data/apis/gameids/
play by play csvs are saved to s3://ncaaf-data/espn-pbp-data/pbp/csvs/

# create_date_api_url_strings.py

The create_urls function takes three arguments; prefix, suffix, and data. Prefix and suffix are api urls. The data is from the date list that is generated.

# create_game_id_api_url_strings.py

create_urls function creates the urls that are going to pull the game_ids.

# date_generation.py

date_list_generation is the only function and all it does is generate a range of dates from two input dates.

# json_to_csv.py

Does all the formatting to turn the json into a usable csv file.

# retrieve_game_ids.py

Takes the URL list and scrapes the game ids from those URLs.

# retrieve_pbp_data.py

Takes the play by play URL list and scrapes the data from those URLs.

# run.py

Combines the .py files into one script, runs them, and saves data into a .csv file.

## collect_espn_teams

Collects the espn team names.

# collect_espn_teams.sh

Simple script that sets scraper in motion and saves the team names to s3://ncaaf-data/espn-teams-data/

# json_to_csv.py

Does all the formatting to turn the json into a usable csv file.

# retrieve_team_data.py

Takes the team names URL list and scrapes the data from those URLs.

# run.py

Combines the .py files into one script, runs them, and saves data into a .csv file.

### format

## format_espn_games

# create_api_urls.py

create_urls function creates urls using a predetermined suffix, prefix, and data.

# create_calendar.py

Similar to other run.py files. Combines all the individual files into one, runable script.

# format_espn_games.sh

Pulls down data from s3 buckets. Combines them all into one file. Put into temp/games.csv and then copied to s3://ncaaf-data/espn-games-data/games/csvs/

# format_input_date.py

Creates a list of years used in create_calendar.py.

# json_to_csv.py

Goes through the json file scraped from espn and converts it to a Pandas DataFrame and then to a csv file.

# merge_dfs.py

Merges the schedule/dates dataframe with the games.csv dataframe.

# retrieve_calendar_data.py

Scrapes calendar data from espn.com.

## format_espn_pbp

# edit_pbp_file.py

Take in play by play stats and use them to calculate new features.  Decode play type ids (listed in code). Calculate desired fields. Examples: yards per play, cumulative yards, play success rate.

# format_espn_pbp.sh

Pull all the seasons in and then run the files with run.py.

# run.py

Runs the edit_pbp_file and does the various other functions.

## format.sh

Runs the two format.sh files. Where you should input dates if I want new ones.

### summarize

# summarize.sh

Read in games from aws and run summarize_games.py

# summarize_games.py

NOTE: medians are used not averages.

read_and_edit_files takes 3 arguments. pbp_file_loc, games_file_loc, and summary_stats.  The first two are file locations.  summary_stats is a list of columns we want to keep in our final version.  The rest of the function is about merging those two files.

summarize_games takes two arguments, the previous merged df, called df, and statistics which is a list of summary stats.  First, find the max play of each game and get the sum of all their pbp statistics.  Do this for home O, home D, away O, and away D.  Each of these is then their own seperate df and returned.  This will be all game summaries.

season_summaries takes those game summaries from above and turns them into a team's cumulative season statistics.

game_summaries does the same thing except on a game by game basis.

rolling_game_summaries gets a rolling median. So game 1 is game 1's stats, game 2 would then have the median of 1 and 2. etc.

The season_summaries, game_by_game_summaries, and rolling_summaries are all saved to individual files.

#### flask_app
#### line_scraper
#### model_training

## preseason_model

# create_experiment.sh

Allows user to go through and make a custom model. Defines years to train, test years/size, features, and the type of algo to use.

# preprocess.py

read_data takes 3 arguments, games_df_file_loc, season_summary_df_file_loc, experiment_info_txt_file_loc.  It pulls these three files.

edit_files takes 3 arguments, games_df, season_summary_df, features, start_year, end_year.  Columns are selected as well as statuses and years.  New columns are added. The new columns, at least in the preseason model, are summarized stats from the previous seasons.  FY is the summarized data from the previous year. FY-1 is the summarized data from two years ago, FY-2 from 3 years ago, etc.

merge_games_and_stats takes two arguments, games_df and stats_df.  The games_df and their participants are merged to their respective stats.

split_data takes the data and the input from the create_experiment file and splits the data into a train and test split.

final_edits scales the data and breaks the x's and y's into seperate dfs.

the file outputs a train_X_scaled file, a test_X_scaled file, a train_y and test_y file, a train_id_cols and test_id_cols file.

# preprocess.sh

Pulls in games and season_summaries to be used in the preprocess.py file.

# train_model.py

import_data takes the preprocess data.

train_model trains the model on the train data using whatever the create_experiment file says the algorithm is going to be.

output_model_results outputs the model results and other pertinent files on the test data. 

# train_model.sh

Runs train_model.py.

#### scheduled_games

# scrape_scheduled_games.py 

date_list_generation creates a list of dates.

create_urls creates urls from the date list and and the schedule url.

retrieve_espn_game_data pulls the data from espn.

transform_espn_ncaaf_game_data transforms the json into usable data.


