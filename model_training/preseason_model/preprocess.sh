#!/usr/bin/env bash

aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/games.csv temp/games.csv
aws s3 cp s3://ncaaf-data/model_data/season_summaries.csv temp/season_summaries.csv
aws s3 cp s3://ncaaf-data/espn-pbp-data/pbp/csvs/pbp_edit.csv temp/pbp.csv

file=$(python -c "import glob; print(glob.glob('temp/preseason_experiment*')[0])")

SUB='fbs_only_ind'
while read p; do
if [[ "$p" == *"$SUB"* ]]; then
  FBS_ind=$(echo "$p"|awk -F ": " '{print $NF}')
fi
done < "$file"
#### if fbs ind = True, go to the summarize script, join the games file to the teams file,
#### and remove any games that contain a team that isn't in the FBS list
if [[ "$FBS_ind" == "True" ]]; then

  # read in teams
  aws s3 cp s3://ncaaf-data/espn-teams-data/teams.csv temp/teams.csv

  # transform teams and games data to only include fbs teams
  python -c "import pandas as pd;\
    games=pd.read_csv('temp/games.csv');\
    teams=pd.read_csv('temp/teams.csv');\
    teams = teams.loc[teams['fbs_ind'] == 1.0][['id', 'fbs_ind']];\
    teams.rename({'id':'team_id'}, axis=1, inplace=True);\
    home_merged = pd.merge(games, teams, left_on='home_team_id', right_on='team_id');\
    pd.merge(home_merged, teams, left_on='away_team_id', right_on='team_id').to_csv('temp/games.csv');"

  python -c "import pandas as pd;\
    pbp=pd.read_csv('temp/pbp.csv');\
    teams=pd.read_csv('temp/teams.csv');\
    teams = teams.loc[teams['fbs_ind'] == 1.0][['id', 'fbs_ind']];\
    teams.rename({'id':'team_id'}, axis=1, inplace=True);\
    pd.merge(pbp, teams, left_on='team_id', right_on='team_id').to_csv('temp/pbp.csv');"


  files_loc=$(pwd)
  games_loc=temp/games.csv
  pbp_loc=temp/pbp.csv
  teams_loc=temp/teams.csv

  cd ../..
  cd etl/summarize

  # need to change the file locations and need to output this to where I need it
  # that might involved editting the summarize_games.py file
  python summarize_games.py --games_file_loc /home/bill/ncaaf/model_training/preseason_model/temp/games.csv \
  --pbp_file_loc /home/bill/ncaaf/model_training/preseason_model/temp/pbp.csv \
  --output_file_loc /home/bill/ncaaf/model_training/preseason_model/temp/ \
  --summary_stats play_success yards_per_play

  cd ../..
  cd model_training/preseason_model

fi

#### if fbs ind = False then dont worry about it
python preprocess.py