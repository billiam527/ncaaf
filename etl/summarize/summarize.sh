#!/usr/bin/env bash

rm -r temp
mkdir temp

# read in games and play by play data
aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/games.csv temp/games.csv
aws s3 cp s3://ncaaf-data/espn-pbp-data/pbp/csvs/pbp_edit.csv temp/pbp.csv

python summarize_games.py --games_file_loc temp/games.csv \
  --pbp_file_loc temp/pbp.csv \
  --summary_stats play_success yards_per_play

aws s3 cp temp/game_by_game_summaries.csv s3://ncaaf-data/model_data/game_by_game_summaries.csv
aws s3 cp temp/season_summaries.csv s3://ncaaf-data/model_data/season_summaries.csv
aws s3 cp temp/rolling_summaries.csv s3://ncaaf-data/model_data/rolling_summaries.csv
