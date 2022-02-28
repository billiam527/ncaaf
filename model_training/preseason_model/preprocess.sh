#!/usr/bin/env bash

aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/games.csv temp/games.csv
aws s3 cp s3://ncaaf-data/model_data/season_summaries.csv temp/season_summaries.csv

python preprocess.py