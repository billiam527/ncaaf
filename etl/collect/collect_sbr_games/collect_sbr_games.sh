#!/usr/bin/env bash

rm -r temp
mkdir temp
args=$#

python3 collect_sbr_games.py

cd temp

echo -e "\tSaving files to s3"
aws s3 cp . s3://ncaaf-data/sbr-data/spreads/csvs/ --recursive --exclude "*" --include "sbr_*" --quiet

cd ..