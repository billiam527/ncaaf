#!/usr/bin/env bash

mkdir temp

python3 run.py

cd temp

echo saving files to s3
aws s3 sync . s3://ncaaf-data/espn-teams-data/ --quiet

cd ..
#rm -r temp
