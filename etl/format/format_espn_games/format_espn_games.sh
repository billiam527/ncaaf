#!/usr/bin/env bash

rm -r temp
mkdir temp
args=$#

if [ $args -eq 1 ]; then
  let 2=${1}+1
  years=$(seq ${1} ${2})
  files=$(aws s3 ls s3://ncaaf-data/espn-games-data/games/csvs/ | awk '{print $4}')
  for file in $files
    do
    for year in $years
      do
        if [[ $file == *$year* ]]; then
          aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/${file} temp/.
        fi
      done
    done

elif [ $args -eq 2 ]; then
  years=$(seq ${1} ${2})
  files=$(aws s3 ls s3://ncaaf-data/espn-games-data/games/csvs/ | awk '{print $4}')
  for file in $files
    do
    for year in $years
      do
        if [[ $file == *$year* ]]; then
          aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/${file} temp/.
        fi
      done
    done
fi

# combine games file into a single file
temp_files=$(ls temp)
for file in $temp_files
  do
    head -n 1 temp/${file} > temp/header.csv
    sed '1d' temp/${file} >> temp/data.csv
  done
cat temp/data.csv >> temp/header.csv
mv temp/header.csv temp/games.csv

# DATECOL will equal the column number of the 'date' column
DATECOL=$(awk -v RS=',' '/date/{print NR; exit}' temp/games.csv)

# take the games.csv file and pull out the date column with the DATECOL variable, creates dates file
cat temp/games.csv | cut -d ',' -f$DATECOL > temp/dates

# edit the dates file to only be years and pull out the min and max years
sed -i -e "1d" temp/dates # remove top row that currently says "date"
sed -i 's/-.*//' temp/dates # keep only the year
# define MIN and MAX YEAR variables to determine calendar dates to pull
MINYEAR=$(head -1 temp/dates)
MAXYEAR=$(tail -1 temp/dates)

# pull calendar data and merge to games data
python create_calendar.py --start_date ${MINYEAR}-01-01 --end_date ${MAXYEAR}-01-01
python merge_dfs.py --file1 temp/games.csv \
    --file2 temp/schedule_${MINYEAR}_to_${MAXYEAR}.csv  \
    --on 'date' \
    --how 'left' \
    --fillna 'bowl'

mv temp/new_games.csv temp/games.csv
aws s3 cp temp/games.csv s3://ncaaf-data/espn-games-data/games/csvs/

rm -r temp
