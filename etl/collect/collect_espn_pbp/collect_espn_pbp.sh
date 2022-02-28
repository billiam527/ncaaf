#!/bin/bash

rm -r temp
mkdir temp
args=$#

if [ $args -eq 0 ]; then # if there are no arguments; -eq is equal to
    echo -e "\tRunning Python collection scripts"
    for i in $(seq 2010 $(date +"%Y"))
    do
        let SECONDYEAR=${i}+1
        START_DATE=${i}-08-01
        END_DATE=${SECONDYEAR}-02-01
        echo -e "\tScraping from ${START_DATE} to ${END_DATE}"
        python3 run.py --start_date $START_DATE --end_date $END_DATE
  done

elif [ $args -eq 1 ]; then # if there is one argument, just scrape that one year
    echo -e "\tRunning Python collection scripts"
    let SECONDYEAR=$(($1 + 1))
    START_DATE=$1-08-01
    END_DATE=${SECONDYEAR}-02-01
    echo -e "\tScraping from ${START_DATE} to ${END_DATE}"
    python3 run.py --start_date $START_DATE --end_date $END_DATE

elif [ $args -eq 2 ]; then # if there are two arguments, scrape that range
    echo -e "\tRunning Python collection scripts"
    for i in $(seq $1 $2)
    do
        let SECONDYEAR=$(($i + 1))
        START_DATE=${i}-08-01
        END_DATE=${SECONDYEAR}-02-01
        echo -e "\tScraping from ${START_DATE} to ${END_DATE}"
        python3 run.py --start_date $START_DATE --end_date $END_DATE
    done

else
    echo too many arguments

fi

cd temp

echo -e "\tSaving files to s3"
aws s3 cp . s3://ncaaf-data/espn-pbp-data/dates/ --recursive --exclude "*" --include "dates_*" --quiet
aws s3 cp . s3://ncaaf-data/espn-pbp-data/apis/dates/ --recursive --exclude "*" --include "date_urls*" --quiet
aws s3 cp . s3://ncaaf-data/espn-pbp-data/gameids/ --recursive --exclude "*" --include "game_ids*" --quiet
aws s3 cp . s3://ncaaf-data/espn-pbp-data/apis/gameids/ --recursive --exclude "*" --include "gameid_urls*" --quiet
aws s3 cp . s3://ncaaf-data/espn-pbp-data/pbp/csvs/ --recursive --exclude "*" --include "play-by-play*" --quiet
aws s3 sync pbpjsons s3://ncaaf-data/espn-pbp-data/pbp/jsons --quiet

cd ..
rm -r temp
