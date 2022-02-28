#!/usr/bin/env bash

rm -r temp
mkdir temp
args=$#


if [ $args -eq 0 ]; then # if there are no arguments pull all years; -eq is equal to
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
aws s3 cp . s3://ncaaf-data/espn-games-data/dates/ --recursive --exclude "*" --include "dates_*" --quiet
aws s3 cp . s3://ncaaf-data/espn-games-data/apis/dates/ --recursive --exclude "*" --include "urls_*" --quiet
aws s3 cp . s3://ncaaf-data/espn-games-data/games/csvs/ --recursive --exclude "*" --include "games_*" --quiet
aws s3 sync gamejsons s3://ncaaf-data/espn-games-data/games/jsons --quiet

cd ..

rm -r temp