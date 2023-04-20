#!/usr/bin/env bash

rm -r temp
rm seasons
mkdir temp
args=$#

# get the seasons we want
if [[ ${args} -eq 2 ]]; then
    for i in $(seq $1 $2)
    do
        echo ${i} >> temp/seasons
    done

elif [[ ${args} -eq 1 ]]; then
    echo ${1} >> temp/seasons
fi

# pull the columns of the games
SECOND_YEAR=$((${1} + 1))
aws s3 cp s3://ncaaf-data/espn-pbp-data/pbp/csvs/play-by-play_${1}-08-01_to_${SECOND_YEAR}-02-01.csv file.csv
COLS=$(head -n1 file.csv)
rm file.csv
LINES=$(wc -l < temp/seasons)

#pull all the column names
#cat temp/* > temp/games.csv
for i in $(seq 1 ${LINES}) #loop through years
do
    YEAR=$(sed -n ${i}p temp/seasons)
    SECOND_YEAR=$((YEAR + 1))
    aws s3 cp s3://ncaaf-data/espn-pbp-data/pbp/csvs/play-by-play_${YEAR}-08-01_to_${SECOND_YEAR}-02-01.csv \
        temp/pbp_${YEAR}.csv
    python3 run.py --input_file_path "temp/pbp_${YEAR}.csv" --output_file_path "temp/pbp_edit_${YEAR}.csv"
    head -1 temp/pbp_edit_${YEAR}.csv > temp/pbp_edit.csv
done

cd temp
files=(*)
tail -n +2 -q pbp_edit_*.csv >> pbp_edit.csv
header=(head -n 1 "$files")
sed -i $header pbp_edit.csv

aws s3 cp . s3://ncaaf-data/espn-pbp-data/pbp/csvs/ --recursive --exclude "*" --include "pbp_edit*"
cd ..

#rm -r temp
