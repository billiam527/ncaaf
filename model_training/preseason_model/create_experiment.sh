#!/usr/bin/env bash

## Create file to record model specifics and results ##
rm -r temp
mkdir temp

now=$(date +"%Y_%m_%d_%H_%m_%M_%S")
file="preseason_experiment_$now.txt"

touch temp/$file
echo $now >> temp/$file

## specify experiment specifics ##
aws s3 cp s3://ncaaf-data/model_data/season_summaries.csv temp/season_summaries.csv

# define train years
printf "\n"
printf "Select years to train the model on\n"
read -p "start (2022 will pull in the 2022-2023 season) train season (YYYY): " start_year
read -p "end (2022 will pull in the 2022-2023 season) train season (YYYY): " end_year
# shellcheck disable=SC2086
echo train_season_range: "${start_year}" - ${end_year} >> temp/$file

# test size or test year
printf "\n"
printf "select a test size (decimal) or select a year (int) to hold out as the test year \n"
read -p "when selecting a year select beginning year of the season (ex 2021 will select the 2021-2022 season): " test
if [[ $test =~ ^[+-]?[0-9]+$ ]]; then
echo "Input is a year."
echo test_year: "${test}" >> temp/"$file"
fi

if [[ $test =~ ^[+-]?[0-9]*\.[0-9]+$ ]]; then
echo "Input is a decimal."
echo test_size: "${test}" >> temp/"$file"
fi

# select features to train on
printf "\nselect features to train on and select FEATURES_SELECTED when done\n"

# take features from the header of season sums
features=$(head -n 1 temp/season_summaries.csv)

# only take features after the second , (team_id, and season)
cut_features=$(cut -d',' -f3- <<< "$features")
for i in ${cut_features//,/ }
do
  data_features="${data_features} $i"
done
features="${data_features} FEATURES_SELECTED"

PS3="Enter a number: "
select feature in $features
do
    if [[ $feature != FEATURES_SELECTED ]]; then
        echo "selected feature: $feature"
        selected_features="${selected_features},$feature"
    else
        break
    fi
done
echo data_features: ${selected_features} | sed s/,// >> temp/$file

# FBS filter
printf '\n'
while true; do
    read -p "Filter out non-FBS teams? (answer y/n) " yn
    case $yn in
        # join games to teams/fbs_ind and filter if yes
        [Yy]* ) echo "FBS teams will be filtered out";
          echo "fbs_only_ind: True" >> temp/$file;
          break;;
        [Nn]* ) echo "FBS teams will not be filtered out";
          echo "fbs_only_ind: False" >> temp/$file;
          break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# algo
printf '\n'
echo choose training algorithm
PS3="Enter a number: "
select algo in linear_regression
do
    echo "selected algorithm: $algo"
    echo "training_algorithm: $algo" >> temp/$file
    break
done
printf '\n'

./preprocess.sh
./train_model.sh
