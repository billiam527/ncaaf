#!/usr/bin/env bash

## Usage: ./create_experiment.sh [--search]
##   --search  re-derive the estimator hyperparameters with a nested search
##             (~10 min). Without it the last known-good params are carried
##             forward, so a routine retrain never silently reverts to library
##             defaults - which on this data cost about 1.1 MAE.
SEARCH_PARAMS=false
for arg in "$@"; do
    case $arg in
        --search) SEARCH_PARAMS=true ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--search]"; exit 1 ;;
    esac
done

## Create file to record model specifics and results ##
rm -rf temp
mkdir temp

now=$(date +"%Y_%m_%d_%H_%m_%M_%S")
file="in_season_experiment_$now.txt"

touch temp/$file
echo $now >> temp/$file

## specify experiment specifics ##
aws s3 cp s3://ncaaf-data/model_data/season_summaries.csv temp/season_summaries.csv
#aws s3 cp s3://ncaaf-data/model_data/rolling_summaries.csv temp/rolling_summaries.csv

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
#### EDIT
# added team name into the season summaries file
# f8 -> data starts at the 8th column
cut_features=$(cut -d',' -f4- <<< "$features")
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
select algo in linear_regression xgboost_reg
do
    echo "selected algorithm: $algo"
    echo "training_algorithm: $algo" >> temp/$file
    break
done
printf '\n'

# ==============================================================================
# Estimator hyperparameters
# ==============================================================================
# Written as a model_params line that train_model.py reads. Omitting it entirely
# means library defaults (max_depth=6, learning_rate=0.3), which overfit this
# dataset badly, so the last known-good config is carried forward by default.
selected_features_clean=$(echo "${selected_features}" | sed 's/^,//')

if [ "$SEARCH_PARAMS" = true ]; then
    printf "Searching for hyperparameters (nested walk-forward, a few minutes)...\n"
    PARAMS=$(python ../tune_hyperparams.py \
        --model in_season \
        --features "${selected_features_clean}" \
        --train-start "${start_year}" \
        --train-end "${end_year}")
    if [ -z "$PARAMS" ]; then
        echo "  search produced nothing; falling back to library defaults"
    else
        echo "model_params: ${PARAMS}" >> temp/$file
    fi
else
    # Newest previously trained model wins; its params were validated.
    PREV=$(ls -dt xgboost_reg_* 2>/dev/null | head -1)
    PREV_PARAMS=""
    if [ -n "$PREV" ]; then
        PREV_PARAMS=$(grep -h '^model_params:' "$PREV"/*experiment*.txt 2>/dev/null \
                      | head -1 | sed 's/^model_params: //')
    fi
    if [ -n "$PREV_PARAMS" ]; then
        echo "model_params: ${PREV_PARAMS}" >> temp/$file
        printf "Carrying forward hyperparameters from %s\n" "$PREV"
        printf "  %s\n" "$PREV_PARAMS"
        printf "  (re-run with --search to re-derive them)\n"
    else
        printf "No previous hyperparameters found; training at library defaults.\n"
        printf "  Consider ./create_experiment.sh --search\n"
    fi
fi
printf '\n'

chmod 755 preprocess.sh
./preprocess.sh
python train_model.py
