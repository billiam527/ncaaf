#!/usr/bin/env bash

## Create file to record model specifics and results ##
rm -r temp
mkdir temp

now=$(date +"%m_%d_%Y_%H_%m_%M_%S")
file="preseason_experiment_$now.txt"

touch temp/$file
echo $now >> temp/$file

## specify experiment specifics ##
aws s3 cp s3://ncaaf-data/model_data/season_summaries.csv temp/season_summaries.csv

# define train years
printf "\n"
read -p "start (inclusive) train season (YYYY): " start_year
read -p "end (ex 2020 will train from start_year until 2019-2020 season) train season (YYYY): " end_year
echo train_season_range: ${start_year} - ${end_year} >> temp/$file

# select features to train on
printf "\nselect features to train on and FEATURES_SELECTED when done\n"

# take features from the header of season sums
features=$(head -n 1 temp/season_summaries.csv)

# cut after the second , (team_id, and season)
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

# test size
printf "\n"
read -p "test size (decimal): " test_size
echo test_size: ${test_size} >> temp/$file

# assign previous seasons and weights
printf "\n"
printf "how many seasons back should we train on and assign weights"
printf '\n'
read -p "seasons: " seasons_back
#echo "assign weights to each of the prev seasons starting with the most recent season"
#echo "make sure season weights are in decimal values and add up to 1.00"

#for i in $(seq $seasons_back); do
#    read -p "weight value: " weight
#    weights="${weights},$weight"
#done

echo seasons_back: $seasons_back >> temp/$file
#echo weights: ${weights} | sed s/,// >> temp/$file