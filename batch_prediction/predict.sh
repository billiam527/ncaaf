# select model

# Derive the repo root from this script's own location rather than hard-coding
# /home/bill/ncaaf, so this works from any checkout.
BATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$BATCH_DIR")"

rm -rf prediction_file
mkdir prediction_file


############### Load Preseason Model ###############
cd ..
cd model_training/preseason_model
aws s3 cp s3://ncaaf-data/cfbd-data/cfbd_spread_data.csv temp/cfbd_spread_data.csv

directories=$(ls -d */)

string='temp/'
dirs=""
for dir in ${directories}
do
  if [[ $string == *${dir}* ]]; then
    continue
  else
    dirs="${dirs}, ${dir::-1}"
  fi
done

dirs="${dirs/, /}"
dirs="${dirs//,/}"
dirs="${dirs/prediction_file//}"
dirs="${dirs/\//}"

PS3="Select a preseason model by entering a number: "
select dir in $dirs
do
    echo "selected model: $dir"
    selected_preseason_model=$dir
    break
done
path="$REPO_ROOT/model_training/preseason_model/"
preseason_model_file="$path$selected_preseason_model"

############### Load In Season Model ###############
cd ..
cd in_season_model

directories=$(ls -d */)

string='temp/'
dirs=""
for dir in ${directories}
do
  if [[ $string == *${dir}* ]]; then
    continue
  else
    dirs="${dirs}, ${dir::-1}"
  fi
done

dirs="${dirs/, /}"
dirs="${dirs//,/}"
dirs="${dirs/prediction_file//}"
dirs="${dirs/\//}"

PS3="Select an in season model by entering a number: "
select dir in $dirs
do
    echo "selected model: $dir"
    selected_in_season_model=$dir
    break
done
path="$REPO_ROOT/model_training/in_season_model/"
in_season_model_file="$path$selected_in_season_model"

###################################################################

cd ..
cd ..
cd batch_prediction
#python -c "import predict; predict.echo_features('$model_file')"

echo scraping schedules
python scrape_scheduled_games.py
mv scheduled_games.csv prediction_file
directory="$BATCH_DIR/prediction_file/"
file="scheduled_games.csv"

echo predicting games
python predict.py --preseason_model_file "$preseason_model_file" \
    --in_season_model_file "$in_season_model_file" \
    --predict_dir "$directory" \
    --predict_file "$file" \
    --model_blender "False"

echo creating blended model
rm -r temp
mkdir temp   
python predict.py --preseason_model_file "$preseason_model_file" \
    --in_season_model_file "$in_season_model_file" \
    --predict_dir "$directory" \
    --predict_file "$file" \
    --model_blender "True"

# take preaseaon model predictions and in season model predictions
cd ..
cd model_training/model_blender
python model_blender.py

cd ../..
cd batch_prediction
python implement_blended_model.py

python weekly_predictions.py

echo building odds table
python build_odds_table.py
echo creating week by week files
python merge_odds_table_and_preds.py

#echo predictions sent to "$directory"predictions.csv