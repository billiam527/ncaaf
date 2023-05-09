# select model

cd ..
cd model_training/preseason_model

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

PS3="Select a model by entering a number: "
select dir in $dirs
do
    echo "selected model: $dir"
    selected_model=$dir
    break
done
path='/home/bill/ncaaf/model_training/preseason_model/'
model_file="$path$selected_model"

cd ..
cd ..
cd batch_prediction
python -c "import predict; predict.echo_features('$model_file')"

python scrape_scheduled_games.py
directory="/home/bill/ncaaf/batch_prediction/prediction_file/"
file="scheduled_games.csv"

python predict.py --model_file "$model_file" --predict_dir "$directory" --predict_file "$file"
echo predictions sent to "$directory"predictions.csv