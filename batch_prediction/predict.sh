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

echo "Is file we are looking to predict here?"
read -p "/home/bill/ncaaf/model_training/preseason_model/prediction_file/ (y/n) " answer
if [[ $answer == y ]]; then
  directory="/home/bill/ncaaf/model_training/preseason_model/prediction_file/"
else
  read -p "type in the name of the directory where the file with predictions is located " directory
fi
echo ""
read -p "What is the name of the file we are looking to predict? (include any extensions like .csv) " file

predict_dir="$directory"
predict_file="$file"
python predict.py --model_file $model_file --predict_dir $predict_dir --predict_file $predict_file