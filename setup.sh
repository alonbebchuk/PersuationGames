for model_type in bert whisper
do
  rm -rf /dev/shm/${model_type}/data
  for dataset in Ego4D Youtube
  do
    python3.10 ${model_type}/load_data.py --dataset ${dataset}
  done
done