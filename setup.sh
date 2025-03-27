rm -rf /dev/shm/data
for model_type in bert whisper
do
  for dataset in Ego4D Youtube
  do
    python3.10 ${model_type}/load_data.py --dataset ${dataset}
  done
done