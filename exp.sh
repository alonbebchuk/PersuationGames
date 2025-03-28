model_types=(bert whisper)
datasets=(Youtube)
strategies=("Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence")
seeds=(42)

for model_type in ${model_types[@]}
do
  for dataset in ${datasets[@]}
  do
    python3.10 ${model_type}/load_data.py --dataset ${dataset}
  done
done

for model_type in ${model_types[@]}
do
  for dataset in ${datasets[@]}
  do
    for strategy in "${strategies[@]}"
    do
      for seed in ${seeds[@]}
      do
        python3.10 ${model_type}/main.py --dataset ${dataset} --strategy="${strategy}" --seed ${seed}
      done
    done
  done
done

cp -r /dev/shm/data ./
cp -r /dev/shm/out ./

rm -rf /dev/shm/*