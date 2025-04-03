model_types=(bert whisper)
datasets=(Youtube)
strategies=("Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence")
seeds=(12)

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
        if [ "$model_type" == "whisper" ]
        then
          python3.10 ${model_type}/main.py --dataset ${dataset} --strategy="${strategy}" --seed ${seed} --with_transcript
        fi
      done
    done
  done
done

cp -r /dev/shm/out ./

python3.10 generate_metric_tables.py

rm -rf /dev/shm/*