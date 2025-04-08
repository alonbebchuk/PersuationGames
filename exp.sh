model_types=(bert bert-mt roberta roberta-mt whisper whisper-mt)
datasets=(Youtube)
strategies=("Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence")
seeds=(12 42 87)

for dataset in ${datasets[@]}
do
  python3.10 common/load_data.py --dataset ${dataset}
done

for model_type in ${model_types[@]}
do
  for dataset in ${datasets[@]}
  do

    for strategy in "${strategies[@]}"
    do
      for seed in ${seeds[@]}
      do
        if [ "$model_type" == "whisper" ]
        then
          python3.10 ${model_type}/main.py --dataset ${dataset} --strategy="${strategy}" --seed ${seed}
          python3.10 ${model_type}/main.py --dataset ${dataset} --strategy="${strategy}" --seed ${seed} --with_transcript
        elif [ "$model_type" != *"-mt" ]
        then
          python3.10 ${model_type}/main.py --model ${model_type} --dataset ${dataset} --strategy="${strategy}" --seed ${seed}
        fi
      done
    done

    for seed in ${seeds[@]}
    do
      if [ "$model_type" == "whisper_mt" ]
      then
        python3.10 ${model_type}/main.py --dataset ${dataset} --seed ${seed}
        python3.10 ${model_type}/main.py --dataset ${dataset} --seed ${seed} --with_transcript
      elif [ "$model_type" == *"-mt" ]
      then
        python3.10 ${model_type}/main.py --model ${model_type} --dataset ${dataset} --seed ${seed}
      fi
    done

  done
done

cp -r /dev/shm/out_mt ./

python3.10 common/generate_metric_tables.py

rm -rf /dev/shm/*