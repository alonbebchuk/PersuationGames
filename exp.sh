# setup
for model_type in bert whisper
do
  rm -rf /dev/shm/${model_type}/data
  rm -rf ${model_type}/out
  for dataset in Ego4D Youtube
  do
    python3.10 ${model_type}/load_data.py --dataset ${dataset}
  done
done

# main
for model_type in bert whisper
do
  for dataset in Ego4D Youtube
  do
    for strategy in "Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence"
    do
      for seed in 13 42 87
      do
        python3.10 ${model_type}/main.py --dataset ${dataset} --strategy ${strategy} --seed ${seed}
      done
    done
  done
done