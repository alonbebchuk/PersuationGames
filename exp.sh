for model_type in bert whisper
do
  rm -rf ${model_type}/out
  for dataset in Ego4D Youtube
  do
    for seed in 13 42 87
    do
      python3.10 ${model_type}/main.py \
        --dataset ${dataset} \
        --seed ${seed}
    done
  done
  python3.10 get_results.py \
    --dataset ${dataset} \
    --model_type ${model_type}
done