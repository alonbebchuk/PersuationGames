context_size=5
batch_size=16
learning_rate=3e-5

for dataset in Ego4D Youtube
do
  for seed in 13 42 87
  do
    python3.10 baselines/main_bert.py \
      --dataset ${dataset} \
      --context_size ${context_size} \
      --batch_size ${batch_size} \
      --learning_rate ${learning_rate} \
      --seed ${seed} \
      --output_dir out/bert/${dataset}/${seed} \
      --overwrite_output_dir
  done
done