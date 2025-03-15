for dataset in Ego4D Youtube
do
  for model_type in bert roberta
  do
    for video in False True
    do
      for seed in $(if [ "$model_type" = "bert" ]; then echo "13 42 87"; else echo "227 624 817"; fi)
      do
        python3 baselines/main.py \
          --dataset ${dataset} \
          --model_type ${model_type} \
          $(if [ "$video" = "True" ]; then echo "--video"; fi) \
          --context_size 5 \
          --batch_size 16 \
          --learning_rate 3e-5 \
          --seed ${seed} \
          --output_dir out/${dataset}/${model_type}$(if [ "$video" = "True" ]; then echo "_video"; fi)/${seed} \
          --overwrite_output_dir
        done
      done
  done
done