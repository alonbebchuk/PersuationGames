for dataset in Ego4D Youtube "Ego4D Youtube"
do
  for model_type in bert roberta
  do
    for video in False True
    do
      for context_size in 0 1 3 5 7 9
      do
        for batch_size in 8 16
        do
          for learning_rate in 1e-5 3e-5 5e-5
          do
            for seed in $(if [ "$model_type" = "bert" ]; then echo "13 42 87"; else echo "227 624 817"; fi)
            do
              "/mnt/c/Users/user/anaconda3/envs/PersuasionGames/python.exe" baselines/main.py \
              --dataset ${dataset} \
              --model_type ${model_type} \
              $(if [ "$video" = "True" ]; then echo "--video"; fi) \
              --context_size ${context_size} \
              --batch_size ${batch_size} \
              --learning_rate ${learning_rate} \
              --seed ${seed} \
              --output_dir out/${dataset}/${model_type}$(if [ "$video" = "True" ]; then echo "_video"; fi)/${batch_size}_${learning_rate}_${context_size}/${seed} \
              --overwrite_output_dir
            done
          done
        done
      done
    done
  done
done