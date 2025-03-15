for dataset in Ego4D Youtube
do
  for model_type in bert roberta
  do
    for seed in $(if [ "$model_type" = "bert" ]; then echo "13 42 87"; else echo "227 624 817"; fi)
    do
      python3 baselines/main.py \
        --dataset ${dataset} \
        --model_type ${model_type} \
        --context_size 5 \
        --batch_size 16 \
        --learning_rate 3e-5 \
        --seed ${seed} \
        --output_dir out/${dataset}/${model_type}/${seed} \
        --overwrite_output_dir
    done
  done
done

# This experiment can be executed in Google Colab by running the following commands:
# %cd /content/drive/MyDrive
# !git clone https://github.com/alonbebchuk/PersuationGames.git
# %cd /content/drive/MyDrive/PersuationGames
# !bash exp.sh