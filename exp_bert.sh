for dataset in Ego4D Youtube
do
  for seed in 13 42 87
  do
    python3 baselines/main.py \
      --dataset ${dataset} \
      --context_size 5 \
      --batch_size 16 \
      --learning_rate 3e-5 \
      --seed ${seed} \
      --output_dir out/bert/${dataset}/${seed} \
      --overwrite_output_dir
  done
done

# This experiment can be executed in Google Colab by running the following commands:
# %cd /content/drive/MyDrive
# !git clone https://github.com/alonbebchuk/PersuationGames.git
# %cd /content/drive/MyDrive/PersuationGames
# !bash exp.sh