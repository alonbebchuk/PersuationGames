#!/bin/bash


python3.10 bert/load_data.py
python3.10 whisper/load_data.py


seeds=(12 42 87)
strategies=("Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence")


for seed in "${seeds[@]}"
do
  for strategy in "${strategies[@]}"
  do
    # single task bert and whisper
    python3.10 bert/single_task/main.py --strategy="${strategy}" --seed ${seed}
    python3.10 whisper/single_task/main_yes_no.py --strategy="${strategy}" --seed ${seed}
  done
done


for seed in "${seeds[@]}"
do
  # multi task binary label bert and whisper
  python3.10 bert/multi_task_binary_label/main.py --seed ${seed}
  python3.10 whisper/multi_task_binary_label/main_yes_no.py --seed ${seed}
done


for seed in "${seeds[@]}"
do
  for strategy in "${strategies[@]}"
  do
    # single task whisper with layer normalization
    python3.10 whisper/single_task/main_projection.py --strategy="${strategy}" --seed ${seed}
  done
  # multi task binary label whisper with layer normalization
  python3.10 whisper/multi_task_binary_label/main_projection.py --seed ${seed}
  # multi task multi label bert
  python3.10 bert/multi_task_multi_label/main.py --seed ${seed}
  # multi task multi label whisper with layer normalization
  python3.10 whisper/multi_task_multi_label/main.py --seed ${seed}
done
