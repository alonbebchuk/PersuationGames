#!/bin/bash
seeds=(12 42 87)
strategies=("Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence")


python3.10 bert/load_data.py
python3.10 whisper/load_data.py


# bert
for seed in "${seeds[@]}"
do
  for strategy in "${strategies[@]}"
  do
    python3.10 bert/single_task/main.py --strategy="${strategy}" --seed ${seed}
  done
  python3.10 bert/multi_task_binary_label/main.py --seed ${seed}
  python3.10 bert/multi_task_multi_label/main.py --seed ${seed}
done


# whisper projection
for seed in "${seeds[@]}"
do
  for strategy in "${strategies[@]}"
  do
    python3.10 whisper/single_task/main_projection.py --strategy="${strategy}" --seed ${seed}
  done
  python3.10 whisper/multi_task_binary_label/main_projection.py --seed ${seed}
  python3.10 whisper/multi_task_multi_label/main_projection.py --seed ${seed}
done


# whisper yes no
for seed in "${seeds[@]}"
do
  for strategy in "${strategies[@]}"
  do
    python3.10 whisper/single_task/main_yes_no.py --strategy="${strategy}" --seed ${seed}
  done
  python3.10 whisper/multi_task_binary_label/main_yes_no.py --seed ${seed}
done
