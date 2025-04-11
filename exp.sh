#!/bin/bash

python3.10 bert/load_data.py
python3.10 whisper/load_data.py

seeds=(12 42 87)
strategies=("Identity Declaration" "Accusation" "Interrogation" "Call for Action" "Defense" "Evidence")

for seed in "${seeds[@]}"
do
  for strategy in "${strategies[@]}"
  do
    python3.10 bert/single_task/main.py --strategy ${strategy} --seed ${seed}
    python3.10 whisper/single_task/main_v1.py --strategy ${strategy} --seed ${seed}
    python3.10 whisper/single_task/main_v2.py --strategy ${strategy} --seed ${seed}
  done

  python3.10 bert/multi_task_binary_label/main.py --seed ${seed}
  python3.10 whisper/multi_task_binary_label/main_v1.py --seed ${seed}
  python3.10 whisper/multi_task_binary_label/main_v2.py --seed ${seed}
  python3.10 bert/multi_task_multi_label/main.py --seed ${seed}
  python3.10 whisper/multi_task_multi_label/main.py --seed ${seed}
done
