#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

# python3.10 scripts/load_data.py

seeds=(12 42 87)
strategies=("Accusation" "Call for Action" "Defense" "Evidence" "Identity Declaration" "Interrogation")

for seed in "${seeds[@]}"
do
  python3.10 scripts/bert.py --seed ${seed}
  # python3.10 scripts/whisper.py --seed ${seed}
  for strategy in "${strategies[@]}"
  do
    python3.10 scripts/bert.py --seed ${seed} --strategy ${strategy}
    # python3.10 scripts/whisper_mt.py --seed ${seed} --strategy ${strategy}
  done
done

cp -r /dev/shm/out ./
