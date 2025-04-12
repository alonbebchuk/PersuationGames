#!/bin/bash

# Script to run appropriate model commands based on input parameters
# Usage: ./run_model.sh <model_type> <model_size> <task_type> <train_projector> <num_classes> [seed] [strategy]

# Check if required arguments are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <model_type> <model_size> <task_type> <train_projector> <num_classes> [seed] [strategy]"
    echo "  model_type: bert or whisper"
    echo "  model_size: small, medium, large, etc."
    echo "  task_type: strat or multi-task"
    echo "  train_projector: true or false"
    echo "  num_classes: integer (e.g., 2 or 6)"
    echo "  seed: optional, defaults to 42"
    echo "  strategy: optional, defaults to 'default'"
    exit 1
fi

# Parse arguments
model_type=$1
model_size=$2
task_type=$3
train_projector=$4
num_classes=$5

# Set default values for optional arguments
seed=${6:-42}
strategy=${7:-"default"}

# Convert train_projector to boolean (case insensitive)
if [[ ${train_projector,,} == "true" ]]; then
    train_projector=true
else
    train_projector=false
fi

# Convert num_classes to integer
num_classes=$(($num_classes))

echo "Running with parameters:"
echo "Model Type: $model_type"
echo "Model Size: $model_size"
echo "Task Type: $task_type"
echo "Train Projector: $train_projector"
echo "Number of Classes: $num_classes"
echo "Seed: $seed"
echo "Strategy: $strategy"
echo ""


python3.10 bert/load_data.py
python3.10 whisper/load_data.py


# Determine which command to run based on the parameters
if [ "$model_type" == "bert" ] && [ "$task_type" == "strat" ] && [ "$train_projector" == true ] && [ "$num_classes" == 2 ]; then
    echo "Running: python3.10 bert/single_task/main.py --strategy=\"${strategy}\" --seed ${seed}"
    python3.10 bert/single_task/main.py --strategy="${strategy}" --seed ${seed}

elif [ "$model_type" == "whisper" ] && [ "$task_type" == "strat" ] && [ "$train_projector" == false ] && [ "$num_classes" == 2 ]; then
    echo "Running: python3.10 whisper/single_task/main_v1.py --strategy=\"${strategy}\" --seed ${seed}"
    python3.10 whisper/single_task/main_v1.py --strategy="${strategy}" --seed ${seed}

elif [ "$model_type" == "whisper" ] && [ "$task_type" == "strat" ] && [ "$train_projector" == true ] && [ "$num_classes" == 2 ]; then
    echo "Running: python3.10 whisper/single_task/main_v2.py --strategy=\"${strategy}\" --seed ${seed}"
    python3.10 whisper/single_task/main_v2.py --strategy="${strategy}" --seed ${seed}

elif [ "$model_type" == "whisper" ] && [ "$task_type" == "multi-task" ] && [ "$train_projector" == false ] && [ "$num_classes" == 2 ]; then
    echo "Running: python3.10 whisper/multi_task_binary_label/main_v1.py --seed ${seed}"
    python3.10 whisper/multi_task_binary_label/main_v1.py --seed ${seed}

elif [ "$model_type" == "whisper" ] && [ "$task_type" == "multi-task" ] && [ "$train_projector" == true ] && [ "$num_classes" == 6 ]; then
    echo "Running: python3.10 whisper/multi_task_multi_label/main.py --seed ${seed}"
    python3.10 whisper/multi_task_multi_label/main.py --seed ${seed}

elif [ "$model_type" == "bert" ] && [ "$task_type" == "multi-task" ] && [ "$train_projector" == true ] && [ "$num_classes" == 2 ]; then
    echo "Running: python3.10 bert/multi_task_binary_label/main.py --seed ${seed}"
    python3.10 bert/multi_task_binary_label/main.py --seed ${seed}

elif [ "$model_type" == "bert" ] && [ "$task_type" == "multi-task" ] && [ "$train_projector" == true ] && [ "$num_classes" == 6 ]; then
    echo "Running: python3.10 bert/multi_task_multi_label/main.py --seed ${seed}"
    python3.10 bert/multi_task_multi_label/main.py --seed ${seed}

elif [ "$model_type" == "whisper" ] && [ "$task_type" == "multi-task" ] && [ "$train_projector" == true ] && [ "$num_classes" == 2 ]; then
    echo "Running: python3.10 whisper/multi_task_binary_label/main_v2.py --seed ${seed}"
    python3.10 whisper/multi_task_binary_label/main_v2.py --seed ${seed}

else
    echo "Error: No matching configuration found for the provided parameters."
    echo "Please check your input and try again."
    exit 1
fi