# Persuasion Modeling with Audio for Social Deduction Games

This repository contains code for training and evaluating persuasion strategy prediction models using text and audio features.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python3.10 bert/load_data.py
python3.10 whisper/load_data.py
```

## Training Models

### BERT Models

To train and evaluate the BERT models (Single-Task, Multi-Task Binary-Label, and Multi-Task Multi-Label), run:

```bash
python3.10 bert/single_task/main.py --strategy="${strategy}" --seed ${seed}
python3.10 bert/multi_task_binary_label/main.py --seed ${seed}
python3.10 bert/multi_task_multi_label/main.py --seed ${seed}
```

Results are written to `out/bert`.

### Whisper Yes-No Models

To train and evaluate the Whisper Yes-No models (Single-Task and Multi-Task Binary-Label), run:

```bash
python3.10 whisper/single_task/main_yes_no.py --strategy="${strategy}" --seed ${seed}
python3.10 whisper/multi_task_binary_label/main_yes_no.py --seed ${seed}
```

Results are written to `out/whisper/yes_no`.

### Whisper Projection Models

To train and evaluate the Whisper Projection models (Single-Task, Multi-Task Binary-Label, and Multi-Task Multi-Label), run:

```bash
python3.10 whisper/single_task/main_projection.py --strategy="${strategy}" --seed ${seed}
python3.10 whisper/multi_task_binary_label/main_projection.py --seed ${seed}
python3.10 whisper/multi_task_multi_label/main_projection.py --seed ${seed}
```

Results are written to `out/whisper/projection`.

## Evaluation

To train and evaluate all models and generate tables with F1 scores and accuracy metrics, run:

```bash
bash exp.sh
```

Results are written to `out`.

## Downloading Audio Samples

To download audio samples:

```bash
python3.10 audio_samples.py
```

Results are written to `audios`.