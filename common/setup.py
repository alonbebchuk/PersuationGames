import argparse
import logging as log
import multiprocessing as mp
import os
from typing import Any


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True, help="Random seed for initialization")
    parser.add_argument("--strategy", type=str, default=None, help="Name of strategy for Single Label Classification or None for Multi Label Classification")

    args = parser.parse_args()
    return args


def bert_setup_args() -> argparse.Namespace:
    args = setup_args()

    args.adam_b1 = 0.9
    args.adam_b2 = 0.999
    args.adam_epsilon = 1e-8
    args.batch_size = 64
    args.context_size = 5
    args.eval_batch_size = 256
    args.evaluate_period = 2
    args.learning_rate = 3e-5
    args.logging_steps = 40
    args.max_seq_length = 256
    args.model_name = "google-bert/bert-large-uncased"
    args.model_type = "bert" if args.strategy is not None else "bert-mt"
    args.num_train_epochs = 10
    args.num_workers = min(8, mp.cpu_count())
    args.out_dir = f"/dev/shm/out/{args.model_type}/{args.seed}/{args.strategy}" if args.strategy is not None else f"/dev/shm/out/{args.model_type}/{args.seed}"
    args.prefetch_factor = 2
    args.warmup_steps = 0

    os.makedirs(args.out_dir, exist_ok=True)

    return args


def whisper_setup_args() -> argparse.Namespace:
    args = setup_args()

    args.adam_b1 = 0.9
    args.adam_b2 = 0.99
    args.adam_epsilon = 1e-8
    args.batch_size = 16
    args.context_size = 5
    args.eval_batch_size = 16
    args.evaluate_period = 2
    args.learning_rate = 1e-5
    args.logging_steps = 40
    args.max_seq_length = 448
    args.model_name = "openai/whisper-medium"
    args.model_type = "whisper" if args.strategy is not None else "whisper-mt"
    args.num_train_epochs = 10
    args.num_workers = min(8, mp.cpu_count())
    args.out_dir = f"/dev/shm/out/{args.model_type}/{args.seed}/{args.strategy}" if args.strategy is not None else f"/dev/shm/out/{args.model_type}/{args.seed}"
    args.prefetch_factor = 2
    args.warmup_steps = 200

    os.makedirs(args.out_dir, exist_ok=True)

    return args


def setup_logger(args: Any, name: str) -> log.Logger:
    logger = log.getLogger(name)
    logger.setLevel(log.INFO)

    formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

    ch = log.StreamHandler()
    ch.setLevel(log.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = log.FileHandler(os.path.join(args.out_dir, "log.txt"))
    fh.setLevel(log.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
