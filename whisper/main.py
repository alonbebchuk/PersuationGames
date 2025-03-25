import argparse
import jax
import jax.numpy as jnp
import json
import logging as log
import multiprocessing as mp
import numpy as np
import optax
import os
import pandas as pd
import random
import shutil
import wandb
from collections import defaultdict
from datasets import Dataset, DatasetDict
from flax import struct, traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot
from load_dataset import load_data, load_dataset, SAMPLING_RATE
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import FlaxWhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
)


FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
TOKENIZER = WhisperTokenizer.from_pretrained("openai/whisper-small")


def MODEL_CLASS() -> FlaxWhisperForConditionalGeneration:
    return FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
# required
parser.add_argument("--dataset", type=str, help="Name of dataset, Ego4D or Youtube")
parser.add_argument("--seed", type=int, help="Random seed for initialization")
parser.add_argument("--no_evaluate_during_training", action="store_true", help="Whether to run evaluation during training.")
parser.add_argument("--no_eval", action="store_true", help="Whether to run eval on the val set.")
parser.add_argument("--no_pin_memory", action="store_true", help="Pin memory for faster data transfer")
parser.add_argument("--no_test", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--no_train", action="store_true", help="Whether to run training.")
# optional
parser.add_argument("--adam_b1", default=0.9, type=float, help="Adam b1")
parser.add_argument("--adam_b2", default=0.999, type=float, help="Adam b2")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--context_size", type=int, default=5, help="Size of the context")
parser.add_argument("--early_stopping_patience", default=10000, type=int, help="Patience for early stopping.")
parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size for evaluation.")
parser.add_argument("--evaluate_period", default=2, type=int, help="Evaluate every * epochs.")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="The initial learning rate for Adam.")
parser.add_argument("--logging_steps", default=40, type=int, help="Log every X updates steps.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_seq_length", default=448, type=int, help="The maximum sequence length for the model.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--num_workers", type=int, default=min(8, mp.cpu_count()), help="Number of worker processes for data loading")
parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
args = parser.parse_args()

args.data_dir = f"/dev/shm/whisper/data/{args.dataset}"
args.temp_dir = f"/dev/shm/whisper/tmp"

os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(args.temp_dir, exist_ok=True)

args.out_dir = os.path.join(os.path.dirname(__file__), "out", args.dataset, str(args.seed))

os.makedirs(args.out_dir, exist_ok=True)

logger.setLevel(log.INFO)
formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

fh = log.FileHandler(os.path.join(args.out_dir, "log.txt"))
fh.setLevel(log.INFO)
fh.setFormatter(formatter)

ch = log.StreamHandler()
ch.setLevel(log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

AUDIO_CACHE = {}
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]

MAX_SAMPLE_LENGTH = SAMPLING_RATE * 30
YES_TOKEN_ID = 6054
NO_TOKEN_ID = 4540


class TrainState(train_state.TrainState):
    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)


def create_train_state(
    model: FlaxWhisperForConditionalGeneration,
    learning_rate_fn: Callable[[int], float],
    weight_decay: float,
) -> TrainState:
    def decay_mask_fn(
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_candidates = ["layer_norm", "self_attn_layer_norm", "final_layer_norm", "encoder_attn_layer_norm"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    optimizer = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=args.adam_b1,
        b2=args.adam_b2,
        eps=args.adam_epsilon,
        weight_decay=weight_decay,
        mask=decay_mask_fn,
    )

    def logits_fn(logits: jnp.ndarray) -> jnp.ndarray:
        completion_logits = logits[:, -1, :]

        yes_logits = completion_logits[:, YES_TOKEN_ID]
        no_logits = completion_logits[:, NO_TOKEN_ID]

        yes_no_logits = jnp.stack([no_logits - yes_logits, yes_logits - no_logits], axis=1)

        return yes_no_logits

    def cross_entropy_loss(
        logits: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        xentropy = optax.softmax_cross_entropy(logits=logits, labels=onehot(labels, num_classes=2))
        return jnp.mean(xentropy)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        logits_fn=logits_fn,
        loss_fn=cross_entropy_loss,
    )


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
) -> Callable[[int], float]:
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=args.learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(init_value=args.learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def collate_fn(
    batch: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    global AUDIO_CACHE

    end_samples = [
        AUDIO_CACHE[sample["audio_path"]]["length"] if sample["end_sample"] == -1 else sample["end_sample"]
        for sample in batch
    ]
    start_samples = [
        max(end_samples[i] - MAX_SAMPLE_LENGTH, sample["start_sample"])
        for i, sample in enumerate(batch)
    ]
    audio_arrays = [
        AUDIO_CACHE[sample["audio_path"]]["array"][start:end]
        for sample, start, end in zip(batch, start_samples, end_samples)
    ]

    features = FEATURE_EXTRACTOR(audio_arrays, sampling_rate=SAMPLING_RATE, return_attention_mask=True, return_tensors="np")
    input_features = features["input_features"]
    attention_mask = features["attention_mask"]
    return {
        "decoder_input_ids": np.array([sample["decoder_input_ids"] for sample in batch]),
        "decoder_attention_mask": np.array([sample["decoder_attention_mask"] for sample in batch]),
        "input_features": input_features,
        "attention_mask": attention_mask,
        "labels": np.array([sample["labels"] for sample in batch]),
    }


def replicate_train_state(
    state: TrainState,
    devices: List[Any],
) -> TrainState:
    return jax.device_put_replicated(state, devices)


def get_adjusted_batch_size(
    original_batch_size: int,
    n_devices: int,
    name: str,
) -> int:
    min_batch_size = n_devices * 2
    global_batch_size = max(original_batch_size, min_batch_size)

    if global_batch_size != original_batch_size:
        logger.warning(f"Adjusting {name} size from {original_batch_size} to {global_batch_size} to ensure at least 2 samples per device")

    per_device_batch_size = global_batch_size // n_devices
    if global_batch_size % n_devices != 0:
        adjusted_batch_size = per_device_batch_size * n_devices
        logger.warning(f"Further adjusting {name} size from {global_batch_size} to {adjusted_batch_size} to be divisible by {n_devices} devices")
        global_batch_size = adjusted_batch_size

    return global_batch_size


def train(
    model: FlaxWhisperForConditionalGeneration,
    strategy: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Tuple[int, float, float, str]:
    worker_id = jax.process_index()
    if worker_id == 0:
        wandb.init(project="werewolf", name=f"whisper-{args.dataset}-seed{args.seed}-{strategy}", tags=["whisper", args.dataset, f"seed{args.seed}", strategy], config=vars(args))

    devices = jax.local_devices()
    n_devices = len(devices)
    logger.info(f"Training using {n_devices} devices")

    global_batch_size = get_adjusted_batch_size(args.batch_size, n_devices, "batch")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True,
    )

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        global_batch_size,
        args.num_train_epochs,
        args.warmup_steps,
    )

    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(model, learning_rate_fn, weight_decay=args.weight_decay)
    state = replicate_train_state(state, devices)

    @jax.pmap
    def train_step(
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: jnp.ndarray,
    ) -> Tuple[TrainState, jnp.ndarray]:
        def loss_fn(
            params: Dict[str, Any],
        ) -> jnp.ndarray:
            outputs = state.apply_fn(
                **{"params": params},
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                input_features=batch["input_features"],
                attention_mask=batch["attention_mask"],
                train=True,
                dropout_rng=dropout_rng,
            )
            logits = state.logits_fn(outputs.logits)
            return state.loss_fn(logits, batch["labels"])

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    global_step = 0
    wait_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    best_f1 = 0
    tr_loss, logging_loss = 0.0, 0.0

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = {
                k: jnp.array(v).reshape((n_devices, -1) + v.shape[1:])
                for k, v in batch.items()
            }

            rng, dropout_rng = jax.random.split(rng)
            dropout_rngs = jax.random.split(dropout_rng, n_devices)

            state, loss = train_step(state, batch, dropout_rngs)
            tr_loss += jnp.mean(loss).item()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                current_lr = learning_rate_fn(global_step)
                if isinstance(current_lr, jnp.ndarray):
                    current_lr = np.array(current_lr)

                if worker_id == 0:
                    wandb.log({"loss": (tr_loss - logging_loss) / args.logging_steps, "lr": current_lr})
                logging_loss = tr_loss
                logger.info("logging train info!!!")
                logger.info("*")

        if not args.no_evaluate_during_training and epoch % args.evaluate_period == 0:
            results = evaluate(state, val_dataset, mode="val", prefix=str(global_step))
            if worker_id == 0:
                wandb.log({f"eval_{key}": value for key, value in results.items()})
            logging_loss = tr_loss
            logger.info(f"{results}")

            if results["f1"] >= best_f1:
                best_f1 = results["f1"]
                wait_step = 0
                best_dir = os.path.join(args.temp_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                logger.info("Saving best model to %s", best_dir)

                devices = jax.local_devices()
                n_devices = len(devices)
                unreplicated_params = jax.tree.map(
                    lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == n_devices else x,
                    state.params,
                )

                save_model = MODEL_CLASS()
                save_model.params = unreplicated_params
                del save_model.config.__dict__["max_length"]
                del save_model.config.__dict__["suppress_tokens"]
                del save_model.config.__dict__["begin_suppress_tokens"]
                save_model.save_pretrained(best_dir)

                with open(os.path.join(best_dir, "training_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=2)
            else:
                wait_step += 1
                if wait_step >= args.early_stopping_patience:
                    train_iterator.close()
                    break

    if worker_id == 0:
        wandb.finish()

    return global_step, tr_loss / global_step, best_f1, best_dir


def evaluate(
    state: TrainState,
    eval_dataset: Dataset,
    mode: str,
    prefix: str,
) -> Dict[str, Any]:
    devices = jax.local_devices()
    n_devices = len(devices)

    global_eval_batch_size = get_adjusted_batch_size(args.eval_batch_size, n_devices, "eval batch")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=global_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True,
    )

    logger.info("***** Running evaluation %s *****", mode + "-" + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", global_eval_batch_size)

    dummy_lr_fn = create_learning_rate_fn(1, 1, 1, 0)

    if isinstance(state.params, dict) and any(isinstance(v, jax.Array) and v.sharding.device_set for v in jax.tree_util.tree_leaves(state.params)):
        logger.info("Extracting parameters from replicated state for evaluation")
        params = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == n_devices else x,
            state.params,
        )
    else:
        params = state.params

    eval_model = MODEL_CLASS()
    eval_model.params = params

    eval_state = create_train_state(eval_model, dummy_lr_fn, weight_decay=0.0)

    total_samples = len(eval_dataset)
    all_preds = np.zeros(total_samples, dtype=np.int32)
    all_labels = np.zeros(total_samples, dtype=np.int32)
    running_loss = 0.0
    sample_idx = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch_size = len(batch["labels"])

        outputs = eval_state.apply_fn(
            **{"params": eval_state.params},
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            train=False,
        )
        logits = eval_state.logits_fn(outputs.logits)
        loss = eval_state.loss_fn(logits, batch["labels"])
        pred = logits.argmax(-1)

        running_loss += loss.item()

        all_preds[sample_idx:sample_idx + batch_size] = np.array(pred)
        all_labels[sample_idx:sample_idx + batch_size] = np.array(batch["labels"])
        sample_idx += batch_size

    eval_loss = running_loss / len(eval_dataloader)

    correct = (all_preds == all_labels).astype(np.int32)

    if prefix == "final":
        results = {
            "f1": f1_score(y_true=all_labels, y_pred=all_preds),
            "precision": precision_score(y_true=all_labels, y_pred=all_preds),
            "recall": recall_score(y_true=all_labels, y_pred=all_preds),
            "accuracy": accuracy_score(y_true=all_labels, y_pred=all_preds),
            "report": classification_report(y_true=all_labels, y_pred=all_preds),
            "correct": correct.tolist(),
            "preds": all_preds.tolist(),
        }
    else:
        results = {
            "f1": f1_score(y_true=all_labels, y_pred=all_preds),
            "loss": eval_loss,
        }
    logger.info(results["f1"])
    return results


def log_predictions(
    splits: List[str],
    preds: Dict[str, Dict[str, List[int]]],
) -> None:
    for split in splits:
        with open(f"{args.data_dir}/{split}.json", "r") as f:
            games = json.load(f)
            id = -1
            data = defaultdict(list)
            for game in games:
                for record in game["Dialogue"]:
                    id += 1
                    pred = []
                    for strategy in STRATEGIES:
                        if preds[split][strategy][id] == 1:
                            pred.append(strategy)
                    if len(pred) == 0:
                        pred.append("No Strategy")
                    for key, val in record.items():
                        data[key].append(val)
                    data["prediction"].append(pred)
            df = pd.DataFrame.from_dict(data)
            df.to_csv(f"{args.out_dir}/predictions_{split}.csv")


def write_json_file(
    data: Dict[str, Any],
    filepath: str,
) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def set_seeds() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)


def process_strategy(
    strategy: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    logger.info(f"Training for strategy {strategy}")

    set_seeds()

    model = MODEL_CLASS()
    results = {}

    if not args.no_train and train_dataset is not None:
        global_step, tr_loss, best_f1, best_dir = train(model, strategy, train_dataset, val_dataset)
        logger.info(" global_step = %s, average loss = %s, best eval f1 = %s", global_step, tr_loss, best_f1)
        logger.info("Reloading best model")
        model = FlaxWhisperForConditionalGeneration.from_pretrained(
            best_dir,
            local_files_only=True,
        )
        shutil.rmtree(best_dir)

    state = create_train_state(model, create_learning_rate_fn(1, 1, 1, 0), weight_decay=0.0)

    strategy_out_dir = f"{args.out_dir}/{strategy}"
    os.makedirs(strategy_out_dir, exist_ok=True)

    if not args.no_eval and val_dataset is not None:
        results["val"] = evaluate(state, val_dataset, mode="val", prefix="final")
        write_json_file(results["val"], f"{strategy_out_dir}/results_val.json")

    if not args.no_test and test_dataset is not None:
        results["test"] = evaluate(state, test_dataset, mode="test", prefix="final")
        write_json_file(results["test"], f"{strategy_out_dir}/results_test.json")

    return strategy, results


def load_strategy_datasets() -> Dict[str, DatasetDict]:
    global AUDIO_CACHE
    
    load_data(args, "train")
    load_data(args, "val")
    load_data(args, "test")

    for split in ["train", "val", "test"]:
        with open(f"{args.data_dir}/{split}.json", "r") as f:
            games = json.load(f)
            for game in games:
                audio_path = game["audio_path"]
                if audio_path not in AUDIO_CACHE:
                    audio_array = np.load(audio_path)
                    AUDIO_CACHE[audio_path] = {"array": audio_array, "length": len(audio_array)}

    strategy_datasets = {}

    for strategy in STRATEGIES:
        strategy_datasets[strategy] = DatasetDict()
        if not args.no_train:
            strategy_datasets[strategy]["train"] = load_dataset(args, strategy, TOKENIZER, "train")
        if not args.no_eval:
            strategy_datasets[strategy]["val"] = load_dataset(args, strategy, TOKENIZER, "val")
        if not args.no_test:
            strategy_datasets[strategy]["test"] = load_dataset(args, strategy, TOKENIZER, "test")

    return strategy_datasets


def format_results(
    split: str,
    all_result: Dict[str, Dict[str, Dict[str, Any]]],
    all_correct: Dict[str, List[int]],
    averaged_f1: Dict[str, float],
) -> None:
    result = all_result[split]

    cnt = 0
    for x in all_correct[split]:
        if x == len(STRATEGIES):
            cnt += 1
    result["overall_accuracy"] = cnt / len(all_correct[split])
    result["averaged_f1"] = averaged_f1[split] / len(STRATEGIES)

    write_json_file(result, f"{args.out_dir}/results_{split}.json")

    with open(f"{args.out_dir}/results_{split}.txt", "w") as f:
        for strategy in STRATEGIES:
            f.write(f"{result[strategy]['f1'] * 100:.1f}\t")
        f.write(f"{result['averaged_f1'] * 100:.1f}\t{result['overall_accuracy'] * 100:.1f}\n")

        for strategy in STRATEGIES:
            report = result[strategy]["report"]
            result[strategy].pop("report")
            f.write(f"{strategy}\n")
            json.dump(result[strategy], f, indent=4)
            f.write(report)
            f.write("\n")


def main() -> None:
    mp.set_start_method("spawn", force=True)

    logger.info("------NEW RUN-----")
    logger.info("random seed %s", args.seed)
    logger.info("Training/evaluation parameters %s", args)
    logger.info(f"Using {args.num_workers} workers for data loading")

    set_seeds()

    all_result = {"val": {}, "test": {}}
    all_correct = {"val": None, "test": None}
    preds = {"val": {}, "test": {}}
    averaged_f1 = {"val": 0.0, "test": 0.0}
    splits = []

    if not args.no_eval:
        splits.append("val")
    if not args.no_test:
        splits.append("test")

    strategy_datasets = load_strategy_datasets()

    for strategy in STRATEGIES:
        datasets = strategy_datasets[strategy]
        strategy_results = process_strategy(
            strategy,
            datasets.get("train"),
            datasets.get("val"),
            datasets.get("test"),
        )

        strategy, results = strategy_results
        for split in splits:
            if split in results:
                all_result[split][strategy] = results[split]

                if all_correct[split] is None:
                    all_correct[split] = results[split]["correct"]
                else:
                    all_correct[split] = [x + y for x, y in zip(all_correct[split], results[split]["correct"])]

                preds[split][strategy] = results[split]["preds"]
                averaged_f1[split] += results[split]["f1"]

    log_predictions(splits, preds)

    for split in splits:
        format_results(split, all_result, all_correct, averaged_f1)


if __name__ == "__main__":
    main()
