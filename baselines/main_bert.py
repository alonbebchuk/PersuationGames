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
from collections import defaultdict
from flax import struct, traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot
from read_dataset import load_werewolf_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import FlaxBertForSequenceClassification, BertTokenizer
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)
from datasets import Dataset, DatasetDict

MODEL_CLASS = lambda: FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
TOKENIZER_CLASS = lambda: BertTokenizer.from_pretrained("bert-base-uncased")

logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="+", type=str, help="Name of dataset, Ego4D or Youtube or Ego4D Youtube")
parser.add_argument("--context_size", type=int, help="Size of the context")
parser.add_argument("--learning_rate", type=float, help="The initial learning rate for Adam.")
parser.add_argument("--seed", type=int, help="Random seed for initialization")
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
parser.add_argument("--no_train", action="store_true", help="Whether to run training.")
parser.add_argument("--no_eval", action="store_true", help="Whether to run eval on the val set.")
parser.add_argument("--no_test", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--no_evaluate_during_training", action="store_true", help="Whether to run evaluation every epoch.")
parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster data transfer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--evaluate_period", default=2, type=int, help="evaluate every * epochs.")
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--early_stopping_patience", default=10000, type=int, help="Patience for early stopping.")
parser.add_argument("--logging_steps", default=40, type=int, help="Log every X updates steps.")
parser.add_argument("--num_workers", type=int, default=min(8, mp.cpu_count()), help="Number of worker processes for data loading")
parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")

args = parser.parse_args()

if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.no_train and not args.overwrite_output_dir):
    raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
logger.setLevel(log.INFO)
formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

fh = log.FileHandler(os.path.join(args.output_dir, "log.txt"))
fh.setLevel(log.INFO)
fh.setFormatter(formatter)

ch = log.StreamHandler()
ch.setLevel(log.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


class TrainState(train_state.TrainState):
    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)


def create_train_state(
    model: FlaxBertForSequenceClassification,
    learning_rate_fn: Callable[[int], float],
    weight_decay: float = 0.0,
) -> TrainState:
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
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
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=weight_decay,
        mask=decay_mask_fn,
    )

    def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        xentropy = optax.softmax_cross_entropy(logits=logits, labels=onehot(labels, num_classes=2))
        return jnp.mean(xentropy)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        logits_fn=lambda logits: logits.argmax(-1),
        loss_fn=cross_entropy_loss,
    )


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
) -> Callable[[int], float]:
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {
        "input_ids": np.array([x["input_ids"] for x in batch]),
        "attention_mask": np.array([x["attention_mask"] for x in batch]),
        "labels": np.array([x["labels"] for x in batch])
    }

def replicate_train_state(state: TrainState, devices: List[Any]) -> TrainState:
    return jax.device_put_replicated(state, devices)

def get_adjusted_batch_size(original_batch_size: int, n_devices: int, name: str = "batch") -> int:
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
    model: FlaxBertForSequenceClassification,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Tuple[int, float, float]:
    tb_writer = SummaryWriter(args.output_dir)

    devices = jax.local_devices()
    n_devices = len(devices)
    logger.info(f"Training using {n_devices} devices")

    global_batch_size = get_adjusted_batch_size(args.batch_size, n_devices, "batch")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True
    )

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        global_batch_size,
        args.num_train_epochs,
        args.warmup_steps,
        args.learning_rate,
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
        def loss_fn(params: Dict[str, Any]) -> jnp.ndarray:
            outputs = state.apply_fn(
                **{"params": params},
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                train=True,
                dropout_rng=dropout_rng,
            )
            return state.loss_fn(outputs.logits, batch["labels"])

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
                
                tb_writer.add_scalar("lr", current_lr, global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
                logger.info("logging train info!!!")
                logger.info("*")

        if not args.no_evaluate_during_training and epoch % args.evaluate_period == 0:
            results = evaluate(state, val_dataset, mode="val", prefix=str(global_step))
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, epoch)
            logging_loss = tr_loss
            logger.info(f"{results}")
            
            if results["f1"] >= best_f1:
                best_f1 = results["f1"]
                wait_step = 0
                output_dir = os.path.join(args.output_dir, "best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving best model to %s", output_dir)
                
                devices = jax.local_devices()
                n_devices = len(devices)
                unreplicated_params = jax.tree.map(
                    lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == n_devices else x, 
                    state.params
                )
                
                save_model = MODEL_CLASS()
                save_model.params = unreplicated_params
                save_model.save_pretrained(output_dir)
                
                with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=2)
            else:
                wait_step += 1
                if wait_step >= args.early_stopping_patience:
                    train_iterator.close()
                    break

    tb_writer.close()
    return global_step, tr_loss / global_step, best_f1


def evaluate(
    state: TrainState,
    eval_dataset: Optional[Dataset] = None,
    mode: str = "val",
    prefix: str = "",
) -> Dict[str, Any]:
    devices = jax.local_devices()
    n_devices = len(devices)
    
    global_eval_batch_size = get_adjusted_batch_size(args.eval_batch_size, n_devices, "eval batch")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=global_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True,
    )

    logger.info("***** Running evaluation %s *****", mode + "-" + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", global_eval_batch_size)

    dummy_lr_fn = create_learning_rate_fn(1, 1, 1, 0, args.learning_rate)
    
    if isinstance(state.params, dict) and any(isinstance(v, jax.Array) and v.sharding.device_set for v in jax.tree_util.tree_leaves(state.params)):
        logger.info("Extracting parameters from replicated state for evaluation")
        params = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == n_devices else x, state.params)
    else:
        params = state.params
    
    eval_model = MODEL_CLASS()
    eval_model.params = params
    
    eval_state = create_train_state(eval_model, dummy_lr_fn)

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
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            train=False,
        )
        loss = eval_state.loss_fn(outputs.logits, batch["labels"])
        pred = eval_state.logits_fn(outputs.logits)
        
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
    for dataset in args.dataset:
        for split in splits:
            with open(os.path.join("data", dataset, f"{split}.json"), "r") as f:
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
            df.to_csv(os.path.join(args.output_dir, f"predictions_{split}.csv"))


def write_json_file(data: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)


def process_strategy(
    strategy: str,
    output_dir: str,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    logger.info(f"Training for strategy {strategy}")
    args.output_dir = os.path.join(output_dir, strategy)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seeds(args.seed)

    model = MODEL_CLASS()
    results = {}

    if not args.no_train and train_dataset is not None:
        global_step, tr_loss, best_f1 = train(model, train_dataset, val_dataset)
        logger.info(" global_step = %s, average loss = %s, best eval f1 = %s", global_step, tr_loss, best_f1)
        logger.info("Reloading best model")
        best_model_path = os.path.join(args.output_dir, "best")
        model = FlaxBertForSequenceClassification.from_pretrained(
            best_model_path, 
            num_labels=2,
            local_files_only=True
        )
        
    state = create_train_state(model, create_learning_rate_fn(1, 1, 1, 0, args.learning_rate))

    if not args.no_eval and val_dataset is not None:
        results["val"] = evaluate(state, val_dataset, mode="val", prefix="final")
        write_json_file(results["val"], os.path.join(args.output_dir, "results_val.json"))

    if not args.no_test and test_dataset is not None:
        results["test"] = evaluate(state, test_dataset, mode="test", prefix="final")
        write_json_file(results["test"], os.path.join(args.output_dir, "results_test.json"))

    return strategy, results


def load_strategy_datasets() -> Dict[str, DatasetDict]:
    base_tokenizer = TOKENIZER_CLASS()
    strategy_datasets = {}
    
    for strategy in STRATEGIES:
        strategy_datasets[strategy] = DatasetDict()
        if not args.no_train:
            strategy_datasets[strategy]["train"] = load_werewolf_dataset(args, strategy, base_tokenizer, mode="train")
        if not args.no_eval:
            strategy_datasets[strategy]["val"] = load_werewolf_dataset(args, strategy, base_tokenizer, mode="val")
        if not args.no_test:
            strategy_datasets[strategy]["test"] = load_werewolf_dataset(args, strategy, base_tokenizer, mode="test")
    
    return strategy_datasets


def format_results(split, all_result, all_correct, averaged_f1, output_dir):
    result = all_result[split]
    
    cnt = 0
    for x in all_correct[split]:
        if x == len(STRATEGIES):
            cnt += 1
    result["overall_accuracy"] = cnt / len(all_correct[split])
    result["averaged_f1"] = averaged_f1[split] / len(STRATEGIES)

    write_json_file(result, os.path.join(output_dir, f"results_{split}.json"))

    with open(os.path.join(output_dir, f"results_{split}_beaut.txt"), "w") as f:
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

    set_seeds(args.seed)

    all_result = {"val": {}, "test": {}}
    all_correct = {"val": None, "test": None}
    output_dir = args.output_dir
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
            output_dir,
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

    args.output_dir = output_dir
    log_predictions(splits, preds)

    for split in splits:
        format_results(split, all_result, all_correct, averaged_f1, args.output_dir)


if __name__ == "__main__":
    main()
