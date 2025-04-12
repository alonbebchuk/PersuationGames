import wandb
import os
if not os.path.exists("/tmp/wandb_lock"):
    wandb.init(project="werewolf")
    with open("/tmp/wandb_lock", "w") as f:
        f.write("1")
import argparse
import jax
import jax.numpy as jnp
import json
import logging as log
import multiprocessing as mp
import numpy as np
import optax
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from models.flax_whisper_for_sequence_classification import FlaxWhisperForSequenceClassification
from whisper.multi_task_multi_label.load_dataset import load_dataset, STRATEGIES, DATA_DIR
from datasets import Dataset
from flax import jax_utils, struct, traverse_util
from flax.training import train_state
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from typing import Any, Callable, Dict, List, Tuple


logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
# required
parser.add_argument("--seed", type=int, required=True, help="Random seed for initialization")
# optional
parser.add_argument("--adam_b1", type=float, default=0.9, help="Adam b1")
parser.add_argument("--adam_b2", type=float, default=0.99, help="Adam b2")
parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--context_size", type=int, default=5, help="Size of the context")
parser.add_argument("--eval_batch_size", type=int, default=256, help="Batch size for evaluation.")
parser.add_argument("--evaluate_period", type=int, default=2, help="Evaluate every * epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="The initial learning rate for Adam.")
parser.add_argument("--logging_steps", type=int, default=40, help="Log every X updates steps.")
parser.add_argument("--max_seq_length", type=int, default=448, help="The maximum sequence length for the model.")
parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
parser.add_argument("--num_workers", type=int, default=min(8, mp.cpu_count()), help="Number of worker processes for data loading")
parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")
parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup over warmup_steps.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some.")
args = parser.parse_args()

args.out_dir = os.path.join(ROOT_DIR, f"out/whisper/multi_task_multi_label/v1/{args.seed}")

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


SAMPLING_RATE = 16000
MAX_SAMPLE_LENGTH = SAMPLING_RATE * 30
YES_TOKEN_ID = 6054
NO_TOKEN_ID = 4540


class AudioCollator:
    def __init__(self, feature_extractor: WhisperFeatureExtractor, batch_size: int) -> None:
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.cache = {}

    def load_cache(self, split: str) -> None:
        with open(f"{DATA_DIR}/{split}.json", "r") as f:
            games = json.load(f)
            for game in games:
                audio_path = game["audio_path"]
                if audio_path not in self.cache:
                    audio_array = np.load(audio_path, mmap_mode="r")
                    self.cache[audio_path] = {"array": audio_array, "length": len(audio_array)}

    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        batch_size = len(batch)
        audio_arrays = []

        decoder_input_ids = np.empty((batch_size, len(batch[0]["decoder_input_ids"])), dtype=np.int32)
        decoder_attention_mask = np.empty((batch_size, len(batch[0]["decoder_attention_mask"])), dtype=np.int32)
        labels = np.empty(batch_size, dtype=np.int32)
        ids = []

        for i, sample in enumerate(batch):
            audio_path = sample["audio_path"]
            cached_data = self.cache[audio_path]

            end_sample = cached_data["length"] if sample["end_sample"] == -1 else sample["end_sample"]
            start_sample = max(end_sample - MAX_SAMPLE_LENGTH, sample["start_sample"])

            audio_array = cached_data["array"][start_sample:end_sample]
            audio_arrays.append(audio_array)

            decoder_input_ids[i] = sample["decoder_input_ids"]
            decoder_attention_mask[i] = sample["decoder_attention_mask"]
            labels[i] = sample["labels"]
            ids.append(sample["id"])

        features = self.feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            return_attention_mask=True,
            return_tensors="np",
            padding="max_length",
        )

        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "input_features": features["input_features"],
            "attention_mask": features["attention_mask"],
            "labels": labels,
            "id": ids,
        }


class TrainState(train_state.TrainState):
    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)


def create_train_state(model: FlaxWhisperForSequenceClassification, learning_rate_fn: Callable[[int], float]) -> TrainState:
    def decay_mask_fn(params: Dict[str, Any]) -> Dict[str, Any]:
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_candidates = ["layernorm", "layer_norm", "ln", "self_attn_layer_norm", "final_layer_norm", "encoder_attn_layer_norm"]
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
        weight_decay=args.weight_decay,
        mask=decay_mask_fn,
    )

    def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        xentropy = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
        return jnp.mean(xentropy)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        logits_fn=lambda logits: (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32),
        loss_fn=cross_entropy_loss,
    )


def create_learning_rate_fn(train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int) -> Callable[[int], float]:
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=args.learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(init_value=args.learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def get_adjusted_batch_size(original_batch_size: int, n_devices: int) -> int:
    min_batch_size = n_devices * 2
    global_batch_size = max(original_batch_size, min_batch_size)

    per_device_batch_size = global_batch_size // n_devices
    if global_batch_size % n_devices != 0:
        adjusted_batch_size = per_device_batch_size * n_devices
        global_batch_size = adjusted_batch_size

    return global_batch_size


def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], dropout_rng: jnp.ndarray) -> Tuple[TrainState, jnp.ndarray]:
    def loss_fn(params: Dict[str, Any]) -> jnp.ndarray:
        outputs = state.apply_fn(
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            train=True,
            params=params,
            dropout_rng=dropout_rng,
        )
        loss = state.loss_fn(outputs.logits, batch["labels"])
        return jax.lax.pmean(loss, axis_name="batch")

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    outputs = state.apply_fn(
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
        input_features=batch["input_features"],
        attention_mask=batch["attention_mask"],
        train=False,
        params=state.params,
    )
    logits = outputs.logits
    loss = state.loss_fn(logits, batch["labels"])
    loss = jax.lax.pmean(loss, axis_name="batch")
    preds = state.logits_fn(logits)
    return loss, preds, batch["labels"]


p_train_step = jax.pmap(train_step, axis_name="batch", in_axes=(0, 0, 0), donate_argnums=(0,))

p_eval_step = jax.pmap(eval_step, axis_name="batch", in_axes=(0, 0))

from transformers import FlaxWhisperForConditionalGeneration
def train(tokenizer: WhisperTokenizer, feature_extractor: WhisperFeatureExtractor, model: FlaxWhisperForConditionalGeneration) -> None:
    train_dataset = load_dataset(args, tokenizer, "train")
    val_dataset = load_dataset(args, tokenizer, "val")
    test_dataset = load_dataset(args, tokenizer, "test")

    devices = jax.local_devices()
    n_devices = len(devices)

    worker_id = jax.process_index()
    if worker_id == 0:
        

    global_batch_size = get_adjusted_batch_size(args.batch_size, n_devices)
    per_device_batch_size = global_batch_size // n_devices

    audio_collator = AudioCollator(feature_extractor, global_batch_size)
    audio_collator.load_cache("train")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=audio_collator,
        drop_last=True,
    )

    learning_rate_fn = create_learning_rate_fn(len(train_dataset), global_batch_size, args.num_train_epochs, args.warmup_steps)

    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(model, learning_rate_fn)
    state = jax_utils.replicate(state)

    global_step = 0
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

            _ = batch.pop("id")
            batch = {
                k: jnp.array(v).reshape((n_devices, per_device_batch_size) + v.shape[1:])
                for k, v in batch.items()
            }

            batch = {
                k: jax.device_put_sharded(list(v), devices)
                for k, v in batch.items()
            }

            rng, dropout_rng = jax.random.split(rng)
            dropout_rngs = jax.random.split(dropout_rng, n_devices)
            dropout_rngs = jax.device_put_sharded(list(dropout_rngs), devices)

            state, loss = p_train_step(state, batch, dropout_rngs)
            tr_loss += jnp.mean(loss).item()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                current_lr = learning_rate_fn(global_step)
                if isinstance(current_lr, jnp.ndarray):
                    current_lr = np.array(current_lr)

                if worker_id == 0:
                    wandb.log({"loss": (tr_loss - logging_loss) / args.logging_steps, "lr": current_lr})
                logging_loss = tr_loss

        if (epoch + 1) % args.evaluate_period == 0:
            results_val = evaluate(feature_extractor, state, val_dataset, "val")
            if worker_id == 0:
                wandb.log({f"eval_{key}": value for key, value in results_val.items() if key != "report" and key not in STRATEGIES})
                wandb.log({f"eval_{strategy}_{key}": value for strategy in STRATEGIES for key, value in results_val[strategy].items() if key != "report"})
            logging_loss = tr_loss
            logger.info(f"\n{results_val['report']}")

            if results_val["f1"] >= best_f1:
                best_f1 = results_val["f1"]

                results_test = evaluate(feature_extractor, state, test_dataset, "test")
                write_json_file(results_val, f"{args.out_dir}/results_val.json")
                write_json_file(results_test, f"{args.out_dir}/results_test.json")

    if worker_id == 0:
        wandb.finish()


def evaluate(feature_extractor: WhisperFeatureExtractor, state: TrainState, eval_dataset: Dataset, mode: str) -> Dict[str, Any]:
    devices = jax.local_devices()
    n_devices = len(devices)

    global_eval_batch_size = get_adjusted_batch_size(args.eval_batch_size, n_devices)
    audio_collator = AudioCollator(feature_extractor, global_eval_batch_size)
    audio_collator.load_cache(mode)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=global_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=audio_collator,
        drop_last=True,
    )

    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        _ = batch.pop("id")
        per_device_batch_size = global_eval_batch_size // n_devices
        batch = {
            k: jnp.array(v).reshape((n_devices, per_device_batch_size) + v.shape[1:])
            for k, v in batch.items()
        }

        batch = {
            k: jax.device_put_sharded(list(v), devices)
            for k, v in batch.items()
        }

        loss, preds, labels = p_eval_step(state, batch)

        running_loss += jnp.mean(loss).item()
        all_preds.extend(jax.device_get(preds).reshape(-1, len(STRATEGIES)))
        all_labels.extend(jax.device_get(labels).reshape(-1, len(STRATEGIES)))

    eval_loss = running_loss / len(eval_dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        "loss": eval_loss,
        "f1": f1_score(y_true=all_labels, y_pred=all_preds, average="macro"),
        "precision": precision_score(y_true=all_labels, y_pred=all_preds, average="macro"),
        "recall": recall_score(y_true=all_labels, y_pred=all_preds, average="macro"),
        "accuracy": accuracy_score(y_true=all_labels, y_pred=all_preds),
        "report": classification_report(y_true=all_labels, y_pred=all_preds, target_names=STRATEGIES),
    }

    for i, strategy in enumerate(STRATEGIES):
        strategy_preds = all_preds[:, i].tolist()
        strategy_labels = all_labels[:, i].tolist()
        results[strategy] = {
            "f1": f1_score(y_true=strategy_labels, y_pred=strategy_preds),
            "precision": precision_score(y_true=strategy_labels, y_pred=strategy_preds),
            "recall": recall_score(y_true=strategy_labels, y_pred=strategy_preds),
            "accuracy": accuracy_score(y_true=strategy_labels, y_pred=strategy_preds),
            "report": classification_report(y_true=strategy_labels, y_pred=strategy_preds),
        }

    return results


def write_json_file(data: Dict[str, Any], filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    wandb.save(filepath)



def set_seeds() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)

        logger.info("------NEW RUN-----")
        logger.info("Training/evaluation parameters %s", args)

        set_seeds()

        model_name = "openai/whisper-small"
        tokenizer = WhisperTokenizer.from_pretrained(model_name)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        model = FlaxWhisperForSequenceClassification.from_pretrained(model_name, num_labels=2)

        train(tokenizer, feature_extractor, model)
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}", exc_info=True)
        wandb.finish(exit_code=1)
        raise e


if __name__ == "__main__":
    main()
    wandb.finish()
