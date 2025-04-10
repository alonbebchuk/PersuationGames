import jax
import jax.numpy as jnp
import json
import logging
import numpy as np
import wandb
from common.collators import get_collator
from common.create_train_state import create_train_state, TrainState
from common.load_datasets import load_datasets, STRATEGIES
from flax import jax_utils
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from typing import Any, Dict, Tuple


def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    outputs = state.apply_fn(
        **state.apply_inputs_fn(batch),
        train=False,
        params=state.params,
    )
    logits = outputs.logits
    loss = state.loss_fn(logits, batch["labels"])
    loss = jax.lax.pmean(loss, axis_name="batch")
    preds = state.preds_fn(logits)
    return loss, preds, batch["labels"]


def evaluate(args: Any, state: TrainState, eval_dataloader: DataLoader) -> Dict[str, Any]:
    devices = jax.local_devices()
    n_devices = len(devices)

    p_eval_step = jax.pmap(eval_step, axis_name="batch", in_axes=(0, 0))

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_ids = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        ids = batch.pop("id")

        per_device_batch_size = args.eval_batch_size // n_devices
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
        
        device_preds = jax.device_get(preds)
        device_labels = jax.device_get(labels)
        if args.strategy is not None:
            all_preds.extend(device_preds.reshape(-1))
            all_labels.extend(device_labels.reshape(-1))
        else:
            all_preds.extend(device_preds.reshape(-1, device_preds.shape[-1]))
            all_labels.extend(device_labels.reshape(-1, device_labels.shape[-1]))
            
        all_ids.extend(ids)

    eval_loss = running_loss / len(eval_dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_ids = np.array(all_ids)

    if args.strategy is not None:
        incorrect_mask = all_preds != all_labels
        incorrect_preds = all_preds[incorrect_mask].tolist()
        incorrect_ids = all_ids[incorrect_mask].tolist()
    else:
        incorrect_mask = np.any(all_preds != all_labels, axis=1)
        incorrect_preds = all_preds[incorrect_mask].tolist()
        incorrect_ids = all_ids[incorrect_mask].tolist()

    if args.strategy is not None:
        results = {
            "loss": eval_loss,
            "f1": f1_score(y_true=all_labels, y_pred=all_preds),
            "precision": precision_score(y_true=all_labels, y_pred=all_preds),
            "recall": recall_score(y_true=all_labels, y_pred=all_preds),
            "accuracy": accuracy_score(y_true=all_labels, y_pred=all_preds),
            "report": classification_report(y_true=all_labels, y_pred=all_preds),
            "preds": incorrect_preds,
            "ids": incorrect_ids,
        }
    else:
        results = {
            "loss": eval_loss,
            "f1": f1_score(y_true=all_labels, y_pred=all_preds, average="macro"),
            "precision": precision_score(y_true=all_labels, y_pred=all_preds, average="macro"),
            "recall": recall_score(y_true=all_labels, y_pred=all_preds, average="macro"),
            "accuracy": accuracy_score(y_true=all_labels, y_pred=all_preds),
            "report": classification_report(y_true=all_labels, y_pred=all_preds, target_names=STRATEGIES),
            "preds": incorrect_preds,
            "ids": incorrect_ids,
        }
        for i, strategy in enumerate(STRATEGIES):
            strategy_preds = all_preds[:, i]
            strategy_labels = all_labels[:, i]

            strategy_incorrect_mask = strategy_preds != strategy_labels
            strategy_incorrect_preds = strategy_preds[strategy_incorrect_mask].tolist()
            strategy_incorrect_ids = all_ids[strategy_incorrect_mask].tolist()

            results[strategy] = {
                "f1": f1_score(y_true=strategy_labels, y_pred=strategy_preds),
                "precision": precision_score(y_true=strategy_labels, y_pred=strategy_preds),
                "recall": recall_score(y_true=strategy_labels, y_pred=strategy_preds),
                "accuracy": accuracy_score(y_true=strategy_labels, y_pred=strategy_preds),
                "report": classification_report(y_true=strategy_labels, y_pred=strategy_preds),
                "preds": strategy_incorrect_preds,
                "ids": strategy_incorrect_ids,
            }

    return results


def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], dropout_rng: jnp.ndarray) -> Tuple[TrainState, jnp.ndarray]:
    def loss_fn(params: Dict[str, Any]) -> jnp.ndarray:
        outputs = state.apply_fn(
            **state.apply_inputs_fn(batch),
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


def train(args: Any, logger: logging.Logger) -> None:
    worker_id = jax.process_index()
    if worker_id == 0:
        if args.strategy:
            wandb.init(project="werewolf", name=f"{args.model_type}-{args.strategy}-{args.seed}", tags=[args.model_type, args.strategy, str(args.seed)], config=vars(args))
        else:
            wandb.init(project="werewolf", name=f"{args.model_type}-{args.seed}", tags=[args.model_type, str(args.seed)], config=vars(args))

    train_dataset, val_dataset, test_dataset = load_datasets(args)

    collator = get_collator(args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collator,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collator,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collator,
        drop_last=True,
    )

    state = create_train_state(args, len(train_dataset))
    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(args.seed)

    devices = jax.local_devices()
    n_devices = len(devices)
    per_device_batch_size = args.batch_size // n_devices

    p_train_step = jax.pmap(train_step, axis_name="batch", in_axes=(0, 0, 0), donate_argnums=(0,))

    global_step = 0
    steps_trained_in_current_epoch = 0
    best_f1 = 0
    logging_loss = 0.0
    tr_loss = 0.0

    for epoch in trange(0, int(args.num_train_epochs), desc="Epoch"):
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
                current_lr = state.learning_rate_fn(global_step)
                if isinstance(current_lr, jnp.ndarray):
                    current_lr = np.array(current_lr)

                if worker_id == 0:
                    if args.strategy is not None:
                        wandb.log({f"{args.strategy}_loss": (tr_loss - logging_loss) / args.logging_steps})
                    else:
                        wandb.log({"loss": (tr_loss - logging_loss) / args.logging_steps})
                    wandb.log({"lr": current_lr})
                logging_loss = tr_loss

        if epoch % args.evaluate_period == 0:
            results_val = evaluate(args, state, val_dataloader)
            if worker_id == 0:
                if args.strategy is not None:
                    wandb.log({
                        f"{args.strategy}_eval_{key}": value
                        for key, value in results_val.items()
                        if key not in ["report", "preds", "ids"]
                    })
                else:
                    wandb.log({
                        f"eval_{key}": value
                        for key, value in results_val.items()
                        if key not in ["report", "preds", "ids"] and key not in STRATEGIES
                    })
                    wandb.log({
                        f"{strategy}_eval_{key}": value
                        for strategy in STRATEGIES
                        for key, value in results_val[strategy].items()
                        if key not in ["report", "preds", "ids"]
                    })
            logging_loss = tr_loss
            logger.info(f"\n{results_val['report']}")

            if results_val["f1"] >= best_f1:
                best_f1 = results_val["f1"]

                results_test = evaluate(args, state, test_dataloader)
                with open(f"{args.out_dir}/results_val.json", "w") as f:
                    json.dump(results_val, f)
                with open(f"{args.out_dir}/results_test.json", "w") as f:
                    json.dump(results_test, f)

    if worker_id == 0:
        wandb.finish()
