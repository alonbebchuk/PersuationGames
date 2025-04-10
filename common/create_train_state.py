import jax
import jax.numpy as jnp
import optax
from common.load_datasets import STRATEGIES
from flax import struct
from flax.training import train_state
from flax.training.common_utils import onehot
from models.flax_whisper_for_sequence_classification import FlaxWhisperForSequenceClassification
from transformers import FlaxBertForSequenceClassification
from typing import Any, Callable, Dict, Tuple


class TrainState(train_state.TrainState):
    apply_inputs_fn: Callable = struct.field(pytree_node=False)
    learning_rate_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)
    preds_fn: Callable = struct.field(pytree_node=False)


def create_model(args: Any) -> FlaxBertForSequenceClassification | FlaxWhisperForSequenceClassification:
    model_class = FlaxBertForSequenceClassification if "bert" in args.model_type else FlaxWhisperForSequenceClassification

    if args.strategy is not None:
        model = model_class.from_pretrained(args.model_name, num_labels=2)
    else:
        model = model_class.from_pretrained(args.model_name, problem_type="multi_label_classification", num_labels=len(STRATEGIES))

    return model


def create_learning_rate_fn(args: Any, train_ds_size: int) -> Callable[[int], float]:
    steps_per_epoch = train_ds_size // args.batch_size
    num_train_steps = steps_per_epoch * args.num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=args.learning_rate, transition_steps=args.warmup_steps)
    decay_fn = optax.linear_schedule(init_value=args.learning_rate, end_value=0, transition_steps=num_train_steps - args.warmup_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[args.warmup_steps])
    return schedule_fn


def bert_apply_inputs(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    apply_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    return apply_inputs


def whisper_apply_inputs(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    apply_inputs = {
        "decoder_input_ids": batch["decoder_input_ids"],
        "decoder_attention_mask": batch["decoder_attention_mask"],
        "input_features": batch["input_features"],
        "attention_mask": batch["attention_mask"],
    }
    return apply_inputs


def get_apply_inputs_fn(args: Any) -> Callable[[Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]]:
    apply_inputs_fn = bert_apply_inputs if "bert" in args.model_type else whisper_apply_inputs
    return apply_inputs_fn


def loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=onehot(labels, num_classes=2))
    loss = jnp.mean(xentropy)
    return loss


def loss_mt(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    xentropy = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(xentropy)
    return loss


def preds(logits: jnp.ndarray) -> jnp.ndarray:
    preds = logits.argmax(-1)
    return preds


def preds_mt(logits: jnp.ndarray) -> jnp.ndarray:
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
    return preds


def get_loss_and_preds_fns(args: Any) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]:
    loss_fn, preds_fn = (loss, preds) if args.strategy is not None else (loss_mt, preds_mt)
    return loss_fn, preds_fn


def create_train_state(args: Any, train_ds_size: int) -> TrainState:
    model = create_model(args)
    apply_inputs_fn = get_apply_inputs_fn(args)
    learning_rate_fn = create_learning_rate_fn(args, train_ds_size)
    loss_fn, preds_fn = get_loss_and_preds_fns(args)
    optimizer = optax.adam(b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_epsilon, learning_rate=learning_rate_fn)

    train_state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        apply_inputs_fn=apply_inputs_fn,
        learning_rate_fn=learning_rate_fn,
        loss_fn=loss_fn,
        preds_fn=preds_fn,
        tx=optimizer,
    )
    return train_state
