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
from typing import Callable, List, Dict, Any, Tuple, Optional
from datasets import Dataset, DatasetDict

MODEL_CLASS = lambda: FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
TOKENIZER_CLASS = lambda: BertTokenizer.from_pretrained("bert-base-uncased")

logger = log.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs='+', type=str, help="Name of dataset, Ego4D or Youtube or Ego4D Youtube")
parser.add_argument("--context_size", type=int, help="Size of the context")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--learning_rate", type=float, help="The initial learning rate for Adam.")
parser.add_argument("--seed", type=int, help="Random seed for initialization")
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
# not used
parser.add_argument("--no_train", action="store_true", help="Whether to run training.")
parser.add_argument("--no_eval", action="store_true", help="Whether to run eval on the val set.")
parser.add_argument("--no_test", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--no_evaluate_during_training", action="store_true", help="Whether to run evaluation every epoch.")
parser.add_argument("--evaluate_period", default=1, type=int, help="evaluate every * epochs.")
parser.add_argument('--eval_batch_size', default=128, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--warmup_steps', default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument('--early_stopping_patience', default=10000, type=int, help="Patience for early stopping.")
parser.add_argument('--logging_steps', default=40, type=int, help="Log every X updates steps.")
parser.add_argument("--num_workers", type=int, default=min(8, mp.cpu_count()), help="Number of worker processes for data loading")
parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")
parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster data transfer")

args = parser.parse_args()

if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.no_train and not args.overwrite_output_dir):
    raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
logger.setLevel(log.INFO)
formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

fh = log.FileHandler(os.path.join(args.output_dir, 'log.txt'))
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

    def cross_entropy_loss(logits, labels):
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
        'input_ids': np.array([x['input_ids'] for x in batch]),
        'attention_mask': np.array([x['attention_mask'] for x in batch]),
        'labels': np.array([x['labels'] for x in batch])
    }

def train(
    model: FlaxBertForSequenceClassification,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Tuple[int, float, float]:
    tb_writer = SummaryWriter(args.output_dir)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn
    )

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        args.batch_size,
        args.num_train_epochs,
        args.warmup_steps,
        args.learning_rate,
    )

    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(model, learning_rate_fn, weight_decay=args.weight_decay)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    global_step = 0
    wait_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    best_f1 = 0
    tr_loss, logging_loss = 0.0, 0.0

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc='Epoch')

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            rng, dropout_rng = jax.random.split(rng)
            
            def train_step(state, batch, dropout_rng):
                def loss_fn(params):
                    outputs = state.apply_fn(
                        **{"params": params},
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        train=True,
                        dropout_rng=dropout_rng
                    )
                    return state.loss_fn(outputs.logits, batch["labels"])

                grad_fn = jax.value_and_grad(loss_fn)
                loss, grads = grad_fn(state.params)
                state = state.apply_gradients(grads=grads)
                return state, loss

            # JIT compile the training step
            train_step = jax.jit(train_step)

            state, loss = train_step(state, batch, dropout_rng)
            tr_loss += loss.item()
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
            
            if results['f1'] >= best_f1:
                best_f1 = results['f1']
                wait_step = 0
                output_dir = os.path.join(args.output_dir, "best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving best model to %s", output_dir)
                model.save_pretrained(output_dir, params=state.params)
                with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
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
    mode: str = 'val',
    prefix: str = '',
) -> Dict[str, Any]:
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate_fn
    )

    logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    labels = []

    @jax.jit
    def eval_step(params, batch):
        outputs = state.apply_fn(
            **{"params": params},
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            train=False
        )
        return outputs.loss, state.logits_fn(outputs.logits)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: jnp.array(v) for k, v in batch.items()}
        loss, pred = eval_step(state.params, batch)
        eval_loss += loss.item()
        nb_eval_steps += 1
        preds.extend(pred.tolist())
        labels.extend(batch["labels"].tolist())

    eval_loss /= nb_eval_steps

    assert len(labels) == len(preds), f"{len(labels)}, {len(preds)}"
    correct = [1 if pred == label else 0 for pred, label in zip(preds, labels)]
    
    if prefix == 'final':
        results = {
            'f1': f1_score(y_true=labels, y_pred=preds),
            'precision': precision_score(y_true=labels, y_pred=preds),
            'recall': recall_score(y_true=labels, y_pred=preds),
            'accuracy': accuracy_score(y_true=labels, y_pred=preds),
            'report': classification_report(y_true=labels, y_pred=preds),
            'correct': correct,
            'preds': preds,
        }
    else:
        results = {
            'f1': f1_score(y_true=labels, y_pred=preds),
            'loss': eval_loss,
        }
    logger.info(results['f1'])
    return results


def log_predictions(
    splits: List[str],
    preds: Dict[str, Dict[str, List[int]]],
) -> None:
    for dataset in args.dataset:
        for split in splits:
            with open(os.path.join('data', dataset, f'{split}.json'), 'r') as f:
                games = json.load(f)
            id = -1
            data = defaultdict(list)
            for game in games:
                for record in game['Dialogue']:
                    id += 1
                    pred = []
                    for strategy in STRATEGIES:
                        if preds[split][strategy][id] == 1:
                            pred.append(strategy)
                    if len(pred) == 0:
                        pred.append("No Strategy")
                    for key, val in record.items():
                        data[key].append(val)
                    data['prediction'].append(pred)
            df = pd.DataFrame.from_dict(data)
            df.to_csv(os.path.join(args.output_dir, f'predictions_{split}.csv'))


def process_strategy(
    strategy: str,
    output_dir: str,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    logger.info(f"Training for strategy {strategy}")
    strategy_output_dir = os.path.join(output_dir, strategy)
    if not os.path.exists(strategy_output_dir):
        os.makedirs(strategy_output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    jax.random.PRNGKey(args.seed)

    model = MODEL_CLASS()

    results = {}

    if not args.no_train and train_dataset is not None:
        global_step, tr_loss, best_f1 = train(model, train_dataset, val_dataset)
        logger.info(" global_step = %s, average loss = %s, best eval f1 = %s", global_step, tr_loss, best_f1)
        logger.info("Reloading best model")
        model = MODEL_CLASS.from_pretrained(os.path.join(strategy_output_dir, 'best'))

    if not args.no_eval and val_dataset is not None:
        results['val'] = evaluate(model, val_dataset, mode="val", prefix='final')
        filename = os.path.join(strategy_output_dir, 'results_val.json')
        with open(filename, 'w') as f:
            json.dump(results['val'], f)

    if not args.no_test and test_dataset is not None:
        results['test'] = evaluate(model, test_dataset, mode="test", prefix='final')
        filename = os.path.join(strategy_output_dir, 'results_test.json')
        with open(filename, 'w') as f:
            json.dump(results['test'], f)

    return strategy, results


def main() -> None:
    mp.set_start_method('spawn', force=True)

    logger.info("------NEW RUN-----")
    logger.info("random seed %s", args.seed)
    logger.info("Training/evaluation parameters %s", args)
    logger.info(f"Using {args.num_workers} workers for data loading")

    all_result = {'val': {}, 'test': {}}
    all_correct = {'val': None, 'test': None}
    output_dir = args.output_dir
    preds = {'val': {}, 'test': {}}
    averaged_f1 = {'val': 0.0, 'test': 0.0}
    splits = []

    if not args.no_eval:
        splits.append('val')
    if not args.no_test:
        splits.append('test')

    base_tokenizer = TOKENIZER_CLASS()

    strategy_datasets = {}
    for strategy in STRATEGIES:
        datasets = {}
        if not args.no_train:
            datasets['train'] = load_werewolf_dataset(args, strategy, base_tokenizer, mode='train')
        if not args.no_eval:
            datasets['val'] = load_werewolf_dataset(args, strategy, base_tokenizer, mode='val')
        if not args.no_test:
            datasets['test'] = load_werewolf_dataset(args, strategy, base_tokenizer, mode='test')
        strategy_datasets[strategy] = DatasetDict(datasets)

    for strategy in STRATEGIES:
        datasets = strategy_datasets[strategy]
        strategy_results = process_strategy(
            strategy,
            output_dir,
            datasets.get('train'),
            datasets.get('val'),
            datasets.get('test')
        )
        
        strategy, results = strategy_results
        if 'val' in results:
            all_result['val'][strategy] = results['val']
            if all_correct['val'] is None:
                all_correct['val'] = results['val']['correct']
            else:
                all_correct['val'] = [x + y for x, y in zip(all_correct['val'], results['val']['correct'])]
            preds['val'][strategy] = results['val']['preds']
            averaged_f1['val'] += results['val']['f1']

        if 'test' in results:
            all_result['test'][strategy] = results['test']
            if all_correct['test'] is None:
                all_correct['test'] = results['test']['correct']
            else:
                all_correct['test'] = [x + y for x, y in zip(all_correct['test'], results['test']['correct'])]
            preds['test'][strategy] = results['test']['preds']
            averaged_f1['test'] += results['test']['f1']

    args.output_dir = output_dir
    log_predictions(splits, preds)

    for split in splits:
        result = all_result[split]
        cnt = 0
        for x in all_correct[split]:
            if x == len(STRATEGIES):
                cnt += 1
        result['overall_accuracy'] = cnt / len(all_correct[split])
        result['averaged_f1'] = averaged_f1[split] / len(STRATEGIES)

        filename = os.path.join(args.output_dir, f'results_{split}.json')
        with open(filename, 'w') as f:
            json.dump(result, f)

        with open(os.path.join(args.output_dir, f"results_{split}_beaut.txt"), 'w') as f:
            for strategy in STRATEGIES:
                f.write(f"{result[strategy]['f1'] * 100:.1f}\t")
            f.write(f"{result['averaged_f1'] * 100:.1f}\t{result['overall_accuracy'] * 100:.1f}\n")

            for strategy in STRATEGIES:
                report = result[strategy]['report']
                result[strategy].pop('report')
                f.write(f"{strategy}\n")
                json.dump(result[strategy], f, indent=4)
                f.write(report)
                f.write("\n")


if __name__ == "__main__":
    main()
