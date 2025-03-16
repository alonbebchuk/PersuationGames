import json
import os
import requests
import torch
from torch.utils.data import TensorDataset
from typing import Any

HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]
STRATEGIES_TO_ID = {
    "No Strategy": 0,
    "Identity Declaration": 1,
    "Accusation": 2,
    "Interrogation": 3,
    "Call for Action": 4,
    "Defense": 5,
    "Evidence": 6,
}
SUPPORTED_DATASETS = ['Ego4D', 'Youtube']
SUPPORTED_MODES = ['test', 'train', 'val']


def load_werewolf_dataset(args: Any, strategy: str, tokenizer: Any, mode: str) -> TensorDataset:
    if strategy not in STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {STRATEGIES}")

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {SUPPORTED_MODES}")

    all_input_ids, all_input_mask, all_label = [], [], []

    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)

    for dataset in args.dataset:
        if dataset not in SUPPORTED_DATASETS:
            raise NotImplementedError(f"Dataset {dataset} not supported")

        local_path = os.path.join('data', dataset, 'split', f'{mode}.json')
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                games = json.load(f)
        else:
            json_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/split/{mode}.json"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            response = requests.get(json_url)
            response.raise_for_status()
            games = response.json()

            with open(local_path, 'w') as f:
                json.dump(games, f)

        id = 0

        for game in games:
            dialogues = game["Dialogue"]
            context = [[]] * args.context_size

            for record in dialogues:
                id += 1
                label = 1 if strategy in record['annotation'] else 0
                utterance = record['utterance']

                tokens = [tokenizer.cls_token]
                if args.context_size != 0:
                    for cxt in context[-args.context_size:]:
                        tokens += cxt + ['<end of text>']
                    tokens += [tokenizer.sep_token]
                context.append(tokenizer.tokenize(utterance))
                tokens += context[-1] + [tokenizer.sep_token]

                if len(tokens) > args.max_seq_length:
                    tokens = [tokenizer.cls_token] + tokens[-args.max_seq_length + 1:]

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                assert len(tokens) <= args.max_seq_length, f"{len(tokens)}, {utterance}"

                padding_length = args.max_seq_length - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * padding_length
                input_mask += [0] * padding_length

                assert len(input_ids) == args.max_seq_length
                assert len(input_mask) == args.max_seq_length

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_label.append(label)

    all_input_ids_array = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask_array = torch.tensor(all_input_mask, dtype=torch.long)
    all_label_array = torch.tensor(all_label, dtype=torch.long)

    Dataset = TensorDataset(all_input_ids_array, all_input_mask_array, all_label_array)
    return Dataset
