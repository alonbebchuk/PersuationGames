import json
import os
import requests
import numpy as np
from datasets import Dataset
from typing import Any
from transformers import BertTokenizer

HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"


def load_dataset(
    args: Any,
    strategy: str,
    tokenizer: BertTokenizer,
    mode: str,
) -> Dataset:
    all_input_ids, all_input_mask, all_label = [], [], []

    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)

    for dataset in args.dataset:
        local_path = os.path.join("bert", "data", dataset, f"{mode}.json")
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                games = json.load(f)
        else:
            json_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/split/{mode}.json"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            response = requests.get(json_url)
            response.raise_for_status()
            games = response.json()

            with open(local_path, "w") as f:
                json.dump(games, f)

        id = 0
        for game in games:
            dialogues = game["Dialogue"]
            context = [[]] * args.context_size

            for record in dialogues:
                id += 1
                label = 1 if strategy in record["annotation"] else 0
                utterance = record["utterance"]

                tokens = []
                if args.context_size != 0:
                    for cxt in context[-args.context_size:]:
                        tokens += cxt + ["[unused0]"]
                    tokens += [tokenizer.sep_token]
                context.append(tokenizer.tokenize(utterance))
                tokens += context[-1]
                tokens = [tokenizer.cls_token] + tokens[-args.max_seq_length + 2:] + [tokenizer.sep_token]

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                padding_length = args.max_seq_length - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * padding_length
                input_mask += [0] * padding_length

                assert len(input_ids) == args.max_seq_length
                assert len(input_mask) == args.max_seq_length

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_label.append(label)

    dataset_dict = {
        "input_ids": np.array(all_input_ids, dtype=np.int32),
        "attention_mask": np.array(all_input_mask, dtype=np.int32),
        "labels": np.array(all_label, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
