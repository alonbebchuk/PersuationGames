import json
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer
from typing import Any


DATASET_TO_VIDEO_NAME_KEY = {"Ego4D": "EG_ID", "Youtube": "video_name"}
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def load_dataset(
    args: Any,
    tokenizer: BertTokenizer,
    mode: str,
) -> Dataset:
    all_id, all_input_ids, all_input_mask, all_labels = [], [], [], []

    json_path = f"{args.data_dir}/{mode}.json"
    with open(json_path, "r") as f:
        games = json.load(f)

    for game in games:
        dialogues = game["Dialogue"]
        context = [[]] * args.context_size

        for i, record in enumerate(dialogues):
            id = f"{game[DATASET_TO_VIDEO_NAME_KEY[args.dataset]]}_{game['Game_ID']}_{i + 1}"

            labels = [1 if strategy in record["annotation"] else 0 for strategy in STRATEGIES]
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

            all_id.append(id)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_labels.append(labels)

    dataset_dict = {
        "id": all_id,
        "input_ids": np.array(all_input_ids, dtype=np.int32),
        "attention_mask": np.array(all_input_mask, dtype=np.int32),
        "labels": np.array(all_labels, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
