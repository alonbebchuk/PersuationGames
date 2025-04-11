import json
import numpy as np
from bert.load_data import DATA_DIR
from bert.prompt_builder import PromptBuilder
from datasets import Dataset
from transformers import BertTokenizer
from typing import Any, List, Tuple


def tokens_to_input_ids_and_attention_mask(tokens: List[str], max_seq_length: int, tokenizer: BertTokenizer) -> Tuple[List[int], List[int]]:
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    input_mask = input_mask + [0] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask


def load_dataset(args: Any, tokenizer: BertTokenizer, mode: str) -> Dataset:
    all_id, all_labels, all_input_ids, all_input_mask = [], [], [], []

    json_path = f"{DATA_DIR}/{mode}.json"
    with open(json_path, "r") as f:
        games = json.load(f)

    prompt_builder = PromptBuilder(args, tokenizer)

    for game in games:
        dialogues = game["Dialogue"]
        context = []

        for i, record in enumerate(dialogues):
            id = f"{game['video_name']}_{game['Game_ID']}_{i + 1}"

            utterance = f"{record['utterance']}\n"
            utterance_tokens = tokenizer.tokenize(utterance)

            labels = [1 if strategy in record["annotation"] else 0 for strategy in STRATEGIES]

            tokens = prompt_builder.get_prompt_tokens(context, utterance_tokens)
            input_ids, input_mask = tokens_to_input_ids_and_attention_mask(tokens, args.max_seq_length, tokenizer)

            all_id.append(id)
            all_labels.append(labels)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)

            context.append(utterance_tokens)
            if len(context) > args.context_size:
                context.pop(0)

    dataset_dict = {
        "id": all_id,
        "labels": np.array(all_labels, dtype=np.int32),
        "input_ids": np.array(all_input_ids, dtype=np.int32),
        "attention_mask": np.array(all_input_mask, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
