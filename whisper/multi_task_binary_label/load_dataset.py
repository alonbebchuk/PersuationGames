import json
import numpy as np
from whisper.load_data import DATA_DIR
from whisper.prompt_builder import PromptBuilder
from datasets import Dataset
from transformers import WhisperTokenizer
from typing import Any, List, Tuple


STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def tokens_to_input_ids_and_attention_mask(tokens: List[str], max_seq_length: int, tokenizer: WhisperTokenizer) -> Tuple[List[int], List[int]]:
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    input_mask = input_mask + [0] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask


def load_dataset(args: Any, tokenizer: WhisperTokenizer, mode: str) -> Dataset:
    all_id, all_audio_path, all_start_sample, all_end_sample, all_strategy, all_label, all_input_ids, all_input_mask = [], [], [], [], [], [], [], []

    json_path = f"{DATA_DIR}/{mode}.json"
    with open(json_path, "r") as f:
        games = json.load(f)

    prompt_builder = PromptBuilder(args, tokenizer)

    for game in games:
        dialogues = game["Dialogue"]
        context = []

        for i, record in enumerate(dialogues):
            id = f"{game['video_name']}_{game['Game_ID']}_{i + 1}"

            audio_path = game["audio_path"]
            start_sample = dialogues[max(0, i - args.context_size)]["sample"]
            end_sample = dialogues[i + 1]["sample"] if i + 1 < len(dialogues) else -1

            utterance = f"{record['utterance']}\n"
            utterance_tokens = tokenizer.tokenize(utterance)

            for strategy in STRATEGIES:
                label = 1 if strategy in record["annotation"] else 0

                tokens = prompt_builder.get_strategy_prompt_tokens(strategy, context, utterance_tokens)
                input_ids, input_mask = tokens_to_input_ids_and_attention_mask(tokens, args.max_seq_length, tokenizer)

                all_id.append(id)
                all_audio_path.append(audio_path)
                all_start_sample.append(start_sample)
                all_end_sample.append(end_sample)
                all_strategy.append(strategy)
                all_label.append(label)
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)

            context.append(utterance_tokens)
            if len(context) > args.context_size:
                context.pop(0)

    dataset_dict = {
        "id": all_id,
        "audio_path": all_audio_path,
        "start_sample": np.array(all_start_sample, dtype=np.int32),
        "end_sample": np.array(all_end_sample, dtype=np.int32),
        "strategy": all_strategy,
        "labels": np.array(all_label, dtype=np.int32),
        "decoder_input_ids": np.array(all_input_ids, dtype=np.int32),
        "decoder_attention_mask": np.array(all_input_mask, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
