import json
import numpy as np
from datasets import Dataset
from prompt_builder import PromptBuilderWithoutTranscript, PromptBuilderWithTranscript
from transformers import WhisperTokenizer
from typing import Any


DATASET_TO_VIDEO_NAME_KEY = {"Ego4D": "EG_ID", "Youtube": "video_name"}


def load_dataset(
    args: Any,
    tokenizer: WhisperTokenizer,
    mode: str,
) -> Dataset:
    all_id, all_input_ids, all_input_mask, all_audio_path, all_start_sample, all_end_sample, all_label = [], [], [], [], [], [], []

    json_path = f"{args.data_dir}/{mode}.json"
    with open(json_path, "r") as f:
        games = json.load(f)

    if args.with_transcript:
        prompt_builder = PromptBuilderWithTranscript(args, tokenizer)
    else:
        prompt_builder = PromptBuilderWithoutTranscript(args, tokenizer)

    for game in games:
        dialogues = game["Dialogue"]
        previous_utterence_tokens_list = []

        for i, record in enumerate(dialogues):
            id = f"{game[DATASET_TO_VIDEO_NAME_KEY[args.dataset]]}_{game['Game_ID']}_{i + 1}"

            label = 1 if args.strategy in record["annotation"] else 0

            if args.with_transcript:
                utterance_tokens = tokenizer.tokenize(record["utterance"])
                tokens = prompt_builder.get_prompt_tokens(previous_utterence_tokens_list, utterance_tokens)
                if len(previous_utterence_tokens_list) == args.context_size:
                    previous_utterence_tokens_list.pop(0)
                previous_utterence_tokens_list.append(utterance_tokens)
            else:
                tokens = prompt_builder.get_prompt_tokens()

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = args.max_seq_length - len(input_ids)
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            input_mask = [0] * padding_length + input_mask

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length

            start_sample = dialogues[max(0, i - args.context_size)]["sample"]
            end_sample = dialogues[i + 1]["sample"] if i + 1 < len(dialogues) else -1

            all_id.append(id)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_audio_path.append(game["audio_path"])
            all_start_sample.append(start_sample)
            all_end_sample.append(end_sample)
            all_label.append(label)

    dataset_dict = {
        "id": all_id,
        "decoder_input_ids": np.array(all_input_ids, dtype=np.int32),
        "decoder_attention_mask": np.array(all_input_mask, dtype=np.int32),
        "audio_path": all_audio_path,
        "start_sample": np.array(all_start_sample, dtype=np.int32),
        "end_sample": np.array(all_end_sample, dtype=np.int32),
        "labels": np.array(all_label, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
