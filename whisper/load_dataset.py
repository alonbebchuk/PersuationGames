import json
import numpy as np
import os
import requests
from datasets import Dataset
from huggingface_hub import hf_hub_download
from prompt_builder import PromptBuilder
from pydub import AudioSegment
from transformers import WhisperTokenizer
from typing import Any


DATASET_TO_VIDEO_NAME_KEY = {
    "Ego4D": "EG_ID",
    "Youtube": "video_name",
}
DATASET_REPO_ID = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
SAMPLING_RATE = 16000


def timestamp_to_sample(
    timestamp: str
) -> int:
    m, s = map(int, timestamp.split(":"))
    return (m * 60 + s) * SAMPLING_RATE

def get_audio_array(
    dataset: str,
    filename: str,
) -> np.ndarray:
    mp4_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=f"{dataset}/videos/{filename}", repo_type="dataset", local_dir="/dev/shm")
    audio = AudioSegment.from_file(mp4_path, format="mp4").set_frame_rate(SAMPLING_RATE).set_channels(1)
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    os.remove(mp4_path)
    return audio_array


def load_data(
    args: Any,
    mode: str,
) -> None:
    json_path = f"{args.data_dir}/{mode}.json"
    if not os.path.exists(json_path):
        json_url = f"{HUGGINGFACE_DATASET_URL}/{args.dataset}/split/{mode}.json"

        response = requests.get(json_url)
        response.raise_for_status()
        games = response.json()

        with open(json_path, "w") as f:
            json.dump(games, f)

    audio_dir = f"{args.data_dir}/audios/{mode}"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

        for game in games:
            if args.dataset == "Ego4D":
                game_timestamp_url = f"{HUGGINGFACE_DATASET_URL}/Ego4D/game_timestamp/{game['EG_ID']}.txt"

                response = requests.get(game_timestamp_url)
                response.raise_for_status()
                num_splits = sum(line.split()[0] == game["Game_ID"] for line in response.text.strip().split("\n"))

                audio_arrays = []
                for i in range(num_splits):
                    audio_array = get_audio_array(args.dataset, f"{game['EG_ID']}_{game['Game_ID']}_{i + 1}.mp4")
                    audio_arrays.append(audio_array)

                np.save(f"{audio_dir}/{game['EG_ID']}_{game['Game_ID']}.npy", np.concatenate(audio_arrays))

                split = 0
                sample_offset = 0
                for i, dialogue in enumerate(game["Dialogue"]):
                    dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"])
                    if i > 0 and dialogue["sample"] == 0 and game["Dialogue"][i - 1]["sample"] != 0:
                        sample_offset += len(audio_arrays[split])
                        split += 1
                    dialogue["sample"] += sample_offset
            else:
                audio_array = get_audio_array(args.dataset, f"{game['video_name']}_{game['Game_ID']}.mp4")

                np.save(f"{audio_dir}/{game['video_name']}_{game['Game_ID']}.npy", audio_array)

                for dialogue in game["Dialogue"]:
                    dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"])

        with open(json_path, "w") as f:
            json.dump(games, f)



def load_dataset(
    args: Any,
    strategy: str,
    tokenizer: WhisperTokenizer,
    mode: str,
) -> Dataset:
    all_input_ids, all_input_mask, all_audio_path, all_start_sample, all_end_sample, all_label = [], [], [], [], [], []

    json_path = f"{args.data_dir}/{mode}.json"
    with open(json_path, "r") as f:
        games = json.load(f)

    audio_dir = f"{args.data_dir}/audios/{mode}"

    prompt_builder = PromptBuilder(args, strategy, tokenizer)

    for game in games:
        dialogues = game["Dialogue"]
        previous_utterence_tokens_list = []

        audio_path = f"{audio_dir}/{game[DATASET_TO_VIDEO_NAME_KEY[args.dataset]]}_{game['Game_ID']}.npy"

        for i, record in enumerate(dialogues):
            label = 1 if strategy in record["annotation"] else 0
            utterance = record["utterance"] + "\n"

            utterance_tokens = tokenizer.tokenize(utterance)
            tokens = prompt_builder.build_prompt_tokens(previous_utterence_tokens_list, utterance_tokens)
            if len(previous_utterence_tokens_list) == args.context_size:
                previous_utterence_tokens_list.pop(0)
            previous_utterence_tokens_list.append(utterance_tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = args.max_seq_length - len(input_ids)
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            input_mask = [0] * padding_length + input_mask

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length

            start_sample = dialogues[max(0, i - args.context_size)]["sample"]
            end_sample = dialogues[i + 1]["sample"] if i + 1 < len(dialogues) else -1

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_audio_path.append(audio_path)
            all_start_sample.append(start_sample)
            all_end_sample.append(end_sample)
            all_label.append(label)

    dataset_dict = {
        "decoder_input_ids": np.array(all_input_ids, dtype=np.int32),
        "decoder_attention_mask": np.array(all_input_mask, dtype=np.int32),
        "audio_path": all_audio_path,
        "start_sample": np.array(all_start_sample, dtype=np.int32),
        "end_sample": np.array(all_end_sample, dtype=np.int32),
        "labels": np.array(all_label, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
