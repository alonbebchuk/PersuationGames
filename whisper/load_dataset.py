import json
import numpy as np
import os
import requests
import soundfile as sf
from datasets import Dataset
from huggingface_hub import hf_hub_download
from prompt_builder import PromptBuilder
from pydub import AudioSegment
from transformers import WhisperTokenizer
from typing import Any


BASE_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET_TO_VIDEO_NAME_KEY = {
    "Ego4D": "EG_ID",
    "Youtube": "video_name",
}
DATASET_REPO_ID = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
SAMPLING_RATE = 16000
MAX_SAMPLE_LENGTH = SAMPLING_RATE * 30


def DATASET_TO_BASE_PATH(dataset: str, mode: str) -> str:
    return f"{BASE_PATH}/data/{dataset}/audios/{mode}"


def load_dataset_audios(dataset: str, mode: str, games: list) -> str:
    audio_dir = DATASET_TO_BASE_PATH(dataset, mode)

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

        for game in games:
            if dataset == "Ego4D":
                game_timestamp_url = f"{HUGGINGFACE_DATASET_URL}/Ego4D/game_timestamp/{game['EG_ID']}.txt"
                response = requests.get(game_timestamp_url)
                response.raise_for_status()
                num_splits = sum(line.split()[0] == game["Game_ID"] for line in response.text.strip().split("\n"))
                video_filenames = [f"{game[DATASET_TO_VIDEO_NAME_KEY[dataset]]}_{game['Game_ID']}_{i + 1}" for i in range(num_splits)]
            else:
                video_filenames = [f"{game[DATASET_TO_VIDEO_NAME_KEY[dataset]]}_{game['Game_ID']}"]

            for video_filename in video_filenames:
                mp4_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=f"{dataset}/videos/{video_filename}.mp4", repo_type="dataset", local_dir="/tmp")
                wav_path = os.path.join(audio_dir, f"{video_filename}.wav")
                audio = AudioSegment.from_file(mp4_path, format="mp4").set_frame_rate(SAMPLING_RATE).set_channels(1)
                audio.export(wav_path, format="wav")
                os.remove(mp4_path)

    return audio_dir


def load_dataset_json(dataset: str, mode: str) -> list:
    mode_path = f"{BASE_PATH}/data/{dataset}/{mode}.json"

    if os.path.exists(mode_path):
        with open(mode_path, "r") as f:
            games = json.load(f)
    else:
        json_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/split/{mode}.json"
        os.makedirs(os.path.dirname(mode_path), exist_ok=True)

        response = requests.get(json_url)
        response.raise_for_status()

        games = response.json()
        with open(mode_path, "w") as f:
            json.dump(games, f)

    return games


def get_audio_array(audio_path: str) -> np.ndarray:
    audio_array, _ = sf.read(audio_path, dtype='float32')
    return audio_array


def ego4d_set_samples(audio_dir: str, game: dict) -> None:
    game_timestamp_url = f"{HUGGINGFACE_DATASET_URL}/Ego4D/game_timestamp/{game['EG_ID']}.txt"
    response = requests.get(game_timestamp_url)
    response.raise_for_status()

    num_splits = sum(line.split()[0] == game["Game_ID"] for line in response.text.strip().split("\n"))

    audio_paths = [f"{audio_dir}/{game['EG_ID']}_{game['Game_ID']}_{i + 1}.wav" for i in range(num_splits)]
    audio_arrays = [get_audio_array(audio_path) for audio_path in audio_paths]

    split = 0
    sample_offset = 0
    for i, dialogue in enumerate(game["Dialogue"]):
        if i != 0 and dialogue["sample"] == 0 and game["Dialogue"][i - 1]["sample"] != 0:
            sample_offset += len(audio_arrays[split])
            split += 1
        dialogue["sample"] += sample_offset

    audio_path = f"{audio_dir}/{game['EG_ID']}_{game['Game_ID']}.wav"
    if not os.path.exists(audio_path):
        sf.write(audio_path, np.concatenate(audio_arrays), SAMPLING_RATE)


def timestamp_to_sample(timestamp: str) -> int:
    m, s = map(int, timestamp.split(":"))
    return (m * 60 + s) * SAMPLING_RATE


def load_dataset(
    args: Any,
    strategy: str,
    tokenizer: WhisperTokenizer,
    mode: str,
) -> Dataset:
    all_input_ids, all_input_mask, all_audio_path, all_start_sample, all_end_sample, all_label = [], [], [], [], [], []

    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)

    prompt_builder = PromptBuilder(args, strategy, tokenizer)

    for dataset in args.dataset:
        games = load_dataset_json(dataset, mode)
        audio_dir = load_dataset_audios(dataset, mode, games)
        for game in games:
            dialogues = game["Dialogue"]
            previous_utterence_tokens_list = []

            for dialogue in game["Dialogue"]:
                dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"])
            if dataset == "Ego4D":
                ego4d_set_samples(audio_dir, game)

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

                audio_path = f"{audio_dir}/{game[DATASET_TO_VIDEO_NAME_KEY[dataset]]}_{game['Game_ID']}.wav"

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
