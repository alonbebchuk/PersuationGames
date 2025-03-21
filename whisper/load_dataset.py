import json
import librosa
import numpy as np
import os
import requests
import tempfile
from datasets import Dataset
from prompt_builder import PromptBuilder
from pydub import AudioSegment
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from typing import Any

HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
SAMPLING_RATE = 16000
MAX_SAMPLE_LENGTH = SAMPLING_RATE * 30


def get_audio_array(mp4_url: str) -> np.ndarray:
    response = requests.get(mp4_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_mp4_file, tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav_file:
        temp_mp4_file.write(response.content)
        temp_mp4_file.flush()

        audio = AudioSegment.from_file(temp_mp4_file.name, format="mp4")
        audio.export(temp_wav_file.name, format="wav")

        audio_array, _ = librosa.load(temp_wav_file.name, sr=SAMPLING_RATE)

    return audio_array


def get_youtube_audio_array(game: dict) -> np.ndarray:
    mp4_url = f"{HUGGINGFACE_DATASET_URL}/Youtube/videos/{game['video_name']}_{game['Game_ID']}.mp4"
    return get_audio_array(mp4_url)


def timestamp_to_sample(timestamp: str) -> int:
    m, s = map(int, timestamp.split(":"))
    return (m * 60 + s) * SAMPLING_RATE


def get_ego4d_audio_array(game: dict) -> np.ndarray:
    game_timestamp_url = f"{HUGGINGFACE_DATASET_URL}/Ego4D/game_timestamp/{game['EG_ID']}.txt"
    response = requests.get(game_timestamp_url)
    response.raise_for_status()

    num_splits = sum(line.split()[0] == game["Game_ID"] for line in response.text.strip().split("\n"))

    mp4_urls = [
        f"{HUGGINGFACE_DATASET_URL}/Ego4D/videos/{game['EG_ID']}_{game['Game_ID']}_{i + 1}.mp4"
        for i in range(num_splits)
    ]
    audio_arrays = [get_audio_array(mp4_url) for mp4_url in mp4_urls]

    split = 0
    sample_offset = 0
    for i, dialogue in enumerate(game["Dialogue"]):
        if i != 0 and dialogue["timestamp"] == "00:00":
            sample_offset += len(audio_arrays[split])
            split += 1
        dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"]) + sample_offset

    audio_array = np.concatenate(audio_arrays)
    return audio_array


DATASET_TO_GET_AUDIO_ARRAY = {
    "Ego4D": get_ego4d_audio_array,
    "Youtube": get_youtube_audio_array,
}


def load_dataset(
    args: Any,
    strategy: str,
    tokenizer: WhisperTokenizer,
    feature_extractor: WhisperFeatureExtractor,
    mode: str,
) -> Dataset:
    all_input_ids, all_input_mask, all_input_features, all_attention_mask, all_label = [], [], [], [], []

    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)

    prompt_builder = PromptBuilder(args, strategy, tokenizer)

    for dataset in args.dataset:
        local_path = os.path.join("whisper", "data", dataset, f"{mode}.json")
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

        for game in games:
            audio_array = DATASET_TO_GET_AUDIO_ARRAY[dataset](game)

            dialogues = game["Dialogue"]
            previous_utterence_tokens_list = []

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

                start_id = max(0, i - args.context_size)
                end_id = i + 1
                if "sample" in dialogues[i]:
                    record_start_sample = dialogues[start_id]["sample"]
                    record_end_sample = dialogues[end_id]["sample"] if end_id < len(dialogues) else len(audio_array)
                else:
                    record_start_sample = timestamp_to_sample(dialogues[start_id]["timestamp"])
                    record_end_sample = timestamp_to_sample(dialogues[end_id]["timestamp"]) if end_id < len(dialogues) else len(audio_array)
                record_audio_array = audio_array[record_start_sample:record_end_sample][-MAX_SAMPLE_LENGTH:]

                features = feature_extractor(record_audio_array, sampling_rate=SAMPLING_RATE, return_attention_mask=True, return_tensors="np")
                input_features = features.input_features.squeeze()
                attention_mask = features.attention_mask.squeeze()

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_input_features.append(input_features)
                all_attention_mask.append(attention_mask)
                all_label.append(label)

    dataset_dict = {
        "decoder_input_ids": np.array(all_input_ids, dtype=np.int32),
        "decoder_attention_mask": np.array(all_input_mask, dtype=np.int32),
        "input_features": np.array(all_input_features, dtype=np.float32),
        "attention_mask": np.array(all_attention_mask, dtype=np.int32),
        "labels": np.array(all_label, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
