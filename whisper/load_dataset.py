import io
import json
import librosa
import os
import requests
import numpy as np
from datasets import Dataset
from typing import Any
from transformers import WhisperTokenizer, WhisperFeatureExtractor

DATASET_TO_VIDEO_NAME_KEY = {"Ego4d": "EG_ID", "Youtube": "video_name"}
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
SAMPLING_RATE = 16000
MAX_SAMPLE_LENGTH = SAMPLING_RATE * 30

def load_dataset(
    args: Any,
    strategy: str,
    tokenizer: WhisperTokenizer,
    feature_extractor: WhisperFeatureExtractor,
    mode: str,
) -> Dataset:
    all_tokens, all_label, all_input_features, all_attention_mask = [], [], [], []

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

                tokens = ["<|startoftranscript|>", "<|notimestamps|>", "<|startofprev|>"]
                if args.context_size != 0:
                    for cxt in context[-args.context_size:]:
                        tokens += cxt + ["<|startofprev|>"]
                context.append(tokenizer.tokenize(utterance))
                tokens += context[-1] + ["<|startofprev|>"]

                if len(tokens) > args.max_seq_length:
                    tokens = ["<|startoftranscript|>", "<|notimestamps|>", "<|startofprev|>"] + tokens[-args.max_seq_length + 3:]

                assert len(tokens) <= args.max_seq_length, f"{len(tokens)}, {utterance}"

                all_tokens.append(tokens)
                all_label.append(label)

                video_name_key = DATASET_TO_VIDEO_NAME_KEY[dataset]
                mp4_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/videos/{game[video_name_key]}_{game['Game_ID']}_{record['Rec_Id']}.mp4"
                
                response = requests.get(mp4_url)
                response.raise_for_status()
                
                audio_array, _ = librosa.load(io.BytesIO(response.content), sr=SAMPLING_RATE)
                if len(audio_array) > MAX_SAMPLE_LENGTH:
                    audio_array = audio_array[-MAX_SAMPLE_LENGTH:]
                
                features = feature_extractor(audio_array, sampling_rate=SAMPLING_RATE, return_tensors="np")
                input_features = features.input_features.squeeze()
                attention_mask = features.attention_mask.squeeze()
                
                all_input_features.append(input_features)
                all_attention_mask.append(attention_mask)

    dataset_dict = {
        "tokens": all_tokens,
        "labels": np.array(all_label, dtype=np.int32),
        "input_features": np.array(all_input_features, dtype=np.float32),
        "attention_mask": np.array(all_attention_mask, dtype=np.int32),
    }

    return Dataset.from_dict(dataset_dict)
