import json
import librosa
import numpy as np
import os
import requests
import tempfile
from datasets import Dataset
from prompt_builder import PromptBuilder
from pydub import AudioSegment
from typing import Any
from transformers import WhisperTokenizer, WhisperFeatureExtractor

DATASET_TO_VIDEO_NAME_KEY = {"Ego4D": "EG_ID", "Youtube": "video_name"}
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
            video_name_key = DATASET_TO_VIDEO_NAME_KEY[dataset]
            mp4_urls = [f"{HUGGINGFACE_DATASET_URL}/{dataset}/videos/{game[video_name_key]}_{game['Game_ID']}_{rid + 1}.mp4" for rid in range(2)]

            dialogues = game["Dialogue"]
            context = [[]] * args.context_size

            for record in dialogues:
                label = 1 if strategy in record["annotation"] else 0
                utterance = record["utterance"] + "\n"

                utterance_tokens = tokenizer.tokenize(utterance)
                tokens = prompt_builder.build_prompt_tokens(context, utterance_tokens)
                context.append(utterance_tokens)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                padding_length = args.max_seq_length - len(input_ids)
                input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
                input_mask = [0] * padding_length + input_mask

                assert len(input_ids) == args.max_seq_length
                assert len(input_mask) == args.max_seq_length

                video_name_key = DATASET_TO_VIDEO_NAME_KEY[dataset]
                mp4_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/videos/{game[video_name_key]}_{game['Game_ID']}_{record['Rec_Id'] - 1}.mp4"

                with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_mp4_file, tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav_file:
                    response = requests.get(mp4_url)
                    response.raise_for_status()
                    temp_mp4_file.write(response.content)
                    temp_mp4_file.flush()
                    
                    audio = AudioSegment.from_file(temp_mp4_file.name, format="mp4")
                    audio.export(temp_wav_file.name, format="wav")
                    
                    audio_array, _ = librosa.load(temp_wav_file.name, sr=SAMPLING_RATE)
                    audio_array = audio_array[-MAX_SAMPLE_LENGTH:]

                features = feature_extractor(audio_array, sampling_rate=SAMPLING_RATE, return_attention_mask=True, return_tensors="np")
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
