import os
import numpy as np
import requests
import shutil
import torch
import zipfile
import json
from logging import Logger
from torch.utils.data import TensorDataset
from typing import Dict, Any

HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]
STRATEGIES_TO_ID = {
    "No Strategy": 0,
    "Identity Declaration": 1,
    "Accusation": 2,
    "Interrogation": 3,
    "Call for Action": 4,
    "Defense": 5,
    "Evidence": 6,
}
SUPPORTED_DATASETS = ['Ego4D', 'Youtube']
SUPPORTED_MODES = ['test', 'train', 'val']


def get_feature_filename(dataset: str, game: Dict[str, Any]) -> str:
    if dataset == 'Ego4D':
        return f'{game["EG_ID"]}_{game["Game_ID"]}.npy'
    else:
        return f'{game["video_name"]}_{game["Game_ID"]}.npy'


def download_and_extract_features(dataset: str) -> str:
    dataset_dir = os.path.join('data', dataset)
    features_dir = os.path.join(dataset_dir, 'mvit_24_k400_features')
    zip_path = os.path.join(dataset_dir, f'mvit_24_k400_features.zip')

    if os.path.exists(features_dir):
        return features_dir

    if not os.path.exists(zip_path):
        feature_zip_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/mvit_24_k400_features.zip"
        os.makedirs(dataset_dir, exist_ok=True)

        response = requests.get(feature_zip_url)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    os.remove(zip_path)
    macosx_folder = os.path.join(dataset_dir, '__MACOSX')
    shutil.rmtree(macosx_folder)

    return features_dir


def load_werewolf_dataset(args: Any, strategy: str, tokenizer: Any, mode: str) -> TensorDataset:
    if strategy not in STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {STRATEGIES}")

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {SUPPORTED_MODES}")

    all_input_ids, all_input_mask, all_label, all_video_features = [], [], [], []

    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)

    for dataset in args.dataset:
        if dataset not in SUPPORTED_DATASETS:
            raise NotImplementedError(f"Dataset {dataset} not supported")

        local_path = os.path.join('data', dataset, 'split', f'{mode}.json')
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                games = json.load(f)
        else:
            json_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/split/{mode}.json"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            response = requests.get(json_url)
            response.raise_for_status()
            games = response.json()

            with open(local_path, 'w') as f:
                json.dump(games, f)

        if args.video:
            features_dir = download_and_extract_features(dataset)

        id = 0

        for game in games:
            dialogues = game["Dialogue"]
            context = [[]] * args.context_size

            video_features = None
            if args.video:
                feature_file = get_feature_filename(dataset, game)
                feature_path = os.path.join(features_dir, feature_file)
                video_features = np.load(feature_path)

            for record in dialogues:
                id += 1
                label = 1 if strategy in record['annotation'] else 0
                utterance = record['utterance']

                tokens = [tokenizer.cls_token]
                if args.context_size != 0:
                    for cxt in context[-args.context_size:]:
                        tokens += cxt + [tokenizer.sep_token]
                context.append(tokenizer.tokenize(utterance))
                tokens += context[-1] + [tokenizer.sep_token]

                if len(tokens) > args.max_seq_length:
                    tokens = [tokenizer.cls_token] + tokens[-args.max_seq_length + 1:]

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                assert len(tokens) <= args.max_seq_length, f"{len(tokens)}, {utterance}"

                padding_length = args.max_seq_length - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * padding_length
                input_mask += [0] * padding_length

                assert len(input_ids) == args.max_seq_length
                assert len(input_mask) == args.max_seq_length

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_label.append(label)

                if args.video:
                    video_feature = video_features[record["Rec_Id"] - 1]
                    all_video_features.append(video_feature)

    all_input_ids_array = np.array(all_input_ids, dtype=np.int64)
    all_input_mask_array = np.array(all_input_mask, dtype=np.int64)
    all_label_array = np.array(all_label, dtype=np.int64)

    if args.video:
        all_video_features_array = np.stack(all_video_features)
        Dataset = TensorDataset(torch.tensor(all_input_ids_array, dtype=torch.long),
                                torch.tensor(all_input_mask_array, dtype=torch.long),
                                torch.tensor(all_label_array, dtype=torch.long),
                                torch.tensor(all_video_features_array, dtype=torch.float32))
    else:
        Dataset = TensorDataset(torch.tensor(all_input_ids_array, dtype=torch.long),
                                torch.tensor(all_input_mask_array, dtype=torch.long),
                                torch.tensor(all_label_array, dtype=torch.long))
    return Dataset
