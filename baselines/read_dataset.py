import io
import os
import numpy as np
import pandas as pd
import requests
import torch
import zipfile
from datasets import load_dataset
from torch.utils.data import TensorDataset

HUGGINGFACE_DATASET = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/tree/main"
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]
STRATEGIES_TO_ID = {
    "No Strategy": 0,
    "Identity Declaration": 1,
    "Accusation": 2,
    "Interrogation": 3,
    "Call for Action": 4,
    "Defense": 5,
    "Evidence": 6
}
SUPPORTED_DATASETS = ['Ego4D', 'YouTube']
SUPPORTED_MODES = ['test', 'train', 'val']


def get_feature_filename(dataset: str, game: dict) -> str:
    if dataset == 'Ego4D':
        return f'{game["EG_ID"]}_{game["Game_ID"]}.npy'
    else:
        return f'{game["video_name"]}_{game["Game_ID"]}.npy'


def load_werewolf_dataset(args, logger, strategy, tokenizer, mode):
    if strategy not in STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {STRATEGIES}")

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {SUPPORTED_MODES}")

    all_input_ids, all_input_mask, all_label, all_video_features = [], [], [], []

    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)

    for dataset in args.dataset:
        logger.info(f'{dataset} dataset:')
        if dataset not in SUPPORTED_DATASETS:
            raise NotImplementedError(f"Dataset {dataset} not supported")

        hf_dataset = load_dataset(HUGGINGFACE_DATASET, data_files=f"{dataset}/split/{mode}.json")

        id = 0
        video_features = None

        for game in hf_dataset:
            dialogues = game["Dialogue"]
            context = [[]] * args.context_size

            if args.video:
                feature_zip_url = f"{HUGGINGFACE_DATASET_URL}/{dataset}/mvit_24_k400_features.zip"
                response = requests.get(feature_zip_url)

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    feature_file = get_feature_filename(dataset, game)
                    with z.open(feature_file) as f:
                        video_features = np.load(io.BytesIO(f.read()))

                logger.info(f'Loaded video features for {feature_file}')

            for record in dialogues:
                id += 1
                label = 1 if strategy in record['annotation'] else 0
                utterance = record['utterance']

                tokens = [tokenizer.cls_token]
                if args.context_size != 0:
                    for cxt in context[-args.context_size:]:
                        tokens += cxt + ['<end of text>']
                    tokens += [tokenizer.sep_token]
                context.append(tokenizer.tokenize(utterance))
                tokens += context[-1] + [tokenizer.sep_token]

                if len(tokens) > args.max_seq_length:
                    logger.info(f'too long, {len(tokens)}')
                    tokens = [tokenizer.cls_token] + tokens[-args.max_seq_length + 1:]
                    logger.info(len(tokens), tokens)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                assert len(tokens) <= args.max_seq_length, f"{len(tokens)}, {utterance}"

                padding_length = args.max_seq_length - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * padding_length
                input_mask += [0] * padding_length

                assert len(input_ids) == args.max_seq_length
                assert len(input_mask) == args.max_seq_length

                if id % 2000 == 1:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {id}")
                    logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    logger.info(f"label: {label}")

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_label.append(label)

                if args.video:
                    video_feature = video_features[record["Rec_Id"] - 1]
                    all_video_features.append(video_feature)

    if args.video:
        Dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
                                torch.tensor(all_input_mask, dtype=torch.long),
                                torch.tensor(all_label, dtype=torch.long),
                                torch.tensor(all_video_features, dtype=torch.float32))
    else:
        Dataset = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
                                torch.tensor(all_input_mask, dtype=torch.long),
                                torch.tensor(all_label, dtype=torch.long))
    return Dataset
