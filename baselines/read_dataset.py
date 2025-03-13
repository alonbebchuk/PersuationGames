from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import logging

Strategies = ["Identity Declaration", "Accusation",
              "Interrogation", "Call for Action", "Defense", "Evidence"]
strategies2id = {"No Strategy": 0, "Identity Declaration": 1, "Accusation": 2, "Interrogation": 3,
                 "Call for Action": 4, "Defense": 5, "Evidence": 6}
role2id = {'Moderator': 0, 'Villager': 1, 'Werewolf': 2, 'Seer': 3, 'Robber': 4, 'Troublemaker': 5,
           'Tanner': 6, 'Drunk': 7, 'Hunter': 8, 'Mason': 9, 'Insomniac': 10, 'Minion': 11, 'Doppelganger': 12}

def load_werewolf_dataset(args, logger, strategy, tokenizer, mode):
    all_input_ids = []
    all_input_mask = []
    all_label = []
    all_video_features = []
    
    if isinstance(args.dataset, str):
        args.dataset = (args.dataset,)
        
    for dataset in args.dataset:
        logger.info(f'{dataset} dataset:')
        hf_dataset = load_dataset("bolinlai/Werewolf-Among-Us", split=mode)
        
        id = 0
        video_features = None
        
        for game in hf_dataset:
            dialogues = game["Dialogue"]
            context = [[]] * args.context_size
            
            if args.video:
                if dataset == 'Ego4D':
                    video_features = np.load(os.path.join(
                        args.video_path, f'{game["EG_ID"]}_{game["Game_ID"]}.npy'))
                elif dataset == 'Youtube':
                    video_features = np.load(os.path.join(
                        args.video_path, f'{game["video_name"]}_{game["Game_ID"]}.npy'))
                else:
                    raise NotImplementedError
                logger.info(f'Loading video features from {args.video_path}')
            
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

def log_predictions(splits, preds):
    """Log predictions to CSV files.
    
    Args:
        splits: List of dataset splits
        preds: Dictionary of predictions for each split and strategy
    """
    for split in splits:
        data = {}
        for strategy in Strategies:
            data[strategy] = preds[split][strategy]
        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(args.output_dir, f'predictions_{split}.csv')) 