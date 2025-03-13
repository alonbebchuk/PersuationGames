import torch
from transformers import AutoTokenizer
import logging
from collections import namedtuple
import random
import requests
import importlib.util
import sys
from pathlib import Path
from read_dataset import load_werewolf_dataset

def fetch_and_load_read_data():
    # URL of the raw file
    url = "https://raw.githubusercontent.com/SALT-NLP/PersuationGames/main/baselines/read_data.py"
    
    # Create a temporary directory if it doesn't exist
    Path("temp").mkdir(exist_ok=True)
    
    # Download and save the file
    response = requests.get(url)
    with open("temp/read_data.py", "w") as f:
        f.write(response.text)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("read_data", "temp/read_data.py")
    read_data_module = importlib.util.module_from_spec(spec)
    sys.modules["read_data"] = read_data_module
    spec.loader.exec_module(read_data_module)
    
    return read_data_module.read_data

def compare_samples(new_sample, old_sample, tokenizer, video_enabled=False):
    """Helper function to compare and log sample differences"""
    # Compare labels
    assert torch.equal(new_sample[2], old_sample[1]), "Labels don't match"
    
    # Compare text features
    new_tokens = tokenizer.decode(new_sample[0])
    
    # Compare video features if enabled
    if video_enabled:
        assert torch.equal(new_sample[3], old_sample[2]), "Video features don't match"
        return {
            'new_label': new_sample[2].item(),
            'old_label': old_sample[1].item(),
            'new_tokens': new_tokens[:100],
            'new_video_shape': new_sample[3].shape,
            'old_video_shape': old_sample[2].shape,
            'video_features_equal': torch.equal(new_sample[3], old_sample[2])
        }
    else:
        return {
            'new_label': new_sample[2].item(),
            'old_label': old_sample[1].item(),
            'new_tokens': new_tokens[:100]
        }

def test_dataset_loaders_equivalence():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Fetch read_data function from GitHub
    read_data = fetch_and_load_read_data()
    
    # Create mock args
    Args = namedtuple('Args', ['dataset', 'context_size', 'max_seq_length', 'video', 'video_path'])
    args = Args(
        dataset=['Ego4D', 'YouTube'],  # Test both datasets
        context_size=2,
        max_seq_length=512,
        video=False,
        video_path='path/to/video/features'
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Test for all strategies and modes
    strategies = ["Identity Declaration", "Accusation", "Interrogation", 
                 "Call for Action", "Defense", "Evidence"]
    modes = ['train', 'test', 'val']
    
    for strategy in strategies:
        for mode in modes:
            logger.info(f"\nTesting strategy: {strategy}, mode: {mode}")
            
            # Test without video features
            new_dataset = load_werewolf_dataset(args, logger, strategy, tokenizer, mode)
            old_dataset = read_data(args, logger, strategy, tokenizer, mode)
            
            # Basic shape checks
            assert len(new_dataset) == len(old_dataset), \
                f"Dataset lengths don't match for {strategy} - {mode}"
            
            # Compare random sample
            idx = random.randint(0, len(new_dataset) - 1)
            new_sample = new_dataset[idx]
            old_sample = old_dataset[idx]
            
            # Compare and log samples
            comparison = compare_samples(new_sample, old_sample, tokenizer, video_enabled=False)
            logger.info(f"\nRandom sample comparison (index {idx}):")
            for key, value in comparison.items():
                logger.info(f"{key}: {value}")
            
            # Test with video features
            args = args._replace(video=True)
            new_dataset_with_video = load_werewolf_dataset(args, logger, strategy, tokenizer, mode)
            old_dataset_with_video = read_data(args, logger, strategy, tokenizer, mode)
            
            assert len(new_dataset_with_video) == len(old_dataset_with_video), \
                f"Dataset lengths don't match for {strategy} - {mode} with video"
            
            # Compare random sample with video
            idx = random.randint(0, len(new_dataset_with_video) - 1)
            new_sample = new_dataset_with_video[idx]
            old_sample = old_dataset_with_video[idx]
            
            # Compare and log samples with video
            comparison = compare_samples(new_sample, old_sample, tokenizer, video_enabled=True)
            logger.info(f"\nRandom sample comparison with video (index {idx}):")
            for key, value in comparison.items():
                logger.info(f"{key}: {value}")
            
            # Reset video setting for next iteration
            args = args._replace(video=False)

if __name__ == "__main__":
    test_dataset_loaders_equivalence()