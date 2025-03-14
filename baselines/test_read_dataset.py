import itertools
import logging
import random
import torch
from collections import namedtuple
from read_dataset import load_werewolf_dataset, STRATEGIES, SUPPORTED_MODES, SUPPORTED_DATASETS
from transformers import AutoTokenizer


def test_load_werewolf_dataset():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    Args = namedtuple(
        'Args', ['dataset', 'context_size', 'max_seq_length', 'video'])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    dataset_combinations = []
    for i in range(1, len(SUPPORTED_DATASETS) + 1):
        dataset_combinations.extend(list(itertools.combinations(SUPPORTED_DATASETS, i)))

    dataset_combinations = [list(combo) for combo in dataset_combinations]
    context_sizes = list(range(10))
    max_seq_lengths = [128, 256, 512]
    video_options = [False, True]

    test_configs = []

    for dataset_combo in dataset_combinations:
        for context_size in context_sizes:
            for max_seq_length in max_seq_lengths:
                for video in video_options:
                    for strategy in STRATEGIES:
                        for mode in SUPPORTED_MODES:
                            test_configs.append({
                                'args': Args(dataset=dataset_combo,
                                             context_size=context_size,
                                             max_seq_length=max_seq_length,
                                             video=video),
                                'strategy': strategy,
                                'mode': mode,
                                'expected_sample_length': 4 if video else 3,
                                'name': f"Test with {'+'.join(dataset_combo)}, context={context_size}, "
                                f"seq_len={max_seq_length}, video={video}, strategy={strategy}, mode={mode}"
                            })

    for config in test_configs:
        logger.info(f"\n=== Running {config['name']} ===")
        args = config['args']
        strategy = config['strategy']
        mode = config['mode']

        try:
            loaded_dataset = load_werewolf_dataset(args, strategy, tokenizer, mode)

            assert isinstance(loaded_dataset, torch.utils.data.TensorDataset)
            assert len(loaded_dataset) > 0

            idx = random.randint(0, len(loaded_dataset) - 1)
            sample = loaded_dataset[idx]

            assert len(sample) == config['expected_sample_length']

            input_ids = sample[0]
            input_mask = sample[1]
            label = sample[2]

            assert input_ids.shape[0] == args.max_seq_length
            assert input_mask.shape[0] == args.max_seq_length
            assert label.dim() == 0
            assert label.item() in [0, 1]

            decoded_text = tokenizer.decode(input_ids)
            assert isinstance(decoded_text, str)
            assert len(decoded_text) > 0

            if args.video:
                video_features = sample[3]
                assert video_features.shape == torch.Size([3, 768])

            logger.info(f"Dataset length: {len(loaded_dataset)}")
            logger.info(f"Sample decoded text: {decoded_text[:100]}...")
            logger.info(f"Sample label: {label.item()}")
            logger.info(f"Test passed")
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


if __name__ == "__main__":
    test_load_werewolf_dataset()
