import json
import numpy as np
from scripts.load_data import DATA_DIR, SAMPLING_RATE
from transformers import WhisperFeatureExtractor
from typing import Any, Callable, Dict, List

def bert_collator(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    collated_batch = {
        "id": [sample["id"] for sample in batch],
        "labels": np.array([sample["labels"] for sample in batch]),
        "input_ids": np.array([sample["input_ids"] for sample in batch]),
        "attention_mask": np.array([sample["attention_mask"] for sample in batch]),
    }
    return collated_batch


class WhisperCollator:
    MAX_SAMPLE_LENGTH = SAMPLING_RATE * 30

    def __init__(self, args: Any) -> None:
        self.args = args
        self.cache = self.load_cache()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)

    def load_cache(self) -> None:
        for mode in ["train", "val", "test"]:
            with open(f"{DATA_DIR}/{mode}.json", "r") as f:
                games = json.load(f)
                for game in games:
                    audio_path = game["audio_path"]
                    if audio_path not in self.cache:
                        audio_array = np.load(audio_path, mmap_mode="r")
                        self.cache[audio_path] = {
                            "array": audio_array, "length": len(audio_array)}

    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        ids = []
        labels = np.empty(len(batch), dtype=np.int32)
        decoder_input_ids = np.empty((len(batch), len(batch[0]["input_ids"])), dtype=np.int32)
        decoder_attention_mask = np.empty((len(batch), len(batch[0]["attention_mask"])), dtype=np.int32)
        audio_arrays = []

        for i, sample in enumerate(batch):
            ids.append(sample["id"])
            labels[i] = sample["labels"]
            decoder_input_ids[i] = sample["decoder_input_ids"]
            decoder_attention_mask[i] = sample["decoder_attention_mask"]

            audio = self.cache[sample["audio_path"]]
            end_sample = audio["length"] if sample["end_sample"] == -1 else sample["end_sample"]
            start_sample = max(end_sample - self.MAX_SAMPLE_LENGTH, sample["start_sample"])
            audio_arrays.append(audio["array"][start_sample:end_sample])

        features = self.feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            return_attention_mask=True,
            return_tensors="np",
            padding="max_length",
        )
        input_features = features["input_features"]
        attention_mask = features["attention_mask"]

        collated_batch = {
            "id": ids,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return collated_batch


def get_collator(args: Any) -> Callable[[List[Dict[str, np.ndarray]]], Dict[str, np.ndarray]]:
    collator = bert_collator if "bert" in args.model_type else WhisperCollator(args)
    return collator
