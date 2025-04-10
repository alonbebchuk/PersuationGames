import json
import numpy as np
from common.prompt_builder import PromptBuilder
from datasets import Dataset
from scripts.load_data import DATA_DIR
from typing import Any, Tuple


STRATEGIES = ["Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation"]


def load_datasets(args: Any) -> Tuple[Dataset, Dataset, Dataset]:
    prompt_builder = PromptBuilder(args)

    dataset_dict = {}
    for mode in ["train", "val", "test"]:
        dataset = {
            "id": [],
            "labels": [],
            "input_ids": [],
            "attention_mask": [],
            "audio_path": [],
            "start_sample": [],
            "end_sample": [],
        }

        json_path = f"{DATA_DIR}/{mode}.json"
        with open(json_path, "r") as f:
            games = json.load(f)

        for game in games:
            prompt_builder.reset_context()

            dialogues = game["Dialogue"]
            for i, dialogue in enumerate(dialogues):
                video_game_dialogue_id = f"{game['video_name']}_{game['Game_ID']}_{i + 1}"
                dataset["id"].append(video_game_dialogue_id)

                if args.strategy is not None:
                    labels = 1 if args.strategy in dialogue["annotation"] else 0
                else:
                    labels = [1 if s in dialogue["annotation"] else 0 for s in STRATEGIES]
                dataset["labels"].append(labels)

                input_ids, attention_mask = prompt_builder.get_input_ids_and_attention_mask(dialogue["utterance"])
                dataset["input_ids"].append(input_ids)
                dataset["attention_mask"].append(attention_mask)

                audio_path = game["audio_path"]
                dataset["audio_path"].append(audio_path)

                start_sample = dialogues[max(0, i - args.context_size)]["sample"]
                end_sample = dialogues[i + 1]["sample"] if i + 1 < len(dialogues) else -1
                dataset["start_sample"].append(start_sample)
                dataset["end_sample"].append(end_sample)

        for key in dataset:
            if key not in ["id", "audio_path"]:
                dataset[key] = np.array(dataset[key], dtype=np.int32)

        dataset_dict[mode] = Dataset.from_dict(dataset)

    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["val"]
    test_dataset = dataset_dict["test"]
    return train_dataset, val_dataset, test_dataset
