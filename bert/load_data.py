import argparse
import json
import os
import requests
from typing import Any


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Name of dataset, Ego4D or Youtube")
args = parser.parse_args()

args.data_dir = f"/dev/shm/data/bert/{args.dataset}"

os.makedirs(args.data_dir, exist_ok=True)


HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"


def load_data(
    args: Any,
    mode: str,
) -> None:
    json_path = f"{args.data_dir}/{mode}.json"
    if not os.path.exists(json_path):
        json_url = f"{HUGGINGFACE_DATASET_URL}/{args.dataset}/split/{mode}.json"

        response = requests.get(json_url)
        response.raise_for_status()
        games = response.json()

        with open(json_path, "w") as f:
            json.dump(games, f)


if __name__ == "__main__":
    load_data(args, "train")
    load_data(args, "val")
    load_data(args, "test")
