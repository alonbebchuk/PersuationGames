import json
import os
import requests


DATA_DIR = f"/dev/shm/data/bert"

os.makedirs(DATA_DIR, exist_ok=True)


HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main/Youtube/split"


def load_data(mode: str) -> None:
    json_path = f"{DATA_DIR}/{mode}.json"
    json_url = f"{HUGGINGFACE_DATASET_URL}/{mode}.json"
    if os.path.exists(json_path):
        print(f"Data file {json_path} already exists. Skipping download.")
        return

    response = requests.get(json_url)
    response.raise_for_status()
    games = response.json()

    with open(json_path, "w") as f:
        json.dump(games, f)


if __name__ == "__main__":
    load_data("train")
    load_data("val")
    load_data("test")
