import json
import numpy as np
import os
import requests
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from typing import Any


DATA_DIR = f"/dev/shm/data/whisper"
AUDIO_DATA_DIR = f"{DATA_DIR}/audios"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DATA_DIR, exist_ok=True)


DATASET_REPO_ID = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main/Youtube/split"
SAMPLING_RATE = 16000


def timestamp_to_sample(timestamp: str) -> int:
    m, s = map(int, timestamp.split(":"))
    return (m * 60 + s) * SAMPLING_RATE


def get_audio_array(filename: str) -> np.ndarray:
    mp4_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=f"Youtube/videos/{filename}", repo_type="dataset", local_dir=AUDIO_DATA_DIR)
    audio = AudioSegment.from_file(mp4_path, format="mp4").set_frame_rate(SAMPLING_RATE).set_channels(1)
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    os.remove(mp4_path)
    return audio_array


def load_data(mode: str) -> None:
    json_path = f"{DATA_DIR}/{mode}.json"
    if os.path.exists(json_path):
        print(f"Data file {json_path} already exists. Skipping download.")
        return
    json_url = f"{HUGGINGFACE_DATASET_URL}/{mode}.json"

    response = requests.get(json_url)
    response.raise_for_status()
    games = response.json()

    for game in games:
        audio_array = get_audio_array(f"{game['video_name']}_{game['Game_ID']}.mp4")

        np.save(f"{AUDIO_DATA_DIR}/{game['video_name']}_{game['Game_ID']}.npy", audio_array)
        game["audio_path"] = f"{AUDIO_DATA_DIR}/{game['video_name']}_{game['Game_ID']}.npy"

        for dialogue in game["Dialogue"]:
            dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"])

    with open(json_path, "w") as f:
        json.dump(games, f)


if __name__ == "__main__":
    load_data("train")
    load_data("val")
    load_data("test")
