import argparse
import json
import numpy as np
import os
import requests
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from typing import Any


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Name of dataset, Ego4D or Youtube")
args = parser.parse_args()

args.data_dir = f"/dev/shm/whisper/data/{args.dataset}"

os.makedirs(args.data_dir, exist_ok=True)


DATASET_TO_VIDEO_NAME_KEY = {"Ego4D": "EG_ID", "Youtube": "video_name"}
DATASET_REPO_ID = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main"
SAMPLING_RATE = 16000


def timestamp_to_sample(
    timestamp: str,
) -> int:
    m, s = map(int, timestamp.split(":"))
    return (m * 60 + s) * SAMPLING_RATE


def get_audio_array(
    dataset: str,
    filename: str,
) -> np.ndarray:
    mp4_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=f"{dataset}/videos/{filename}", repo_type="dataset", local_dir="/dev/shm")
    audio = AudioSegment.from_file(mp4_path, format="mp4").set_frame_rate(SAMPLING_RATE).set_channels(1)
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    os.remove(mp4_path)
    return audio_array


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

    audio_dir = f"{args.data_dir}/audios/{mode}"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

        for game in games:
            if args.dataset == "Ego4D":
                game_timestamp_url = f"{HUGGINGFACE_DATASET_URL}/Ego4D/game_timestamp/{game['EG_ID']}.txt"

                response = requests.get(game_timestamp_url)
                response.raise_for_status()
                num_splits = sum(line.split()[0] == game["Game_ID"] for line in response.text.strip().split("\n"))

                audio_arrays = []
                for i in range(num_splits):
                    audio_array = get_audio_array(args.dataset, f"{game['EG_ID']}_{game['Game_ID']}_{i + 1}.mp4")
                    audio_arrays.append(audio_array)

                np.save(f"{audio_dir}/{game['EG_ID']}_{game['Game_ID']}.npy", np.concatenate(audio_arrays))
                game["audio_path"] = f"{audio_dir}/{game['EG_ID']}_{game['Game_ID']}.npy"

                split = 0
                sample_offset = 0
                for i, dialogue in enumerate(game["Dialogue"]):
                    dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"])
                    if i > 0 and dialogue["sample"] == 0 and game["Dialogue"][i - 1]["sample"] != 0:
                        sample_offset += len(audio_arrays[split])
                        split += 1
                    dialogue["sample"] += sample_offset
            else:
                audio_array = get_audio_array(args.dataset, f"{game['video_name']}_{game['Game_ID']}.mp4")

                np.save(f"{audio_dir}/{game['video_name']}_{game['Game_ID']}.npy", audio_array)
                game["audio_path"] = f"{audio_dir}/{game['video_name']}_{game['Game_ID']}.npy"

                for dialogue in game["Dialogue"]:
                    dialogue["sample"] = timestamp_to_sample(dialogue["timestamp"])

        with open(json_path, "w") as f:
            json.dump(games, f)


if __name__ == "__main__":
    load_data(args, "train")
    load_data(args, "val")
    load_data(args, "test")
