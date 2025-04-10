import json
import numpy as np
import os
import requests
import shutil
from huggingface_hub import hf_hub_download
from pydub import AudioSegment


DATA_DIR = "/dev/shm/data"
DATASET_REPO_ID = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main/Youtube/split"
SAMPLING_RATE = 16000


if __name__ == "__main__":
    for mode in ["train", "val", "test"]:
        json_file_url = f"{HUGGINGFACE_DATASET_URL}/{mode}.json"
        response = requests.get(json_file_url)
        response.raise_for_status()
        games = response.json()

        audio_dir = f"{DATA_DIR}/audios"
        for game in games:
            video_game_id = f"{game['video_name']}_{game['Game_ID']}"

            mp4_file_name = f"Youtube/videos/{video_game_id}.mp4"
            mp4_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=mp4_file_name, repo_type="dataset", local_dir=audio_dir)
            audio = AudioSegment.from_file(mp4_path, format="mp4").set_frame_rate(SAMPLING_RATE).set_channels(1)
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            os.remove(mp4_path)

            npy_file_name = f"{audio_dir}/{video_game_id}.npy"
            np.save(npy_file_name, audio_array)
            game["audio_path"] = npy_file_name

            dialogues = game["Dialogue"]
            for dialogue in dialogues:
                m, s = map(int, dialogue["timestamp"].split(":"))
                dialogue["sample"] = (m * 60 + s) * SAMPLING_RATE

        json_path = f"{DATA_DIR}/{mode}.json"
        with open(json_path, "w") as f:
            json.dump(games, f)

        video_dir = f"{audio_dir}/Youtube"
        shutil.rmtree(video_dir)
