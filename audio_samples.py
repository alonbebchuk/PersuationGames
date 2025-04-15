import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

DATASET_REPO_ID = "bolinlai/Werewolf-Among-Us"
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/resolve/main/Youtube/split"
SAMPLING_RATE = 16000


def timestamp_to_sample(timestamp: str) -> int:
    m, s = map(int, timestamp.split(":"))
    return (m * 60 + s) * SAMPLING_RATE


def download_and_convert_audio(game, output_dir):
    video_name = game['video_name']
    game_id = game['Game_ID']
    filename = f"{video_name}_{game_id}.mp4"
    output_filename = f"{video_name}_{game_id}.wav"
    output_path = os.path.join(output_dir, output_filename)
    mp4_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=f"Youtube/videos/{filename}", repo_type="dataset")
    audio = AudioSegment.from_file(mp4_path, format="mp4").set_frame_rate(SAMPLING_RATE).set_channels(1)
    audio.export(output_path, format="wav")
    os.remove(mp4_path)


def process_split(split, output_dir):
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)
    json_url = f"{HUGGINGFACE_DATASET_URL}/{split}.json"
    response = requests.get(json_url)
    response.raise_for_status()
    games = response.json()
    for game in tqdm(games):
        download_and_convert_audio(game, split_output_dir)


def main():
    output_dir = "audios"
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        process_split(split, output_dir)


if __name__ == "__main__":
    main()
