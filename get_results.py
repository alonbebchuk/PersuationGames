import argparse
import json
import numpy as np
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Name of dataset, Ego4D or Youtube")
parser.add_argument("--model_type", type=str, help="Model type, bert or whisper")
args = parser.parse_args()

args.output_dir = os.path.join(args.model_type, "out", args.dataset)

def write_metrics(f: Any, results: Dict[str, List[float]]) -> None:
    for strategy in STRATEGIES:
        f.write(f"{np.mean(results[strategy]):.1f}({np.std(results[strategy]):.1f})\t")
    f.write(f"{np.mean(results['averaged_f1']):.1f}({np.std(results['averaged_f1']):.1f})\t"
            f"{np.mean(results['overall_accuracy']):.1f}({np.std(results['overall_accuracy']):.1f})\n")


def get_result() -> DefaultDict[str, List[float]]:
    results = defaultdict(list)
    for seed in os.listdir(args.output_dir):
        if not os.path.isdir(os.path.join(args.output_dir, seed)):
            continue
        with open(os.path.join(args.output_dir, seed, f"results_test.json"), "r") as f:
            result = json.load(f)
            results["averaged_f1"].append(result["averaged_f1"] * 100)
            results["overall_accuracy"].append(result["overall_accuracy"] * 100)
            for strategy in STRATEGIES:
                results[strategy].append(result[strategy]["f1"] * 100)
    with open(os.path.join(args.output_dir, f"results_test.txt"), "w") as f:
        write_metrics(f, results)
    return results


if __name__ == "__main__":
    get_result()
