import argparse
import json
import numpy as np
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="+", type=str, help="Name of dataset, Ego4D or Youtube or Ego4D Youtube")
parser.add_argument("--model_type", type=str, help="Model type, bert or roberta")
parser.add_argument("--seed", type=int, help="Random seed for initialization")
parser.add_argument("--output_dir", type=str, help="Output directory")
parser.add_argument("--get_result", action="store_true")
args = parser.parse_args()

STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def write_metrics(f: Any, results: Dict[str, List[float]]) -> None:
    for strategy in STRATEGIES:
        f.write(f"{np.mean(results[strategy]):.1f}({np.std(results[strategy]):.1f})\t")
    f.write(f"{np.mean(results['averaged_f1']):.1f}({np.std(results['averaged_f1']):.1f})\t"
            f"{np.mean(results['overall_accuracy']):.1f}({np.std(results['overall_accuracy']):.1f})\n")


def get_result(output_dir: str, mode: str) -> DefaultDict[str, List[float]]:
    results = defaultdict(list)
    for seed in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, seed)):
            continue
        with open(os.path.join(output_dir, seed, f"results_{mode}.json"), "r") as f:
            result = json.load(f)
            results["averaged_f1"].append(result["averaged_f1"] * 100)
            results["overall_accuracy"].append(result["overall_accuracy"] * 100)
            for strategy in STRATEGIES:
                results[strategy].append(result[strategy]["f1"] * 100)
    with open(os.path.join(output_dir, f"results_{mode}.txt"), "w") as f:
        write_metrics(f, results)
    return results


if __name__ == "__main__":
    get_result(args.output_dir, "test")
