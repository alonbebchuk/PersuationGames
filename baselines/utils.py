from typing import Dict, List, Any, DefaultDict
import argparse
import json
import numpy as np
import os
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", default='out', type=str)
parser.add_argument("--model", default='bert', type=str)
parser.add_argument("--batch_size", nargs='+', default=('8', '16'), type=str)
parser.add_argument("--learning_rate", nargs='+', default=('1e-5', '3e-5', '5e-5'), type=str)
parser.add_argument("--seed", default=None, type=str)
parser.add_argument("--context_size", nargs='+', default=(1, 3, 5, 7, 9), type=int)
parser.add_argument("--suffix", default='', type=str)
parser.add_argument("--analyze_context", action="store_true")
parser.add_argument("--get_best_hp", action="store_true")
parser.add_argument("--get_result", action="store_true")
args = parser.parse_args()

if args.model == 'bert':
    args.seed = args.seed or ['13', '42', '87']
elif args.model == 'roberta':
    args.seed = args.seed or ['227', '624', '817']

STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]


def write_metrics(f: Any, results: Dict[str, List[float]]) -> None:
    for strategy in STRATEGIES:
        f.write(f"{np.mean(results[strategy]):.1f}({np.std(results[strategy]):.1f})\t")
    f.write(f'{np.mean(results["averaged_f1"]):.1f}({np.std(results["averaged_f1"]):.1f})\t'
            f'{np.mean(results["overall_accuracy"]):.1f}({np.std(results["overall_accuracy"]):.1f})\n')


def get_result(result_dir: str, mode: str) -> DefaultDict[str, List[float]]:
    results = defaultdict(list)
    for seed in args.seed:
        with open(os.path.join(result_dir, seed, f"results_{mode}.json"), 'r') as f:
            result = json.load(f)
            results["averaged_f1"].append(result['averaged_f1'] * 100)
            results["overall_accuracy"].append(result['overall_accuracy'] * 100)
            for strategy in STRATEGIES:
                results[strategy].append(result[strategy]['f1'] * 100)
    with open(os.path.join(result_dir, f'results_{mode}.txt'), 'w') as f:
        write_metrics(f, results)
    return results


def analyze_context():
    with open(os.path.join(args.result_dir, args.model, "context", "result.txt"), 'w') as f:
        for context_size in args.context_size:
            f.write(f"context_size {context_size}: \n")
            dir_name = os.path.join(args.result_dir, args.model, "context", str(context_size))
            results = get_result(dir_name, "test")
            write_metrics(f, results)


def get_best_hp():
    best_results = None
    with open(os.path.join(args.result_dir, args.model, f"final_results{args.suffix}.txt"), "a") as f_final:
        for bs in args.batch_size:
            for lr in args.learning_rate:
                id = f"{bs}_{lr}{args.suffix}"
                f_final.write(f"{id}'s dev result:\n")
                result_dir = os.path.join(args.result_dir, args.model, id)
                results = get_result(result_dir, "dev")
                write_metrics(f_final, results)

                if best_results is None or np.mean(results["averaged_f1"]) > np.mean(best_results["dev"]["averaged_f1"]):
                    best_results = {"dev": results, "id": id, "test": None}
                    best_results["test"] = get_result(result_dir, "test")

        f_final.write(f"best hyper parameters: {best_results['id']}\n")
        f_final.write(f"test result:\n")
        results = best_results['test']
        write_metrics(f_final, results)


if __name__ == "__main__":
    if args.analyze_context:
        analyze_context()
    if args.get_best_hp:
        get_best_hp()
    if args.get_result:
        get_result(args.result_dir, "test")
