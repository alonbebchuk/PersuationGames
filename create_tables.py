import json
import numpy as np
from pathlib import Path

SEEDS = [12, 42, 87]
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]

MODELS = [
    ("BERT ST", "bert/single_task", "single"),
    ("BERT MTBL", "bert/multi_task_binary_label", "multi"),
    ("BERT MTML", "bert/multi_task_multi_label", "multi"),
    ("Whisper YN ST", "whisper/yes_no/single_task", "single"),
    ("Whisper YN MTBL", "whisper/yes_no/multi_task_binary_label", "multi"),
    ("Whisper Proj ST", "whisper/projection/single_task", "single"),
    ("Whisper Proj MTBL", "whisper/projection/multi_task_binary_label", "multi"),
    ("Whisper Proj MTML", "whisper/projection/multi_task_multi_label", "multi"),
]

def load_single_task_results(base_path):
    results = {seed: {strategy: None for strategy in STRATEGIES} for seed in SEEDS}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            results_file = base_path / strategy / str(seed) / "results_test.json"
            with open(results_file) as f:
                data = json.load(f)
                results[seed][strategy] = {'f1': data['f1'], 'accuracy': data['accuracy']}
    return results

def load_multi_task_results(base_path):
    results = {strategy: [] for strategy in STRATEGIES}
    overall_results = []

    for seed in SEEDS:
        results_file = base_path / str(seed) / "results_test.json"
        with open(results_file) as f:
            data = json.load(f)
            overall_results.append({'f1': data['f1'], 'accuracy': data['accuracy']})
            for strategy in STRATEGIES:
                results[strategy].append({'f1': data[strategy]['f1'], 'accuracy': data[strategy]['accuracy']})
    return results, overall_results

def calculate_metrics(results, metric_key):
    values = np.array([r[metric_key] for r in results])
    return np.mean(values), np.std(values)

def calculate_st_metrics(results, metric_key):
    seed_avgs = []
    for seed in SEEDS:
        values = [results[seed][strategy][metric_key] for strategy in STRATEGIES]
        seed_avgs.append(np.mean(values))
    return np.mean(seed_avgs), np.std(seed_avgs)

def format_cell(mean, std, rank=None):
    cell = f"{100*mean:.1f}±{100*std:.1f}"
    if rank is not None:
        if rank == 1:
            cell = f"**{cell}**"
        cell = f"{rank}. {cell}"
    return cell

def get_column_values(results, result_type, metric):
    if result_type == "single":
        values = []
        for strategy in STRATEGIES:
            strategy_values = [results[seed][strategy][metric] for seed in SEEDS]
            mean, std = np.mean(strategy_values), np.std(strategy_values)
            values.append((mean, std))
        
        avg_mean, avg_std = calculate_st_metrics(results, metric)
        values.append((avg_mean, avg_std))
    else:
        values = []
        for strategy in STRATEGIES:
            mean, std = calculate_metrics(results[0][strategy], metric)
            values.append((mean, std))
        mean, std = calculate_metrics(results[1], metric)
        values.append((mean, std))
    
    return values

def create_table(metric, out_dir):
    all_results = []
    
    for model_name, model_path, result_type in MODELS:
        if result_type == "single":
            results = load_single_task_results(out_dir / model_path)
            all_results.append((model_name, get_column_values(results, result_type, metric)))
        else:
            results = load_multi_task_results(out_dir / model_path)
            all_results.append((model_name, get_column_values(results, result_type, metric)))
    
    n_cols = len(STRATEGIES) + 1
    rankings = []
    for col in range(n_cols):
        col_values = [(i, res[1][col][0]) for i, res in enumerate(all_results)]
        sorted_indices = sorted(col_values, key=lambda x: x[1], reverse=True)
        col_ranking = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_indices)}
        rankings.append(col_ranking)
    
    header = ["Method"] + STRATEGIES + ["Avg"]
    separator = ["-" * max(len(h), 25) for h in header]
    
    rows = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(separator) + "|"
    ]
    
    for i, (model_name, values) in enumerate(all_results):
        row = [model_name]
        for col, (mean, std) in enumerate(values):
            rank = rankings[col][i]
            row.append(format_cell(mean, std, rank))
        rows.append("| " + " | ".join(row) + " |")
    
    return rows

def main():
    out_dir = Path("out")
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    for metric in ['f1', 'accuracy']:
        rows = create_table(metric, out_dir)
        with open(tables_dir / f"{metric}_table.md", "w") as f:
            f.write("\n".join(rows) + "\n")

if __name__ == "__main__":
    main()
