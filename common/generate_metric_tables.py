import json
import numpy as np
from typing import Dict, List, Tuple


def read_results_file(file_path: str) -> Tuple[float, float]:
    with open(file_path, "r") as f:
        data = json.load(f)
        return data["f1"] * 100, data["accuracy"] * 100


def get_metrics_for_models(out_dir: str, models: List[str], strategies: List[str], seeds: List[int]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    metrics = {}
    
    for model in models:
        metrics[model] = {"f1": {}, "accuracy": {}}
        
        for strategy in strategies:
            metrics[model]["f1"][strategy] = []
            metrics[model]["accuracy"][strategy] = []
            
            for seed in seeds:
                results_file = f"{out_dir}/{model}/Youtube/{strategy}/{seed}/results_test.json"
                f1, accuracy = read_results_file(results_file)
                metrics[model]["f1"][strategy].append(f1)
                metrics[model]["accuracy"][strategy].append(accuracy)
    
    return metrics


def calculate_stats(metrics_data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    stats = {}
    
    for model in metrics_data:
        stats[model] = {"f1": {}, "accuracy": {}}
        
        for strategy in metrics_data[model]["f1"]:
            f1_values = metrics_data[model]["f1"][strategy]
            accuracy_values = metrics_data[model]["accuracy"][strategy]
            
            stats[model]["f1"][strategy] = {
                "mean": np.mean(f1_values),
                "std": np.std(f1_values, ddof=1),
            }
            
            stats[model]["accuracy"][strategy] = {
                "mean": np.mean(accuracy_values),
                "std": np.std(accuracy_values, ddof=1),
            }
    
    return stats


def generate_markdown_table(metric: str, models: List[str], strategies: List[str], stats_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> str:
    model_width = max(len(model) for model in models)
    
    best_models = {}
    for strategy in strategies:
        max_mean = max(stats_data[model][metric][strategy]["mean"] for model in models)
        best_models[strategy] = [model for model in models if abs(stats_data[model][metric][strategy]["mean"] - max_mean) < 1e-2]
    
    avg_means_by_model = {}
    for model in models:
        means = [stats_data[model][metric][strategy]["mean"] for strategy in strategies]
        avg_means_by_model[model] = np.mean(means)
    
    max_avg_mean = max(avg_means_by_model.values())
    best_models["Average"] = [model for model in models if abs(avg_means_by_model[model] - max_avg_mean) < 1e-2]
    
    cell_contents = {}
    for model in models:
        cell_contents[model] = {}
        avg_means = []
        
        for strategy in strategies:
            mean = stats_data[model][metric][strategy]["mean"]
            std = stats_data[model][metric][strategy]["std"]
            
            if model in best_models[strategy]:
                cell_contents[model][strategy] = f"{mean:.2f}({std:.2f})*"
            else:
                cell_contents[model][strategy] = f"{mean:.2f}({std:.2f})"
                
            avg_means.append(mean)
            
        avg_mean = np.mean(avg_means)
        
        if model in best_models["Average"]:
            cell_contents[model]["Average"] = f"{avg_mean:.2f}*"
        else:
            cell_contents[model]["Average"] = f"{avg_mean:.2f}"
    
    strategy_widths = {}
    for strategy in strategies + ["Average"]:
        content_width = max([len(cell_contents[model][strategy]) for model in models])
        header_width = len(strategy)
        strategy_widths[strategy] = max(content_width, header_width)
    
    header = f"| {'Model'.ljust(model_width)} |"
    for strategy in strategies:
        header += f" {strategy.center(strategy_widths[strategy])} |"
    header += f" {'Average'.center(strategy_widths['Average'])} |"
    
    separator = f"|{'-' * (model_width + 2)}|"
    for strategy in strategies + ["Average"]:
        separator += f"{'-' * (strategy_widths[strategy] + 2)}|"
    
    rows = []
    for model in models:
        row = f"| {model.ljust(model_width)} |"
        
        for strategy in strategies:
            cell = cell_contents[model][strategy]
            row += f" {cell.center(strategy_widths[strategy])} |"
            
        avg_cell = cell_contents[model]["Average"]
        row += f" {avg_cell.center(strategy_widths['Average'])} |"
        rows.append(row)
    
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    
    footer = "\n\nBest score in each column is marked with an asterisk (\\*)"
    
    return table + footer


def main():
    models = ["bert", "whisper-audio", "whisper-audio-and-text"]
    strategies = [
        "Identity Declaration",
        "Accusation",
        "Interrogation",
        "Call for Action",
        "Defense",
        "Evidence",
    ]
    seeds = [12, 42, 87]
    out_dir = "out"

    metrics_data = get_metrics_for_models(out_dir, models, strategies, seeds)
    
    stats_data = calculate_stats(metrics_data)
    
    f1_table = generate_markdown_table("f1", models, strategies, stats_data)
    accuracy_table = generate_markdown_table("accuracy", models, strategies, stats_data)

    print("F1 Scores Table:")
    print(f1_table)
    print("\nAccuracy Scores Table:")
    print(accuracy_table)

    with open("out/metric_tables.md", "w") as f:
        f.write("# Model Performance Tables\n\n")
        f.write("## F1 Scores\n")
        f.write(f1_table)
        f.write("\n\n## Accuracy Scores\n")
        f.write(accuracy_table)


if __name__ == "__main__":
    main()
