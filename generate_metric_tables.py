import json
import os
from typing import Dict, List, Tuple


def read_results_file(file_path: str) -> Tuple[float, float]:
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data['f1'] * 100, data['accuracy'] * 100


def get_metrics_for_model(model_dir: str, strategies: List[str]) -> Dict[str, Dict[str, float]]:
    metrics = {'f1': {}, 'accuracy': {}}

    for strategy in strategies:
        results_file = os.path.join(model_dir, 'Youtube', strategy, '12', 'results_test.json')
        f1, accuracy = read_results_file(results_file)
        metrics['f1'][strategy] = f1
        metrics['accuracy'][strategy] = accuracy

    return metrics


def generate_markdown_table(metric: str, models: List[str], strategies: List[str], metrics_data: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    strategy_width = max([len(f"{metric} {strategy}") for strategy in strategies] + [len(f"{metric} Average")])
    model_widths = [max(len(model), 5) for model in models]
    
    header = f"| {' ' * (strategy_width - 5)}Class |"
    for i, model in enumerate(models):
        header += f" {model.center(model_widths[i])} |"
    table = header + "\n"
    
    separator = f"|{'-' * (strategy_width + 2)}|"
    for width in model_widths:
        separator += f"{'-' * (width + 2)}|"
    table += separator + "\n"
    
    for strategy in strategies:
        strategy_name = f"{metric} {strategy}"
        row = f"| {strategy_name.ljust(strategy_width)} |"
        for i, model in enumerate(models):
            value = metrics_data[model][metric][strategy]
            row += f" {f'{value:.3f}'.rjust(model_widths[i])} |"
        table += row + "\n"
    
    avg_row = f"| {f'{metric} Average'.ljust(strategy_width)} |"
    for i, model in enumerate(models):
        values = [metrics_data[model][metric][strategy] for strategy in strategies]
        avg = sum(values) / len(values) if values else 0.0
        avg_row += f" {f'{avg:.3f}'.rjust(model_widths[i])} |"
    table += avg_row + "\n"
    
    return table


def main():
    models = ['bert', 'whisper-audio', 'whisper-audio-and-text']
    strategies = [
        'Identity Declaration',
        'Accusation',
        'Interrogation',
        'Call for Action',
        'Defense',
        'Evidence'
    ]

    metrics_data = {}
    for model in models:
        model_dir = os.path.join('out', model)
        if os.path.exists(model_dir):
            metrics_data[model] = get_metrics_for_model(model_dir, strategies)

    f1_table = generate_markdown_table('F1', models, strategies, metrics_data)
    accuracy_table = generate_markdown_table('Accuracy', models, strategies, metrics_data)

    print("F1 Scores Table:")
    print(f1_table)
    print("\nAccuracy Scores Table:")
    print(accuracy_table)

    with open('metric_tables.md', 'w') as f:
        f.write("# Model Performance Tables\n\n")
        f.write("## F1 Scores\n")
        f.write(f1_table)
        f.write("\n## Accuracy Scores\n")
        f.write(accuracy_table)


if __name__ == "__main__":
    main()
