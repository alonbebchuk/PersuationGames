import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from typing import List, Dict


def parse_classification_report(
    report_str: str,
) -> dict:
    lines = [line for line in report_str.split('\n') if line.strip()]

    class_metrics = {}
    for line in lines:
        if line.strip().startswith('0') or line.strip().startswith('1'):
            parts = [p for p in line.split() if p.strip()]
            class_label = parts[0]
            class_metrics[f'class_{class_label}'] = {
                'precision': float(parts[1]),
                'recall': float(parts[2]),
                'f1': float(parts[3]),
                'support': int(parts[4]),
            }
    return class_metrics


def load_results(
    models: List[str],
    datasets: List[str],
    strategies: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    records = []

    for model in models:
        for dataset in datasets:
            for strategy in strategies:
                for seed in seeds:
                    file_path = f"./out/{model}/{dataset}/{strategy}/{seed}/results_test.json"
                    with open(file_path, 'r') as f:
                        results = json.load(f)

                        class_metrics = parse_classification_report(results['report'])

                        record = {
                            'model': f"{model}-{dataset}",
                            'strategy': strategy,
                            'seed': seed,
                            'loss': results['loss'],
                            'f1': results['f1'],
                            'precision': results['precision'],
                            'recall': results['recall'],
                            'accuracy': results['accuracy'],
                            'class_0_precision': class_metrics['class_0']['precision'],
                            'class_0_recall': class_metrics['class_0']['recall'],
                            'class_0_f1': class_metrics['class_0']['f1'],
                            'class_0_support': class_metrics['class_0']['support'],
                            'class_1_precision': class_metrics['class_1']['precision'],
                            'class_1_recall': class_metrics['class_1']['recall'],
                            'class_1_f1': class_metrics['class_1']['f1'],
                            'class_1_support': class_metrics['class_1']['support'],
                        }
                        records.append(record)

    return pd.DataFrame(records)


def create_comparison_tables(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    metrics = [
        'accuracy', 'f1', 'precision', 'recall',
        'class_0_precision', 'class_0_recall', 'class_0_f1',
        'class_1_precision', 'class_1_recall', 'class_1_f1',
    ]
    tables = {}

    overall = df.groupby('model')[metrics].mean().round(3)
    tables['overall'] = overall

    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        strategy_table = strategy_df.groupby('model')[metrics].mean().round(3)
        tables[f'strategy_{strategy}'] = strategy_table

    return tables


def plot_metrics_comparison(
    df: pd.DataFrame,
    analysis_dir: str,
) -> None:
    overall_metrics = ['accuracy', 'f1', 'precision', 'recall']

    os.makedirs(analysis_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(overall_metrics):
        plt.subplot(2, 2, i+1)
        sns.boxplot(data=df, x='model', y=metric)
        plt.xticks(rotation=45)
        plt.title(f'{metric.capitalize()} by Model')
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/overall_comparison.png')
    plt.close()

    plt.figure(figsize=(15, 10))
    class_metrics = ['precision', 'recall', 'f1']
    for i, metric in enumerate(class_metrics):
        plt.subplot(2, 3, i+1)
        sns.boxplot(data=df, x='model', y=f'class_0_{metric}')
        plt.xticks(rotation=45)
        plt.title(f'Class 0 {metric.capitalize()}')

        plt.subplot(2, 3, i+4)
        sns.boxplot(data=df, x='model', y=f'class_1_{metric}')
        plt.xticks(rotation=45)
        plt.title(f'Class 1 {metric.capitalize()}')
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/class_specific_comparison.png')
    plt.close()

    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]

        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(overall_metrics):
            plt.subplot(2, 2, i+1)
            sns.boxplot(data=strategy_df, x='model', y=metric)
            plt.xticks(rotation=45)
            plt.title(f'{metric.capitalize()} by Model - Strategy {strategy}')
        plt.tight_layout()
        plt.savefig(f'{analysis_dir}/strategy_{strategy}_overall_comparison.png')
        plt.close()

        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(class_metrics):
            plt.subplot(2, 3, i+1)
            sns.boxplot(data=strategy_df, x='model', y=f'class_0_{metric}')
            plt.xticks(rotation=45)
            plt.title(f'Class 0 {metric.capitalize()} - Strategy {strategy}')

            plt.subplot(2, 3, i+4)
            sns.boxplot(data=strategy_df, x='model', y=f'class_1_{metric}')
            plt.xticks(rotation=45)
            plt.title(f'Class 1 {metric.capitalize()} - Strategy {strategy}')
        plt.tight_layout()
        plt.savefig(f'{analysis_dir}/strategy_{strategy}_class_specific_comparison.png')
        plt.close()


def main():
    base_path = "out"
    models = ["bert", "whisper"]
    datasets = ["Youtube"]

    strategies = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]
    seeds = [12]

    df = load_results(models, datasets, strategies, seeds)

    tables = create_comparison_tables(df)

    os.makedirs('analysis', exist_ok=True)
    for name, table in tables.items():
        table.to_csv(f'analysis/{name}_comparison.csv')

    plot_metrics_comparison(df, 'analysis')


if __name__ == "__main__":
    main()
