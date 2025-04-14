import json
import numpy as np
from pathlib import Path

SEEDS = [12, 42, 87]
STRATEGIES = ["Identity Declaration", "Accusation", "Interrogation", "Call for Action", "Defense", "Evidence"]

# Calculate max width needed for each column
METHOD_WIDTH = 20
STRATEGY_WIDTHS = {s: max(len(s), 20) for s in STRATEGIES}  # At least 20 chars for data
AVG_WIDTH = 20

def load_single_task_results(base_path):
    results = {seed: {strategy: None for strategy in STRATEGIES} for seed in SEEDS}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            results_file = base_path / strategy / str(seed) / "results_test.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    results[seed][strategy] = {'f1': data['f1'], 'accuracy': data['accuracy']}
    return results


def load_multi_task_results(base_path):
    results = {strategy: [] for strategy in STRATEGIES}
    overall_results = []

    for seed in SEEDS:
        results_file = base_path / str(seed) / "results_test.json"
        if results_file.exists():
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


def format_cell(mean, std, width, is_best=False):
    mean_str = f"{100*mean:.1f}"
    std_str = f"±{100*std:.1f}"
    cell = f"{mean_str}{std_str}"
    padding = width - len(cell)
    if is_best:
        mean_str = f"**{mean_str}**"
        cell = f"{mean_str}{std_str}"
    return cell + " " * max(0, padding)  # Ensure padding is not negative


def create_row(name, results, overall_results=None, metric='f1', all_results=None):
    row_data = []

    for strategy in STRATEGIES:
        if overall_results is None:
            values = [results[seed][strategy][metric] for seed in SEEDS]
            mean, std = np.mean(values), np.std(values)
        else:
            mean, std = calculate_metrics(results[strategy], metric)
        row_data.append((mean, std))

    if overall_results is not None:
        mean, std = calculate_metrics(overall_results, metric)
    else:
        mean, std = calculate_st_metrics(results, metric)
    row_data.append((mean, std))

    if all_results is None:
        return row_data

    row = [name.ljust(METHOD_WIDTH)]
    for i, (mean, std) in enumerate(row_data):
        # Find max value for this column
        max_val = max(r[i][0] for r in all_results)
        # Check if current value is equal to max (within floating point precision)
        is_best = abs(mean - max_val) < 1e-10
        width = STRATEGY_WIDTHS[STRATEGIES[i]] if i < len(STRATEGIES) else AVG_WIDTH
        row.append(format_cell(mean, std, width, is_best))

    return "| " + " | ".join(row) + " |"  # Add pipes with spaces


def main():
    out_dir = Path("out")
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    bert_st = load_single_task_results(out_dir / "bert" / "single_task")
    whisper_st = load_single_task_results(out_dir / "whisper" / "single_task" / "yes_no")

    bert_mt_results, bert_mt_overall = load_multi_task_results(out_dir / "bert" / "multi_task_binary_label")
    whisper_mt_results, whisper_mt_overall = load_multi_task_results(out_dir / "whisper" / "multi_task_binary_label" / "yes_no")

    for metric, metric_name in [('f1', 'F1'), ('accuracy', 'Accuracy')]:
        all_results = [
            create_row("BERT ST", bert_st, metric=metric),
            create_row("BERT MTBL", bert_mt_results, bert_mt_overall, metric=metric),
            create_row("Whisper YN ST", whisper_st, metric=metric),
            create_row("Whisper YN MTBL", whisper_mt_results, whisper_mt_overall, metric=metric),
        ]

        header = "| " + " | ".join([metric_name.ljust(METHOD_WIDTH)] + [s.ljust(STRATEGY_WIDTHS[s]) for s in STRATEGIES] + ["Avg".ljust(AVG_WIDTH)]) + " |"
        separator = "|" + "|".join(["-" * (METHOD_WIDTH + 2)] + ["-" * (STRATEGY_WIDTHS[s] + 2) for s in STRATEGIES] + ["-" * (AVG_WIDTH + 2)]) + "|"

        rows = [
            header,
            separator,
            create_row("BERT ST", bert_st, metric=metric, all_results=all_results),
            create_row("BERT MTBL", bert_mt_results, bert_mt_overall, metric=metric, all_results=all_results),
            create_row("Whisper YN ST", whisper_st, metric=metric, all_results=all_results),
            create_row("Whisper YN MTBL", whisper_mt_results, whisper_mt_overall, metric=metric, all_results=all_results),
        ]

        with open(tables_dir / f"{metric}_table.md", "w") as f:
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
