#!/usr/bin/env python3
"""
Task1 AUROC Metrics Calculator.

Reads results from:
    results/{project_name}/{model_name}/{organ}/task1/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json

Computes organ-wise AUROC with mean ± std across seeds.
Reports missing seeds and computes metrics with available seeds.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# Configuration
# ==============================================================================

# Default seeds to check
DEFAULT_SEEDS = [0, 1, 2]

# Default organs
DEFAULT_ORGANS = [
    "bone_marrow",
    "breast",
    "cervix",
    "colon",
    "lung",
    "peripheral_blood",
    "prostate",
    "skin",
]


def is_valid_number(value) -> bool:
    """Check if a value is a valid number (not None, NaN, or invalid)."""
    if value is None:
        return False
    if isinstance(value, str):
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except (TypeError, ValueError):
        return False


def calculate_mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    """Calculate mean and standard deviation from a list of values."""
    valid_values = [v for v in values if is_valid_number(v)]
    if len(valid_values) == 0:
        return None, None, 0

    mean = sum(valid_values) / len(valid_values)
    if len(valid_values) == 1:
        std = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in valid_values) / (len(valid_values) - 1)
        std = math.sqrt(variance)

    return mean, std, len(valid_values)


def compute_auroc(ground_truths: List[int], predicted_labels: List[int]) -> Optional[float]:
    """
    Compute AUROC from ground truth and predicted labels.

    For binary classification, AUROC can be computed as:
    AUROC = (sum of ranks of positive class - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Args:
        ground_truths: List of ground truth labels (0 or 1)
        predicted_labels: List of predicted labels (0 or 1)

    Returns:
        AUROC score or None if computation fails
    """
    if len(ground_truths) != len(predicted_labels):
        return None

    if len(ground_truths) == 0:
        return None

    # Count positives and negatives
    n_pos = sum(1 for gt in ground_truths if gt == 1)
    n_neg = sum(1 for gt in ground_truths if gt == 0)

    if n_pos == 0 or n_neg == 0:
        return None  # Cannot compute AUROC with single class

    # Compute accuracy as a simple metric
    # For binary predictions (not probabilities), we use accuracy-based AUROC approximation
    # True positive rate (TPR) = TP / (TP + FN)
    # False positive rate (FPR) = FP / (FP + TN)

    tp = sum(1 for gt, pred in zip(ground_truths, predicted_labels) if gt == 1 and pred == 1)
    fn = sum(1 for gt, pred in zip(ground_truths, predicted_labels) if gt == 1 and pred == 0)
    fp = sum(1 for gt, pred in zip(ground_truths, predicted_labels) if gt == 0 and pred == 1)
    tn = sum(1 for gt, pred in zip(ground_truths, predicted_labels) if gt == 0 and pred == 0)

    # TPR (sensitivity) and FPR (1 - specificity)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # For binary predictions, AUROC is approximated as:
    # AUROC = (TPR + (1 - FPR)) / 2 = (TPR + TNR) / 2
    # This is equivalent to balanced accuracy
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    auroc = (tpr + tnr) / 2

    return auroc


def load_json_safe(filepath: Path) -> Optional[List[Dict]]:
    """Safely load a JSON file, returning None if it fails."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return None


def find_result_files(base_dir: Path, organ: str, seed: int) -> List[Path]:
    """
    Find all result JSON files for a given organ and seed.

    Path structure:
        {base_dir}/{organ}/task1/{test_case_id}/{gt_moa}/{compound}/{candidate_moa}_{seed}.json
    """
    organ_task_dir = base_dir / organ / "task1"
    if not organ_task_dir.exists():
        return []

    result_files = []
    pattern = f"*_{seed}.json"

    # Walk through all subdirectories
    for root, dirs, files in os.walk(organ_task_dir):
        for file in files:
            if file.endswith(f"_{seed}.json"):
                result_files.append(Path(root) / file)

    return result_files


def compute_accuracy(ground_truths: List[int], predicted_labels: List[int]) -> Optional[float]:
    """Compute accuracy from ground truth and predicted labels."""
    if len(ground_truths) != len(predicted_labels) or len(ground_truths) == 0:
        return None
    correct = sum(1 for gt, pred in zip(ground_truths, predicted_labels) if gt == pred)
    return correct / len(ground_truths)


def collect_organ_results(
    base_dir: Path,
    organ: str,
    seeds: List[int],
) -> Tuple[Dict[int, Dict], List[int], List[int]]:
    """
    Collect results for a single organ across all seeds.

    Returns:
        - Dict mapping seed -> {"auroc": float|None, "accuracy": float|None, "n_samples": int, "n_pos": int, "n_neg": int, "n_files": int}
        - List of available seeds (seeds with data)
        - List of missing seeds (seeds without any data files)
    """
    seed_results = {}
    available_seeds = []
    missing_seeds = []

    for seed in seeds:
        result_files = find_result_files(base_dir, organ, seed)

        if not result_files:
            missing_seeds.append(seed)
            continue

        # Collect all ground truths and predictions from all files for this seed
        all_ground_truths = []
        all_predicted_labels = []

        for filepath in result_files:
            data = load_json_safe(filepath)
            if data is None:
                continue

            # data is a list of result dictionaries
            for result in data:
                gt = result.get("ground_truth")
                pred = result.get("predicted_label")

                if gt is not None and pred is not None:
                    all_ground_truths.append(int(gt))
                    all_predicted_labels.append(int(pred))

        if len(all_ground_truths) > 0:
            n_pos = sum(1 for gt in all_ground_truths if gt == 1)
            n_neg = sum(1 for gt in all_ground_truths if gt == 0)
            auroc = compute_auroc(all_ground_truths, all_predicted_labels)
            accuracy = compute_accuracy(all_ground_truths, all_predicted_labels)

            seed_results[seed] = {
                "auroc": auroc,
                "accuracy": accuracy,
                "n_samples": len(all_ground_truths),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_files": len(result_files),
            }
            available_seeds.append(seed)
        else:
            missing_seeds.append(seed)

    return seed_results, available_seeds, missing_seeds


def collect_all_results(
    base_dir: Path,
    organs: List[str],
    seeds: List[int],
) -> Dict[str, Dict]:
    """
    Collect results for all organs.

    Returns:
        Dict mapping organ -> {
            "seed_results": {seed: {"auroc": float|None, "accuracy": float, ...}},
            "available_seeds": [seeds],
            "missing_seeds": [seeds],
            "auroc_mean": float,
            "auroc_std": float,
            "auroc_count": int,
            "accuracy_mean": float,
            "accuracy_std": float,
            "accuracy_count": int,
            "total_samples": int,
            "total_pos": int,
            "total_neg": int,
        }
    """
    all_results = {}

    for organ in organs:
        seed_results, available_seeds, missing_seeds = collect_organ_results(
            base_dir, organ, seeds
        )

        # Calculate mean and std for AUROC
        auroc_values = [r["auroc"] for r in seed_results.values() if r["auroc"] is not None]
        auroc_mean, auroc_std, auroc_count = calculate_mean_std(auroc_values)

        # Calculate mean and std for Accuracy
        accuracy_values = [r["accuracy"] for r in seed_results.values() if r["accuracy"] is not None]
        accuracy_mean, accuracy_std, accuracy_count = calculate_mean_std(accuracy_values)

        # Total samples across all seeds
        total_samples = sum(r["n_samples"] for r in seed_results.values())
        total_pos = sum(r["n_pos"] for r in seed_results.values())
        total_neg = sum(r["n_neg"] for r in seed_results.values())

        all_results[organ] = {
            "seed_results": seed_results,
            "available_seeds": available_seeds,
            "missing_seeds": missing_seeds,
            "auroc_mean": auroc_mean,
            "auroc_std": auroc_std,
            "auroc_count": auroc_count,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "accuracy_count": accuracy_count,
            "total_samples": total_samples,
            "total_pos": total_pos,
            "total_neg": total_neg,
        }

    return all_results


def print_separator(char: str = "=", length: int = 100):
    """Print a separator line."""
    print(char * length)


def print_results(results: Dict[str, Dict], seeds: List[int]):
    """Print results in a formatted table."""
    print_separator("=")
    print("Task 1 - Organ-wise AUROC Results")
    print_separator("=")
    print()

    # Define column widths
    organ_width = 20
    seed_width = 10
    mean_width = 10
    std_width = 10
    count_width = 7
    samples_width = 9

    # Print header
    header = f"{'Organ':<{organ_width}}"
    for seed in seeds:
        header += f" | {'Seed ' + str(seed):^{seed_width}}"
    header += f" | {'Mean':^{mean_width}} | {'Std':^{std_width}} | {'Count':^{count_width}} | {'Samples':^{samples_width}}"
    print(header)
    print_separator("-")

    # Print results for each organ
    overall_aurocs = []

    for organ in sorted(results.keys()):
        data = results[organ]
        row = f"{organ:<{organ_width}}"

        for seed in seeds:
            if seed in data["seed_results"]:
                seed_data = data["seed_results"][seed]
                auroc = seed_data["auroc"]
                if auroc is not None:
                    row += f" | {auroc:^{seed_width}.4f}"
                    overall_aurocs.append(auroc)
                else:
                    # AUROC not computable (single class)
                    row += f" | {'*':^{seed_width}}"
            else:
                row += f" | {'N/A':^{seed_width}}"

        if data["auroc_mean"] is not None:
            row += f" | {data['auroc_mean']:^{mean_width}.4f} | {data['auroc_std']:^{std_width}.4f} | {data['auroc_count']:^{count_width}} | {data['total_samples']:^{samples_width}}"
        else:
            row += f" | {'N/A':^{mean_width}} | {'N/A':^{std_width}} | {0:^{count_width}} | {data['total_samples']:^{samples_width}}"

        print(row)

    print_separator("-")
    print("  * = Data found but AUROC not computable (single class in ground truth)")

    # Print overall statistics
    if overall_aurocs:
        overall_mean, overall_std, overall_count = calculate_mean_std(overall_aurocs)
        print(f"\nOverall AUROC: {overall_mean:.4f} +/- {overall_std:.4f} (n={overall_count})")

    print()

    # Print Accuracy table
    print_separator("=")
    print("Task 1 - Organ-wise Accuracy Results")
    print_separator("=")
    print()

    # Define column widths for accuracy table
    pos_neg_width = 12

    header = f"{'Organ':<{organ_width}}"
    for seed in seeds:
        header += f" | {'Seed ' + str(seed):^{seed_width}}"
    header += f" | {'Mean':^{mean_width}} | {'Std':^{std_width}} | {'Pos/Neg':^{pos_neg_width}}"
    print(header)
    print_separator("-")

    overall_accuracies = []

    for organ in sorted(results.keys()):
        data = results[organ]
        row = f"{organ:<{organ_width}}"

        for seed in seeds:
            if seed in data["seed_results"]:
                seed_data = data["seed_results"][seed]
                acc = seed_data["accuracy"]
                if acc is not None:
                    row += f" | {acc:^{seed_width}.4f}"
                    overall_accuracies.append(acc)
                else:
                    row += f" | {'N/A':^{seed_width}}"
            else:
                row += f" | {'N/A':^{seed_width}}"

        if data["accuracy_mean"] is not None:
            pos_neg = f"{data['total_pos']}/{data['total_neg']}"
            row += f" | {data['accuracy_mean']:^{mean_width}.4f} | {data['accuracy_std']:^{std_width}.4f} | {pos_neg:^{pos_neg_width}}"
        else:
            row += f" | {'N/A':^{mean_width}} | {'N/A':^{std_width}} | {'0/0':^{pos_neg_width}}"

        print(row)

    print_separator("-")

    # Print overall statistics
    if overall_accuracies:
        overall_mean, overall_std, overall_count = calculate_mean_std(overall_accuracies)
        print(f"\nOverall Accuracy: {overall_mean:.4f} +/- {overall_std:.4f} (n={overall_count})")

    print()


def print_missing_seeds_report(results: Dict[str, Dict]):
    """Print a report of missing seeds and data summary."""
    has_missing = any(len(data["missing_seeds"]) > 0 for data in results.values())
    has_single_class = any(
        any(r["auroc"] is None and r["n_samples"] > 0 for r in data["seed_results"].values())
        for data in results.values()
    )

    if not has_missing and not has_single_class:
        print("All seeds available for all organs with valid AUROC.")
        return

    if has_missing:
        print_separator("-")
        print("Missing Seeds Report (no data files found):")
        print_separator("-")

        for organ in sorted(results.keys()):
            data = results[organ]
            if data["missing_seeds"]:
                missing_str = ", ".join(str(s) for s in data["missing_seeds"])
                available_str = ", ".join(str(s) for s in data["available_seeds"]) if data["available_seeds"] else "None"
                print(f"  {organ}:")
                print(f"    - Missing: [{missing_str}]")
                print(f"    - Available: [{available_str}]")

        print()

    if has_single_class:
        print_separator("-")
        print("Single-Class Data Report (AUROC not computable):")
        print_separator("-")

        for organ in sorted(results.keys()):
            data = results[organ]
            for seed, seed_data in data["seed_results"].items():
                if seed_data["auroc"] is None and seed_data["n_samples"] > 0:
                    print(f"  {organ} (seed {seed}):")
                    print(f"    - Samples: {seed_data['n_samples']} (pos: {seed_data['n_pos']}, neg: {seed_data['n_neg']})")
                    print(f"    - Files: {seed_data['n_files']}")
                    print(f"    - Accuracy: {seed_data['accuracy']:.4f}" if seed_data['accuracy'] else "    - Accuracy: N/A")

        print()


def print_latex_table(results: Dict[str, Dict]):
    """Print results as a LaTeX table row."""
    print_separator("=")
    print("LaTeX Table Format (mean +/- std):")
    print_separator("=")

    organs_sorted = sorted(results.keys())

    # Header
    header = "Organ"
    for organ in organs_sorted:
        header += f" & {organ.replace('_', ' ').title()}"
    header += " \\\\"
    print(header)

    # AUROC Values
    row = "AUROC"
    for organ in organs_sorted:
        data = results[organ]
        if data["auroc_mean"] is not None:
            row += f" & {data['auroc_mean']*100:.2f}$\\pm${data['auroc_std']*100:.2f}"
        else:
            row += " & N/A"
    row += " \\\\"
    print(row)

    # Accuracy Values
    row = "Accuracy"
    for organ in organs_sorted:
        data = results[organ]
        if data["accuracy_mean"] is not None:
            row += f" & {data['accuracy_mean']*100:.2f}$\\pm${data['accuracy_std']*100:.2f}"
        else:
            row += " & N/A"
    row += " \\\\"
    print(row)

    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate organ-wise AUROC for Task1 results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Base results directory (e.g., results/pbioagent_project/deepseek-ai-DeepSeek-R1-Distill-Llama-8B)"
    )
    parser.add_argument(
        "--organs",
        type=str,
        nargs="+",
        default=DEFAULT_ORGANS,
        help="List of organs to evaluate"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="List of seeds to check (default: 0, 1, 2)"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also print LaTeX table format"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.results_dir)

    if not base_dir.exists():
        print(f"Error: Results directory does not exist: {base_dir}")
        return

    print("\n" + "=" * 80)
    print("Task 1 AUROC Metrics Calculator")
    print("=" * 80)
    print(f"Results Directory: {base_dir}")
    print(f"Organs: {args.organs}")
    print(f"Seeds: {args.seeds}")
    print("=" * 80 + "\n")

    # Collect results
    results = collect_all_results(base_dir, args.organs, args.seeds)

    # Print results
    print_results(results, args.seeds)

    # Print missing seeds report
    print_missing_seeds_report(results)

    # Print LaTeX table if requested
    if args.latex:
        print_latex_table(results)


if __name__ == "__main__":
    main()
