import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt


def collect_results(results_root: str = "results") -> List[Dict[str, Any]]:
    """
    Traverse the results directory and collect all metrics.json files into a list of dicts.
    """
    root = Path(results_root)
    result_dicts: List[Dict[str, Any]] = []

    for metrics_path in root.glob("*/*/*/metrics.json"):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        # Attach path info for later grouping
        path_parts = metrics_path.parts
        dataset = path_parts[-4]
        mechanism = path_parts[-3]
        data["_dataset"] = dataset
        data["_mechanism"] = mechanism
        data["_path"] = str(metrics_path)
        result_dicts.append(data)
    return result_dicts


def group_by_dataset_and_mechanism(results: List[Dict[str, Any]]):
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (r["_dataset"], r["_mechanism"])
        grouped.setdefault(key, []).append(r)
    return grouped


def plot_accuracy_vs_epsilon(grouped_results: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> None:
    """
    For each (dataset, mechanism), plot test accuracy vs epsilon (where available).
    """
    for (dataset, mechanism), runs in grouped_results.items():
        epsilons = []
        accuracies = []
        for r in runs:
            eps = r["args"].get("epsilon", None)
            if eps is None:
                continue
            epsilons.append(float(eps))
            accuracies.append(float(r.get("final_test_accuracy", 0.0)))

        if not epsilons:
            continue

        # Sort by epsilon
        eps_acc = sorted(zip(epsilons, accuracies), key=lambda x: x[0])
        eps_sorted, acc_sorted = zip(*eps_acc)

        plt.figure()
        plt.plot(eps_sorted, acc_sorted, marker="o")
        plt.xlabel("Target epsilon")
        plt.ylabel("Final test accuracy")
        plt.title(f"Accuracy vs epsilon ({dataset}, {mechanism})")
        plt.grid(True)
        plt.tight_layout()
        out_path = Path("results") / f"accuracy_vs_epsilon_{dataset}_{mechanism}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def plot_mi_auc_vs_epsilon(grouped_results: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> None:
    """
    For each (dataset, mechanism), plot membership inference AUC vs epsilon.
    """
    for (dataset, mechanism), runs in grouped_results.items():
        epsilons = []
        aucs = []
        for r in runs:
            eps = r["args"].get("epsilon", None)
            if eps is None:
                continue
            epsilons.append(float(eps))
            aucs.append(float(r.get("membership_inference_auc", 0.0)))

        if not epsilons:
            continue

        eps_auc = sorted(zip(epsilons, aucs), key=lambda x: x[0])
        eps_sorted, auc_sorted = zip(*eps_auc)

        plt.figure()
        plt.plot(eps_sorted, auc_sorted, marker="o")
        plt.xlabel("Target epsilon")
        plt.ylabel("MI AUC (higher = easier to attack)")
        plt.title(f"Membership inference AUC vs epsilon ({dataset}, {mechanism})")
        plt.grid(True)
        plt.tight_layout()
        out_path = Path("results") / f"mi_auc_vs_epsilon_{dataset}_{mechanism}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")


def main():
    results = collect_results("results")
    if not results:
        print("No results found under 'results/'. Run some experiments first.")
        return

    grouped = group_by_dataset_and_mechanism(results)
    plot_accuracy_vs_epsilon(grouped)
    plot_mi_auc_vs_epsilon(grouped)


if __name__ == "__main__":
    main()
