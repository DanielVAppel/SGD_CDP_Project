import itertools
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any


def run_one(config: Dict[str, Any]) -> None:
    """
    Call train_experiment_tf.py with the given configuration.
    """
    cmd = ["python", "train_experiment_tf.py"]
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.append(flag)
        cmd.append(str(value))
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # Configuration grid
    datasets = ["mnist", "adult"]
    mechanisms = ["dpsgd", "cdp"]
    epsilons = [1.0, 2.0, 4.0, 8.0]

    base_config = {
        "delta": 1e-5,
        "batch_size": 256,
        "epochs": 5,
        "learning_rate": 0.15,
        "l2_norm_clip": 1.0,
        "results_dir": "results",
        "seed": 42,
        # Composite DP base parameters (can be tuned later)
        "cdp_L": 1.0,
        "cdp_m": 0.5,
        "cdp_y": 0.05,
    }

    # We can hand-tune noise_multiplier for each epsilon
    # For now, we use the same noise_multiplier for all epsilons as a starting point.
    noise_multiplier_for_all = 1.1

    configs: List[Dict[str, Any]] = []
    for dataset, mechanism, epsilon in itertools.product(datasets, mechanisms, epsilons):
        cfg = base_config.copy()
        cfg["dataset"] = dataset
        cfg["mechanism"] = mechanism
        cfg["epsilon"] = epsilon
        cfg["noise_multiplier"] = noise_multiplier_for_all
        configs.append(cfg)

    for cfg in configs:
        run_one(cfg)


if __name__ == "__main__":
    main()
