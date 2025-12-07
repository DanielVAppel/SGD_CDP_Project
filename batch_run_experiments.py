import itertools
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import itertools
import json
import sys
import time


def run_one(config: Dict[str, Any]) -> None:
    """
    Call train_experiment_tf.py with the given configuration.
    """
    cmd = [sys.executable, "train_experiment_tf.py"]
    for key, value in config.items():
        flag = f"--{key}"
        cmd.append(flag)
        cmd.append(str(value))
    print("\n" + "="*80)
    print("Running:", " ".join(cmd))
    print("="*80)
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
        # Composite DP parameters (from Example1.py)
        "cdp_k": 0.5,
        "cdp_m": 0.4,
        "cdp_y": 0.3,
        "cdp_index": 1,  # Use Perturbation-1 (Activation-1 + Base-1)
    }

    # Different noise multipliers for different epsilon targets
    # These are approximate and may need tuning
    noise_multiplier_map = {
        1.0: 2.0,
        2.0: 1.1,
        4.0: 0.7,
        8.0: 0.4,
    }

    configs: List[Dict[str, Any]] = []
    for dataset, mechanism, epsilon in itertools.product(datasets, mechanisms, epsilons):
        cfg = base_config.copy()
        cfg["dataset"] = dataset
        cfg["mechanism"] = mechanism
        cfg["epsilon"] = epsilon
        cfg["noise_multiplier"] = noise_multiplier_map[epsilon]
        configs.append(cfg)

    print(f"\nTotal experiments to run: {len(configs)}")
    print("="*80)
    
    for i, cfg in enumerate(configs, 1):
        print(f"\n\nExperiment {i}/{len(configs)}")
        print(f"Dataset: {cfg['dataset']}, Mechanism: {cfg['mechanism']}, Epsilon: {cfg['epsilon']}")
        try:
            run_one(cfg)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Experiment failed with error code {e.returncode}")
            print("Continuing with next experiment...")
            continue


if __name__ == "__main__":
    main()