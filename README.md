# SGD_CDP_Project
We are comparing two differential privacy mechanisms: standard DP-SGD and the Bounded and Unbiased Composite Differential Privacy (C-DP) method (Zhang et al., 2023) under matched privacy budgets.

We are comparing two differential privacy mechanisms: standard DP-SGD and the Bounded and Unbiased Composite Differential Privacy (C-DP) method (Zhang et al., 2023) under matched privacy budgets. Using small, reproducible models (a multilayer perceptron for tabular data and a shallow CNN for images), we will train on benchmark datasets MNIST and UCI Adult Income across several privacy levels. The study will measure utility (accuracy, F1, calibration), training efficiency (loss convergence, and runtime), and privacy robustness through membership inference (MIA) and model-inversion attacks. This applications project will result in a side-by-side evaluation of their empirical privacy/utility trade-offs. The outcome of which will showcase how C-DP’s bounded-unbiased design impacts learning stability, resistance to privacy attacks, and overall efficiency compared with DP-SGD.

Our goal is to use empirical evaluation of utility and privacy performance to demonstrate that combining CDP with SGD performs significantly better than using DP with SGD, utilizing the performance evaluation in the paper that first proposed DP-SGD (https://arxiv.org/pdf/1607.00133) to understand how DP-SGD was evaluated. we follow the same experimental setup to evaluate CDP-SGD (by replacing DP noise with CDP noise) and then compare their performance. CDP also has publicly available code, which is used for parts of the experiments. https://github.com/CompositeDP/CompositeDP.git

SGD_CDP_Project/
├─ requirements.txt
├─ train_experiment_tf.py          # main training + evaluation script (MNIST + Adult, DP-SGD + C-DP)
├─ batch_run_experiments.py        # runs a grid of experiments (epsilons, mechanisms, datasets)
├─ plot_results.py                 # plots accuracy vs epsilon, membership AUC vs epsilon
├─ datasets.py                     # MNIST and Adult Income loaders
├─ models.py                       # CNN for MNIST, MLP for Adult
├─ tensorflow_privacy_utils.py     # helper for DP-SGD optimizer + epsilon computation
├─ composite_dp_utils.py           # wrapper around CompositeDP + k auto-sweep + logging
├─ membership_inference.py         # loss-based membership inference attack
└─ results/                        # (auto-created) metrics and models per run

# How to Run:

Ensure you are using Python 3.11 or below

# 1) (optional) create venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) install
python -m pip install -r requirements.txt
python -m pip install tensorflow-privacy==0.8.12 --no-deps
python -m pip install dp-accounting==0.4.3
python -m pip install tensorflow-probability==0.19.0

# Quick Test Runs:

- MNIST, DP-SGD (TF Privacy):
python train_experiment_tf.py --dataset mnist --mechanism dpsgd --epochs 1 --epsilon 2.0 --delta 1e-5 --batch_size 256 --noise_multiplier 1.1

- MNIST, CompositeDP:
python train_experiment_tf.py --dataset mnist --mechanism cdp --epochs 1 --epsilon 2.0 --batch_size 256 --cdp_k 0.5 --cdp_m 0.4 --cdp_y 0.3 --cdp_index 1

- MNIST — NON-DP baseline
python train_experiment_tf.py --dataset mnist --mechanism none --epochs 1 --batch_size 256

- Adult Income (tabular), DP-SGD:
python train_experiment_tf.py --dataset adult --mechanism dpsgd --epochs 1 --epsilon 2.0 --delta 1e-5 --batch_size 256 --noise_multiplier 1.1

- Adult Income — CompositeDP (C-DP)
python train_experiment_tf.py --dataset adult --mechanism cdp --epochs 1 --epsilon 2.0 --delta 1e-5 --batch_size 256 --cdp_k 0.5 --cdp_m 0.4 --cdp_y 0.3 --cdp_index 1

Each run will save:
results/<dataset>/<mechanism>/<timestamp>/metrics.json
results/<dataset>/<mechanism>/<timestamp>/model.keras

# Batch Experiments + plots
(To sweep over ε ∈ {1, 2, 4, 8} for both MNIST and Adult, both DP-SGD and C-DP:)

1) python batch_run_experiments.py
2) python plot_results.py

The metrics.json includes:
- Training curves
- Final train and test accuracy
- Mean train and test loss
- Membership inference AUC
- For DP-SGD: list of epsilon estimates per epoch and final one
- For C-DP: composite_dp_calibration containing (L, m, y, k, target_variance, empirical_variance, index)