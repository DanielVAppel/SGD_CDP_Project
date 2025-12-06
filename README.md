# SGD_CDP_Project
We are comparing two differential privacy mechanisms: standard DP-SGD and the Bounded and Unbiased Composite Differential Privacy (C-DP) method (Zhang et al., 2023) under matched privacy budgets.

We are comparing two differential privacy mechanisms: standard DP-SGD and the Bounded and Unbiased Composite Differential Privacy (C-DP) method (Zhang et al., 2023) under matched privacy budgets. Using small, reproducible models (a multilayer perceptron for tabular data and a shallow CNN for images), we will train on benchmark datasets MNIST and UCI Adult Income across several privacy levels. The study will measure utility (accuracy, F1, calibration), training efficiency (loss convergence, and runtime), and privacy robustness through membership inference (MIA) and model-inversion attacks. This applications project will result in a side-by-side evaluation of their empirical privacy/utility trade-offs. The outcome of which will showcase how C-DP’s bounded-unbiased design impacts learning stability, resistance to privacy attacks, and overall efficiency compared with DP-SGD.

Our goal is to use empirical evaluation of utility and privacy performance to demonstrate that combining CDP with SGD performs significantly better than using DP with SGD, utilizing the performance evaluation in the paper that first proposed DP-SGD (https://arxiv.org/pdf/1607.00133) to understand how DP-SGD was evaluated. we follow the same experimental setup to evaluate CDP-SGD (by replacing DP noise with CDP noise) and then compare their performance. CDP also has publicly available code, which is used for parts of the experiments. https://github.com/CompositeDP/CompositeDP.git

# How to Run:

# 1) (optional) create venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) install
pip install -r requirements.txt

# 3) DP-SGD (Gaussian) on MNIST @ ε≈2 (heuristic sigma=1.2; read the reported ε every epoch)
python run_experiment.py --dataset mnist --mechanism dpsgd --epsilon 2.0 --delta 1e-5 --epochs 5

# 4) Composite-DP on MNIST (bounded + unbiased)
python run_experiment.py --dataset mnist --mechanism cdp --epsilon 2.0 --delta 1e-5 --epochs 5
