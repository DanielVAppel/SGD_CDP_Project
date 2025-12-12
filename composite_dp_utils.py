import os
import sys
from typing import Dict, Any

import numpy as np

# Ensure we can import from the CompositeDP repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPOSITE_DP_DIR = os.path.join(BASE_DIR, "CompositeDP")
if COMPOSITE_DP_DIR not in sys.path:
    sys.path.append(COMPOSITE_DP_DIR)

from CompositeDP.Perturbation_Mechanism import perturbation_fun_multipleCall  # type: ignore
from CompositeDP.Mechanism.Parameter_Optimization import parameter_optimization  # type: ignore


def generate_composite_dp_noise_samples(
    epsilon: float,
    sensitivity: float,
    lower_bound: float,
    k: float,
    m: float,
    y: float,
    index: int,
    sample_count: int,
) -> np.ndarray:
    """
    Use perturbation_fun_multipleCall from CompositeDP to generate composite DP noise samples.

    We treat the raw query f(D) as 0 for calibration and only look at the added noise.
    """
    fd = 0.0  # raw query result f(D); we use 0 for calibration
    results = perturbation_fun_multipleCall(
        epsilon,
        fd,
        sensitivity,
        lower_bound,
        k,
        m,
        y,
        index,
        sample_count,
    )
    results_array = np.array(results, dtype=float)
    noise = results_array - fd
    return noise


def auto_calibrate_composite_parameters(
    epsilon: float,
    target_variance: float,
    sensitivity: float,
    lower_bound: float,
    L: float,
    m: float,
    y: float,
    index: int = 2,
    calibration_samples: int = 2000,
) -> Dict[str, Any]:
    """
    Enumeration-based parameter selection used in the paper.

    Instead of manually sweeping over many (k, m, y) combinations, we delegate to
    CompositeDP.Mechanism.Parameter_Optimization.parameter_optimization, which
    searches over the grid defined in the original CompositeDP library.

    We then empirically estimate the variance of the resulting mechanism so we can
    log how close it is to the target_variance (typically the matching Gaussian variance).

    Returns:
        Dictionary with L, m, y, k, index, target_variance, empirical_variance.
    """
    # Run the library's enumeration-based optimizer.
    # NOTE: This optimizer ignores the L, m, y arguments you pass here and returns its
    # own best (k, m, y) for the given epsilon and index.
    k_opt, m_opt, y_opt = parameter_optimization(epsilon, index=index)

    # Empirically estimate the variance of the resulting mechanism for logging.
    noise = generate_composite_dp_noise_samples(
        epsilon=epsilon,
        sensitivity=sensitivity,
        lower_bound=lower_bound,
        k=k_opt,
        m=m_opt,
        y=y_opt,
        index=index,
        sample_count=calibration_samples,
    )
    empirical_variance = float(np.var(noise))

    return {
        "L": float(L),
        "m": float(m_opt),
        "y": float(y_opt),
        "k": float(k_opt),
        "index": int(index),
        "target_variance": float(target_variance),
        "empirical_variance": empirical_variance,
    }


def add_composite_dp_noise_to_gradient(
    gradient_array: np.ndarray,
    epsilon: float,
    sensitivity: float,
    lower_bound: float,
    L: float,
    m: float,
    y: float,
    k: float,
    index: int,
) -> np.ndarray:
    """
    Apply CompositeDP noise element-wise to a flattened gradient array and reshape back.

    This calls perturbation_fun_multipleCall once to generate as many samples as there are
    gradient components, then adds them to the original gradient.
    """
    flat = gradient_array.reshape(-1)
    sample_count = flat.shape[0]

    # Draw composite-DP perturbed values around 0 and interpret them as noise.
    results = perturbation_fun_multipleCall(
        epsilon,
        0.0,  # fd, the raw f(D); we take 0 so that outputs are just noise
        sensitivity,
        lower_bound,
        k,
        m,
        y,
        index,
        sample_count,
    )
    noise_array = np.array(results, dtype=float)
    if noise_array.shape[0] != flat.shape[0]:
        raise ValueError(
            f"CompositeDP returned {noise_array.shape[0]} samples "
            f"but gradient has {flat.shape[0]} components."
        )

    noisy_flat = flat + noise_array
    return noisy_flat.reshape(gradient_array.shape)
