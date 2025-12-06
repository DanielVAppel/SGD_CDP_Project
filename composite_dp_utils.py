import os
import sys
from typing import Dict, Any, List

import numpy as np

# Ensure we can import from the CompositeDP repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPOSITE_DP_DIR = os.path.join(os.path.dirname(BASE_DIR), "CompositeDP")
if COMPOSITE_DP_DIR not in sys.path:
    sys.path.append(COMPOSITE_DP_DIR)

from Perturbation_Mechanism import perturbation_fun_multipleCall  # type: ignore


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
    # fd: raw query result f(D); we use 0 for calibration
    fd = 0.0
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
    # The library returns a list of perturbed outputs; noise = output - fd
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
    candidate_k_values: List[float] = None,
    calibration_samples: int = 2000,
) -> Dict[str, Any]:
    """
    Sweep over candidate k values and pick the one whose empirical variance is closest to target_variance.

    This uses only the CompositeDP noise mechanism (no training data), so it does not consume privacy budget.
    """
    if candidate_k_values is None:
        candidate_k_values = list(np.linspace(0.05, 0.95, 10))

    best_k = None
    best_variance = None
    best_error = float("inf")

    for k in candidate_k_values:
        noise = generate_composite_dp_noise_samples(
            epsilon=epsilon,
            sensitivity=sensitivity,
            lower_bound=lower_bound,
            k=k,
            m=m,
            y=y,
            index=index,
            sample_count=calibration_samples,
        )
        empirical_variance = float(np.var(noise))
        error = abs(empirical_variance - target_variance)

        if error < best_error:
            best_error = error
            best_k = float(k)
            best_variance = empirical_variance

    return {
        "L": float(L),
        "m": float(m),
        "y": float(y),
        "k": float(best_k),
        "index": int(index),
        "target_variance": float(target_variance),
        "empirical_variance": float(best_variance),
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

    This calls perturbation_fun_multipleCall once to generate as many samples as there are gradient components.
    """
    flat = gradient_array.reshape(-1)
    sample_count = flat.shape[0]

    results = perturbation_fun_multipleCall(
        epsilon,
        0.0,  # fd, raw f(D); we focus on added noise
        sensitivity,
        lower_bound,
        k,
        m,
        y,
        index,
        sample_count,
    )
    noise_array = np.array(results, dtype=float)  # output - fd is noise
    # If fd was 0, noise is just the returned values
    noisy_flat = flat + noise_array
    return noisy_flat.reshape(gradient_array.shape)
