import os
import sys
from typing import Dict, Any, List

import numpy as np

# Ensure CompositeDP is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPOSITE_DP_DIR = os.path.join(os.path.dirname(BASE_DIR), "CompositeDP")
if COMPOSITE_DP_DIR not in sys.path:
    sys.path.append(COMPOSITE_DP_DIR)

from CompositeDP.Perturbation_Mechanism import perturbation_fun_multipleCall  # type: ignore


# ---------------------------------------------------------------------
# Utility: generate composite DP noise samples with better error handling
# ---------------------------------------------------------------------
def _safe_generate_noise(
    epsilon: float,
    sensitivity: float,
    lower_bound: float,
    k: float,
    m: float,
    y: float,
    index: int,
    sample_count: int,
):
    """
    Runs CompositeDP noise sampling.
    Returns:
        - noise array (float)
        - success flag (bool)
    Never crashes; handles any CompositeDP constraint errors silently.
    """
    try:
        # fd = 0 for pure noise testing
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
        noise = np.asarray(results, dtype=float) - fd
        if noise.size == 0 or np.isnan(noise).any():
            return None, False
        return noise, True

    except Exception:
        # CompositeDP throws exceptions when constraints fail — treat as failure
        return None, False


# ---------------------------------------------------------------------
# Auto calibration: choose k that matches target variance best
# ---------------------------------------------------------------------
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
    Sweeps several candidate k values and chooses the one whose empirical
    noise variance best matches the target Gaussian variance.
    Suppresses individual constraint errors; prints one summary count.
    """
    if candidate_k_values is None:
        candidate_k_values = [0.05, 0.1, 0.2, 0.3, 0.5]  # good reasonable defaults

    best_k = None
    best_variance = None
    lowest_error = float("inf")
    constraint_failures = 0

    for k in candidate_k_values:
        noise, ok = _safe_generate_noise(
            epsilon=epsilon,
            sensitivity=sensitivity,
            lower_bound=lower_bound,
            k=k,
            m=m,
            y=y,
            index=index,
            sample_count=calibration_samples,
        )

        if not ok:
            constraint_failures += 1
            continue

        emp_var = float(np.var(noise))
        error = abs(emp_var - target_variance)

        if error < lowest_error:
            lowest_error = error
            best_k = k
            best_variance = emp_var

    # If everything failed → fallback
    if best_k is None:
        print("[C-DP] WARNING: All candidate k values failed constraints. "
              "Falling back to k = 0.1 with empirical variance = 0.0.")
        best_k = 0.1
        best_variance = 0.0

    # Print summary of failures once (not spam)
    if constraint_failures > 0:
        print(f"[C-DP] auto_calibrate: {constraint_failures} constraint failures during k search.")

    return {
        "L": float(L),
        "m": float(m),
        "y": float(y),
        "k": float(best_k),
        "index": int(index),
        "target_variance": float(target_variance),
        "empirical_variance": float(best_variance),
    }


# ---------------------------------------------------------------------
# Add CompositeDP noise to gradient vector
# ---------------------------------------------------------------------
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
    Adds CompositeDP noise element-wise to a gradient array.

    Generates exactly one noise sample per gradient component.

    If CompositeDP fails for this batch, we return the gradient unchanged
    (this avoids crashing training).
    """
    flat_grad = gradient_array.reshape(-1)
    sample_count = flat_grad.shape[0]

    noise, ok = _safe_generate_noise(
        epsilon=epsilon,
        sensitivity=sensitivity,
        lower_bound=lower_bound,
        k=k,
        m=m,
        y=y,
        index=index,
        sample_count=sample_count,
    )

    if not ok:
        # Fail silently — but warn once
        print("[C-DP] WARNING: CompositeDP noise generation failed for a gradient batch. "
              "Applying *no* noise for this batch.")
        return gradient_array  # unchanged gradient

    noisy_flat = flat_grad + noise
    return noisy_flat.reshape(gradient_array.shape)
