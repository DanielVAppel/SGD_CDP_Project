import os
import sys
from typing import Dict, Any, List

import numpy as np

# Ensure CompositeDP is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPOSITE_DP_DIR = os.path.join(BASE_DIR, "CompositeDP")
if COMPOSITE_DP_DIR not in sys.path:
    sys.path.insert(0, COMPOSITE_DP_DIR)

try:
    from CompositeDP.Perturbation_Mechanism import perturbation_fun_multipleCall
    from CompositeDP.Mechanism.Constraints import checkConstraints
except ImportError as e:
    print(f"Error importing CompositeDP: {e}")
    print(f"Make sure CompositeDP directory is at: {COMPOSITE_DP_DIR}")
    raise


def add_composite_dp_noise_to_gradient(
    gradient_array: np.ndarray,
    epsilon: float,
    sensitivity: float,
    lower_bound: float,
    k: float,
    m: float,
    y: float,
    index: int,
) -> np.ndarray:
    """
    Adds CompositeDP noise element-wise to a gradient array.
    
    This function generates noise samples using the Composite DP mechanism
    and adds them to each gradient component.
    
    Args:
        gradient_array: The gradient array to add noise to
        epsilon: Privacy budget
        sensitivity: Sensitivity of the query (L2 norm clip)
        lower_bound: Lower bound of the query output range
        k: CompositeDP parameter (bump height)
        m: CompositeDP parameter (bump width)
        y: CompositeDP parameter (base density)
        index: CompositeDP perturbation function index (1-6)
    
    Returns:
        The gradient array with CompositeDP noise added
    """
    flat_grad = gradient_array.reshape(-1)
    sample_count = flat_grad.shape[0]
    
    # fd = 0 means we want noise centered at 0
    fd = 0.0
    
    try:
        # Generate noise samples using CompositeDP
        noisy_samples = perturbation_fun_multipleCall(
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
        
        # Convert to numpy array
        noisy_samples = np.asarray(noisy_samples, dtype=float)
        
        # Extract pure noise by subtracting fd
        noise = noisy_samples - fd
        
        # Check for invalid values
        if noise.size == 0 or np.isnan(noise).any() or np.isinf(noise).any():
            print("[C-DP] WARNING: Invalid noise detected. Using zero noise for this batch.")
            return gradient_array
        
        # Add noise to gradient
        noisy_flat = flat_grad + noise
        return noisy_flat.reshape(gradient_array.shape)
        
    except Exception as e:
        print(f"[C-DP] WARNING: CompositeDP noise generation failed: {e}")
        print("[C-DP] Using zero noise for this batch.")
        return gradient_array


def test_composite_parameters(
    epsilon: float,
    sensitivity: float,
    lower_bound: float,
    k: float,
    m: float,
    y: float,
    index: int,
    test_samples: int = 100,
) -> Dict[str, Any]:
    """
    Test if the given CompositeDP parameters satisfy the constraints
    and can generate valid noise samples.
    
    Returns:
        Dictionary with test results including:
        - constraints_ok: whether constraints are satisfied
        - can_generate: whether noise can be generated
        - sample_variance: variance of test samples (if generation succeeds)
        - sample_mean: mean of test samples (if generation succeeds)
    """
    # Test fd = 0 (pure noise)
    fd = 0.0
    
    # Check constraints
    # Map fd to Cp space for constraint checking
    from CompositeDP.Mechanism.Mapping import mapping_fromRealToL
    Cp = mapping_fromRealToL(fd, sensitivity, lower_bound, epsilon, k, m, y, index)
    
    constraints_result = checkConstraints(epsilon, k, m, y, Cp, index)
    constraints_ok = (constraints_result == 0)
    
    result = {
        "constraints_ok": constraints_ok,
        "constraint_code": constraints_result,
        "can_generate": False,
        "sample_variance": None,
        "sample_mean": None,
    }
    
    if not constraints_ok:
        return result
    
    # Try to generate samples
    try:
        noisy_samples = perturbation_fun_multipleCall(
            epsilon, fd, sensitivity, lower_bound, k, m, y, index, test_samples
        )
        noisy_samples = np.asarray(noisy_samples, dtype=float)
        noise = noisy_samples - fd
        
        if noise.size > 0 and not np.isnan(noise).any() and not np.isinf(noise).any():
            result["can_generate"] = True
            result["sample_variance"] = float(np.var(noise))
            result["sample_mean"] = float(np.mean(noise))
    except Exception as e:
        result["error"] = str(e)
    
    return result


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
    Auto-calibrate k parameter to match target variance.
    This function is kept for backward compatibility but is not recommended.
    Better to use fixed parameters from the CompositeDP paper.
    """
    if candidate_k_values is None:
        candidate_k_values = [0.05, 0.1, 0.2, 0.3, 0.5]

    best_k = None
    best_variance = None
    lowest_error = float("inf")
    
    print(f"[C-DP] Auto-calibrating k parameter...")
    print(f"[C-DP] Target variance: {target_variance:.6f}")
    
    for k in candidate_k_values:
        test_result = test_composite_parameters(
            epsilon, sensitivity, lower_bound, k, m, y, index, calibration_samples
        )
        
        if not test_result["can_generate"]:
            continue
        
        emp_var = test_result["sample_variance"]
        error = abs(emp_var - target_variance)
        
        print(f"[C-DP]   k={k:.3f}: variance={emp_var:.6f}, error={error:.6f}")
        
        if error < lowest_error:
            lowest_error = error
            best_k = k
            best_variance = emp_var
    
    if best_k is None:
        print("[C-DP] WARNING: All candidate k values failed. Using default k=0.1")
        best_k = 0.1
        best_variance = 0.0
    else:
        print(f"[C-DP] Best k={best_k:.3f} with variance={best_variance:.6f}")
    
    return {
        "L": float(L),
        "m": float(m),
        "y": float(y),
        "k": float(best_k),
        "index": int(index),
        "target_variance": float(target_variance),
        "empirical_variance": float(best_variance),
    }