import math
from typing import Tuple

import tensorflow as tf

# DP-SGD optimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

# New-style import for epsilon computation:
# In recent tensorflow-privacy versions, compute_dp_sgd_privacy is exposed at the
# top-level package and backed by compute_dp_sgd_privacy_lib.
from tensorflow_privacy import compute_dp_sgd_privacy as _tfp_compute_dp_sgd_privacy


def build_dp_sgd_optimizer(
    learning_rate: float,
    l2_norm_clip: float,
    noise_multiplier: float,
    num_microbatches: int,
) -> tf.keras.optimizers.Optimizer:
    """
    Build a TensorFlow Privacy DPKerasSGDOptimizer with the given parameters.
    """
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate,
    )
    return optimizer


def compute_epsilon(
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    """
    Compute epsilon for DP-SGD training using tensorflow-privacy's
    compute_dp_sgd_privacy helper.

    Args:
        num_examples: Size of the training set (N).
        batch_size: Batch size used in training.
        num_epochs: Number of epochs run so far.
        noise_multiplier: Gaussian noise multiplier used in DP-SGD.
        delta: Target delta.

    Returns:
        epsilon: The DP epsilon value.
    """
    if noise_multiplier == 0.0:
        # Non-private case, formally eps = inf
        return float("inf")

    eps, _ = _tfp_compute_dp_sgd_privacy(
        n=num_examples,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=num_epochs,
        delta=delta,
    )
    return float(eps)
