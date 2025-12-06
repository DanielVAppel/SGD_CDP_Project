from typing import Tuple

import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def build_dp_sgd_optimizer(
    learning_rate: float,
    l2_norm_clip: float,
    noise_multiplier: float,
    num_microbatches: int,
) -> tf.keras.optimizers.Optimizer:
    """
    Create a DPKerasSGDOptimizer instance for DP-SGD training.
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
    Compute epsilon using the TF-Privacy DP-SGD privacy analysis helper.
    """
    steps_per_epoch = num_examples // batch_size
    if steps_per_epoch == 0:
        return float("nan")

    eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=num_examples,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=num_epochs,
        delta=delta,
    )
    return float(eps)
