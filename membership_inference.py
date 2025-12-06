from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def loss_threshold_membership_inference_auc(
    train_losses: Sequence[float],
    test_losses: Sequence[float],
) -> float:
    """
    Simple membership inference attack:
    - Members (train) tend to have lower loss than non-members (test).
    - We use -loss as the score and compute ROC AUC.
    """
    train_losses = np.array(train_losses, dtype=float)
    test_losses = np.array(test_losses, dtype=float)

    labels = np.concatenate([np.ones_like(train_losses), np.zeros_like(test_losses)])
    scores = np.concatenate([-train_losses, -test_losses])

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float("nan")
    return float(auc)
