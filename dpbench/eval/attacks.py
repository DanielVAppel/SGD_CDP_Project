import torch
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def loss_threshold_attack(train_losses, test_losses):
    # Yeom-style: members (train) have lower loss --> use -loss as score
    y = [1]*len(train_losses) + [0]*len(test_losses)
    s = [-float(l) for l in train_losses] + [-float(l) for l in test_losses]
    try:
        return roc_auc_score(y, s)
    except Exception:
        return float('nan')
