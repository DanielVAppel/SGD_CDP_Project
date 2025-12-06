import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers=0, drop_last=True):
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=drop_last)
    val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train, val, test

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    ys, yhats, losses = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
        ys.append(yb.detach().cpu())
        yhats.append(logits.argmax(dim=1).detach().cpu())
    ys, yhats = torch.cat(ys).numpy(), torch.cat(yhats).numpy()
    acc = accuracy_score(ys, yhats)
    f1  = f1_score(ys, yhats, average='macro')
    return acc, f1, sum(losses)/max(1,len(losses)), losses

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))
