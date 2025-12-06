import argparse, os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from dpbench.datasets.mnist import get_mnist
from dpbench.models.cnn import SmallCNN
from dpbench.utils.training import make_loaders, evaluate, train_epoch
from dpbench.datasets.adult import get_adult

from dpbench.models.mlp import MLP

# DP-SGD baseline (Opacus/Gaussian)
from dpbench.privacy.dpsgd import attach_privacy, get_epsilon
# Composite-DP (bounded + unbiased sine-bump)
from dpbench.privacy.cdp_noise import CompositeSineBump
from dpbench.privacy.cdp_optimizer import CDPOptimizer

from dpbench.eval.attacks import loss_threshold_attack

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mnist'], default='mnist')
    ap.add_argument('--mechanism', choices=['dpsgd','cdp','none'], default='dpsgd')
    ap.add_argument('--epsilon', type=float, default=2.0)
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--cdp-L', type=float, default=1.0)
    ap.add_argument('--cdp-m', type=float, default=0.5)
    ap.add_argument('--cdp-y', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--runs-dir', type=str, default='runs')
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset/model
    train_ds, val_ds, test_ds = get_mnist()
    model = SmallCNN(num_classes=10).to(device)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, args.batch_size)
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    base_opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    privacy_engine = None
    logs = dict(vars(args))
    logs['device'] = device
    logs['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

    if args.mechanism == 'dpsgd':
        # Heuristic start for Gaussian noise; report achieved ε each epoch.
        noise_multiplier = 1.2  # tune toward your target ε using accountant readouts
        privacy_engine, model, base_opt, train_loader = attach_privacy(
            model, base_opt, train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            target_delta=args.delta
        )
        logs['noise_multiplier'] = noise_multiplier

    elif args.mechanism == 'cdp':
        # Composite-DP: bounded, unbiased sampler replaces Gaussian noise.
        # We target a rough variance; tune (L,m,y,target_var) during calibration.
        target_var = 0.2
        sampler = CompositeSineBump(L=args.cdp_L, m=args.cdp_m, y=args.cdp_y,
                                    target_var=target_var, device=device)
        base_opt = CDPOptimizer(model.parameters(), lr=args.lr, momentum=0.9,
                                max_grad_norm=args.max_grad_norm, noise_sampler=sampler)
        logs['cdp_sampler'] = {'L': args.cdp_L, 'm': args.cdp_m, 'y': args.cdp_y, 'var': sampler.variance()}

    # Train
    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'eps': []}
    for epoch in range(args.epochs):
        tl = train_epoch(model, train_loader, base_opt, loss_fn, device)
        acc, f1, vloss, _ = evaluate(model, val_loader, loss_fn, device)
        history['train_loss'].append(tl)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        if privacy_engine is not None:
            eps = get_epsilon(privacy_engine, args.delta)
            history['eps'].append(eps)
            print(f"Epoch {epoch+1}: train_loss={tl:.4f} val_acc={acc:.4f} val_f1={f1:.4f} eps≈{eps}")
        else:
            print(f"Epoch {epoch+1}: train_loss={tl:.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

    # Final eval + membership inference (loss-threshold AUC)
    train_acc, train_f1, train_loss_mean, train_losses = evaluate(model, train_loader, loss_fn, device)
    test_acc,  test_f1,  test_loss_mean,  test_losses  = evaluate(model, test_loader,  loss_fn, device)
    mi_auc = loss_threshold_attack(train_losses, test_losses)

    logs.update({
        'val_acc_last': history['val_acc'][-1],
        'val_f1_last': history['val_f1'][-1],
        'train_acc': train_acc, 'train_f1': train_f1, 'train_loss': train_loss_mean,
        'test_acc': test_acc,   'test_f1': test_f1,   'test_loss': test_loss_mean,
        'mi_auc': mi_auc,
        'history': history
    })

    out_dir = Path(args.runs_dir) / args.dataset / args.mechanism / time.strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/'metrics.json', 'w') as f:
        json.dump(logs, f, indent=2)
    torch.save(model.state_dict(), out_dir/'model.pt')
    print(f"Saved run artifacts to: {out_dir}")
    print(json.dumps({'val_acc': logs['val_acc_last'], 'test_acc': logs['test_acc'], 'mi_auc': logs['mi_auc']}, indent=2))

if __name__ == '__main__':
    main()
