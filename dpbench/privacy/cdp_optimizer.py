import torch
from torch.optim import Optimizer

class CDPOptimizer(torch.optim.SGD):
    """
    SGD + Composite-DP noise: clip total grad norm to max_grad_norm, then add bounded composite noise.
    """
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0,
                 max_grad_norm=1.0, noise_sampler=None):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.noise_sampler = noise_sampler

    @torch.no_grad()
    def step(self, closure=None):
        # Clip global grad norm
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norms.append(p.grad.norm(2))
        if norms:
            total = torch.norm(torch.stack(norms), 2)
            if total > self.max_grad_norm and total > 0:
                scale = self.max_grad_norm / (total + 1e-12)
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.mul_(scale)

        # Add composite noise
        if self.noise_sampler is not None:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        noise = self.noise_sampler.sample(p.grad.shape).to(p.grad.device)
                        p.grad.add_(noise)

        return super().step(closure)
