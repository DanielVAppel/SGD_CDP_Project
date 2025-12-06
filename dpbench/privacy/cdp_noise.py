import torch, math
from torch.distributions.uniform import Uniform

class CompositeSineBump:
    """
    Bounded, unbiased composite noise on [-L, L] with base y and a sine "bump" of width m
    centered at 0 (we want zero-mean for gradient noise). Approximates A2B1:
    P(x) = y + k*sin(pi*(x+ m/2)/m) on |x|<=m/2, else P(x)=y, normalized.

    If target_var is provided, we crude-tune k in (0,1) to roughly match it.
    """
    def __init__(self, L=1.0, m=0.5, y=0.05, k=None, target_var=None, device='cpu'):
        assert L>0 and 0<m<=2*L, "Invalid (L, m)"
        self.L, self.m, self.y = float(L), float(m), float(y)
        self.device = device
        self.k = 0.5 if k is None else float(k)
        if target_var is not None:
            self._tune_k_for_variance(target_var)
        self.u_base = Uniform(-self.L, self.L)

    def _pdf(self, x):
        p = torch.full_like(x, self.y)
        a, b = -self.m/2, self.m/2
        mask = (x>=a) & (x<=b)
        p[mask] = p[mask] + self.k*torch.sin(math.pi*(x[mask]-a)/self.m)
        Z = 2*self.y*self.L + 2*self.k*self.m/math.pi
        return p / Z

    def _tune_k_for_variance(self, target_var, steps=50):
        ks = torch.linspace(0.05, 0.95, steps)
        best = (None, 1e9)
        for k in ks:
            self.k = float(k)
            v = self.variance()
            err = abs(v-target_var)
            if err < best[1]:
                best = (float(k), float(err))
        self.k = best[0]

    def sample(self, shape):
        # Rejection sampling
        max_pdf = (self.y + self.k) / (2*self.y*self.L + 2*self.k*self.m/math.pi)
        samples, needed = [], int(torch.tensor(shape).numel())
        while len(samples) < needed:
            x = self.u_base.sample((needed*2,)).to(self.device)
            u = torch.rand_like(x) * max_pdf
            accept = u <= self._pdf(x)
            samples.extend(x[accept].tolist())
        out = torch.tensor(samples[:needed], device=self.device).reshape(shape)
        # Unbiased (mean≈0), bounded in [-L, L]
        return out

    def variance(self, n=20000):
        x = self.u_base.sample((n,)).to(self.device)
        p = self._pdf(x)
        mu = (x*p).sum() / p.sum()
        var = ((x-mu)**2 * p).sum() / p.sum()
        return var.item()
