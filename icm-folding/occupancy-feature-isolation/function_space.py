"""Exact training objective optimized over arbitrary scalar functions of distance.

The sensor is a deterministic function of distance, so every encoder is a scalar
function f(log d). This module removes the MLP and optimizes the *expected*
training objective of the controlled endpoint studies directly over a
piecewise-linear f on a fine log-distance grid:

    0.9 * mean over all source-131 transitions (f(q) - f(q'))^2
  + 0.1 * E_batch[SIGReg1D(1024 iid endpoints from P)]

The SIGReg expectation is exact for iid batches of size N:
    N * |psi - phi|^2 + (1 - |psi|^2)   at each frequency,
integrated with the module's own trapezoid/Gaussian weights. psi is the exact
characteristic function of the code under the empirical endpoint law P.

It compares the best monotone (order-preserving) function with the best
unconstrained function from several starts. All functions are evaluated only
through actual source observations; no states are synthesized.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fs_feature', HERE/'run.py')
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
R, torch, np, B = S.R, S.torch, S.np, S.B
QMAX = float(np.log(16.))
N_BATCH = 1024


class Problem:
    """Expected objective for endpoint atoms q_P with weights w_P."""

    def __init__(self, q_end, w_end, q_a, q_b, knots=321, align_weight=.9, reg_weight=.1):
        self.grid = torch.linspace(0, QMAX, knots, dtype=torch.float64)
        self.h = self.grid[1] - self.grid[0]
        self.w_end = torch.as_tensor(w_end, dtype=torch.float64)
        self.end = self._interp(torch.as_tensor(q_end, dtype=torch.float64))
        self.a, self.b = self._interp(q_a), self._interp(q_b)
        reg = B.SIGReg1D().double()
        self.t, self.phi, self.wt = reg.frequency, reg.target, reg.weight
        self.aw, self.rw = align_weight, reg_weight

    def _interp(self, q):
        q = torch.as_tensor(q, dtype=torch.float64).clamp(0, QMAX)
        k = (q/self.h).floor().long().clamp(0, len(self.grid) - 2)
        return k, (q - self.grid[k])/self.h

    def values(self, f, ix):
        k, r = ix
        return f[k]*(1 - r) + f[k + 1]*r

    def parts(self, f):
        y = self.values(f, self.end)
        ph = y[:, None]*self.t
        c = (self.w_end[:, None]*ph.cos()).sum(0); s = (self.w_end[:, None]*ph.sin()).sum(0)
        bias = ((c - self.phi).square() + s.square())*self.wt
        var = (1 - c.square() - s.square())*self.wt
        align = (self.values(f, self.a) - self.values(f, self.b)).square().mean()
        sig = N_BATCH*bias.sum() + var.sum()
        return dict(alignment=align, sigreg=sig, loss=self.aw*align + self.rw*sig,
                    sigreg_bias=N_BATCH*bias.sum(), sigreg_variance=var.sum())

    def loss(self, f):
        return self.parts(f)['loss']


def monotone(theta, h):
    # f = f0 + cumulative positive increments: strictly order-preserving.
    return torch.cat([theta[:1], theta[:1] + torch.cumsum(torch.nn.functional.softplus(theta[1:])*h, 0)])


def optimize(problem, start, constrained, steps=6000, lr=.02):
    g = problem.grid
    if constrained:
        d = torch.diff(start)/problem.h
        d = d.clamp_min(1e-4)
        inv = torch.where(d > 20, d, torch.log(torch.expm1(d)))
        theta = torch.cat([start[:1], inv]).clone().requires_grad_(True)
        make = lambda th: monotone(th, problem.h)
    else:
        theta = start.clone().requires_grad_(True); make = lambda th: th
    opt = torch.optim.Adam([theta], lr=lr)
    for i in range(steps):
        frac = i/steps
        opt.param_groups[0]['lr'] = lr*(.5*(1 + np.cos(np.pi*frac)))*.98 + lr*.02
        loss = problem.loss(make(theta)); opt.zero_grad(); loss.backward(); opt.step()
    lbfgs = torch.optim.LBFGS([theta], lr=1, max_iter=2000, tolerance_grad=1e-12, tolerance_change=1e-15,
                              history_size=50, line_search_fn='strong_wolfe')

    def closure():
        lbfgs.zero_grad(); l = problem.loss(make(theta)); l.backward(); return l
    lbfgs.step(closure)
    f = make(theta).detach()
    return f, {k: float(v) for k, v in problem.parts(f).items()}


def fold_stats(f, problem, lo=None, hi=None):
    """Turns and backward variation of f restricted to [lo, hi] in log-distance."""
    g = problem.grid.numpy(); y = f.numpy()
    keep = np.ones_like(g, bool)
    if lo is not None: keep &= g >= lo - 1e-12
    if hi is not None: keep &= g <= hi + 1e-12
    y = y[keep]; g = g[keep]; dy = np.diff(y)
    up, down = np.maximum(dy, 0).sum(), np.maximum(-dy, 0).sum()
    back = float(min(up, down)/max(up + down, 1e-12))
    sgn = np.sign(dy); nz = sgn != 0; sg = sgn[nz]; gg = g[1:][nz]
    turns = [float(np.exp(gg[i])) for i in np.flatnonzero(sg[:-1]*sg[1:] < 0)]
    return dict(backward_fraction=back, material=back > .001, turns=turns)


def alignment_pairs():
    data = R.trials.load(R.LEGACY/'output/sources/icm-131.pt')['data']
    return torch.log(1/data['x'][:, 1].double()), torch.log(1/data['xp'][:, 1].double())


def mixture(near_q, far_q, alpha):
    q = torch.cat([near_q, far_q])
    w = torch.cat([torch.full((len(near_q),), alpha/len(near_q)),
                   torch.full((len(far_q),), (1 - alpha)/len(far_q))]).double()
    return q, w


def mlp_function(state, grid):
    m = B.Encoder().double(); m.load_state_dict({k: v.double() for k, v in state.items()})
    c = B.Config()
    with torch.no_grad():
        return m(B.sensor(grid.exp(), c)).double()
