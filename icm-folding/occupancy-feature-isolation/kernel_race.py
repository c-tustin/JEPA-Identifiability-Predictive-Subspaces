"""Kernel race statistic: does SIGReg's force, filtered through the encoder's tangent kernel at initialization,
bend the code curve back faster than it removes the skewness monotonically?

For a law P (atoms d, weights w), an input map and an initializer:
  1. Bin atoms onto a 240-point log-distance grid; z = standardized initial code f0 on the grid under P.
  2. g_b = dSIGReg/dz_b for bin mass w_b (exact expected force; standardized units).
  3. Function-space velocity v = -J M J^T g, with J the Jacobian of the grid codes w.r.t. the parameters and M either the
     identity (gradient descent, 'gd') or Adam's first-step sign map ('sign': v = -J sign(J^T g)).
  4. Remove the affine part of v (a + b z, weighted by P): affine motion never changes the standardized shape.
  5. fold_rate = max over occupied adjacent grid pairs of the rate at which their order closes: -(dv)/(dz).
     skew_rate = -dγ/dt under v, relative to γ.  race = fold_rate / max(skew_rate, 1e-9).
A large race means the kernel bends the curve shut before it can Gaussianize monotonically.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import geometry as G

K, R, B = G.K, G.R, G.B
GRID = torch.linspace(0, np.log(16.), 241, dtype=torch.float64)
_JAC = {}


def jacobian(input_map, init_seed, c):
    key = (input_map, init_seed)
    if key not in _JAC:
        m = G.MappedEncoder(input_map).double()
        m.load_state_dict({k: v.double() for k, v in G.initial(c, init_seed).items()})
        X = B.sensor(GRID.exp(), c).double()
        params = list(m.parameters())
        f = m(X); J = []
        for i in range(len(GRID)):
            gr = torch.autograd.grad(f[i], params, retain_graph=True); J.append(torch.cat([q.flatten() for q in gr]))
        _JAC[key] = (torch.stack(J).numpy(), f.detach().numpy())
    return _JAC[key]


def sigreg_grad(z, w):
    reg = B.SIGReg1D().double(); t = reg.frequency.numpy(); wt = reg.weight.numpy(); phi = reg.target.numpy()
    C = w @ np.cos(np.outer(z, t)); S = w @ np.sin(np.outer(z, t))
    # d/dz_b of sum_t wt (C-phi)^2 + S^2, with C = sum_b w_b cos(t z_b)
    return w*((2*(C - phi)*(-t*np.sin(np.outer(z, t))) + 2*S*(t*np.cos(np.outer(z, t))))@wt)


def race(d, w, input_map, init_seed, c, mode='sign'):
    J, f0 = jacobian(input_map, init_seed, c)
    b = np.clip(np.rint((np.log(d) - 0)/(np.log(16.)/(len(GRID) - 1))).astype(int), 0, len(GRID) - 1)
    wb = np.bincount(b, weights=w/w.sum(), minlength=len(GRID))
    mu = wb @ f0; sd = np.sqrt(wb @ (f0 - mu)**2); z = (f0 - mu)/sd
    if np.corrcoef(GRID.numpy(), z)[0, 1] < 0: z = -z; J = -J
    g = sigreg_grad(z, wb)
    v = -(J @ (np.sign(J.T @ g) if mode == 'sign' else J.T @ g))/sd
    A = np.stack([np.ones_like(z), z], 1); coef = np.linalg.lstsq(A*np.sqrt(wb)[:, None], v*np.sqrt(wb), rcond=None)[0]
    v = v - A @ coef
    occ = np.flatnonzero(wb > 1e-6); dz = np.diff(z[occ]); dv = np.diff(v[occ])
    fold_rate = max(float(np.max(-dv/np.maximum(np.abs(dz), 1e-9)*np.sign(dz))), 0.)
    gam = float(wb @ z**3); dgam = float((3*wb*z**2 - 3*wb*(wb @ z**2)) @ v)  # d/dt of E z^3 with mean/var fixed
    skew_rate = max(-dgam*np.sign(gam)/max(abs(gam), 1e-3), 1e-9)
    return dict(fold_rate=fold_rate, skew_rate=skew_rate, race=fold_rate/skew_rate, skew=gam)
