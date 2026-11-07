"""Check the population preference separately from trained-network behavior.

For independent stationary pairs, derive the finite paired-batch SIGReg
expectation exactly for a linear Gaussian code; integrate a Gaussianized fold
by Monte Carlo. The fold is a measurable-code comparator, not a trained model.
"""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from zoo import behavior, source_rho
import numpy as np
from scipy import optimize, special
import torch

OUT = HERE/"output"
N = 1024


def main():
    torch.set_num_threads(1)
    regularizer = behavior.SIGReg1D().double()
    t = regularizer.frequency.numpy()
    target = regularizer.target.numpy()
    w = regularizer.weight.numpy()
    def linear(s, rho):
        phi = np.exp(-s*s*t*t/2)
        psi = np.exp(-s*s*t*t*(1-rho))
        penalty = np.sum(w*(N*(phi-target)**2+1+psi-2*phi**2))
        return float(.9*2*s*s*(1-rho)+.1*penalty)

    checks = []
    for rho in (.9, -.45):
        for scale in (.6, 1.):
            rng = np.random.default_rng(20261104)
            empirical = []
            for _ in range(512):
                z, noise = rng.standard_normal((2, N//2))
                y, yp = scale*z, scale*(rho*z+np.sqrt(1-rho*rho)*noise)
                penalty = float(regularizer(torch.from_numpy(np.concatenate([y, yp]))))
                empirical.append(.9*np.mean((y-yp)**2)+.1*penalty)
            predicted = linear(scale, rho)
            se = float(np.std(empirical, ddof=1)/np.sqrt(len(empirical)))
            difference = float(np.mean(empirical)-predicted)
            assert abs(difference) < 5*se+.001, (rho, scale, difference, se)
            checks.append(dict(rho=rho, scale=scale, analytic=predicted,
                               monte_carlo=float(np.mean(empirical)), mc_standard_error=se))

    rows = []
    count = 2**20
    for name in ("ou", "ou_hp_w1", "ou_hp2_w1", "ou_hp2_w3"):
        rho = source_rho(name)
        # Nonnegative SIGReg gives a finite global search bound from alignment.
        upper = np.sqrt(linear(1.,rho)/(1.8*(1-rho)))+1e-8
        grid = np.linspace(0, upper, 1001)
        losses = np.array([linear(s,rho) for s in grid])
        candidates = [(losses[0],0.),(losses[-1],upper)]
        for k in range(1,len(grid)-1):
            if losses[k] <= losses[k-1] and losses[k] <= losses[k+1]:
                fit = optimize.minimize_scalar(lambda s: linear(s,rho),bounds=(grid[k-1],grid[k+1]),method="bounded")
                candidates.append((fit.fun,fit.x))
        best_loss, scale = min(candidates)
        rng = np.random.default_rng(20261104)
        total, squares = 0., 0.
        phi = np.exp(-t*t/2)
        constant = N*(phi-target)**2+1-2*phi**2
        for offset in range(0,count,65536):
            z, noise = rng.standard_normal((2,min(65536,count-offset)))
            zp = rho*z+np.sqrt(1-rho*rho)*noise
            def fold(x):
                return special.ndtri(np.clip(2*special.ndtr(np.abs(x))-1,1e-12,1-1e-12))
            difference = fold(z)-fold(zp)
            value = .9*difference**2+.1*((constant+np.cos(difference[:,None]*t))*w).sum(1)
            total += value.sum()
            squares += (value**2).sum()
        mean = total/count
        se = np.sqrt(max(0,squares/count-mean**2)/count)
        rows.append(dict(source=name,rho=rho,best_linear_scale=float(scale),linear_expected_loss=float(best_loss),
                         gaussianized_fold_expected_loss=float(mean),fold_mc_se=float(se),pairs=count,
                         fold_better_by_more_than_five_mc_se=bool(mean+5*se<best_loss)))
    OUT.mkdir(exist_ok=True)
    (OUT/"loss-theory.json").write_text(json.dumps(dict(batch_endpoints=N,formula_checks=checks,comparators=rows,
        scope="Independent stationary pairs; implemented finite paired-batch loss. Fold comparator is not a fitted network or proof of its representability by the fixed architecture."),indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
