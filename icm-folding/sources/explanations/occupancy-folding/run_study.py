"""Run matched occupancy interventions; safe to resume completed run files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/occupancy-folding-mpl')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # repository root for provenance paths
MECH = HERE.parents[1] / 'predictive-closure-workspace/behavior-policy-experiments/qualitative-study/fold-location/training-mechanism'
sys.path.insert(0, str(MECH))
import run_mechanism as old
import numpy as np
import torch

OUT = HERE / 'output'
METHODS = ('ordinary', 'inverse_gradient', 'weighted_moments')
PEAKS = {'near2': [2.], 'bimodal': [2., 8.], 'trimodal': [2., 5., 12.], 'uniform': []}


def penalty(y, weights, method, reg):
    """Keep batch-size multiplier fixed, including for importance weights."""
    if method == 'ordinary':
        return reg(y)
    weights = weights / weights.mean()
    if method == 'inverse_gradient':
        y.register_hook(lambda g: g * weights)
        return reg(y)
    phase = y[:, None] * reg.frequency
    w = weights / weights.sum()
    co = (w[:, None] * phase.cos()).sum(0)
    si = (w[:, None] * phase.sin()).sum(0)
    return len(y) * (((co - reg.target).square() + si.square()) * reg.weight).sum()


def roots(q, slope):
    k = np.flatnonzero(slope[:-1] * slope[1:] < 0)
    return np.exp(q[k] - slope[k] * (q[k+1] - q[k]) / (slope[k+1] - slope[k])).tolist()


def curve(model, c):
    q = torch.linspace(0, np.log(16), 1025, requires_grad=True)
    y = model(old.trials.behavior.sensor(q.exp(), c))
    s = torch.autograd.grad(y.sum(), q)[0]
    return q.detach().numpy(), y.detach().numpy(), s.detach().numpy()


def synthetic_mass(name):
    q = (torch.arange(64) + .5) * np.log(16) / 64
    if not PEAKS[name]:
        return torch.ones(64) / 64
    peak = torch.stack([torch.exp(-.5 * ((q - np.log(d)) / .14)**2) for d in PEAKS[name]])
    peak = (peak / peak.sum(1, keepdim=True)).mean(0)
    return .1 / 64 + .9 * peak


def fit(seed, distribution, method):
    path = OUT / 'runs' / f'{distribution}-{method}-{seed}.pt'
    if path.exists():
        return
    started = time.perf_counter()
    settings = json.loads((MECH / 'output/settings.json').read_text())
    c = old.trials.behavior.Config(**settings['config'])
    data = old.trials.load(old.ORIGINAL / 'sources' / f'icm-{seed}.pt')['data']
    with old.encoder_gain(4):
        model, plan = old.trials.study.make_plan(c, seed, seed, 'icm')
    generator = torch.Generator().manual_seed(1900000 + seed)
    reg_indices = torch.randint(len(data['x']), plan['indices'].shape, generator=generator)
    reg = old.trials.behavior.SIGReg1D()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    if distribution == 'native':
        endpoints = torch.cat([data['x'], data['xp']])
        d = old.trials.distances(endpoints)
        bins = (d.log() / np.log(16) * 32).long().clamp(0, 31)
        mass = torch.bincount(bins, minlength=32).float() / len(bins)
        inverse = 1 / mass[bins]
    else:
        mass = synthetic_mass(distribution)
    frames = []
    q, y, slope = curve(model, c)
    frames.append(dict(step=0, y=y, slope=slope, roots=roots(q, slope)))
    for step in range(c.jepa_steps):
        opt.param_groups[0]['lr'] = old.trials.study.controls.learning_rate(step, c.jepa_steps)
        ix = plan['indices'][step]
        y = model(torch.cat([data['x'][ix], data['xp'][ix]]))
        alignment = (y[:c.batch] - y[c.batch:]).square().mean()
        if distribution == 'native':
            ri = reg_indices[step]
            ri = torch.cat([ri, ri + len(data['x'])])
            rx, weights = endpoints[ri], inverse[ri]
        else:
            bi = torch.multinomial(mass, 2*c.batch, replacement=True, generator=generator)
            rq = (bi + torch.rand(2*c.batch, generator=generator)) * np.log(16) / 64
            rx = old.trials.behavior.sensor(rq.exp(), c)
            weights = 1 / mass[bi]
        ry = model(rx)
        regloss = penalty(ry, weights, method, reg)
        loss = .9 * alignment + .1 * regloss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1) % 25 == 0:
            q, y, slope = curve(model, c)
            frames.append(dict(step=step+1, y=y, slope=slope, roots=roots(q, slope)))
    q, y, slope = curve(model, c)
    sign = 1 if np.cov(q, y)[0, 1] >= 0 else -1
    metrics = dict(roots=roots(q, slope), folded=bool(roots(q, slope)),
                   r2=float(np.corrcoef(q, y)[0, 1]**2),
                   reversal_fraction=float(np.mean(sign * slope < 0)),
                   seconds=time.perf_counter()-started)
    old.trials.save_torch(path, dict(seed=seed, distribution=distribution, method=method,
        encoder=model.state_dict(), q=q, frames=frames, mass=mass.numpy(), metrics=metrics,
        config=vars(c), final_optimizer=opt.state_dict()))
    print(path.stem, json.dumps(metrics), flush=True)


def audit():
    """Gradient checks independent of the training network."""
    torch.manual_seed(41)
    reg = old.trials.behavior.SIGReg1D().double()
    y = torch.randn(23, dtype=torch.float64, requires_grad=True)
    w = torch.rand(23, dtype=torch.float64) + .1
    raw = torch.autograd.grad(reg(y), y)[0]
    hooky = y.detach().clone().requires_grad_()
    hookgrad = torch.autograd.grad(penalty(hooky, w, 'inverse_gradient', reg), hooky)[0]
    torch.testing.assert_close(hookgrad, raw*w/w.mean())
    unity = penalty(y, torch.ones_like(w), 'weighted_moments', reg)
    torch.testing.assert_close(unity, reg(y))
    ok = torch.autograd.gradcheck(lambda z: penalty(z, w, 'weighted_moments', reg), (y,))
    old.trials.atomic_json(OUT/'gradient-checks.json', dict(inverse_gradient=True,
        uniform_weight_equivalence=True, weighted_moments_gradcheck=ok))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        audit()
        return
    jobs = [(s, 'native', m) for s in range(130, 140) for m in METHODS]
    jobs += [(s, d, m) for s in (130, 131, 136) for d in PEAKS
             for m in (('ordinary',) if d == 'uniform' else METHODS)]
    for i, job in enumerate(jobs):
        if i % args.shards == args.shard:
            fit(*job)


if __name__ == '__main__':
    main()
