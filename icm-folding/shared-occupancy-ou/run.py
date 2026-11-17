"""Generate multimodal temporal pairs; train both losses on identical endpoints."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/shared-occupancy-ou-mpl')
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import numpy as np
from scipy.special import ndtr, ndtri
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared_ou_reference', HERE.parent/'occupancy-study/run_study.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
old = base.old
OUT = HERE/'output'
EDGES = np.linspace(0, np.log(16), 65)
SEEDS = (130, 131, 136)
TARGETS = ('uniform', 'bimodal', 'trimodal')
DTS = (.01, .1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def masses(name):
    m = base.synthetic_mass(name).double().numpy()
    return m/m.sum()

def quantile(u, mass):
    cdf = np.r_[0., np.cumsum(mass)]
    cdf[-1] = 1.
    return np.interp(u, cdf, EDGES)

def generate(seed, name, dt, episodes=256, horizon=64):
    if dt <= 0:
        raise ValueError('dt must be positive')
    rng = np.random.default_rng(2300000+seed)
    initial = rng.standard_normal(episodes)
    noise = rng.standard_normal((horizon, episodes))
    z = np.empty((horizon+1, episodes))
    z[0] = initial
    rho = np.exp(-dt)
    for t in range(horizon):
        z[t+1] = rho*z[t] + np.sqrt(-np.expm1(-2*dt))*noise[t]
    q = quantile(ndtr(z), masses(name))
    return dict(z=z, q=q, initial=initial, noise=noise, dt=dt, target=name, seed=seed)

def observations(trajectory, c):
    q = trajectory['q']
    x = old.trials.behavior.sensor(torch.tensor(np.exp(q[:-1].reshape(-1)), dtype=torch.float32), c)
    xp = old.trials.behavior.sensor(torch.tensor(np.exp(q[1:].reshape(-1)), dtype=torch.float32), c)
    return x, xp

def shared_loss(model, x, xp, reg):
    y = model(torch.cat((x, xp)))
    n = len(x)
    align = (y[:n]-y[n:]).square().mean()
    regularizer = reg(y)
    return .9*align+.1*regularizer, align, regularizer, y

def config():
    settings = json.loads((base.MECH/'output/settings.json').read_text())
    return old.trials.behavior.Config(**settings['config'])

def fit(seed, name, dt):
    key = f'{name}-dt{dt:g}-{seed}'
    path = OUT/f'{key}.pt'
    if path.exists():
        saved = old.trials.load(path)
        assert saved['code_sha256'] == sha(__file__), 'Code changed: use a new output directory'
        assert saved['protocol_sha256'] == sha(HERE/'PROTOCOL.md')
        return
    c = config()
    trajectory = generate(seed, name, dt, c.train_episodes, c.horizon)
    x, xp = observations(trajectory, c)
    torch.manual_seed(40000+seed)
    model = old.ScaledEncoder(4)
    initial = copy.deepcopy(model.state_dict())
    indices = torch.randint(len(x), (c.jepa_steps, c.batch), generator=torch.Generator().manual_seed(2400000+seed))
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    reg = old.trials.behavior.SIGReg1D()
    history = []
    for step, ix in enumerate(indices):
        opt.param_groups[0]['lr'] = old.trials.study.controls.learning_rate(step, c.jepa_steps)
        loss, align, regularizer, _ = shared_loss(model, x[ix], xp[ix], reg)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1)%100 == 0:
            q, y, slope = base.curve(model, c)
            history.append(dict(step=step+1, y=y, slope=slope, roots=base.roots(q,slope), alignment=float(align.detach()), sigreg=float(regularizer.detach())))
    old.trials.save_torch(path, dict(seed=seed, target=name, dt=dt, config=vars(c), trajectory=trajectory,
        x=x, xp=xp, indices=indices, initial=initial, encoder=model.state_dict(), history=history,
        code_sha256=sha(__file__), protocol_sha256=sha(HERE/'PROTOCOL.md'),
        reference_sha256=sha(base.__file__), sensor_sha256=sha(old.trials.behavior.__file__)))
    print(key, 'final roots', history[-1]['roots'], flush=True)

def audit():
    m = masses('trimodal')
    u = np.linspace(0, 1, 10001)
    np.testing.assert_allclose(np.interp(quantile(u,m), EDGES, np.r_[0,np.cumsum(m)]), u, atol=1e-14)
    rng = np.random.default_rng(27)
    count, _ = np.histogram(quantile(ndtr(rng.standard_normal(1000000)),m), EDGES)
    iid_tv = float(abs(count/count.sum()-m).sum()/2)
    assert iid_tv < .006
    c = config()
    for name in TARGETS:
        for dt in DTS:
            tr = generate(130,name,dt)
            np.testing.assert_allclose(tr['z'][1:], np.exp(-dt)*tr['z'][:-1]+np.sqrt(-np.expm1(-2*dt))*tr['noise'], atol=0, rtol=0)
            x,xp = observations(tr,c)
            assert torch.equal(xp[:-256],x[256:])
            assert torch.allclose(old.trials.distances(x),torch.tensor(np.exp(tr['q'][:-1].reshape(-1)),dtype=torch.float32),atol=1e-5)
    torch.manual_seed(42)
    model = old.ScaledEncoder(4)
    x = torch.randn(11,6,requires_grad=True)
    xp = torch.randn(11,6,requires_grad=True)
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    seen = []
    real_reg = old.trials.behavior.SIGReg1D()
    def spy(y):
        seen.append(y)
        return real_reg(y)
    loss,a,r,y = shared_loss(model,x,xp,spy)
    handle.remove()
    assert len(calls)==1 and seen[0] is y
    for term in (a,r):
        gx,gxp = torch.autograd.grad(term,(x,xp),retain_graph=True)
        assert gx.abs().sum()>0 and gxp.abs().sum()>0
    ga = torch.autograd.grad(a,y,retain_graph=True)[0]
    expected = 2*(y[:11]-y[11:])/11
    torch.testing.assert_close(ga[:11],expected)
    torch.testing.assert_close(ga[11:],-expected)
    old.trials.atomic_json(OUT/'audit.json',dict(quantile_roundtrip=True, independent_million_sample_tv=iid_tv,
        exact_ou_replay=True, adjacent_pair_continuity=True, sensor_reconstruction=True,
        one_forward_call=True, sigreg_same_tensor=True, both_losses_reach_both_endpoints=True))
    print('Generator and shared-input audits passed', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.check:
        audit()
        return
    for seed in ([args.seed] if args.seed is not None else SEEDS):
        for name in TARGETS:
            for dt in DTS:
                fit(seed,name,dt)

if __name__ == '__main__':
    main()
