"""Observation/action-only encoder training; physical geometry is evaluation-only."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/icm-gradient-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/icm-gradient-cache')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "sources"
LEGACY = ROOT/'predictive-closure-workspace/behavior-policy-experiments/qualitative-study'
sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location('gradient_study_trials', LEGACY/'run_trials.py')
trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trials)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

B = trials.behavior
OUT = HERE/'output'
METHODS = ('sigreg', 'vicreg', 'infonce', 'quantile')
STEPS = 4000
FRAMES = (0, 100, 250, 500, 1000, 2000, 4000)


def sha(tensor):
    return hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest()


def source(seed):
    path = ((LEGACY/'output/sources') if seed < 140 else (OUT/'fresh-sources'))/f'icm-{seed}.pt'
    saved = trials.load(path)
    c = B.Config(**saved['config'])
    # All geometry labels are dropped before constructing a training batch.
    clean = {k: saved['data'][k].clone() for k in ('x', 'xp', 'action')}
    episodes = torch.arange(len(clean['x'])) % c.train_episodes
    mask = episodes < int(.8*c.train_episodes)
    train = {k:v[mask] for k,v in clean.items()}
    test = {k:v[~mask] for k,v in clean.items()}
    assert set(train) == {'x', 'xp', 'action'}
    return c, train, test, hashlib.sha256(path.read_bytes()).hexdigest()


def standardized_successors(train):
    target = torch.tanh(4*train['xp'])
    mean = target.mean(0)
    std = target.std(0, unbiased=False).clamp_min(.05)
    return (target-mean)/std, mean, std


def base_loss(y, yp, method, reg):
    align = (y-yp).square().mean()
    both = torch.cat([y, yp])
    if method == 'sigreg':
        return .9*align + .1*reg(both)
    if method == 'vicreg':
        variance = .5*(F.relu(1-torch.sqrt(y.var()+1e-4)) +
                        F.relu(1-torch.sqrt(yp.var()+1e-4)))
        return 25*align + 25*variance
    if method == 'quantile':
        n = len(both)
        u = (torch.arange(n, device=both.device, dtype=both.dtype)+.5)/n
        normal = np.sqrt(2)*torch.erfinv(2*u-1)
        return .9*align + .1*n*(both.sort().values-normal).square().mean()
    if method == 'infonce':
        # A scalar distance score, not a one-dimensional cosine score.
        scale = both.std(unbiased=False).clamp_min(.05)
        yy, zz = y[:128]/scale, yp[:128]/scale
        logits = -(yy[:, None]-zz[None, :]).square()/(2*.1)
        labels = torch.arange(len(yy))
        return .5*(F.cross_entropy(logits, labels)+F.cross_entropy(logits.T, labels))
    raise ValueError(method)


def collision_loss(y, yp, action, future):
    n = min(128, len(y))
    z = y[:n]/torch.cat([y, yp]).std(unbiased=False).detach().clamp_min(.05)
    distance = (z[:, None]-z[None, :]).abs()
    same = action[:n, None] == action[None, :n]
    same.fill_diagonal_(False)
    search = distance.detach().masked_fill(~same, float('inf'))
    values, ids = search.topk(8, dim=1, largest=False)
    valid = torch.isfinite(values)
    disparity = (future[:n, None, :]-future[ids]).square().mean(-1).sqrt().clamp_max(2)
    gap = distance.gather(1, ids)
    hinge = F.relu(.5*disparity-gap).square()
    return (hinge*valid).sum()/valid.sum().clamp_min(1)


def geometry(model, c, points=2049):
    # Evaluation only: labels/order/simulator derivatives never enter training.
    q = torch.linspace(0, np.log(16), points, dtype=torch.float64, requires_grad=True)
    m = copy.deepcopy(model).double()
    y = m(B.sensor(q.exp(), c))
    slope = torch.autograd.grad(y.sum(), q)[0]
    qq, yy, ss = [v.detach().numpy() for v in (q, y, slope)]
    ids = np.flatnonzero(ss[:-1]*ss[1:] < 0)
    roots = np.exp(qq[ids]-ss[ids]*np.diff(qq)[ids]/np.diff(ss)[ids]).tolist()
    diff = np.diff(yy)
    up, down = np.maximum(diff, 0).sum(), np.maximum(-diff, 0).sum()
    backward = float(min(up, down)/max(up+down, 1e-12))
    return dict(q=qq, y=yy, slope=ss, roots=roots, folded=bool(roots),
                backward_fraction=backward, material_fold=bool(roots) and backward>.001,
                std=float(yy.std()), affine_r2=float(np.corrcoef(qq, yy)[0, 1]**2))


def fit(seed, method, coefficient, steps=STEPS, evaluation=True):
    key = f'{seed}-{method}-c{coefficient:g}-n{steps}'
    path = OUT/'runs'/f'{key}.pt'
    if path.exists():
        return
    c, train, test, source_hash = source(seed)
    torch.manual_seed(40000+seed)
    model = B.Encoder()
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    reg = B.SIGReg1D()
    gen = torch.Generator().manual_seed(2100000+seed)
    plan = torch.randint(len(train['x']), (steps, 512), generator=gen)
    future, mean, std = standardized_successors(train)
    frames = [dict(step=0, **geometry(model, c))] if evaluation else []
    partial = OUT/'partials'/f'{key}.pt'
    start = 0
    if partial.exists():
        prior = trials.load(partial)
        assert prior['source_hash'] == source_hash and prior['plan_hash'] == sha(plan)
        model.load_state_dict(prior['model'])
        opt.load_state_dict(prior['optimizer'])
        frames = prior['frames']
        start = prior['step']
    started = time.perf_counter()
    for step in range(start, steps):
        phase = max(0, (step-steps/2)/max(1, steps/2-1))
        opt.param_groups[0]['lr'] = .001*.5*(1+np.cos(np.pi*phase))
        ix = plan[step]
        out = model(torch.cat([train['x'][ix], train['xp'][ix]]))
        y, yp = out[:512], out[512:]
        base = base_loss(y, yp, method, reg)
        collision = collision_loss(y, yp, train['action'][ix], future[ix]) if coefficient else out.sum()*0
        loss = base+coefficient*collision
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if evaluation and (step+1 in FRAMES or step+1 == steps):
            frames.append(dict(step=step+1, base_loss=float(base.detach()),
                collision_loss=float(collision.detach()), **geometry(model, c)))
        if (step+1)%1000 == 0:
            trials.save_torch(partial, dict(step=step+1, model=model.state_dict(), optimizer=opt.state_dict(),
                frames=frames, source_hash=source_hash, plan_hash=sha(plan)))
    saved = dict(seed=seed, method=method, coefficient=coefficient, steps=steps,
        model=model.state_dict(), initial=initial, source_hash=source_hash,
        plan_hash=sha(plan), train_keys=sorted(train), frames=frames,
        training_seconds=time.perf_counter()-started, config=vars(c))
    if evaluation:
        saved['geometry'] = geometry(model, c, points=10001)
        saved['prediction'] = downstream(model, train, test, c, seed)
    trials.save_torch(path, saved)
    if evaluation:
        print(key, 'fold', saved['geometry']['folded'], 'backward', round(saved['geometry']['backward_fraction'], 4),
              'heldout', round(saved['prediction']['native_mse'], 4), 'seconds', round(saved['training_seconds'], 1), flush=True)
    else:
        print(key, 'seconds', round(saved['training_seconds'], 2), flush=True)


def common_data(c):
    generator = torch.Generator().manual_seed(80261115)
    d = 1+15*torch.rand(8192, generator=generator)
    a = torch.randint(3, (8192,), generator=generator)
    return dict(x=B.sensor(d,c), xp=B.sensor(B.move(d,a,c),c), action=a)


def downstream(model, train, test, c, seed, raw=False):
    # Fresh prediction network; encoder frozen and no gradients through labels.
    _, mean, std = standardized_successors(train)
    common = common_data(c)
    with torch.no_grad():
        code = model(train['x'])
        center, scale = code.mean(), code.std(unbiased=False).clamp_min(.05)
        def features(data):
            x = torch.tanh(4*data['x']) if raw else ((model(data['x'])-center)/scale)[:,None]
            return torch.cat([x, F.one_hot(data['action'],3).float()],1)
        xx = features(train)
        target = (torch.tanh(4*train['xp'])-mean)/std
        tx, cx = features(test), features(common)
        ty = (torch.tanh(4*test['xp'])-mean)/std
        cy = (torch.tanh(4*common['xp'])-mean)/std
    torch.manual_seed(5100000+seed)
    decoder = nn.Sequential(nn.Linear(xx.shape[1],64), nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,6))
    opt = torch.optim.Adam(decoder.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(5200000+seed)
    for _ in range(1500):
        ix = torch.randint(len(xx),(256,), generator=gen)
        loss = F.mse_loss(decoder(xx[ix]),target[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        result = dict(native_mse=float(F.mse_loss(decoder(tx),ty)),
            common_mse=float(F.mse_loss(decoder(cx),cy)),
            train_mse=float(F.mse_loss(decoder(xx),target)))
        means = torch.stack([target[train['action']==a].mean(0) for a in range(3)])
        result.update(action_native_mse=float(F.mse_loss(means[test['action']],ty)),
            action_common_mse=float(F.mse_loss(means[common['action']],cy)))
    result['decoder'] = decoder.state_dict()
    return result


def checks():
    torch.manual_seed(177)
    reg = B.SIGReg1D().double()
    for method in METHODS:
        y = torch.randn(16,dtype=torch.float64,requires_grad=True)
        z = torch.randn(16,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(lambda a,b:base_loss(a,b,method,reg),(y,z))
    # VICReg variance is per branch, matching its official unbiased variance.
    y=torch.linspace(-2,2,512);z=y*.8
    expected=25*(y-z).square().mean()+12.5*(F.relu(1-torch.sqrt(y.var()+1e-4))+F.relu(1-torch.sqrt(z.var()+1e-4)))
    torch.testing.assert_close(base_loss(y,z,'vicreg',reg),expected)
    # Collision loss may use no order; flipping all codes preserves its value.
    a=torch.arange(128)%3;f=torch.randn(128,6);y=torch.randn(128,requires_grad=True);z=torch.randn(128)
    torch.testing.assert_close(collision_loss(y,z,a,f),collision_loss(-y,-z,a,f))
    loss=collision_loss(y,z,a,f);loss.backward();assert torch.isfinite(y.grad).all() and y.grad.abs().sum()>0
    # Controlled gradient sign: an isolated conflicting pair is pushed apart.
    y=torch.linspace(-.5,.5,128,requires_grad=True);z=y.detach().clone()
    a=torch.arange(128);a[1]=a[0]
    future=torch.zeros(128,6);future[1]=2
    grad=torch.autograd.grad(collision_loss(y,z,a,future),y)[0]
    assert grad[0]>0 and grad[1]<0
    OUT.mkdir(exist_ok=True)
    trials.atomic_json(OUT/'formula-checks.json',dict(four_loss_gradchecks=True,
        vicreg_branch_variance=True, collision_global_sign_invariance=True,
        collision_pair_separation=True, training_keys=['action','x','xp']))
    print('Loss gradient and collision-direction checks passed.',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.check:
        checks()
    elif args.benchmark:
        fit(130,'infonce',1,steps=100,evaluation=False)
    else:
        jobs=[(s,m,c) for s in range(130,140) for m in METHODS for c in (0.,1.)]
        for i,job in enumerate(jobs):
            if i%args.shards==args.shard:
                fit(*job)
