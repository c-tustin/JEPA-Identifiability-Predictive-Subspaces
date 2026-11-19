"""Scalar projected-area observations for raw/own-code InfoNCE."""
import argparse
import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('area_infonce_reference', HERE.parent/'raw-embedding-infonce/run.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
base = reference.base
OUT = HERE/'output'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def area(x):
    v = x.reshape(-1, 3, 2)
    u, w = v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]
    return .5*(u[:, 0]*w[:, 1]-u[:, 1]*w[:, 0]).abs()[:, None]


def convert(data):
    return {'x': area(data['x']), 'xp': area(data['xp']), 'action': data['action']}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        return self.net(torch.tanh(4*x)).flatten()


def critic():
    return nn.Sequential(nn.Linear(1,48), nn.Tanh(), nn.Linear(48,48), nn.Tanh(), nn.Linear(48,1))


def geometry(model, c, points=20001):
    d = torch.exp(torch.linspace(0, np.log(16), points, dtype=torch.float64)).requires_grad_()
    m = copy.deepcopy(model).double()
    a = area(base.B.sensor(d, c))
    y = m(a)
    slope = torch.autograd.grad(y.sum(), d)[0]
    dd, aa, yy, ss = [t.detach().numpy() for t in (d, a.flatten(), y, slope)]
    ids = np.flatnonzero(ss[:-1]*ss[1:] < 0)
    roots = dd[ids]-ss[ids]*np.diff(dd)[ids]/np.diff(ss)[ids]
    increments = np.diff(yy)
    up, down = np.maximum(increments,0).sum(), np.maximum(-increments,0).sum()
    return dict(distance=dd, area=aa, code=yy, slope=ss, roots=roots.tolist(), folded=bool(len(roots)),
                backward_fraction=float(min(up,down)/max(up+down,1e-12)), std=float(yy.std()),
                log_distance_r2=float(np.corrcoef(np.log(dd), yy)[0,1]**2))


def decoder():
    return nn.Sequential(nn.Linear(4,64), nn.ELU(), nn.Linear(64,64), nn.ELU(), nn.Linear(64,1))


def prediction_features(model, train, test, c):
    common = convert(base.common_data(c))
    with torch.no_grad():
        code = model(train['x'])
        center, scale = code.mean(), code.std(unbiased=False).clamp_min(.05)
        mean, std = train['xp'].mean(0), train['xp'].std(0, unbiased=False).clamp_min(1e-6)
        def pair(data):
            x = torch.cat([((model(data['x'])-center)/scale)[:,None], F.one_hot(data['action'],3).float()],1)
            return x, (data['xp']-mean)/std
        pairs = {name: pair(data) for name,data in [('train',train),('heldout',test),('coverage',common)]}
    return pairs, dict(center=center, scale=scale, mean=mean, std=std), common


def downstream(model, train, test, c, seed):
    pairs, stats, common = prediction_features(model, train, test, c)
    torch.manual_seed(5100000+seed)
    dec = decoder()
    opt = torch.optim.Adam(dec.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(5200000+seed)
    history=[]
    def evaluate(step):
        with torch.no_grad():
            history.append(dict(step=step, **{name: F.mse_loss(dec(x),y).item() for name,(x,y) in pairs.items()}))
    evaluate(0)
    for step in range(1500):
        ix = torch.randint(len(train['x']), (256,), generator=gen)
        x,y = pairs['train']
        loss = F.mse_loss(dec(x[ix]), y[ix])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step+1)%100==0: evaluate(step+1)
    means = torch.stack([pairs['train'][1][train['action']==a].mean(0) for a in range(3)])
    baseline = {name: F.mse_loss(means[data['action']], pairs[name][1]).item()
                for name,data in [('heldout',test),('coverage',common)]}
    return dict(model=dec.state_dict(), stats=stats, history=history, action_only=baseline)


def setup(seed, steps):
    c, raw_train, raw_test, source_hash = base.source(seed)
    train,test = convert(raw_train),convert(raw_test)
    torch.manual_seed(40000+seed); model=Encoder()
    torch.manual_seed(6100000+seed); scorer=critic()
    gen=torch.Generator().manual_seed(2100000+seed)
    plan=torch.randint(len(train['x']), (steps,512), generator=gen)
    endpoints=torch.cat([train['x'],train['xp']])
    mean,std=endpoints.mean(0),endpoints.std(0,unbiased=False).clamp_min(1e-6)
    return c,train,test,source_hash,model,scorer,plan,mean,std


def fit(seed, steps=4000):
    OUT.mkdir(exist_ok=True)
    path=OUT/f'{seed}-n{steps}.pt'
    if path.exists(): return
    started=time.monotonic()
    c,train,test,source_hash,model,scorer,plan,mean,std=setup(seed,steps)
    initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.Adam(list(model.parameters())+list(scorer.parameters()),lr=.001)
    fixed={}
    for j,(name,data) in enumerate([('train',train),('heldout',test)]):
        gen=torch.Generator().manual_seed(7400000+10*seed+j)
        ix=torch.randperm(len(data['x']),generator=gen)[:128]
        fixed[name]=torch.cat([data['x'][ix],data['xp'][ix]])
    history=[]; losses=[]
    def frame(step):
        with torch.no_grad():
            scores={name: reference.objective((x-mean)/std,model(x),scorer).item() for name,x in fixed.items()}
        history.append(dict(step=step, infonce=scores, geometry=geometry(model,c,2049)))
    frame(0)
    for step,ix in enumerate(plan):
        phase=max(0,(step-steps/2)/max(1,steps/2-1))
        opt.param_groups[0]['lr']=.001*.5*(1+np.cos(np.pi*phase))
        x=torch.cat([train['x'][ix[:128]], train['xp'][ix[:128]]])
        loss=reference.objective((x-mean)/std,model(x),scorer)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step+1)%100==0 or step+1==steps: frame(step+1)
    g=geometry(model,c)
    pred=downstream(model,train,test,c,seed)
    record=dict(seed=seed, steps=steps, config=vars(c), source_hash=source_hash,
                plan_hash=base.sha(plan), initial=initial, model=model.state_dict(), critic=scorer.state_dict(),
                raw_mean=mean, raw_std=std, optimizer=opt.state_dict(), history=history, losses=losses,
                geometry=g, prediction=pred, code_hash=sha(__file__), protocol_hash=sha(HERE/'PROTOCOL.md'),
                seconds=time.monotonic()-started)
    base.trials.save_torch(path,record)
    print(seed, 'folded',g['folded'],'heldout',pred['history'][-1]['heldout'], 'seconds',round(record['seconds'],1),flush=True)


def checks():
    torch.manual_seed(54)
    vertices=torch.randn(11,3,2,dtype=torch.float64)
    x,y=vertices[:,:,0],vertices[:,:,1]
    expected=.5*(x*y.roll(-1,1)-y*x.roll(-1,1)).sum(1).abs()
    torch.testing.assert_close(area(vertices).flatten(),expected)
    theta=torch.tensor(.7,dtype=torch.float64)
    rotation=torch.stack([torch.stack([theta.cos(),-theta.sin()]),torch.stack([theta.sin(),theta.cos()])])
    torch.testing.assert_close(area(vertices@rotation),area(vertices))
    torch.testing.assert_close(area(vertices+torch.tensor([4.,-3.])),area(vertices))
    torch.testing.assert_close(area(3*vertices),9*area(vertices))
    c=base.B.Config(); d=torch.linspace(1,16,100,dtype=torch.float64)
    a=area(base.B.sensor(d,c)).flatten()
    torch.testing.assert_close(a,a[0]/d.square())
    assert bool((a[1:]<a[:-1]).all())
    raw=torch.randn(7,1,dtype=torch.float64,requires_grad=True)
    z=torch.randn(7,dtype=torch.float64,requires_grad=True); scorer=critic().double()
    score=reference.logits(raw,z,scorer)
    explicit=torch.stack([torch.stack([-(scorer(zz[None])-xx).square().mean()/.2 for xx in raw.detach()]) for zz in z])
    torch.testing.assert_close(score,explicit)
    assert torch.autograd.gradcheck(lambda zz: reference.objective(raw,zz,scorer),(z,))
    reference.objective(raw,z,scorer).backward()
    assert raw.grad is None and z.grad.abs().sum()>0
    model=Encoder(); xx=torch.rand(7,1)
    reference.objective(xx,model(xx),critic()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    OUT.mkdir(exist_ok=True)
    base.trials.atomic_json(OUT/'formula-checks.json',dict(area_shoelace=True,translation=True,rotation=True,
        quadratic_scaling=True,inverse_square_distance=True,strictly_decreasing_area=True,
        scalar_input_output=True,explicit_logits=True,finite_difference_gradient=True,raw_detached=True,encoder_gradient=True))
    print('Scalar area and InfoNCE checks passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(130,145)))
    parser.add_argument('--steps',type=int,default=4000);args=parser.parse_args()
    torch.set_num_threads(1)
    if args.check: checks()
    else:
        for seed in args.seeds: fit(seed,args.steps)
