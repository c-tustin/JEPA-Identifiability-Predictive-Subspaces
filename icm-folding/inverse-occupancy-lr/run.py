"""Paired historical-source inverse-occupancy learning-rate follow-up."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/occupancy-lr-mpl')
from pathlib import Path
import sys, json, hashlib, time
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'occupancy-study'
sys.path.insert(0, str(SOURCE.resolve()))
import importlib.util
spec = importlib.util.spec_from_file_location('occupancy_reference', SOURCE.resolve()/'run_study.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
import numpy as np
import torch
OUT = HERE / 'output'
SEEDS = (130, 131, 136)

def fit(seed, method, factor=1., compensate=False):
    name = f'{seed}-{method}-lr{factor:g}' + ('-normmatched' if compensate else '')
    path = OUT / f'{name}.pt'
    if path.exists(): return
    old = base.old
    settings = json.loads((base.MECH/'output/settings.json').read_text())
    c = old.trials.behavior.Config(**settings['config'])
    source = old.ORIGINAL/'sources'/f'icm-{seed}.pt'
    data = old.trials.load(source)['data']
    with old.encoder_gain(4): model, plan = old.trials.study.make_plan(c, seed, seed, 'icm')
    gen = torch.Generator().manual_seed(1900000+seed)
    indices = torch.randint(len(data['x']), plan['indices'].shape, generator=gen)
    endpoints = torch.cat([data['x'], data['xp']])
    bins = (old.trials.distances(endpoints).log()/np.log(16)*32).long().clamp(0,31)
    mass = torch.bincount(bins, minlength=32).float()/len(bins)
    inverse = 1/mass[bins]
    reg = old.trials.behavior.SIGReg1D()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    params = list(model.parameters())
    ratios=[]; frames=[]
    for step in range(c.jepa_steps):
        ix=plan['indices'][step]
        y=model(torch.cat([data['x'][ix],data['xp'][ix]]))
        align=(y[:c.batch]-y[c.batch:]).square().mean()
        ri=indices[step]; ri=torch.cat([ri,ri+len(data['x'])])
        ry=model(endpoints[ri]); weights=inverse[ri]
        if compensate:
            ordinary=.9*align+.1*reg(ry)
            reference=torch.autograd.grad(ordinary,params,retain_graph=True)
            reference_norm=torch.stack([g.square().sum() for g in reference]).sum().sqrt()
        loss=.9*align+.1*base.penalty(ry,weights,method,reg)
        opt.zero_grad(set_to_none=True); loss.backward()
        norm=torch.stack([p.grad.square().sum() for p in params]).sum().sqrt()
        ratio=float(reference_norm/norm.clamp_min(1e-30)) if compensate else 1.
        assert np.isfinite(ratio) and torch.isfinite(loss)
        lr=old.trials.study.controls.learning_rate(step,c.jepa_steps)*factor*ratio
        opt.param_groups[0]['lr']=lr; opt.step(); ratios.append(ratio)
        if (step+1)%25==0:
            q,z,s=base.curve(model,c)
            frames.append(dict(step=step+1,y=z,slope=s,roots=base.roots(q,s)))
    q,z,s=base.curve(model,c)
    metrics=dict(folded=bool(base.roots(q,s)),roots=base.roots(q,s),r2=float(np.corrcoef(q,z)[0,1]**2))
    old.trials.save_torch(path,dict(seed=seed,method=method,factor=factor,compensate=compensate,
        encoder=model.state_dict(),config=vars(c),q=q,frames=frames,metrics=metrics,
        lr_ratios=ratios,final_optimizer=opt.state_dict(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(name,metrics,flush=True)

def main():
    torch.set_num_threads(1); OUT.mkdir(exist_ok=True)
    for seed in SEEDS:
        for method in ('inverse_gradient','weighted_moments'):
            for factor in (.25,4.): fit(seed,method,factor)
        fit(seed,'inverse_gradient',compensate=True)
if __name__=='__main__': main()
