"""Fixed paired validation and continuation of existing folded encoders."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/icm-weighted-validation-mpl')
from pathlib import Path
import sys, importlib.util, copy, hashlib, json, argparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if (HERE.parent/'sources').exists(): ROOT=HERE.parent/'sources'
spec=importlib.util.spec_from_file_location('weighted_validation_occupancy',HERE.parent/'occupancy-study/run_study.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
old=base.old
import numpy as np
import torch
OUT=HERE/'output'
FRESH=HERE.parent/'mlp-gradient-study/output/fresh-sources'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()
def geometry(model,c,points=20001):
    m=copy.deepcopy(model).double()
    q=torch.linspace(0,np.log(16),points,dtype=torch.float64,requires_grad=True)
    y=m(old.trials.behavior.sensor(q.exp(),c)); s=torch.autograd.grad(y.sum(),q)[0]
    q,y,s=[x.detach().numpy() for x in (q,y,s)]
    turns=base.roots(q,s);diff=np.diff(y)
    backward=float(min(np.maximum(diff,0).sum(),np.maximum(-diff,0).sum())/max(np.abs(diff).sum(),1e-30))
    return dict(q=q,y=y,slope=s,roots=turns,folded=bool(turns),material_fold=bool(turns) and backward>.001,
        backward_fraction=backward,std=float(y.std()),r2=float(np.corrcoef(q,y)[0,1]**2))
def source(seed):
    path=FRESH/f'icm-{seed}.pt';saved=old.trials.load(path)
    c=old.trials.behavior.Config(**saved['config']);data=saved['data']
    endpoints=torch.cat([data['x'],data['xp']])
    bins=(old.trials.distances(endpoints).log()/np.log(16)*32).long().clamp(0,31)
    mass=torch.bincount(bins,minlength=32).float()/len(bins)
    return c,data,endpoints,1/mass[bins],sha(path),mass

def fit(seed,gain,method,stage='scratch'):
    path=OUT/f'{stage}-g{gain}-{method}-{seed}.pt'
    if path.exists():return
    c,data,endpoints,weights,source_hash,mass=source(seed)
    with old.encoder_gain(gain):model,plan=old.trials.study.make_plan(c,seed,seed,'icm')
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    initial=copy.deepcopy(model.state_dict())
    n=4000 if stage=='scratch' else 8000
    if stage=='repair':
        startpath=OUT/f'scratch-g4-ordinary-{seed}.pt';start=old.trials.load(startpath)
        assert start['final']['folded']
        model.load_state_dict(start['encoder']);initial=copy.deepcopy(model.state_dict())
        if method!='weighted_reset': opt.load_state_dict(start['optimizer'])
        ag=torch.Generator().manual_seed(2100000+seed)
        alignment_plan=torch.randint(len(data['x']),(n,c.batch),generator=ag)
        rg=torch.Generator().manual_seed(2200000+seed)
    else:
        alignment_plan=plan['indices']
        rg=torch.Generator().manual_seed(1900000+seed)
    regularizer_plan=torch.randint(len(data['x']),(n,c.batch),generator=rg)
    reg=old.trials.behavior.SIGReg1D()
    frames=[dict(step=0,**geometry(model,c,2049))]
    for step in range(n):
        opt.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,n) if stage=='scratch' else .0001
        ix=alignment_plan[step];ri=regularizer_plan[step];ri=torch.cat([ri,ri+len(data['x'])])
        y=model(torch.cat([data['x'][ix],data['xp'][ix]]))
        alignment=(y[:c.batch]-y[c.batch:]).square().mean()
        ry=model(endpoints[ri])
        regularizer=base.penalty(ry,weights[ri],'ordinary' if method=='ordinary' else 'weighted_moments',reg)
        loss=.9*alignment+.1*regularizer
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step+1 in (100,250,500,1000,2000,4000,8000):frames.append(dict(step=step+1,**geometry(model,c,2049)))
    final=geometry(model,c)
    saved=dict(seed=seed,gain=gain,method=method,stage=stage,encoder=model.state_dict(),initial=initial,
        optimizer=opt.state_dict(),config=vars(c),source_hash=source_hash,code_hash=sha(__file__),
        protocol_hash=sha(HERE/'PROTOCOL.md'),alignment_hash=tensor_sha(alignment_plan),
        regularizer_hash=tensor_sha(regularizer_plan),frames=frames,final=final,mass=mass.numpy())
    if stage=='repair':saved['start_hash']=sha(startpath)
    old.trials.save_torch(path,saved)
    print(path.stem,{k:final[k] for k in ('folded','roots','backward_fraction','std','r2')},flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gain',type=int,choices=(4,8));args=p.parse_args()
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    for gain in ((args.gain,) if args.gain else (4,8)):
        for seed in range(140,145):
            for method in ('ordinary','weighted'):fit(seed,gain,method)
        if gain==4:
            for seed in range(140,145):
                r=old.trials.load(OUT/f'scratch-g4-ordinary-{seed}.pt')
                if r['final']['folded']:
                    for method in ('ordinary','weighted','weighted_reset'):fit(seed,4,method,'repair')
if __name__=='__main__':main()
