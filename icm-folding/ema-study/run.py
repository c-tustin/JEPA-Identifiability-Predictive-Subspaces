"""Matched original, parameter-average, and stop-gradient EMA-target experiments."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/icm-ema-mpl')
from pathlib import Path
import sys,copy,json,hashlib,importlib.util,argparse
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ema_geometry_reference',HERE.parent/'weighted-moments-validation/run.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
base=reference.base;old=reference.old;geometry=reference.geometry
import numpy as np
import torch
OUT=HERE/'output'
DECAYS=(.9,.99,.999)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()
def frozen_copy(model):
    result=copy.deepcopy(model)
    result.requires_grad_(False)
    return result
@torch.no_grad()
def update_ema(target,student,beta):
    for t,s in zip(target.parameters(),student.parameters()):t.mul_(beta).add_(s,alpha=1-beta)
def target_alignment(student_values,target_values,batch):
    return (student_values[:batch]-target_values[batch:]).square().mean()+(student_values[batch:]-target_values[:batch]).square().mean()
def compact(g):return {k:v for k,v in g.items() if k not in ('q','y','slope')}
def sample(model,c):return compact(geometry(model,c,1025))
def fit(seed,mode,beta=None):
    name=f'{seed}-{mode}'+('' if beta is None else f'-b{beta:g}')
    path=OUT/f'{name}.pt'
    if path.exists():return
    source=old.ORIGINAL/'sources'/f'icm-{seed}.pt'
    saved=old.trials.load(source);c=old.trials.behavior.Config(**saved['config']);data=saved['data']
    with old.encoder_gain(4):student,plan=old.trials.study.make_plan(c,seed,seed,'icm')
    initial=copy.deepcopy(student.state_dict())
    target=frozen_copy(student) if mode=='target' else None
    averages={str(b):frozen_copy(student) for b in DECAYS} if mode=='ordinary' else {}
    opt=torch.optim.Adam(student.parameters(),lr=.001)
    reg=old.trials.behavior.SIGReg1D()
    generator=torch.Generator().manual_seed(1900000+seed)
    regularizer_plan=torch.randint(len(data['x']),plan['indices'].shape,generator=generator)
    endpoints=torch.cat([data['x'],data['xp']])
    history=[]
    def frame(step):
        item=dict(step=step,student=sample(student,c))
        if target is not None:item['target']=sample(target,c)
        if averages:item['averages']={b:sample(m,c) for b,m in averages.items()}
        history.append(item)
    frame(0)
    for step in range(c.jepa_steps):
        opt.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,c.jepa_steps)
        ix=plan['indices'][step];xx=torch.cat([data['x'][ix],data['xp'][ix]])
        y=student(xx)
        if target is None:align=(y[:c.batch]-y[c.batch:]).square().mean()
        else:
            with torch.no_grad():ty=target(xx)
            align=target_alignment(y,ty,c.batch)
        ri=regularizer_plan[step];ri=torch.cat([ri,ri+len(data['x'])])
        loss=.9*align+.1*reg(student(endpoints[ri]))
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        if target is not None:assert all(p.grad is None for p in target.parameters())
        opt.step()
        if target is not None:update_ema(target,student,beta)
        for b,m in averages.items():update_ema(m,student,float(b))
        if (step+1)%25==0:frame(step+1)
    models={'student':student}
    if target is not None:models['target']=target
    models.update({f'average_{b}':m for b,m in averages.items()})
    final={k:geometry(m,c) for k,m in models.items()}
    states={k:m.state_dict() for k,m in models.items()}
    original=old.trials.load(base.OUT/'runs'/f'native-ordinary-{seed}.pt')
    baseline_error=max(float((v-original['encoder'][k]).abs().max()) for k,v in student.state_dict().items())
    if mode=='ordinary' or beta==0:assert baseline_error==0,baseline_error
    lag=None if target is None else float(torch.stack([(s-t).square().sum() for s,t in zip(student.parameters(),target.parameters())]).sum().sqrt())
    old.trials.save_torch(path,dict(seed=seed,mode=mode,beta=beta,config=vars(c),initial=initial,states=states,
        optimizer=opt.state_dict(),history=history,final=final,source_sha256=sha(source),
        alignment_sha256=tensor_sha(plan['indices']),regularizer_sha256=tensor_sha(regularizer_plan),
        code_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.md'),baseline_parameter_error=baseline_error,target_lag=lag))
    print(name,{k:dict(folded=g['folded'],r2=round(g['r2'],4),std=round(g['std'],4)) for k,g in final.items()},flush=True)

def audit():
    torch.set_num_threads(1);torch.manual_seed(54)
    model=old.ScaledEncoder(4).double();target=frozen_copy(model)
    x=torch.randn(24,6,dtype=torch.float64)
    y=model(x);a=(y[:12]-y[12:]).square().mean()
    ga=torch.autograd.grad(a,list(model.parameters()))
    y=model(x);b=target_alignment(y,target(x),12)
    gb=torch.autograd.grad(b,list(model.parameters()))
    for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,rtol=1e-12,atol=1e-12)
    assert all(p.grad is None and not p.requires_grad for p in target.parameters())
    before=copy.deepcopy(target.state_dict())
    with torch.no_grad():
        for p in model.parameters():p.add_(.17)
    update_ema(target,model,.99)
    for k,v in target.state_dict().items():torch.testing.assert_close(v,.99*before[k]+.01*model.state_dict()[k],rtol=1e-12,atol=1e-12)
    update_ema(target,model,0.)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in target.state_dict().items())
    old.trials.atomic_json(OUT/'formula-checks.json',dict(zero_lag_gradient_matches_original=True,teacher_detached=True,ema_recurrence=True,beta_zero_exact=True))
    print('EMA formula, zero-lag gradient and target-detachment checks passed.',flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--check',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    if args.check:audit();return
    jobs=[(s,m,b) for s in range(130,140) for m,b in [('ordinary',None),('target',.99),('target',.999)]]+[(131,'target',0.)]
    for i,job in enumerate(jobs):
        if i%args.shards==args.shard:fit(*job)
if __name__=='__main__':main()
