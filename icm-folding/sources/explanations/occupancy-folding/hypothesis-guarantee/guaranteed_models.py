"""Two globally ordered height splines; train by ordinary or projected gradients."""
import argparse
import copy
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'matched-concentration'))
from run_comparison import old,dataset,target_measure,penalty,metrics,OUT as COMPARISON,key
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT=HERE/'output';MARGIN=.05


def pava(values):
    means=[];counts=[]
    for value in values:
        means.append(float(value));counts.append(1)
        while len(means)>1 and means[-2]>means[-1]:
            n=counts[-2]+counts[-1]
            mean=(means[-2]*counts[-2]+means[-1]*counts[-1])/n
            means[-2:]=[mean];counts[-2:]=[n]
    return np.repeat(means,counts)


class BaseSpline(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('knots',torch.logspace(np.log10(1/16),0,65,dtype=torch.float64))
    def forward(self,x):
        s=x[:,1].double().contiguous()
        i=torch.searchsorted(self.knots,s).sub(1).clamp(0,len(self.knots)-2)
        z=self.values();fraction=(s-self.knots[i])/(self.knots[i+1]-self.knots[i])
        return z[i]+fraction*(z[i+1]-z[i])
    def certificate(self):
        with torch.no_grad():
            slopes=torch.diff(self.values())/torch.diff(self.knots)
        return dict(min_slope=float(slopes.min()),max_slope=float(slopes.max()),
            minimum_increment=float(torch.diff(self.values()).min()))


class PositiveSpline(BaseSpline):
    def __init__(self):
        super().__init__();slope=2/(1-1/16)
        self.offset=nn.Parameter(torch.tensor(-1.,dtype=torch.float64))
        self.raw=nn.Parameter(torch.full((64,),np.log(np.expm1(slope-MARGIN)),dtype=torch.float64))
    def values(self):
        increments=(MARGIN+F.softplus(self.raw))*torch.diff(self.knots)
        return self.offset+torch.cat([torch.zeros(1,dtype=self.knots.dtype),increments.cumsum(0)])
    def project(self):return 0.


class ProjectedSpline(BaseSpline):
    def __init__(self):
        super().__init__();self.z=nn.Parameter(-1+2*(self.knots-self.knots[0])/(self.knots[-1]-self.knots[0]))
    def values(self):return self.z
    def project(self):
        with torch.no_grad():
            old_values=self.z.clone()
            transformed=(self.z-MARGIN*self.knots).numpy()
            self.z.copy_(torch.from_numpy(pava(transformed))+MARGIN*self.knots)
            return float((self.z-old_values).norm())


def cf(y,w,reg,scale=1024):
    phase=y[:,None]*reg.frequency;v=w/w.sum()
    C=(v[:,None]*phase.cos()).sum(0);S=(v[:,None]*phase.sin()).sum(0)
    return scale*(reg.weight*((C-reg.target).square()+S.square())).sum()


def evaluation(model,ds,target,seed):
    data=ds['data'];e=ds['endpoints'].double();reg=old.trials.behavior.SIGReg1D().double()
    ratio=np.zeros_like(target);mask=ds['p']>0;ratio[mask]=target[mask]/ds['p'][mask]
    w=torch.from_numpy(ratio[ds['bins'].numpy()])
    with torch.no_grad():
        y=model(e);n=len(data['x'])
        alignment=float((y[:n]-y[n:]).square().mean())
        population_reg=float(cf(y,w,reg))
        gen=torch.Generator().manual_seed(10000000+seed);values=[];regs=[];aligns=[]
        for _ in range(16):
            ix=torch.randint(n,(512,),generator=gen);ri=torch.cat([ix,ix+n])
            z=y[ri];a=(z[:512]-z[512:]).square().mean();r=cf(z,w[ri],reg)
            values.append(float(.9*a+.1*r));regs.append(float(r));aligns.append(float(a))
    q=torch.linspace(0,np.log(16),10001,dtype=torch.float64,requires_grad=True)
    yy=model(old.trials.behavior.sensor(q.exp(),ds['c']))
    ss=torch.autograd.grad(yy.sum(),q)[0]
    return dict(population_alignment=alignment,population_sigreg=population_reg,
        population_objective=.9*alignment+.1*population_reg,
        batch_objective_mean=float(np.mean(values)),batch_objective_sd=float(np.std(values,ddof=1)),
        batch_sigreg_mean=float(np.mean(regs)),batch_alignment_mean=float(np.mean(aligns)),
        q=q.detach().numpy(),y=yy.detach().numpy(),slope=ss.detach().numpy(),
        geometry=metrics(q.detach().numpy(),yy.detach().numpy(),ss.detach().numpy()))


def name(seed,method,level,tail):
    return f'{seed}-{method}-native' if level is None else f'{seed}-{method}-a{int(level*10000)}-{tail}'


def fit(seed,method,level,tail):
    path=OUT/'guaranteed'/f'{name(seed,method,level,tail)}.pt'
    if path.exists():return
    started=time.perf_counter();ds=dataset(seed);c=ds['c'];data=ds['data']
    _,plan=old.trials.study.make_plan(c,seed,seed,'icm')
    target=ds['p'] if level is None else target_measure(ds,level,tail)
    ratio=np.zeros_like(target);mask=ds['p']>0;ratio[mask]=target[mask]/ds['p'][mask]
    w=torch.from_numpy(ratio[ds['bins'].numpy()])
    model=PositiveSpline() if method=='positive' else ProjectedSpline()
    initial=copy.deepcopy(model.state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=.001);reg=old.trials.behavior.SIGReg1D().double()
    rg=torch.Generator().manual_seed(1900000+seed);reg_plan=torch.randint(len(data['x']),plan['indices'].shape,generator=rg)
    min_slope=model.certificate()['min_slope'];history=[];max_projection=0.
    for step in range(4000):
        opt.param_groups[0]['lr']=old.trials.study.controls.learning_rate(step,4000)
        ix=plan['indices'][step];ri=torch.cat([reg_plan[step],reg_plan[step]+len(data['x'])])
        y=model(torch.cat([data['x'][ix],data['xp'][ix]]));a=(y[:512]-y[512:]).square().mean()
        r=cf(model(ds['endpoints'][ri]),w[ri],reg);loss=.9*a+.1*r
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();max_projection=max(max_projection,model.project())
        certificate=model.certificate();min_slope=min(min_slope,certificate['min_slope'])
        assert certificate['min_slope']>=MARGIN-1e-10 and certificate['minimum_increment']>0
        if (step+1)%100==0:history.append(dict(step=step+1,alignment=float(a.detach()),sigreg=float(r.detach()),**certificate))
    result=evaluation(model,ds,target,seed)
    assert not result['geometry']['folded'] and np.all(result['slope']<0)
    old.trials.save_torch(path,dict(seed=seed,method=method,level=level,tail=tail,initial=initial,
        model=model.state_dict(),evaluation=result,certificate=model.certificate(),
        minimum_training_slope=min_slope,max_projection=max_projection,history=history,
        seconds=time.perf_counter()-started))
    print(path.stem,'objective',round(result['batch_objective_mean'],5),'R2',round(result['geometry']['r2'],4),
          'min_slope',round(min_slope,5),'seconds',round(time.perf_counter()-started,1),flush=True)


def audit():
    rng=np.random.default_rng(42)
    for _ in range(100):
        x=rng.normal(size=65);z=pava(x);dual=np.cumsum(x-z)[:-1]
        assert np.all(np.diff(z)>=-1e-12) and np.min(dual)>=-1e-10
        assert abs(np.sum(z-x))<1e-10 and np.max(abs(dual*np.diff(z)))<1e-10
    a,b=PositiveSpline(),ProjectedSpline()
    torch.testing.assert_close(a.values(),b.values(),rtol=0,atol=1e-12)
    b.z.data.copy_(torch.from_numpy(rng.normal(size=65)));b.project()
    assert b.certificate()['min_slope']>=MARGIN-1e-10
    old.trials.atomic_json(OUT/'spline-audit.json',dict(projection_KKT_cases=100,same_initial_function=True,positive_margin=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.audit_only:audit()
    else:
        jobs=[(s,m,a,t) for s in range(130,140) for m in ('positive','projected') for a,t in [(None,None),(.875,'one'),(.875,'two'),(.9375,'one'),(.9375,'two')]]
        for i,job in enumerate(jobs):
            if i%args.shards==args.shard:fit(*job)
