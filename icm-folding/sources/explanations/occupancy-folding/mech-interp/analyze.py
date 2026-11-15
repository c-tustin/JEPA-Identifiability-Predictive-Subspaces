"""Exact Adam component interventions in the original ICM trainer, seed 131."""
import copy
import inspect
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'gradient-followup'))
from train_gradients import old,source
from torch.func import functional_call
import numpy as np
import torch
from torch.nn import functional as F

OUT=HERE/'output';B=old.trials.behavior
SNAPS=(99,199,299,399,499,599)


def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'icm-adam-components-131.pt'
    if path.exists():return
    c,_,_,_=source(131)
    snapshots=[];pending={};ma={};mf={};max_grad_error=0.;max_moment_error=0.
    states={}
    def before(icm,optimizer,x,xp,action,update):
        nonlocal max_grad_error
        named=list(icm.named_parameters());ps=[p for _,p in named]
        error,logits=icm(x,xp,action)
        forward=error.mean();inverse=F.cross_entropy(logits,action)
        ga=torch.autograd.grad(.8*inverse,ps,allow_unused=True,retain_graph=True)
        gf=torch.autograd.grad(.2*forward,ps,allow_unused=True)
        ga=[torch.zeros_like(p) if g is None else g for p,g in zip(ps,ga)]
        gf=[torch.zeros_like(p) if g is None else g for p,g in zip(ps,gf)]
        norm=torch.linalg.vector_norm(torch.stack([(a+f).norm() for a,f in zip(ga,gf)]))
        clip=min(1.,40./(float(norm)+1e-6))
        for (name,p),a,f in zip(named,ga,gf):
            a=a*clip;f=f*clip
            max_grad_error=max(max_grad_error,float((a+f-p.grad).abs().max()))
            if name not in ma:ma[name]=torch.zeros_like(p);mf[name]=torch.zeros_like(p)
            ma[name].mul_(.9).add_(a,alpha=.1);mf[name].mul_(.9).add_(f,alpha=.1)
        if update in SNAPS:
            pending.clear();pending.update(before={n:p.detach().clone() for n,p in named},
                x=x.detach().clone(),xp=xp.detach().clone(),action=action.detach().clone(),
                raw_forward=float(forward.detach()),inverse_ce=float(inverse.detach()),clip=clip)
    def after(icm,optimizer,update):
        nonlocal max_moment_error
        for name,p in icm.named_parameters():
            max_moment_error=max(max_moment_error,float((ma[name]+mf[name]-optimizer.state[p]['exp_avg']).abs().max()))
        if update not in SNAPS:return
        directions={k:{} for k in ('inverse','forward')}
        lr=optimizer.param_groups[0]['lr'];t=update+1
        for name,p in icm.named_parameters():
            st=optimizer.state[p]
            denominator=(st['exp_avg_sq']/(1-.999**t)).sqrt()+1e-8
            for label,moment in (('inverse',ma),('forward',mf)):
                directions[label][name]=-lr*moment[name]/(1-.9**t)/denominator
        before=pending['before'];actual={n:p.detach().clone() for n,p in icm.named_parameters()}
        reconstruction=max(float((actual[n]-before[n]-directions['inverse'][n]-directions['forward'][n]).abs().max()) for n in before)
        model=copy.deepcopy(icm).double()
        x=pending['x'].double();xp=pending['xp'].double();action=pending['action']
        def measurement(parameters):
            pars={n:v.double() for n,v in parameters.items()}
            with torch.no_grad():
                error,logits=functional_call(model,pars,(x,xp,action))
                fp={n.removeprefix('features.'):v for n,v in pars.items() if n.startswith('features.')}
                target=functional_call(model.features,fp,(torch.tanh(4*xp),))
                variance=.5*(target-target.mean(0)).square().sum(1).mean()
            return dict(forward_error=float(error.mean()),target_variance=float(variance),
                inverse_ce=float(F.cross_entropy(logits,action)))
        probes={'before':measurement(before),'actual_update':measurement(actual)}
        for label in ('inverse','forward'):
            probes[label+'_component']=measurement({n:before[n]+directions[label][n] for n in before})
        for block in ('features.','forward_model.'):
            probes[block.rstrip('.')+'_block']=measurement({n:(actual[n] if n.startswith(block) else before[n]) for n in before})
        snapshots.append(dict(update=update+1,probes=probes,reconstruction_max_error=reconstruction,
            before=before,actual=actual,directions=directions,x=pending['x'],xp=pending['xp'],action=action))
        states[update+1]=copy.deepcopy(icm.state_dict())
        print(update+1,{k:v['forward_error']-probes['before']['forward_error'] for k,v in probes.items()},flush=True)
    trainer=inspect.getsource(B.train_policy)
    anchor='        icm_optimizer.step()'
    assert trainer.count(anchor)==1
    instrumented=trainer.replace(anchor,
        '        before(icm, icm_optimizer, x, xp, action, update)\n'+anchor+
        '\n        after(icm, icm_optimizer, update)')
    ns=dict(B.__dict__,before=before,after=after)
    exec(compile(instrumented,str(HERE/'instrumented-trainer'),'exec'),ns)
    policy,icm,history,_=ns['train_policy'](c,131)
    original=old.trials.load(old.ORIGINAL/'sources/icm-131.pt')
    maximum={label:max(float((v-original[label][n]).abs().max()) for n,v in model.state_dict().items()) for label,model in (('policy',policy),('icm',icm))}
    assert max(maximum.values())==0,maximum
    assert max(s['reconstruction_max_error'] for s in snapshots)<1e-6
    assert max_grad_error<1e-5 and max_moment_error<1e-5
    old.trials.save_torch(path,dict(seed=131,snapshots=snapshots,states=states,history=history,
        original_trainer=trainer,final_parameter_error=maximum,max_gradient_error=max_grad_error,
        max_moment_error=max_moment_error))
    old.trials.atomic_json(OUT/'component-summary.json',dict(seed=131,
        snapshots=[dict(update=s['update'],probes=s['probes'],reconstruction_max_error=s['reconstruction_max_error']) for s in snapshots],
        final_parameter_error=maximum,max_gradient_error=max_grad_error,max_moment_error=max_moment_error))


if __name__=='__main__':run()
