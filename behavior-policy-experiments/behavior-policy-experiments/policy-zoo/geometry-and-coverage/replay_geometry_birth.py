"""Find exact first fold updates and replay all preselected detailed fits."""
import copy
import json
import multiprocessing as mp

import numpy as np
import torch
import common_gc as study
import run_geometry as runner


def geometry_flag(model):
    q=np.linspace(0,np.log(16),1025)
    with torch.no_grad():
        y=model(study.behavior.sensor(torch.tensor(np.exp(q),dtype=torch.float32),study.original.config())).numpy()
    sign=1 if np.cov(q,y)[0,1]>=0 else -1
    normalized=sign*(y-y.mean())/max(1e-8,y.std())
    return bool(np.any(np.diff(normalized)<-1e-6))


def maximum_error(a,b):
    if isinstance(a,torch.Tensor):return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max([maximum_error(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max([maximum_error(x,y) for x,y in zip(a,b)] or [0.])
    assert a==b
    return 0.


def advance(model,optimizer,moments,data,pool,ia,ir,step,c):
    optimizer.param_groups[0]['lr']=study.controls.learning_rate(step,c.jepa_steps)
    regularizer=study.behavior.SIGReg1D()
    y=model(torch.cat([data['x'][ia[step]],data['xp'][ia[step]]]))
    a=.9*(y[:c.batch]-y[c.batch:]).square().mean()
    r=.1*regularizer(model(pool[ir[step]]))
    named=list(model.named_parameters());parameters=[p for _,p in named]
    ga=torch.autograd.grad(a,parameters);gr=torch.autograd.grad(r,parameters)
    optimizer.zero_grad(set_to_none=True)
    beta=optimizer.param_groups[0]['betas'][0]
    for (name,p),g1,g2 in zip(named,ga,gr):
        p.grad=g1+g2
        moments['carry'][name].mul_(beta)
        moments['alignment'][name].mul_(beta).add_(g1,alpha=1-beta)
        moments['regularizer'][name].mul_(beta).add_(g2,alpha=1-beta)
    optimizer.step()
    return {n:g.detach() for (n,_),g in zip(named,ga)},{n:g.detach() for (n,_),g in zip(named,gr)}


def initial_moments(model,optimizer):
    result={key:{} for key in ('carry','alignment','regularizer')}
    for name,p in model.named_parameters():
        result['carry'][name]=optimizer.state[p]['exp_avg'].clone()
        result['alignment'][name]=torch.zeros_like(p)
        result['regularizer'][name]=torch.zeros_like(p)
    return result


def job(task):
    torch.set_num_threads(1)
    _,_,alignment,regularization,seed=task
    name,c,data,pool,model,optimizer,ia,ir,_=runner.load_task(task)
    path=study.OUT/'birth_replays'/f'{name}_seed{seed}.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['signature']==study.signature()
        return old
    moments=initial_moments(model,optimizer)
    first=None;transitions=[];previous=geometry_flag(model)
    assert not previous
    for step in range(c.jepa_steps):
        advance(model,optimizer,moments,data,pool,ia,ir,step,c)
        folded=geometry_flag(model)
        if folded!=previous:transitions.append({'updates':step+1,'folded':folded})
        if folded and first is None:first=step
        previous=folded
    expected=torch.load(study.OUT/f'encoder_{name}_seed{seed}.pt',map_location='cpu')
    model_error=maximum_error(model.state_dict(),expected['encoder'])
    optimizer_error=maximum_error(optimizer.state_dict(),expected['optimizer'])
    assert model_error==optimizer_error==0
    result={'signature':study.signature(),'condition':name,'seed':seed,'first_fold_update':None if first is None else first+1,
            'fold_transitions':transitions,'final_parameter_error':model_error,'final_optimizer_error':optimizer_error}
    if first is not None:
        snap_index=max(index for index in runner.SNAPSHOTS if index<=first)
        snapshot=torch.load(study.OUT/'snapshots'/f'{name}_seed{seed}_update{snap_index}.pt',map_location='cpu')
        model=study.behavior.Encoder();model.load_state_dict(snapshot['before'])
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(snapshot['optimizer_before'])
        moments=copy.deepcopy(snapshot['moments_before'])
        for step in range(snap_index,first+1):
            if step==first:
                before={n:p.detach().clone() for n,p in model.named_parameters()}
                optimizer_before=copy.deepcopy(optimizer.state_dict())
            ga,gr=advance(model,optimizer,moments,data,pool,ia,ir,step,c)
        attribution=runner.attribute(model,optimizer,before,ga,gr,moments,study.controls.learning_rate(first,c.jepa_steps))
        flags={'actual_after':geometry_flag(model)}
        model.load_state_dict(before);flags['before']=geometry_flag(model)
        directions=attribution['directions']
        for key,include in [('regularizer_only',['regularizer']),('without_regularizer',['carry','alignment']),
                            ('without_alignment',['carry','regularizer']),('carry_only',['carry'])]:
            state={n:before[n]+sum(directions[k][n] for k in include) for n in before}
            model.load_state_dict(state);flags[key]=geometry_flag(model)
        assert not flags['before'] and flags['actual_after']
        result['birth_component_fold_flags']=flags
        result['birth_parameter_reconstruction_error']=attribution['parameter_reconstruction_error']
        result['birth_moment_reconstruction_error']=attribution['moment_reconstruction_error']
        study.atomic_torch(study.OUT/'birth_replays'/f'{name}_seed{seed}.pt',{
            'signature':study.signature(),'condition':name,'seed':seed,'update':first,
            'before':before,'optimizer_before':optimizer_before,
            'attribution':attribution,'fold_flags':flags})
    study.save_json(path,result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    tasks=[('factor','icm',a,r,s) for s in study.FACTOR_SEEDS[:5] for a in ('early','late') for r in ('early','late')]
    with mp.get_context('spawn').Pool(4) as pool:
        results=list(pool.imap_unordered(job,tasks))
    study.save_json(study.OUT/'BIRTH_REPLAY_SUMMARY.json',{'signature':study.signature(),'rows':results,
        'scope':'All four cells of five preselected seeds; exact dense-grid fold onset and full-training replays. Not a significance-tested population attribution.'})
