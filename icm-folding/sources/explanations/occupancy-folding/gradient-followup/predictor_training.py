"""Extend exact ICM training; isolate forward fitting with frozen targets."""
import copy
import inspect
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
from train_gradients import OUT, source, old
import numpy as np
import torch
from torch.nn import functional as F
B=old.trials.behavior


@torch.no_grad()
def evaluate(icm,data):
    error,logits=icm(data['x'],data['xp'],data['action'])
    target=icm.features(torch.tanh(4*data['xp']))
    variance=.5*(target-target.mean(0)).square().sum(1).mean()
    return dict(error=float(error.mean()),normalized_error=float(error.mean()/variance.clamp_min(1e-12)),
        target_variance=float(variance),target_rms=float(target.square().mean().sqrt()),
        inverse_accuracy=float((logits.argmax(1)==data['action']).float().mean()))


def fixed_grid(c):
    d=torch.exp(torch.linspace(0,np.log(16),1024)).repeat_interleave(3)
    actions=torch.arange(3).repeat(1024)
    return dict(x=B.sensor(d,c),xp=B.sensor(B.move(d,actions,c),c),action=actions)


def joint(seed):
    path=OUT/f'joint-predictor-{seed}.pt'
    if path.exists():return
    c,_,_,_=source(seed);c.policy_updates=1800
    frames=[];gate={};states={}
    grid=fixed_grid(c)
    def capture(policy,icm,c,seed,update):
        with torch.random.fork_rng():
            heldout=B.collect(c,'icm',64,5100000+seed,policy)
            frame=dict(update=update+1,on_policy=evaluate(icm,heldout),fixed_grid=evaluate(icm,grid),
                fraction_below4=float((old.trials.distances(heldout['x'])<4).float().mean()))
            frames.append(frame)
            if update in (599,1199,1799):
                states[update+1]=dict(policy=copy.deepcopy(policy.state_dict()),icm=copy.deepcopy(icm.state_dict()))
            if update==599:
                original=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
                for label,model in (('policy',policy),('icm',icm)):
                    err=max(float((v-original[label][k]).abs().max()) for k,v in model.state_dict().items())
                    assert err==0,(seed,label,err)
                    gate[label]=err
            print(f'joint seed={seed} update={update+1} onpolicy={frame["on_policy"]["error"]:.6g} fixed={frame["fixed_grid"]["error"]:.6g}',flush=True)
    original=inspect.getsource(B.train_policy)
    anchor='    for update in range(c.policy_updates):'
    assert original.count(anchor)==1
    instrumented=original.replace(anchor,'    capture(policy, icm, c, seed, -1)\n'+anchor)
    anchor='        if update % 25 == 0 or update == c.policy_updates - 1:'
    assert instrumented.count(anchor)==1
    instrumented=instrumented.replace(anchor,'        if (update+1) % 100 == 0:\n            capture(policy, icm, c, seed, update)\n'+anchor)
    namespace=dict(B.__dict__,capture=capture)
    exec(compile(instrumented,str(HERE/'instrumented-joint-training'),'exec'),namespace)
    policy,icm,history,_=namespace['train_policy'](c,seed)
    old.trials.save_torch(path,dict(seed=seed,config=vars(c),frames=frames,history=history,
        states=states,update600_exact=gate,original_trainer=original))


def frozen(seed):
    path=OUT/f'frozen-predictor-{seed}.pt'
    if path.exists():return
    saved=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
    c=B.Config(**saved['config']);data=saved['data']
    policy,icm=B.Policy(),B.ICM()
    policy.load_state_dict(saved['policy']);icm.load_state_dict(saved['icm'])
    with torch.random.fork_rng():test=B.collect(c,'icm',64,5200000+seed,policy)
    for p in icm.parameters():p.requires_grad_(False)
    for p in icm.forward_model.parameters():p.requires_grad_(True)
    def features(data):
        with torch.no_grad():
            phi=icm.features(torch.tanh(4*data['x']))
            phip=icm.features(torch.tanh(4*data['xp']))
        return torch.cat([phi,F.one_hot(data['action'],3).float()],1),phip
    xx,target=features(data);tx,ty=features(test)
    train_var=float(.5*(target-target.mean(0)).square().sum(1).mean())
    test_var=float(.5*(ty-ty.mean(0)).square().sum(1).mean())
    @torch.no_grad()
    def measurement(step):
        tr=.5*(icm.forward_model(xx)-target).square().sum(1).mean()
        te=.5*(icm.forward_model(tx)-ty).square().sum(1).mean()
        return dict(step=step,train_error=float(tr),heldout_error=float(te),
            train_normalized=float(tr)/train_var,heldout_normalized=float(te)/test_var)
    frames=[measurement(0)]
    opt=torch.optim.Adam(icm.forward_model.parameters(),lr=.001)
    gen=torch.Generator().manual_seed(5300000+seed)
    for step in range(6000):
        ix=torch.randint(len(xx),(512,),generator=gen)
        error=.5*(icm.forward_model(xx[ix])-target[ix]).square().sum(1).mean()
        opt.zero_grad(set_to_none=True);error.backward();opt.step()
        if (step+1)%100==0:frames.append(measurement(step+1))
    old.trials.save_torch(path,dict(seed=seed,frames=frames,icm=icm.state_dict(),
        train_target_variance=train_var,test_target_variance=test_var,config=vars(c)))
    print(f'frozen {seed} initial={frames[0]} final={frames[-1]}',flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    for seed in (130,131,136):
        joint(seed);frozen(seed)
