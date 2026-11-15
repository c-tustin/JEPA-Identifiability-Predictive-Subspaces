"""Does ICM's boundary preference follow the visual sensor geometry?"""
import argparse
import inspect
import sys
import copy
from pathlib import Path
import numpy as np
import torch
from guaranteed_models import old,dataset,OUT
B=old.trials.behavior
ORIGINAL_SENSOR=B.sensor


def sensor_for(kind):
    if kind=='native':return ORIGINAL_SENSOR
    if kind=='mirror':return lambda d,c:ORIGINAL_SENSOR(c.distance_min+c.distance_max-d,c)
    def equalized(d,c):
        ends=torch.tensor([c.distance_min,c.distance_max],dtype=d.dtype)
        z=torch.tanh(4*ORIGINAL_SENSOR(ends,c));fraction=(d-c.distance_min)/(c.distance_max-c.distance_min)
        target=z[0]+fraction[:,None]*(z[1]-z[0])
        return torch.atanh(target.clamp(-1+1e-7,1-1e-7))/4
    return equalized


@torch.no_grad()
def diagnostics(policy,icm,c,sensor,seed,episodes=64):
    with torch.random.fork_rng():data=B.collect(c,'icm',episodes,10000+seed,policy)
    physical=4*torch.exp(c.sigma*data['z'])
    grid=torch.linspace(1,16,601);x=sensor(grid,c)
    probabilities=policy(x)[0].probs
    successors=torch.stack([B.move(grid,torch.full_like(grid,a,dtype=torch.long),c) for a in range(3)],1)
    errors=torch.stack([icm(x,sensor(successors[:,a],c),torch.full_like(grid,a,dtype=torch.long))[0] for a in range(3)],1)
    signal=torch.stack([(torch.tanh(4*sensor(successors[:,a],c))-torch.tanh(4*x)).square().sum(1) for a in (0,2)],1).mean(1)
    return dict(distances=physical.numpy(),fraction_below2=float((physical<2).float().mean()),
        fraction_above15=float((physical>15).float().mean()),median_distance=float(physical.median()),
        mean_distance=float(physical.mean()),grid=grid.numpy(),probabilities=probabilities.numpy(),
        drift=(probabilities*(successors-grid[:,None])).sum(1).numpy(),
        expected_error=(probabilities*errors).sum(1).numpy(),visual_response=signal.numpy())


def fit(seed,kind):
    path=OUT/'actors'/f'{kind}-{seed}.pt'
    if path.exists():return
    c=dataset(seed)['c'];sensor=sensor_for(kind);previous=B.sensor;B.sensor=sensor
    frames=[]
    try:
        def capture(policy,icm,c,seed,update):
            frame=diagnostics(policy,icm,c,sensor,seed,64);frame['update']=update+1;frames.append(frame)
        trainer=inspect.getsource(B.train_policy)
        trainer=trainer.replace('    for update in range(c.policy_updates):','    capture(policy, icm, c, seed, -1)\n    for update in range(c.policy_updates):')
        anchor='        if update % 25 == 0 or update == c.policy_updates - 1:'
        assert trainer.count(anchor)==1
        trainer=trainer.replace(anchor,'        if (update+1) % 100 == 0:\n            capture(policy, icm, c, seed, update)\n'+anchor)
        namespace=dict(B.__dict__,capture=capture)
        exec(compile(trainer,str(Path(__file__).resolve()),'exec'),namespace)
        policy,icm,history,_=namespace['train_policy'](c,seed)
        final=diagnostics(policy,icm,c,sensor,seed,256)
        result=dict(seed=seed,kind=kind,policy=policy.state_dict(),icm=icm.state_dict(),
            history=history,frames=frames,final=final,config=vars(c))
        if kind=='native':
            reference=old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
            error={label:max(float((v-reference[label][n]).abs().max()) for n,v in model.state_dict().items()) for label,model in (('policy',policy),('icm',icm))}
            assert max(error.values())==0,error
            result['reproduction']=error
        old.trials.save_torch(path,result)
        print(kind,seed,'mean distance',round(final['mean_distance'],3),'near',round(final['fraction_below2'],3),'far',round(final['fraction_above15'],3),flush=True)
    finally:B.sensor=previous


def audit():
    c=dataset(131)['c'];d=torch.linspace(1.25,15.75,501)
    responses={}
    for kind in ('native','mirror','equalized'):
        sensor=sensor_for(kind);x=torch.tanh(4*sensor(d,c))
        response=torch.stack([(torch.tanh(4*sensor(B.move(d,torch.full_like(d,a,dtype=torch.long),c),c))-x).square().sum(1) for a in (0,2)],1).mean(1)
        responses[kind]=response
    assert float(responses['equalized'].std()/responses['equalized'].mean())<1e-4
    torch.testing.assert_close(responses['mirror'],responses['native'].flip(0),rtol=1e-4,atol=1e-8)
    endpoints=torch.tensor([1.,16.]);eq=torch.tanh(4*sensor_for('equalized')(endpoints,c));native=torch.tanh(4*ORIGINAL_SENSOR(endpoints,c))
    torch.testing.assert_close(eq,native,atol=1e-6,rtol=1e-6)
    old.trials.atomic_json(OUT/'geometry-audit.json',dict(equalized_response_cv=float(responses['equalized'].std()/responses['equalized'].mean()),mirror_response_reverses=True,endpoints_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    if args.audit_only:audit()
    else:
        jobs=[(131,'native')]+[(s,k) for s in range(130,140) for k in ('mirror','equalized')]
        for i,job in enumerate(jobs):
            if i%args.shards==args.shard:fit(*job)
