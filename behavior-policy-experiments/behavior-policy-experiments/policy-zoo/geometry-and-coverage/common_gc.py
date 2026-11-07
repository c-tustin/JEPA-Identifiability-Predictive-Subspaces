"""Common definitions for loss-source and original-policy exposure controls."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'collection-age-replication'))
import common as prior

OUT=HERE/'output'
FACTOR_SEEDS=tuple(range(200,220))
ZOO_SEEDS=tuple(range(8,28))
ZOO_POLICIES=('icm','random','count','rnd','disagreement','ou_actions_lp_w0.1','ou_actions_hp_w0.3')
behavior=prior.behavior
controls=prior.controls
original=prior.original


def signature():
    digest=hashlib.sha256()
    for filename in ('common_gc.py','run_geometry.py','PROTOCOL.md'):
        digest.update((HERE/filename).read_bytes())
    digest.update(prior.signature().encode())
    return digest.hexdigest()


def atomic_torch(path,value):
    prior.atomic_torch(path,value)


def save_json(path,value):
    prior.save_json(path,value)


def source_hash(data):
    return prior.data_hash(data)


def factor_data(seed,age):
    return prior.source('icm',age,seed)


def pooled(data):
    return torch.cat([data['x'],data['xp']])


def frame(model,points=257):
    q=torch.linspace(0,np.log(16),points,requires_grad=True)
    y=model(behavior.sensor(q.exp(),original.config()))
    slope=torch.autograd.grad(y.sum(),q)[0]
    return {'distance':q.detach().exp().numpy(),'curve':y.detach().numpy(),'slope':slope.detach().numpy()}


def old_actor(policy,seed):
    for root in original.run_zoo.EXISTING_CACHES:
        path=root/'sources'/f'{policy}_seed{seed}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu')['policy']
            return original.learned.load(saved)
    saved=torch.load(HERE.parent/'output/policies'/f'{policy}_seed{seed}.pt',map_location='cpu')
    return original.learned.load(saved)


def native_data(policy,seed):
    c=original.config()
    data=None
    if policy in ('icm','rnd','disagreement','count'):
        for root in original.run_zoo.EXISTING_CACHES:
            path=root/'sources'/f'{policy}_seed{seed}.pt'
            if path.exists():
                data=torch.load(path,map_location='cpu')['data']
                break
        if data is None:
            data=original.learned.collect(c,policy,c.train_episodes,10000+seed,old_actor(policy,seed))
    else:
        data,_=original.run_zoo.source(c,policy,seed,HERE.parent/'output')
    actual=original.run_zoo.diagnostics(data,c)['sha256']
    expected=original.raw_record(policy,seed)['source_stats']['sha256']
    if actual!=expected:
        raise RuntimeError(f'Original zoo source changed: {policy}, seed {seed}')
    return data


def reset_pool(seed,size=16384):
    rng=torch.Generator().manual_seed(5200000+seed)
    q=np.log(16)*(torch.arange(size)+torch.rand(size,generator=rng))/size
    q=q[torch.randperm(size,generator=rng)]
    distance=q.exp()
    return distance,behavior.sensor(distance,original.config())


@torch.no_grad()
def reset_data(policy,seed,native=None):
    c=original.config()
    distance,x=reset_pool(seed)
    if policy in ('ou_actions_lp_w0.1','ou_actions_hp_w0.3'):
        # These action processes do not read the scene; reuse their exact native
        # realization, including original warm-up and temporal correlations.
        if native is None:
            native=native_data(policy,seed)
        action=native['action'].clone()
        successor=behavior.move(distance,action,c)
        return {'x':x,'xp':behavior.sensor(successor,c),'action':action,
                'z':torch.log(distance/4)/c.sigma,'zp':torch.log(successor/4)/c.sigma}
    actor=None
    rule=None
    before=None
    if policy in ('icm','rnd','disagreement','count'):
        actor=old_actor(policy,seed)
        before=prior.tensor_hash(actor.state_dict())
    elif policy!='random':
        rule=original.run_zoo.zoo.PHYSICAL_RULES[policy]()
    torch.manual_seed(5300000+seed)
    if rule is not None:
        rule.reset(c.train_episodes)
    actions=[]
    for t in range(c.horizon):
        views=x[t*c.train_episodes:(t+1)*c.train_episodes]
        if actor is not None:
            action=actor(views)[0].sample()
        elif rule is not None:
            action=rule.act(views)
        else:
            action=torch.randint(0,3,(c.train_episodes,))
        actions.append(action)
    action=torch.cat(actions)
    successor=behavior.move(distance,action,c)
    if actor is not None:
        assert before==prior.tensor_hash(actor.state_dict())
    return {'x':x,'xp':behavior.sensor(successor,c),'action':action,
            'z':torch.log(distance/4)/c.sigma,'zp':torch.log(successor/4)/c.sigma}
