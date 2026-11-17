"""Match native ICM occupancy by empirical-quantile OU; shared loss inputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/icm-matched-ou-mpl')
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import numpy as np
from scipy.special import ndtr
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('icm_ou_shared_reference',HERE.parent/'shared-occupancy-ou/run.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
old,base = shared.old,shared.base
sha = shared.sha
OUT = HERE/'output'
SEEDS = (130,131,134,136)
ARMS = {'native':None,'ou-small':.01,'ou-large':.1}

def source(seed):
    path = old.ORIGINAL/'sources'/f'icm-{seed}.pt'
    saved = old.trials.load(path)
    return path,saved['data'],old.trials.behavior.Config(**saved['config'])

def sorted_pool(data):
    endpoints = torch.cat((data['x'],data['xp']))
    distance = old.trials.distances(endpoints).numpy().astype(np.float64)
    order = np.argsort(distance,kind='stable')
    return endpoints[torch.from_numpy(order)],distance[order]

def generate(seed,dt,pool,distance,episodes=256,horizon=64):
    rng = np.random.default_rng(2300000+seed)
    initial = rng.standard_normal(episodes)
    noise = rng.standard_normal((horizon,episodes))
    z = np.empty((horizon+1,episodes))
    z[0] = initial
    rho = np.exp(-dt)
    for t in range(horizon):
        z[t+1] = rho*z[t]+np.sqrt(-np.expm1(-2*dt))*noise[t]
    u = ndtr(z)
    ranks = np.minimum((len(pool)*u).astype(np.int64),len(pool)-1)
    x = pool[torch.from_numpy(ranks[:-1].reshape(-1))]
    xp = pool[torch.from_numpy(ranks[1:].reshape(-1))]
    return dict(z=z,initial=initial,noise=noise,ranks=ranks,distance=distance[ranks]),x,xp

def fit(seed,arm):
    path = OUT/f'{arm}-{seed}.pt'
    if path.exists():
        r = old.trials.load(path)
        assert r['code_sha256']==sha(__file__)
        assert r['protocol_sha256']==sha(HERE/'PROTOCOL.md')
        return
    source_path,data,c = source(seed)
    pool,distance = sorted_pool(data)
    dt = ARMS[arm]
    if dt is None:
        trajectory = None
        x,xp = data['x'],data['xp']
    else:
        trajectory,x,xp = generate(seed,dt,pool,distance,c.train_episodes,c.horizon)
    torch.manual_seed(40000+seed)
    model = old.ScaledEncoder(4)
    initial = copy.deepcopy(model.state_dict())
    indices = torch.randint(len(x),(c.jepa_steps,c.batch),generator=torch.Generator().manual_seed(2400000+seed))
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    reg = old.trials.behavior.SIGReg1D()
    history = []
    for step,ix in enumerate(indices):
        opt.param_groups[0]['lr'] = old.trials.study.controls.learning_rate(step,c.jepa_steps)
        loss,a,r,_ = shared.shared_loss(model,x[ix],xp[ix],reg)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1)%100==0:
            q,y,slope = base.curve(model,c)
            history.append(dict(step=step+1,y=y,slope=slope,roots=base.roots(q,slope),
                alignment=float(a.detach()),sigreg=float(r.detach())))
    old.trials.save_torch(path,dict(seed=seed,arm=arm,dt=dt,config=vars(c),trajectory=trajectory,
        indices=indices,initial=initial,encoder=model.state_dict(),history=history,
        code_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.md'),
        shared_code_sha256=sha(shared.__file__),source_sha256=sha(source_path)))
    print(seed,arm,'final roots',history[-1]['roots'],flush=True)

def audit():
    checks = []
    for seed in SEEDS:
        path,data,c = source(seed)
        pool,distance = sorted_pool(data)
        for dt in (.01,.1):
            tr,x,xp = generate(seed,dt,pool,distance,c.train_episodes,c.horizon)
            np.testing.assert_array_equal(tr['z'][1:],np.exp(-dt)*tr['z'][:-1]+np.sqrt(-np.expm1(-2*dt))*tr['noise'])
            assert torch.equal(xp[:-c.train_episodes],x[c.train_episodes:])
            u = ndtr(tr['z'])
            assert np.all(tr['ranks']/len(pool)<=u)
            assert np.all(u<(tr['ranks']+1)/len(pool))
            assert torch.equal(x,pool[torch.from_numpy(tr['ranks'][:-1].reshape(-1))])
            bins = np.geomspace(1,16,65)
            target,_ = np.histogram(distance,bins)
            measured,_ = np.histogram(np.r_[old.trials.distances(x).numpy(),old.trials.distances(xp).numpy()],bins)
            assert np.all(measured[target==0]==0)
            checks.append(dict(seed=seed,dt=dt,empty_bins_preserved=True,source_sha256=sha(path)))
    # The midpoint of every rank cell must select that endpoint exactly.
    u = (np.arange(len(pool))+.5)/len(pool)
    np.testing.assert_array_equal((len(pool)*u).astype(np.int64),np.arange(len(pool)))
    # Audit the imported objective unchanged; its output lives in the new study.
    previous = shared.OUT
    try:
        shared.OUT = OUT
        shared.audit()
    finally:
        shared.OUT = previous
    old.trials.atomic_json(OUT/'generator-audit.json',dict(runs=checks,all_rank_cells_correct=True,
        exact_recurrence=True,trajectory_continuity=True,observations_exactly_from_source=True))
    print('Native-occupancy generator audit passed',flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.check:
        audit()
        return
    for seed in SEEDS:
        for arm in ARMS:
            fit(seed,arm)

if __name__=='__main__':
    main()
