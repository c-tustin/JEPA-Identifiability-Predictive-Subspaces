"""Independent saved-dataset, initialization, and minibatch-plan factors."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/icm-sampling-mpl')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sampling_base', HERE.parent/'initialization-crossing/run.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
np, torch, B = R.np, R.torch, R.B
OUT = HERE/'output'
DATA = INIT = tuple(range(130,140))
PLANS = tuple(range(5))


def dependencies():
    paths = [Path(__file__), HERE/'PROTOCOL.md', HERE/'REFERENCES.json', HERE.parent/'initialization-crossing/run.py',
             R.LEGACY/'run_trials.py', R.LEGACY.parent/'reliability/run_study.py',
             R.LEGACY.parent/'curiosity/experiment.py',
             R.LEGACY.parent/'identifiability/reproduce.py',
             R.LEGACY.parent/'identifiability/sigreg.py',
             R.LEGACY.parent/'curriculum-controls/run_controls.py']
    return {str(p.relative_to(HERE.parent.parent)):R.digest(p.read_bytes()) for p in paths}


def source(seed):
    path = R.LEGACY/'output/sources'/f'icm-{seed}.pt'
    s=R.trials.load(path)
    return B.Config(**s['config']), {k:s['data'][k] for k in ('x','xp')}, R.digest(path.read_bytes())


def plans(label, n, c):
    shape=(c.jepa_steps,c.batch)
    return tuple(torch.randint(n,shape,generator=torch.Generator().manual_seed(offset+label))
                 for offset in (3100000,3200000))


def initial(label):
    torch.manual_seed(40000+label)
    return B.Encoder()


def update(model,opt,reg,data,ix,ri,c,step):
    opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,c.jepa_steps)
    y=model(torch.cat([data['x'][ix],data['xp'][ix]]))
    alignment=(y[:c.batch]-y[c.batch:]).square().mean()
    penalty=reg(model(torch.cat([data['x'][ri],data['xp'][ri]])))
    loss=(1-c.sigreg_weight)*alignment+c.sigreg_weight*penalty
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(alignment.detach()),float(penalty.detach())


def preflight():
    config=None; n=None; hashes={};init_hashes={};plan_hashes={}
    references=json.loads((HERE/'REFERENCES.json').read_text())
    for d in DATA:
        c,data,h=source(d)
        assert set(data)=={'x','xp'}
        assert data['x'].shape==data['xp'].shape==(16384,6)
        assert torch.isfinite(data['x']).all() and torch.isfinite(data['xp']).all()
        assert config is None or vars(c)==config
        config=vars(c);n=len(data['x']);hashes[d]=h
        assert h==references['source_hashes'][str(d)]
    for j in INIT:
        m=initial(j)
        h=R.state_hash(m.state_dict());assert h==references['initial_hashes'][str(j)]
        init_hashes[j]=h
    for p in PLANS:
        ix,ri=plans(p,n,c)
        assert not torch.equal(ix,ri)
        plan_hashes[p]=[R.digest(a.numpy().tobytes()) for a in (ix,ri)]
    assert len({tuple(v) for v in plan_hashes.values()})==len(PLANS)
    assert len(set(init_hashes.values()))==len(INIT)
    return dict(source_hashes=hashes,initial_hashes=init_hashes,plan_hashes=plan_hashes,
                config=config,dependencies=dependencies())


def validate():
    audit=preflight();replays={}
    for d in (130,131):
        c,data,_=source(d)
        m,p=R.trials.study.make_plan(c,d,d,'icm')
        ix=p['indices'];ri=torch.randint(len(data['x']),ix.shape,
            generator=torch.Generator().manual_seed(1900000+d))
        opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
        for step in range(c.jepa_steps):update(m,opt,reg,data,ix[step],ri[step],c,step)
        expected=R.trials.load(R.LEGACY/'output/encoders'/f'icm-{d}.pt')['encoder']
        error=max(float((v-expected[k]).abs().max()) for k,v in m.state_dict().items())
        assert error==0;replays[d]=error
        print('Historical replay',d,'max error',error,flush=True)
    # Save/load both Adam and model at an actual interruption boundary.
    c,data,_=source(130);ix,ri=plans(0,len(data['x']),c)
    m=initial(130);opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
    for step in range(7):update(m,opt,reg,data,ix[step],ri[step],c,step)
    tmp=OUT/'validation-resume.pt'
    R.trials.save_torch(tmp,dict(model=m.state_dict(),optimizer=opt.state_dict()))
    for step in range(7,20):update(m,opt,reg,data,ix[step],ri[step],c,step)
    target=copy.deepcopy(m.state_dict());target_opt=copy.deepcopy(opt.state_dict())
    saved=R.trials.load(tmp);m=initial(130);m.load_state_dict(saved['model'])
    opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(saved['optimizer'])
    for step in range(7,20):update(m,opt,reg,data,ix[step],ri[step],c,step)
    assert R.state_hash(target)==R.state_hash(m.state_dict())
    for k,values in target_opt['state'].items():
        for key,v in values.items():assert torch.equal(v,opt.state_dict()['state'][k][key])
    tmp.unlink()
    audit.update(passed=True,historical_replay_max_error=replays,resume_model_and_adam_exact=True)
    R.trials.atomic_json(OUT/'preflight.json',audit)
    print('Preflight and resume checks passed',flush=True)


def fit(d,j,p,audit,steps=4000):
    path=OUT/'runs'/f'data{d}-init{j}-plan{p}-n{steps}.pt'
    c,data,h=source(d);ix,ri=plans(p,len(data['x']),c);m=initial(j)
    start_state=copy.deepcopy(m.state_dict())
    identity=dict(data_seed=d,init_seed=j,plan=p,steps=steps,source_hash=h,
        initial_hash=R.state_hash(start_state),plan_hashes=[R.digest(a.numpy().tobytes()) for a in (ix,ri)],
        dependencies=audit['dependencies'])
    assert identity['initial_hash']==audit['initial_hashes'][str(j)]
    assert identity['plan_hashes']==audit['plan_hashes'][str(p)]
    assert h==audit['source_hashes'][str(d)]
    if path.exists():
        assert R.trials.load(path)['identity']==identity
        return
    opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
    frames=[dict(step=0,**R.geometry(m,c,1025))];start=0;elapsed=0.
    partial=OUT/'partials'/path.name
    if partial.exists():
        s=R.trials.load(partial);assert s['identity']==identity
        m.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
        frames,start,elapsed=s['frames'],s['step'],s['seconds']
    started=time.monotonic()
    for step in range(start,steps):
        align,penalty=update(m,opt,reg,data,ix[step],ri[step],c,step)
        if (step+1)%100==0 or step+1==steps:
            frames.append(dict(step=step+1,alignment=align,regularizer=penalty,**R.geometry(m,c,1025)))
        if (step+1)%500==0:
            R.trials.save_torch(partial,dict(identity=identity,model=m.state_dict(),optimizer=opt.state_dict(),
                frames=frames,step=step+1,seconds=elapsed+time.monotonic()-started))
    s=dict(identity=identity,config=vars(c),initial=start_state,model=m.state_dict(),
        optimizer=opt.state_dict(),frames=frames,step=steps,seconds=elapsed+time.monotonic()-started)
    R.trials.save_torch(partial,s)
    check=B.Encoder();check.load_state_dict(R.trials.load(partial)['model'])
    s['geometry']=R.geometry(check,c,20001)
    R.trials.save_torch(path,s);partial.unlink(missing_ok=True)
    row=dict(data_seed=d,init_seed=j,plan=p,**s['geometry'],seconds=s['seconds'])
    R.trials.atomic_json(OUT/'cells'/f'data{d}-init{j}-plan{p}-n{steps}.json',row)
    print(json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.validate:return validate()
    assert 0<=args.shard<args.shards
    audit=json.loads((OUT/'preflight.json').read_text())
    assert audit['passed'] and audit['dependencies']==dependencies()
    cells=[(d,j,p) for p in PLANS for d in DATA for j in INIT]
    for k,cell in enumerate(cells):
        if k%args.shards==args.shard:fit(*cell,audit)

if __name__=='__main__':main()
