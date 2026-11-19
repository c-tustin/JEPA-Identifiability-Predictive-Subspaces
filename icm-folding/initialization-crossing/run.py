"""Cross historical ICM datasets and scalar-encoder initializations independently."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/icm-crossing-mpl')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "sources"
LEGACY = ROOT/'predictive-closure-workspace/behavior-policy-experiments/qualitative-study'
sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location('crossing_trials', LEGACY/'run_trials.py')
trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trials)
import numpy as np
import torch
B = trials.behavior
OUT = HERE/'output'
SEEDS = tuple(range(130, 140))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def state_hash(state):
    return digest(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in sorted(state.items())))


def geometry(model, c, points):
    m = copy.deepcopy(model).double()
    q = torch.linspace(0, np.log(16), points, dtype=torch.float64, requires_grad=True)
    y = m(B.sensor(q.exp(), c))
    slope = torch.autograd.grad(y.sum(), q)[0]
    qq, yy, ss = [a.detach().numpy() for a in (q,y,slope)]
    ids = np.flatnonzero(ss[:-1]*ss[1:] < 0)
    roots = np.exp(qq[ids]-ss[ids]*np.diff(qq)[ids]/np.diff(ss)[ids]).tolist()
    dy = np.diff(yy)
    up, down = np.maximum(dy,0).sum(), np.maximum(-dy,0).sum()
    backward = float(min(up,down)/max(up+down,1e-12))
    return dict(roots=roots, folded=bool(roots), material_fold=bool(roots) and backward>.001,
                backward_fraction=backward, std=float(yy.std()),
                affine_r2=float(np.corrcoef(qq,yy)[0,1]**2))


def fit(data_seed, init_seed, steps):
    path = OUT/'runs'/f'data{data_seed}-init{init_seed}-n{steps}.pt'
    source_path = LEGACY/'output/sources'/f'icm-{data_seed}.pt'
    source = trials.load(source_path)
    c = B.Config(**source['config'])
    # Geometry and action labels cannot enter training.
    data = {k:source['data'][k] for k in ('x','xp')}
    model, plan = trials.study.make_plan(c, data_seed, init_seed, 'icm')
    indices = plan['indices'][:steps]
    reg_indices = torch.randint(len(data['x']), plan['indices'].shape,
        generator=torch.Generator().manual_seed(1900000+data_seed))[:steps]
    initial = copy.deepcopy(model.state_dict())
    identity = dict(data_seed=data_seed, init_seed=init_seed, steps=steps,
        source_hash=digest(source_path.read_bytes()), initial_hash=state_hash(initial),
        alignment_plan_hash=digest(indices.numpy().tobytes()),
        regularizer_plan_hash=digest(reg_indices.numpy().tobytes()),
        code_hash=digest(Path(__file__).read_bytes()),
        protocol_hash=digest((HERE/'PROTOCOL.md').read_bytes()))
    if path.exists():
        saved = trials.load(path)
        assert saved['identity'] == identity, 'Existing run identity mismatch'
        return
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    reg = B.SIGReg1D()
    frames = [dict(step=0, **geometry(model,c,1025))]
    start = 0
    partial = OUT/'partials'/path.name
    elapsed = 0.
    if partial.exists():
        saved = trials.load(partial)
        assert saved['identity'] == identity
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        frames, start, elapsed = saved['frames'],saved['step'],saved['seconds']
    started = time.monotonic()
    for step in range(start,steps):
        opt.param_groups[0]['lr'] = trials.study.controls.learning_rate(step,c.jepa_steps)
        ix, ri = indices[step],reg_indices[step]
        y = model(torch.cat([data['x'][ix],data['xp'][ix]]))
        alignment = (y[:c.batch]-y[c.batch:]).square().mean()
        penalty = reg(model(torch.cat([data['x'][ri],data['xp'][ri]])))
        loss = (1-c.sigreg_weight)*alignment+c.sigreg_weight*penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step+1)%100 == 0 or step+1 == steps:
            frames.append(dict(step=step+1,alignment=float(alignment.detach()),
                regularizer=float(penalty.detach()),**geometry(model,c,1025)))
        if (step+1)%500 == 0:
            trials.save_torch(partial,dict(identity=identity,model=model.state_dict(),
                optimizer=opt.state_dict(),frames=frames,step=step+1,
                seconds=elapsed+time.monotonic()-started))
    saved = dict(identity=identity,config=vars(c),initial=initial,model=model.state_dict(),
        optimizer=opt.state_dict(),frames=frames,seconds=elapsed+time.monotonic()-started)
    # Reload final parameters independently before measuring dense geometry.
    trials.save_torch(partial,saved)
    check = B.Encoder()
    check.load_state_dict(trials.load(partial)['model'])
    saved['geometry'] = geometry(check,c,20001)
    if data_seed == init_seed and steps == c.jepa_steps:
        original = trials.load(LEGACY/'output/encoders'/f'icm-{data_seed}.pt')['encoder']
        saved['diagonal_max_parameter_error'] = max(float((v-original[k]).abs().max())
            for k,v in model.state_dict().items())
        assert saved['diagonal_max_parameter_error'] == 0, 'Historical replay differs'
    trials.save_torch(path,saved)
    partial.unlink(missing_ok=True)
    print(json.dumps(dict(**identity,**saved['geometry'],seconds=saved['seconds'])),flush=True)


def report(steps):
    rows=[]
    initial_hashes, plans = {}, {}
    for path in sorted((OUT/'runs').glob(f'*-n{steps}.pt')):
        s=trials.load(path); i=s['identity']; d,j=i['data_seed'],i['init_seed']
        assert initial_hashes.setdefault(j,i['initial_hash']) == i['initial_hash']
        pair=(i['alignment_plan_hash'],i['regularizer_plan_hash'])
        assert plans.setdefault(d,pair) == pair
        rows.append(dict(data_seed=d,init_seed=j,**s['geometry'],
            diagonal_max_parameter_error=s.get('diagonal_max_parameter_error'),seconds=s['seconds']))
    trials.atomic_json(OUT/f'summary-n{steps}.json',dict(completed=len(rows),planned=100,
        status='complete' if len(rows)==100 else 'in_progress',rows=rows,
        pairing_checks='passed for completed runs'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--data',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--init',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    assert 0 < args.steps <= 4000
    torch.set_num_threads(1)
    if not args.report_only:
        cells=[(d,j) for d in args.data for j in args.init]
        # Validate the historical diagonal first, then finish the crossing.
        cells.sort(key=lambda pair:(pair[0]!=pair[1],pair))
        for d,j in cells:
            fit(d,j,args.steps)
            report(args.steps)
    report(args.steps)

if __name__ == '__main__':
    main()
