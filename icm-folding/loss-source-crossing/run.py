"""Independently cross the data supplied to alignment and SIGReg."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('initialization_crossing', HERE.parent/'initialization-crossing/run.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
import torch

OUT = HERE/'output'


def main():
    torch.set_num_threads(1)
    sources = {s:C.trials.load(C.LEGACY/'output/sources'/f'icm-{s}.pt') for s in C.SEEDS}
    config = C.B.Config(**sources[131]['config'])
    model, plan = C.trials.study.make_plan(config,131,131,'icm')
    initial = copy.deepcopy(model.state_dict())
    ix = plan['indices']
    ri = torch.randint(len(sources[131]['data']['x']),ix.shape,
                       generator=torch.Generator().manual_seed(1900131))
    assert all(len(s['data']['x']) == len(sources[131]['data']['x']) for s in sources.values())
    assert all(s['config']==sources[131]['config'] for s in sources.values())
    shared = dict(initial_hash=C.state_hash(initial),alignment_plan_hash=C.digest(ix.numpy().tobytes()),
                  regularizer_plan_hash=C.digest(ri.numpy().tobytes()),
                  code_hash=C.digest(Path(__file__).read_bytes()),
                  protocol_hash=C.digest((HERE/'PROTOCOL.md').read_bytes()),
                  dependencies={str(p):C.digest(p.read_bytes()) for p in
                                [HERE.parent/'initialization-crossing/run.py',C.LEGACY/'run_trials.py']})
    hashes={s:C.digest((C.LEGACY/'output/sources'/f'icm-{s}.pt').read_bytes()) for s in sources}
    cells=[(a,r) for a in C.SEEDS for r in C.SEEDS]
    cells.sort(key=lambda x:(x!=(131,131),x))
    for a,r in cells:
        path=OUT/'runs'/f'align{a}-reg{r}.pt'
        identity=dict(**shared,alignment_source=a,regularizer_source=r,
                      alignment_source_hash=hashes[a],regularizer_source_hash=hashes[r])
        if path.exists():
            assert C.trials.load(path)['identity']==identity
            continue
        model=C.B.Encoder();model.load_state_dict(initial)
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        reg=C.B.SIGReg1D()
        da,dr=sources[a]['data'],sources[r]['data']
        frames=[dict(step=0,**C.geometry(model,config,1025))]
        started=time.monotonic()
        for step in range(4000):
            opt.param_groups[0]['lr']=C.trials.study.controls.learning_rate(step,4000)
            y=model(torch.cat([da['x'][ix[step]],da['xp'][ix[step]]]))
            alignment=(y[:config.batch]-y[config.batch:]).square().mean()
            penalty=reg(model(torch.cat([dr['x'][ri[step]],dr['xp'][ri[step]]])))
            loss=.9*alignment+.1*penalty
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%100==0:
                frames.append(dict(step=step+1,alignment=float(alignment.detach()),
                                   sigreg=float(penalty.detach()),**C.geometry(model,config,1025)))
        saved=dict(identity=identity,initial=initial,model=model.state_dict(),optimizer=opt.state_dict(),
                   frames=frames,config=vars(config),seconds=time.monotonic()-started)
        if (a,r)==(131,131):
            original=C.trials.load(C.LEGACY/'output/encoders/icm-131.pt')['encoder']
            saved['replay_max_parameter_error']=max(float((p-original[k]).abs().max()) for k,p in model.state_dict().items())
            assert saved['replay_max_parameter_error']==0
        C.trials.save_torch(path,saved)
        check=C.B.Encoder();check.load_state_dict(C.trials.load(path)['model'])
        saved['geometry']=C.geometry(check,config,20001)
        for label,source in [('alignment',da),('regularizer',dr)]:
            ds=1/torch.cat([source['x'],source['xp']])[:,1]
            lo,hi=float(ds.min()),float(ds.max())
            saved['geometry'][label+'_support']=[lo,hi]
            saved['geometry'][label+'_support_roots']=[d for d in saved['geometry']['roots'] if lo<=d<=hi]
        C.trials.save_torch(path,saved)
        rows=[dict(**(s:=C.trials.load(p))['identity'],**s['geometry']) for p in sorted((OUT/'runs').glob('*.pt'))]
        C.trials.atomic_json(OUT/'summary.json',dict(completed=len(rows),planned=100,rows=rows))
        print(json.dumps(dict(alignment=a,regularizer=r,**saved['geometry'],seconds=saved['seconds'])),flush=True)


if __name__=='__main__':main()
