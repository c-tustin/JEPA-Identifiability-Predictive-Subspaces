"""Repeat regularizer-source contrasts with iid endpoint sampling."""
import copy
import json
from pathlib import Path
import time
import run as T
import torch

C=T.C


def main():
    torch.set_num_threads(1)
    sources={s:C.trials.load(C.LEGACY/'output/sources'/f'icm-{s}.pt') for s in C.SEEDS}
    c=C.B.Config(**sources[131]['config'])
    model,plan=C.trials.study.make_plan(c,131,131,'icm');initial=copy.deepcopy(model.state_dict())
    ix=plan['indices'];da=sources[131]['data']
    out=T.OUT/'iid-endpoints';rows=[]
    for seed in [2900131,2900132,2900133]:
        ids=torch.randint(2*len(da['x']),(4000,1024),generator=torch.Generator().manual_seed(seed))
        for source in C.SEEDS:
            path=out/'runs'/f'reg{source}-plan{seed}.pt'
            identity=dict(regularizer_source=source,plan_seed=seed,alignment_source=131,initialization=131,
                initial_hash=C.state_hash(initial),alignment_plan_hash=C.digest(ix.numpy().tobytes()),
                endpoint_plan_hash=C.digest(ids.numpy().tobytes()),
                regularizer_source_hash=C.digest((C.LEGACY/'output/sources'/f'icm-{source}.pt').read_bytes()),
                alignment_source_hash=C.digest((C.LEGACY/'output/sources/icm-131.pt').read_bytes()),
                code_hash=C.digest(Path(__file__).read_bytes()),protocol_hash=C.digest((T.HERE/'IID_PROTOCOL.md').read_bytes()))
            if path.exists():
                saved=C.trials.load(path);assert saved['identity']==identity
            else:
                model=C.B.Encoder();model.load_state_dict(initial)
                opt=torch.optim.Adam(model.parameters(),lr=.001);reg=C.B.SIGReg1D()
                dr=sources[source]['data'];endpoints=torch.cat([dr['x'],dr['xp']])
                frames=[dict(step=0,**C.geometry(model,c,1025))];start=time.monotonic()
                for step in range(4000):
                    opt.param_groups[0]['lr']=C.trials.study.controls.learning_rate(step,4000)
                    y=model(torch.cat([da['x'][ix[step]],da['xp'][ix[step]]]))
                    la=(y[:c.batch]-y[c.batch:]).square().mean();lr=reg(model(endpoints[ids[step]]))
                    loss=.9*la+.1*lr;assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();opt.step()
                    if (step+1)%100==0:frames.append(dict(step=step+1,**C.geometry(model,c,1025)))
                saved=dict(identity=identity,model=model.state_dict(),initial=initial,optimizer=opt.state_dict(),
                           config=vars(c),frames=frames,seconds=time.monotonic()-start)
                C.trials.save_torch(path,saved)
                check=C.B.Encoder();check.load_state_dict(C.trials.load(path)['model'])
                saved['geometry']=C.geometry(check,c,20001)
                C.trials.save_torch(path,saved)
            rows.append(dict(**identity,**saved['geometry']))
            C.trials.atomic_json(out/'summary.json',dict(completed=len(rows),planned=30,rows=rows))
            print(json.dumps(dict(source=source,plan=seed,**saved['geometry'])),flush=True)


if __name__=='__main__':main()
