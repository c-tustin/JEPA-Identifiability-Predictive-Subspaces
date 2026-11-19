"""Test whether temporal alignment is necessary in this configuration."""
import copy
import json
from pathlib import Path
import run as T
import torch

C=T.C


def main():
    torch.set_num_threads(1)
    out=T.OUT/'no-alignment';rows=[]
    for source in C.SEEDS:
        source_path=C.LEGACY/'output/sources'/f'icm-{source}.pt'
        saved_source=C.trials.load(source_path);d=saved_source['data'];c=C.B.Config(**saved_source['config'])
        model,plan=C.trials.study.make_plan(c,131,131,'icm')
        initial=copy.deepcopy(model.state_dict())
        ids=torch.randint(len(d['x']),plan['indices'].shape,generator=torch.Generator().manual_seed(1900131))
        identity=dict(regularizer_source=source,initial_hash=C.state_hash(initial),
            source_hash=C.digest(source_path.read_bytes()),plan_hash=C.digest(ids.numpy().tobytes()),
            code_hash=C.digest(Path(__file__).read_bytes()),protocol_hash=C.digest((T.HERE/'ABLATION_PROTOCOL.md').read_bytes()))
        path=out/'runs'/f'reg{source}.pt'
        if path.exists():
            saved=C.trials.load(path);assert saved['identity']==identity
        else:
            opt=torch.optim.Adam(model.parameters(),lr=.001);reg=C.B.SIGReg1D()
            frames=[dict(step=0,**C.geometry(model,c,1025))]
            for step in range(4000):
                opt.param_groups[0]['lr']=C.trials.study.controls.learning_rate(step,4000)
                ix=ids[step]
                loss=.1*reg(model(torch.cat([d['x'][ix],d['xp'][ix]])))
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
                if (step+1)%100==0:frames.append(dict(step=step+1,**C.geometry(model,c,1025)))
            saved=dict(identity=identity,initial=initial,model=model.state_dict(),optimizer=opt.state_dict(),frames=frames,config=vars(c))
            C.trials.save_torch(path,saved)
            check=C.B.Encoder();check.load_state_dict(C.trials.load(path)['model'])
            saved['geometry']=C.geometry(check,c,20001);C.trials.save_torch(path,saved)
        rows.append(dict(**identity,**saved['geometry']))
        C.trials.atomic_json(out/'summary.json',dict(completed=len(rows),planned=10,rows=rows))
        print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
