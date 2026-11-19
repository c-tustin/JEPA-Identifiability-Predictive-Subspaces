"""Replicate first-moment surgery across historical fold events."""
import copy
import importlib.util
import json
from pathlib import Path
import run as R
import torch
import numpy as np

spec=importlib.util.spec_from_file_location('crossing_geometry',R.HERE.parent/'initialization-crossing/run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def coarse(model,c):
    q=torch.linspace(0,np.log(16),1025,requires_grad=True)
    y=model(C.B.sensor(q.exp(),c));s=torch.autograd.grad(y.sum(),q)[0].detach()
    return bool(torch.any(s[:-1]*s[1:]<0))


def main():
    torch.set_num_threads(1)
    events=[];replays={};initials={};failed=[];states={}
    for seed in [131,132,133,136,139]:
        source_path=C.LEGACY/'output/sources'/f'icm-{seed}.pt'
        source=C.trials.load(source_path);data=source['data'];c=C.B.Config(**source['config'])
        model,plan=C.trials.study.make_plan(c,seed,seed,'icm')
        ri=torch.randint(len(data['x']),plan['indices'].shape,generator=torch.Generator().manual_seed(1900000+seed))
        opt=torch.optim.Adam(model.parameters(),lr=.001);reg=C.B.SIGReg1D()
        mr={k:torch.zeros_like(p) for k,p in model.named_parameters()}
        ma=copy.deepcopy(mr)
        initials[seed]=C.geometry(model,c,20001)
        before_folded=coarse(model,c)
        for step in range(4000):
            opt.param_groups[0]['lr']=C.trials.study.controls.learning_rate(step,4000)
            ix,rx=plan['indices'][step],ri[step]
            ya=model(torch.cat([data['x'][ix],data['xp'][ix]]))
            yr=model(torch.cat([data['x'][rx],data['xp'][rx]]))
            ga=torch.autograd.grad(.9*(ya[:c.batch]-ya[c.batch:]).square().mean(),tuple(model.parameters()))
            gr=torch.autograd.grad(.1*reg(yr),tuple(model.parameters()))
            scan=step<500
            # A fork needs only an ordered before-state; skip snapshots of
            # already-folded states, while still scanning the next state.
            if scan and not before_folded:
                before=copy.deepcopy(model.state_dict());ob=copy.deepcopy(opt.state_dict())
                mb_r,mb_a=copy.deepcopy(mr),copy.deepcopy(ma)
            opt.zero_grad()
            for (k,p),a,r in zip(model.named_parameters(),ga,gr):
                p.grad=a+r;ma[k].mul_(.9).add_(a,alpha=.1);mr[k].mul_(.9).add_(r,alpha=.1)
            opt.step()
            if scan:
                after_folded=coarse(model,c)
                if not before_folded and after_folded:
                    bm=C.B.Encoder();bm.load_state_dict(before)
                    bg=C.geometry(bm,c,20001);ag=C.geometry(model,c,20001)
                    if bg['folded'] or not ag['folded']:
                        failed.append(dict(seed=seed,update=step+1,before=bg,after=ag))
                    else:
                        forks={};key=f'{seed}-{step+1}';states[key]={}
                        for mode in ['full','remove_regularizer_history','remove_alignment_history','remove_all_history']:
                            m=C.B.Encoder();m.load_state_dict(before)
                            o=torch.optim.Adam(m.parameters(),lr=.001);o.load_state_dict(copy.deepcopy(ob))
                            for (k,p),a,r in zip(m.named_parameters(),ga,gr):
                                # The first update has no optimizer history;
                                # every history-removal fork is then a no-op.
                                if 'exp_avg' in o.state[p]:
                                    if mode=='remove_regularizer_history':o.state[p]['exp_avg'].sub_(mb_r[k])
                                    if mode=='remove_alignment_history':o.state[p]['exp_avg'].sub_(mb_a[k])
                                    if mode=='remove_all_history':o.state[p]['exp_avg'].zero_()
                                p.grad=a+r
                            o.step()
                            if mode=='full':assert all(torch.equal(v,model.state_dict()[k]) for k,v in m.state_dict().items())
                            forks[mode]=C.geometry(m,c,20001);states[key][mode]=m.state_dict()
                        events.append(dict(seed=seed,update=step+1,has_prior_momentum=step>0,before=bg,after=ag,forks=forks))
                before_folded=after_folded
        original=C.trials.load(C.LEGACY/'output/encoders'/f'icm-{seed}.pt')['encoder']
        error=max(float((v-original[k]).abs().max()) for k,v in model.state_dict().items())
        assert error==0;replays[seed]=error
        print(json.dumps(dict(seed=seed,replay_error=error,events=[e for e in events if e['seed']==seed])),flush=True)
    result=dict(events=events,initial_geometry=initials,failed_dense_gates=failed,final_replay_errors=replays,
        source_hashes={s:C.digest((C.LEGACY/'output/sources'/f'icm-{s}.pt').read_bytes()) for s in replays},
        code_hash=C.digest(Path(__file__).read_bytes()),protocol_hash=C.digest((R.HERE/'BIRTH_PROTOCOL.md').read_bytes()))
    C.trials.atomic_json(R.HERE/'output/birth_replication.json',result)
    C.trials.save_torch(R.HERE/'output/birth_replication_forks.pt',states)


if __name__=='__main__':main()
