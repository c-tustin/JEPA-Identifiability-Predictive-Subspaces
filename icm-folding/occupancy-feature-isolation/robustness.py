"""Reuse the audited dose trainer under new explicit initializers/plans."""
import argparse
import itertools
import json
from pathlib import Path
import dose as D

S,R,torch,B=D.S,D.R,D.torch,D.B
OUT=S.OUT/'robustness'


def setup():
    ds,c,_,ix,_,mean,std,_=D.setup();initial={};critics={}
    for seed in range(130,135):
        torch.manual_seed(40000+seed);initial[seed]=B.Encoder().state_dict()
        torch.manual_seed(6100000+seed);critics[seed]=D.N.critic().state_dict()
    uniforms={p:torch.rand((4000,1024),dtype=torch.float64,generator=torch.Generator().manual_seed(8800131+p)) for p in [0,1]}
    meta=dict(wrapper_hash=R.digest(Path(__file__).read_bytes()),
        governing_protocol_hash=R.digest((S.HERE/'ROBUSTNESS_PROTOCOL.md').read_bytes()),
        dependencies=D.dependencies(),initial_hashes={s:R.state_hash(v) for s,v in initial.items()},
        critic_hashes={s:R.state_hash(v) for s,v in critics.items()},
        uniform_hashes={p:R.digest(u.numpy().tobytes()) for p,u in uniforms.items()},
        alignment_plan_hash=R.digest(ix.numpy().tobytes()),
        source_hashes={s:d['source_hash'] for s,d in ds.items()},planned=120)
    return ds,c,ix,mean,std,initial,critics,uniforms,meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1)
    ds,c,ix,mean,std,initial,critics,uniforms,meta=setup()
    if a.validate:
        R.trials.atomic_json(OUT/'preflight.json',meta);print('Explicit initializers and plans recorded',flush=True);return
    assert json.loads(json.dumps(meta))==json.loads((OUT/'preflight.json').read_text())
    jobs=list(itertools.product(range(130,135),[0,1],[.5,.75,.9375],[130,136],['sigreg','raw_infonce']))
    for k,(seed,plan,alpha,template,method) in enumerate(jobs):
        if k%a.shards!=a.shard:continue
        D.OUT=OUT/f'init{seed}-plan{plan}'
        D.fit(template,template,alpha,method,ds,c,initial[seed],ix,uniforms[plan],mean,std,critics[seed])
        name=f'n{template}-f{template}-a{alpha:g}-{method}'
        R.trials.atomic_json(D.OUT/'scope'/f'{name}.json',dict(initialization=seed,plan_label=plan,
            governing_protocol_hash=meta['governing_protocol_hash'],wrapper_hash=meta['wrapper_hash'],
            checkpoint_hash=R.digest((D.OUT/'runs'/f'{name}.pt').read_bytes())))


if __name__=='__main__':main()
