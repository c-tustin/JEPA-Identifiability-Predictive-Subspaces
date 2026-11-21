"""Resolve shoulder probability versus fine shape, with exact corner replays."""
import argparse
import itertools
from pathlib import Path
import deep_shape as E

F,D,S,R,torch=E.F,E.D,E.S,E.R,E.torch
J,T=E.J,E.T
OUT=S.OUT/'shoulder'


def distribution(bits,ds,u):
    a,b=9104,9109;parts={};fractions={}
    for s in [a,b]:
        x=ds[s]['far'];d=1/x[:,1];z=x[d<3];q=1/z[:,1]
        parts[s]=[z[q<2.5],z[q>=2.5],x[(d>=3)&(d<6)],x[d>=6]]
        fractions[s]=float((q<2.5).double().mean())
    alpha=fractions[[a,b][bits[0]]]
    nn=[len(x) for x in parts[a]];nf=sum(nn);shoulder=(nn[0]+nn[1])/nf
    weights=[.5,.5*shoulder*alpha,.5*shoulder*(1-alpha),.5*nn[2]/nf,.5*nn[3]/nf]
    arrays=[ds[a]['near'],parts[[a,b][bits[1]]][0],parts[[a,b][bits[2]]][1],parts[a][2],parts[a][3]]
    x=torch.cat(arrays);counts=[len(p) for p in arrays]
    if bits in [(0,0,0),(1,1,1)]:
        xx,ids,_=E.distribution((bits[0],0,0),ds,u);assert torch.equal(xx,x)
    else:ids=J.pair_indices(weights,counts,u)
    return x,ids,dict(bits=bits,shoulder_cdf2_5=alpha,weights=weights,counts=counts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,initial,ix,u,_,_,_=F.setup()
    for k,bits in enumerate(itertools.product([0,1],repeat=3)):
        if k%a.shards!=a.shard:continue
        x,ids,spec=distribution(bits,ds,u);name=''.join(map(str,bits))
        identity=dict(**spec,initial_hash=R.state_hash(initial),index_hash=R.digest(ids.numpy().tobytes()),
            code_hash=R.digest(Path(__file__).read_bytes()),protocol_hash=R.digest((S.HERE/'SHOULDER_PROTOCOL.md').read_bytes()),
            source_hashes={s:ds[s]['source_hash'] for s in [131,9104,9109]})
        s=T.fit(OUT/'runs'/f'{name}.pt',x,ids,identity,ds,c,initial,ix);error=None
        if bits in [(0,0,0),(1,1,1)]:
            ref=R.trials.load(E.OUT/'runs'/('000.pt' if bits[0]==0 else '100.pt'))
            error=max(float((v-ref['model'][key]).abs().max()) for key,v in s['model'].items());assert error==0
        R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,**s['geometry'],corner_max_parameter_error=error))
        print(name,s['geometry'],flush=True)


if __name__=='__main__':main()
