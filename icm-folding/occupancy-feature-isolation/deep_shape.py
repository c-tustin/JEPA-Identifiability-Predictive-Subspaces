"""Exchange far within-interval profiles while all interval masses are fixed."""
import argparse
import itertools
from pathlib import Path
import far_regions as G

F,D,S,R,torch=G.F,G.D,G.S,G.R,G.torch
J,T=G.J,G.T
OUT=S.OUT/'deep-shape'


def distribution(bits,ds,u):
    a,b=9104,9109;regions={}
    for s in [a,b]:
        x=ds[s]['far'];d=1/x[:,1];regions[s]=[x[d<3],x[(d>=3)&(d<6)],x[d>=6]]
    native_counts=[len(x) for x in regions[a]]
    weights=[.5,*[.5*n/sum(native_counts) for n in native_counts]]
    parts=[ds[a]['near'],*[regions[[a,b][bit]][k] for k,bit in enumerate(bits)]]
    x=torch.cat(parts);counts=[len(p) for p in parts]
    ids=S.indices(.5,counts[0],sum(counts[1:]),u) if not any(bits) else J.pair_indices(weights,counts,u)
    return x,ids,dict(bits=bits,weights=weights,counts=counts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,initial,ix,u,_,_,_=F.setup()
    for k,bits in enumerate(itertools.product([0,1],repeat=3)):
        if k%a.shards!=a.shard:continue
        x,ids,spec=distribution(bits,ds,u);name=''.join(map(str,bits))
        identity=dict(**spec,initial_hash=R.state_hash(initial),index_hash=R.digest(ids.numpy().tobytes()),
            code_hash=R.digest(Path(__file__).read_bytes()),protocol_hash=R.digest((S.HERE/'DEEP_SHAPE_PROTOCOL.md').read_bytes()),
            source_hashes={s:ds[s]['source_hash'] for s in [131,9104,9109]})
        s=T.fit(OUT/'runs'/f'{name}.pt',x,ids,identity,ds,c,initial,ix)
        error=None
        if bits in [(0,0,0),(1,1,1)]:
            ref=R.trials.load(G.OUT/'runs'/f'9104-9109-p0q0s{bits[0]}.pt')
            error=max(float((v-ref['model'][key]).abs().max()) for key,v in s['model'].items());assert error==0
        R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,**s['geometry'],corner_max_parameter_error=error))
        print(name,s['geometry'],flush=True)


if __name__=='__main__':main()
