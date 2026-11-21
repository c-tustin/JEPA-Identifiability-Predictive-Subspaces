"""Separate far regional probabilities from within-region shape."""
import argparse
import itertools
from pathlib import Path
import fresh_shape as C
import train_marginal as T
import joint as J

F,D,S,R,torch=C.F,C.D,C.S,C.R,C.torch
OUT=S.OUT/'far-regions'


def distribution(pair,bits,ds,u):
    a,b=pair;p3,p6,shape=bits;far={};cdf={}
    for seed in pair:
        x=ds[seed]['far'];d=1/x[:,1]
        far[seed]=[x[d<3],x[(d>=3)&(d<6)],x[d>=6]]
        cdf[seed]=[float((d<3).double().mean()),float((d<6).double().mean())]
    r3,r6=cdf[pair[p3]][0],cdf[pair[p6]][1];assert 0<r3<r6<1
    ss=pair[shape];parts=[ds[a]['near'],*far[ss]];counts=[len(p) for p in parts]
    weights=[.5,.5*r3,.5*(r6-r3),.5*(1-r6)]
    x=torch.cat(parts);native=p3==p6==shape
    ids=S.indices(.5,counts[0],sum(counts[1:]),u) if native else J.pair_indices(weights,counts,u)
    return x,ids,dict(pair=pair,bits=bits,conditional_far_cdf3=r3,conditional_far_cdf6=r6,
                     region_weights=weights,region_counts=counts,native_cdf_shortcut=native)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,initial,ix,u,_,_,_=F.setup()
    jobs=[(pair,bits) for pair in C.PAIRS for bits in itertools.product([0,1],repeat=3)]
    for k,(pair,bits) in enumerate(jobs):
        if k%a.shards!=a.shard:continue
        x,ids,spec=distribution(pair,bits,ds,u)
        name=f'{pair[0]}-{pair[1]}-p{bits[0]}q{bits[1]}s{bits[2]}'
        identity=dict(**spec,code_hash=R.digest(Path(__file__).read_bytes()),
            governing_protocol_hash=R.digest((S.HERE/'FAR_REGIONS_PROTOCOL.md').read_bytes()),
            initial_hash=R.state_hash(initial),alignment_plan_hash=R.digest(ix.numpy().tobytes()),
            index_hash=R.digest(ids.numpy().tobytes()),uniform_hash=R.digest(u.numpy().tobytes()),
            source_hashes={s:ds[s]['source_hash'] for s in set([*pair,131])})
        saved=T.fit(OUT/'runs'/f'{name}.pt',x,ids,identity,ds,c,initial,ix)
        error=None
        if spec['native_cdf_shortcut']:
            n=pair[0];f=pair[bits[2]]
            reference=R.trials.load(C.OUT/f'{pair[0]}-{pair[1]}'/'runs'/f'n{n}-f{f}-a0.5-sigreg.pt')
            error=max(float((v-reference['model'][key]).abs().max()) for key,v in saved['model'].items());assert error==0
        row=dict(name=name,**identity,**saved['geometry'],native_max_parameter_error=error)
        R.trials.atomic_json(OUT/'cells'/f'{name}.json',row)
        print(name,saved['geometry'],flush=True)


if __name__=='__main__':main()
