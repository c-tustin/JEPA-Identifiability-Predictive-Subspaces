"""Fixed-mass shape swaps motivated by the recorded fresh prediction failures."""
import argparse
import itertools
from pathlib import Path
import fresh_fit as F

D,S,R,torch=F.D,F.S,F.R,F.torch
PAIRS=((9101,9100),(9103,9102),(9104,9109))
OUT=S.OUT/'fresh-shapes'


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,initial,ix,u,mean,std,critic=F.setup()
    jobs=[(pair,bits) for pair in PAIRS for bits in itertools.product([0,1],repeat=2)]
    for k,(pair,bits) in enumerate(jobs):
        if k%a.shards!=a.shard:continue
        n,f=[pair[b] for b in bits];D.OUT=OUT/f'{pair[0]}-{pair[1]}'
        D.fit(n,f,.5,'sigreg',ds,c,initial,ix,u,mean,std,critic)
        name=f'n{n}-f{f}-a0.5-sigreg';path=D.OUT/'runs'/f'{name}.pt';s=R.trials.load(path)
        error=None
        if n==f:
            reference=R.trials.load(F.OUT/'runs'/f'{name}.pt')
            error=max(float((v-reference['model'][key]).abs().max()) for key,v in s['model'].items())
            assert error==0
        R.trials.atomic_json(D.OUT/'scope'/f'{name}.json',dict(pair=pair,bits=bits,
            governing_protocol_hash=R.digest((S.HERE/'FRESH_SHAPE_PROTOCOL.md').read_bytes()),
            wrapper_hash=R.digest(Path(__file__).read_bytes()),checkpoint_hash=R.digest(path.read_bytes()),
            diagonal_max_parameter_error=error))


if __name__=='__main__':main()
