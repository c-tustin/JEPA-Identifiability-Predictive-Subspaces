"""Apply predeclared native/mass interventions after freezing predictions."""
import argparse
import json
from pathlib import Path
import dose as D

S,R,torch,B=D.S,D.R,D.torch,D.B
OUT=S.OUT/'fresh/encoders'


def setup():
    ds,c,initial,ix,_,mean,std,critic=D.setup()
    for seed in range(9100,9110):
        path=S.OUT/'fresh/sources'/f'icm-{seed}.pt';s=R.trials.load(path);assert s['config']==vars(c)
        x=torch.cat([s['data']['x'],s['data']['xp']]);d=1/x[:,1];order=d.argsort(stable=True);x,d=x[order],d[order]
        ds[seed]=dict(x=x,d=d,near=x[d<2],far=x[d>=2],mass=float((d<2).double().mean()),
            data=s['data'],config=s['config'],source_hash=R.digest(path.read_bytes()))
    u=torch.rand((4000,1024),dtype=torch.float64,generator=torch.Generator().manual_seed(8900131))
    return ds,c,initial,ix,u,mean,std,critic


def prediction_lock():
    return dict(files={str(s):R.digest((S.OUT/'fresh/predictions'/f'{s}.json').read_bytes()) for s in range(9100,9110)},
                bank_hash=R.digest((S.HERE/'BANK.json').read_bytes()),
                protocol_hash=R.digest((S.HERE/'FRESH_FIT_PROTOCOL.md').read_bytes()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-predictions',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);lock=prediction_lock()
    if a.freeze_predictions:
        assert not list((OUT/'runs').glob('*.pt'))
        R.trials.atomic_json(S.HERE/'PREDICTIONS_LOCK.json',lock);print('All ten predictions frozen before encoder fits');return
    assert lock==json.loads((S.HERE/'PREDICTIONS_LOCK.json').read_text())
    ds,c,initial,ix,u,mean,std,critic=setup();D.OUT=OUT
    jobs=[(s,alpha,method) for s in range(9100,9110) for alpha in sorted(set([ds[s]['mass'],.5,.75,.9375])) for method in ['sigreg','raw_infonce']]
    for k,(s,alpha,method) in enumerate(jobs):
        if k%a.shards!=a.shard:continue
        prediction=json.loads((S.OUT/'fresh/predictions'/f'{s}.json').read_text())
        assert prediction['source_hash']==ds[s]['source_hash']
        D.fit(s,s,alpha,method,ds,c,initial,ix,u,mean,std,critic)
        name=f'n{s}-f{s}-a{alpha:g}-{method}'
        R.trials.atomic_json(OUT/'scope'/f'{name}.json',dict(source=s,is_native=alpha==ds[s]['mass'],
            prediction_file_hash=lock['files'][str(s)],governing_protocol_hash=lock['protocol_hash'],
            wrapper_hash=R.digest(Path(__file__).read_bytes()),
            checkpoint_hash=R.digest((OUT/'runs'/f'{name}.pt').read_bytes())))


if __name__=='__main__':main()
