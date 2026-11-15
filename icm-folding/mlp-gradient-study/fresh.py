"""Fixed-hyperparameter check on five freshly trained ICM trajectory sources."""
import argparse
from dataclasses import asdict
import time
import torch
from run import B,LEGACY,OUT,trials,fit
from hard_mining import fit as mining_fit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(1)
    c=B.Config(**trials.load(LEGACY/'output/sources/icm-130.pt')['config'])
    for i,seed in enumerate(range(140,145)):
        if i%args.shards!=args.shard:continue
        path=OUT/'fresh-sources'/f'icm-{seed}.pt'
        if not path.exists():
            start=time.perf_counter()
            policy,icm,history,movement=B.train_policy(c,seed)
            data=B.collect(c,'icm',c.train_episodes,10000+seed,policy)
            trials.save_torch(path,dict(config=asdict(c),policy=policy.state_dict(),icm=icm.state_dict(),
                history=history,parameter_change=movement,data=data))
            print('fresh source',seed,'seconds',round(time.perf_counter()-start,1),flush=True)
        for method in ('sigreg','vicreg','infonce','quantile'):
            fit(seed,method,0.)
        mining_fit(seed,100.)
