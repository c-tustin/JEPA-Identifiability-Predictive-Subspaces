"""Observation-space prediction reference, trained without any state labels."""
from run import B, OUT, source, downstream, trials
import torch

torch.set_num_threads(1)
for seed in range(130,140):
    path=OUT/'references'/f'{seed}.pt'
    if path.exists():
        continue
    c,train,test,source_hash=source(seed)
    torch.manual_seed(40000+seed)
    result=downstream(B.Encoder(),train,test,c,seed,raw=True)
    trials.save_torch(path,dict(seed=seed,source_hash=source_hash,**result))
    print(seed,'raw-input heldout',result['native_mse'],'common',result['common_mse'],flush=True)
