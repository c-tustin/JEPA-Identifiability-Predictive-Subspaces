"""Global observed-transition collision mining, with unchanged scalar encoder.

Follow-up motivated by early main-study fits: batch-local collision gradients
barely change the folded SIGReg examples. No physical ordering enters mining.
"""
import argparse
import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from run import source,standardized_successors,B,OUT,trials,sha,geometry,downstream,FRAMES


def mine(model,train,future):
    with torch.no_grad():
        code=model(train['x'])
        scale=code.std(unbiased=False).clamp_min(.05)
        normalized=code/scale
        left=[];right=[]
        for action in range(3):
            ids=torch.nonzero(train['action']==action).flatten()
            ids=ids[normalized[ids].argsort()]
            for gap in (1,2,4,8,16,32):
                if len(ids)>gap:
                    left.append(ids[:-gap]);right.append(ids[gap:])
        i=torch.cat(left);j=torch.cat(right)
        target=.5*(future[i]-future[j]).square().mean(1).sqrt().clamp_max(2)
        violation=target-(normalized[i]-normalized[j]).abs()
        selected=violation.topk(min(512,len(violation))).indices
        return i[selected],j[selected],target[selected],scale


def fit(seed,coefficient):
    path=OUT/'hard-mining'/f'{seed}-c{coefficient:g}.pt'
    if path.exists():return
    c,train,test,source_hash=source(seed)
    torch.manual_seed(40000+seed)
    model=B.Encoder();initial=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);reg=B.SIGReg1D()
    generator=torch.Generator().manual_seed(2100000+seed)
    plan=torch.randint(len(train['x']),(4000,512),generator=generator)
    mg=torch.Generator().manual_seed(7300000+seed)
    sampled=torch.randint(512,(4000,64),generator=mg)
    future,_,_=standardized_successors(train)
    frames=[dict(step=0,**geometry(model,c))]
    begun=time.perf_counter()
    for step in range(4000):
        if step%50==0:
            ii,jj,gap,scale=mine(model,train,future)
        phase=max(0,(step-2000)/1999)
        optimizer.param_groups[0]['lr']=.001*.5*(1+np.cos(np.pi*phase))
        ix=plan[step]
        code=model(torch.cat([train['x'][ix],train['xp'][ix]]))
        base=.9*(code[:512]-code[512:]).square().mean()+.1*reg(code)
        choice=sampled[step]
        a,b=ii[choice],jj[choice]
        zz=model(torch.cat([train['x'][a],train['x'][b]]))/scale
        penalty=F.relu(gap[choice]-(zz[:64]-zz[64:]).abs()).square().mean()
        loss=base+coefficient*penalty
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step+1 in FRAMES:
            frames.append(dict(step=step+1,base_loss=float(base.detach()),
                collision_loss=float(penalty.detach()),**geometry(model,c)))
    final=geometry(model,c,points=10001)
    prediction=downstream(model,train,test,c,seed)
    trials.save_torch(path,dict(seed=seed,coefficient=coefficient,source_hash=source_hash,
        initial=initial,model=model.state_dict(),plan_hash=sha(plan),frames=frames,
        geometry=final,prediction=prediction,train_keys=sorted(train),
        training_seconds=time.perf_counter()-begun))
    print('hard mining',seed,coefficient,'fold',final['folded'],'backward',round(final['backward_fraction'],4),
        'prediction',round(prediction['native_mse'],5),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--coefficient',type=float)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();torch.set_num_threads(1)
    jobs=[(s,c) for s in ([args.seed] if args.seed is not None else range(130,140))
          for c in ((args.coefficient,) if args.coefficient is not None else (1.,10.))]
    for i,job in enumerate(jobs):
        if i%args.shards==args.shard:fit(*job)
