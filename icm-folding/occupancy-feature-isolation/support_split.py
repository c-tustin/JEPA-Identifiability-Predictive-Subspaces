"""Alternative regional decompositions and a strictly common endpoint pool."""
import argparse
import itertools
import json
from pathlib import Path
import time
import dose as D

S,R,torch,np,B=D.S,D.R,D.torch,D.np,D.B
OUT=S.OUT/'split-support'


def own_hashes():
    return dict(wrapper_hash=R.digest(Path(__file__).read_bytes()),
                governing_protocol_hash=R.digest((S.HERE/'SPLIT_SUPPORT_PROTOCOL.md').read_bytes()))


def common_pool(ds):
    counts=[];parts=[]
    for seed in range(130,140):
        x=ds[seed]['x'];d=ds[seed]['d'];x=x[d<=10];d=d[d<=10]
        bins=(d.double().log()/np.log(16)*32).long().clamp(0,31)
        assert torch.equal(bins<8,d<2)
        counts.append(torch.bincount(bins,minlength=32));parts.append((x,bins))
    occupied=torch.stack(counts).min(0).values>0
    xs=[];bs=[]
    for x,b in parts:xs.append(x[occupied[b]]);bs.append(b[occupied[b]])
    x=torch.cat(xs);bins=torch.cat(bs);order=bins.argsort(stable=True);x,bins=x[order],bins[order]
    pooled_counts=torch.bincount(bins,minlength=32);profiles={}
    for seed,count in zip(range(130,140),counts):
        pn=count.double()*occupied*(torch.arange(32)<8);pf=count.double()*occupied*(torch.arange(32)>=8)
        profiles[seed]=(pn/pn.sum(),pf/pf.sum())
    return x,bins,pooled_counts,profiles


def common_fit(pair,bits,ds,c,initial,ix,u,pool):
    x,bins,counts,profiles=pool;ms,ns,fs=[pair[b] for b in bits];alpha=ds[ms]['mass']
    target=alpha*profiles[ns][0]+(1-alpha)*profiles[fs][1]
    point_weights=target[bins]/counts[bins];assert torch.all(point_weights>0)
    assert abs(float(point_weights.sum())-1)<1e-12 and abs(float(target[:8].sum())-alpha)<1e-12
    cdf=point_weights.cumsum(0);cdf[-1]=1
    ids=torch.searchsorted(cdf,u.flatten()).reshape(u.shape).clamp_max(len(x)-1)
    name=f'{pair[0]}-{pair[1]}-m{bits[0]}n{bits[1]}f{bits[2]}'
    path=OUT/'common/runs'/f'{name}.pt'
    identity=dict(pair=pair,bits=bits,alpha=alpha,**own_hashes(),dependencies=D.dependencies(),
        initial_hash=R.state_hash(initial),alignment_plan_hash=R.digest(ix.numpy().tobytes()),
        uniform_hash=R.digest(u.numpy().tobytes()),index_hash=R.digest(ids.numpy().tobytes()),
        pool_hash=R.digest(x.numpy().tobytes()),target_bin_mass=target.tolist(),
        source_hashes={s:ds[s]['source_hash'] for s in ds})
    if path.exists():assert R.trials.load(path)['identity']==identity;return
    m=B.Encoder();m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);reg=B.SIGReg1D()
    data=ds[131]['data'];frames=[dict(step=0,**R.geometry(m,c,1025))];start=time.monotonic()
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        ya=m(torch.cat([data['x'][ix[step]],data['xp'][ix[step]]]))
        loss=.9*(ya[:512]-ya[512:]).square().mean()+.1*reg(m(x[ids[step]]))
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%100==0:frames.append(dict(step=step+1,**R.geometry(m,c,1025)))
    realized=torch.bincount(bins[ids.flatten()],minlength=32).double()/ids.numel()
    saved=dict(identity=identity,initial=initial,model=m.state_dict(),optimizer=opt.state_dict(),config=vars(c),
        frames=frames,realized_bins=realized.tolist(),target_bins=target.tolist(),
        support=[float((1/x[:,1]).min()),float((1/x[:,1]).max())],seconds=time.monotonic()-start)
    R.trials.save_torch(path,saved);check=B.Encoder();check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry']=R.geometry(check,c,20001);R.trials.save_torch(path,saved)
    R.trials.atomic_json(OUT/'common/cells'/f'{name}.json',dict(name=name,**identity,**saved['geometry']))
    print(json.dumps(dict(name='common/'+name,**saved['geometry'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['split','common'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,initial,ix,_,mean,std,critic=D.setup()
    u=torch.rand((4000,1024),dtype=torch.float64,generator=torch.Generator().manual_seed(9200131))
    if a.mode=='split':
        jobs=[(cut,pair,bits) for cut in [1.5,2,2.5,3] for pair in [(130,131),(137,136)] for bits in itertools.product([0,1],repeat=3)]
        for k,(cut,pair,bits) in enumerate(jobs):
            if k%a.shards!=a.shard:continue
            changed={s:dict(v,near=v['x'][v['d']<cut],far=v['x'][v['d']>=cut],mass=float((v['d']<cut).double().mean())) for s,v in ds.items()}
            ms,ns,fs=[pair[b] for b in bits];alpha=changed[ms]['mass']
            D.OUT=OUT/f'split{cut:g}'/f'pair{pair[0]}-{pair[1]}'
            D.fit(ns,fs,alpha,'sigreg',changed,c,initial,ix,u,mean,std,critic)
            name=f'n{ns}-f{fs}-a{alpha:g}-sigreg'
            R.trials.atomic_json(D.OUT/'scope'/f'{name}.json',dict(cut=cut,pair=pair,bits=bits,**own_hashes(),
                checkpoint_hash=R.digest((D.OUT/'runs'/f'{name}.pt').read_bytes())))
    else:
        pool=common_pool(ds);jobs=list(itertools.product(S.PAIRS,itertools.product([0,1],repeat=3)))
        for k,(pair,bits) in enumerate(jobs):
            if k%a.shards==a.shard:common_fit(pair,bits,ds,c,initial,ix,u,pool)


if __name__=='__main__':main()
