"""Train identical restricted critics on fixed folded/ordered candidates."""
import argparse
import copy
import json
from pathlib import Path
import time
import dose as D

S,R,torch,np,B=D.S,D.R,D.torch,D.np,D.B
OUT=S.OUT/'critic-probe'
MASSES=(.5,.75,.9375)
SEEDS=(6100131,6100132,6100133)


def evaluate(critic,raw,code,ids):
    vals=[]
    with torch.no_grad():
        for batch in ids:vals.append(float(D.N.objective(raw[batch],code[batch],critic)))
    return dict(mean=float(np.mean(vals)),std=float(np.std(vals)),batches=len(vals))


def fit(template,trained_mass,alpha,seed,ds,c,mean,std,u,test_u):
    name=f's{template}-encoder{trained_mass:g}-target{alpha:g}-critic{seed}'
    path=OUT/'runs'/f'{name}.pt'
    enc_path=D.OUT/'runs'/f'n{template}-f{template}-a{trained_mass:g}-sigreg.pt'
    enc=R.trials.load(enc_path);m=B.Encoder();m.load_state_dict(enc['model'])
    x=torch.cat([ds[template]['near'],ds[template]['far']]);n=len(ds[template]['near'])
    with torch.no_grad():z=m(x)
    # Exact target-distribution affine conditioning preserves code identity.
    zmean=alpha*z[:n].mean()+(1-alpha)*z[n:].mean()
    second=alpha*z[:n].square().mean()+(1-alpha)*z[n:].square().mean()
    zstd=(second-zmean.square()).clamp_min(1e-12).sqrt()
    code=((z-zmean)/zstd).detach();raw=((x-mean)/std).detach()
    ids=S.indices(alpha,n,len(x)-n,u);test_ids=S.indices(alpha,n,len(x)-n,test_u)
    torch.manual_seed(seed);critic=D.N.critic();initial=copy.deepcopy(critic.state_dict())
    identity=dict(template=template,encoder_trained_mass=trained_mass,target_mass=alpha,critic_seed=seed,
        encoder_checkpoint_hash=R.digest(enc_path.read_bytes()),source_hash=ds[template]['source_hash'],
        initial_critic_hash=R.state_hash(initial),plan_hash=R.digest(ids.numpy().tobytes()),
        test_plan_hash=R.digest(test_ids.numpy().tobytes()),code_hash=R.digest(Path(__file__).read_bytes()),
        protocol_hash=R.digest((S.HERE/'MECHANISM_PROTOCOL.md').read_bytes()))
    if path.exists():
        assert R.trials.load(path)['identity']==identity
        return
    opt=torch.optim.Adam(critic.parameters(),lr=.001);frames=[];started=time.monotonic()
    frames.append(dict(step=0,validation=evaluate(critic,raw,code,test_ids[:8])))
    for step in range(4000):
        opt.param_groups[0]['lr']=R.trials.study.controls.learning_rate(step,4000)
        batch=ids[step];loss=D.N.objective(raw[batch],code[batch],critic)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if (step+1)%500==0:frames.append(dict(step=step+1,training_loss=float(loss.detach()),
                            validation=evaluate(critic,raw,code,test_ids[:8])))
    result=dict(identity=identity,critic=critic.state_dict(),initial_critic=initial,
        optimizer=opt.state_dict(),frames=frames,validation=evaluate(critic,raw,code,test_ids),
        code_mean=zmean,code_std=zstd,raw_mean=mean,raw_std=std,
        encoder_geometry=enc['geometry'],seconds=time.monotonic()-started)
    R.trials.save_torch(path,result)
    check=D.N.critic();check.load_state_dict(R.trials.load(path)['critic'])
    assert evaluate(check,raw,code,test_ids)==result['validation']
    R.trials.atomic_json(OUT/'cells'/f'{name}.json',dict(name=name,**identity,
                         validation=result['validation'],frames=frames,encoder_geometry=enc['geometry']))
    print(json.dumps(dict(name=name,loss=result['validation']['mean'],seconds=result['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);ds,c,_,_,_,mean,std,_=D.setup()
    u=torch.rand((4000,256),dtype=torch.float64,generator=torch.Generator().manual_seed(8600131))
    test_u=torch.rand((64,256),dtype=torch.float64,generator=torch.Generator().manual_seed(8700131))
    jobs=[(t,m,alpha,seed) for t in (130,136) for alpha in MASSES for m in MASSES for seed in SEEDS]
    for k,job in enumerate(jobs):
        if k%a.shards==a.shard:fit(*job,ds,c,mean,std,u,test_u)


if __name__=='__main__':main()
