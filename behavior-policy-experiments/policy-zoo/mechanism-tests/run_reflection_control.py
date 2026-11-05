"""Hold every SIGReg batch fixed; remove reflection vs random alignment."""

import argparse
import hashlib
import json
import multiprocessing as mp
import torch

from run_crossed import HERE,LEVELS,OUT,EVAL_SEED,behavior,controls,reliability,load,save,sha,geometry,reference


def train(args):
    data_seed,init_seed,arm=args
    torch.set_num_threads(1)
    path=OUT/'reflection-controls'/f'{arm}-data{data_seed}-init{init_seed}.pt'
    if path.exists():return path.name
    c=behavior.Config()
    source=load(OUT/'sources'/f'icm-{data_seed}.pt')
    reflected=(source['action']==0)&(source['x'][:,1]>1/1.25)
    remove=torch.zeros_like(reflected)
    if arm=='remove_reflection':remove=reflected
    if arm=='remove_random':
        permutation=torch.randperm(len(remove),generator=torch.Generator().manual_seed(2400000+data_seed))
        remove[permutation[:int(reflected.sum())]]=True
    if arm!='all_pairs':assert int(remove.sum())==int(reflected.sum())
    model,plan=reliability.make_plan(c,data_seed,init_seed,'icm')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    regularizer=behavior.SIGReg1D()
    partial=OUT/'partials'/path.name
    start=0
    if partial.exists():
        saved=load(partial);model.load_state_dict(saved['encoder']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
    for step in range(start,c.jepa_steps):
        ix=plan['indices'][step]
        y=model(torch.cat([source['x'][ix],source['xp'][ix]]))
        alignment=((y[:512]-y[512:]).square()*(~remove[ix]).float()).mean()
        penalty=regularizer(y)
        loss=.9*alignment+.1*penalty
        optimizer.param_groups[0]['lr']=controls.learning_rate(step,c.jepa_steps)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%1000==0:save(partial,dict(encoder=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1))
    metric,_=controls.evaluate(model,reference(c,EVAL_SEED),c,EVAL_SEED)
    result=dict(data_seed=data_seed,init_seed=init_seed,arm=arm,encoder=model.state_dict(),metrics=metric,final=geometry(model,c),
                removed_pairs=int(remove.sum()),removed_fraction=float(remove.float().mean()),
                mask_sha256=hashlib.sha256(remove.numpy().tobytes()).hexdigest())
    if arm=='all_pairs':
        old=load(OUT/'crossed'/f'icm-data{data_seed}-init{init_seed}.pt')
        error=max(float((v-old['encoder'][k]).abs().max()) for k,v in model.state_dict().items())
        assert error==0.,error
        result['max_reproduction_error']=error
    save(path,result)
    print(arm,data_seed,init_seed,metric['common_grid_r2'],flush=True)
    return path.name


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    torch.set_num_threads(1)
    settings=dict(code_sha256=sha(HERE/'run_reflection_control.py'),protocol_sha256=sha(HERE/'REFLECTION_PROTOCOL.md'),levels=list(LEVELS))
    path=OUT/'reflection-control-settings.json'
    if path.exists():assert json.loads(path.read_text())==settings
    else:path.write_text(json.dumps(settings,indent=2)+'\n')
    train((8,8,'all_pairs'))
    jobs=[(d,i,arm) for d in LEVELS for i in LEVELS for arm in ('remove_reflection','remove_random')]
    with mp.get_context('spawn').Pool(args.workers) as pool:list(pool.imap_unordered(train,jobs))


if __name__=='__main__':main()
