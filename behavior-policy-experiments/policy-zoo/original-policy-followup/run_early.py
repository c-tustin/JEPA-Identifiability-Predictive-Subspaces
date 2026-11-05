"""Change collection age, keeping original frozen behavior actors unchanged."""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import original as o
import run_original as runner

HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
NAMES=('icm','random','count')


def signature():
    digest=hashlib.sha256()
    for name in ['run_early.py','EARLY_COLLECTION_PROTOCOL.md','original.py']:
        digest.update((HERE/name).read_bytes())
    return digest.hexdigest()


def actor_for(name,seed):
    if name=='random':return None
    for root in o.run_zoo.EXISTING_CACHES:
        path=root/'sources'/f'{name}_seed{seed}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu')['policy']
            return o.learned.load(saved)
    return o.learned.load(torch.load(o.ZOO/'output/policies'/f'{name}_seed{seed}.pt',map_location='cpu'))


def exposure(data):
    distance=1/torch.cat([data['x'][:,1],data['xp'][:,1]])
    bins=(torch.log(distance)/np.log(16)*32).long().clamp(0,31)
    p=torch.bincount(bins,minlength=32).double();p=p/p.sum()
    return {'mass_below_2':float((distance<2).double().mean()),
            'mass_below_4':float((distance<4).double().mean()),
            'occupied_log_bins':int((p>0).sum()),
            'normalized_log_bin_entropy':float(-(p[p>0]*p[p>0].log()).sum()/np.log(32)),
            'log_bin_probability':p.tolist()}


def job(task):
    name,seed=task
    torch.set_num_threads(1)
    path=OUT/f'early-{name}_seed{seed}.json'
    if path.exists():
        assert json.loads(path.read_text())['code_hash']==signature()
        return
    c=o.config()
    original,source_hash=runner.source_cached(name,seed,OUT)
    collection=o.config();collection.burnin=0
    actor=actor_for(name,seed)
    data=o.behavior.collect(collection,'random' if name=='random' else 'icm',c.train_episodes,10000+seed,actor)
    assert len(data['x'])==len(original['x'])==16384
    # Frozen-actor identity is retained; no training of exploration models occurs.
    early_hash=o.run_zoo.diagnostics(data,c)['sha256']
    model,plan=o.reliability.make_plan(c,seed,seed,'icm')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    regularizer=o.behavior.SIGReg1D()
    partial=OUT/'partials'/f'early-{name}-{seed}.pt'
    start=0
    if partial.exists():
        saved=torch.load(partial,map_location='cpu');assert saved['code_hash']==signature()
        model.load_state_dict(saved['encoder']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
    for step in range(start,c.jepa_steps):
        ix=plan['indices'][step]
        o.controls.update(model,optimizer,regularizer,data['x'][ix],data['xp'][ix],c,o.controls.learning_rate(step,c.jepa_steps))
        if (step+1)%1000==0:
            temporary=partial.with_suffix('.tmp')
            torch.save({'encoder':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'code_hash':signature()},temporary)
            temporary.replace(partial)
    model.eval()
    metrics,curve=o.geometry(model)
    result={'policy':name,'seed':seed,'code_hash':signature(),'original_source_hash':source_hash,
            'early_source_hash':early_hash,'original_exposure':exposure(original),'early_exposure':exposure(data),
            'metrics':metrics,'planning':{}}
    torch.save({'encoder':model.state_dict(),'config':vars(c),'code_hash':signature()},path.with_suffix('.pt'))
    arrays=dict(curve)
    if seed<18:
        common=o.matched.shared_predictor_data(o.matched.make_design(size=16384),seed)
        for kind,predictor_data in [('shared',common),('native',data)]:
            predictor,mean,scale=o.matched.planning.train_predictor(model,predictor_data,common,seed,2000)
            planning_metrics,trajectory=o.matched.planning_evaluation(model,predictor,mean,scale,seed)
            result['planning'][kind]=planning_metrics
            arrays.update({kind+'_'+key:value for key,value in trajectory.items()})
            torch.save({'predictor':predictor.state_dict(),'mean':mean,'scale':scale},path.with_name(path.stem+'_'+kind+'.pt'))
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    o.save_json(path,result)
    partial.unlink(missing_ok=True)
    print(json.dumps({'stage':'early','policy':name,'seed':seed,'r2':metrics['affine_log_r2'],
                      'folded':metrics['folded'],'near_mass_old':result['original_exposure']['mass_below_2'],
                      'near_mass_early':result['early_exposure']['mass_below_2']}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with mp.get_context('spawn').Pool(6) as pool:
        list(pool.imap_unordered(job,[(name,seed) for seed in o.SEEDS for name in NAMES]))
