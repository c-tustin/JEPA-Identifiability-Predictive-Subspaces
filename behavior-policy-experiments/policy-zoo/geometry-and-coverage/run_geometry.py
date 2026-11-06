"""Loss-source factorial and common-exposure tests with original zoo policies."""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call,jvp
import common_gc as study

SNAPSHOTS=(0,100,500,1000,2000)


def prepare_zoo(task):
    policy,seed=task
    torch.set_num_threads(1)
    path=study.OUT/'sources'/f'{policy}_seed{seed}.pt'
    manifest=study.OUT/'source_manifests'/f'{policy}_seed{seed}.json'
    if path.exists() and manifest.exists():
        assert json.loads(manifest.read_text())['signature']==study.signature()
        return
    native=study.native_data(policy,seed)
    reset=study.reset_data(policy,seed,native)
    _,pool=study.reset_pool(seed)
    torch.testing.assert_close(reset['x'],pool,atol=0,rtol=0)
    study.atomic_torch(path,{'signature':study.signature(),'native':native,'reset':reset,'pool':pool})
    study.save_json(manifest,{'signature':study.signature(),'policy':policy,'seed':seed,
                              'native_sha256':study.source_hash(native),'reset_sha256':study.source_hash(reset),
                              'native_exposure':study.prior.exposure(native),'reset_exposure':study.prior.exposure(reset),
                              'reset_start_sha256':hashlib.sha256(pool.numpy().tobytes()).hexdigest()})
    print(json.dumps({'stage':'prepare_zoo','policy':policy,'seed':seed}),flush=True)


def load_task(task):
    experiment,policy,alignment,regularization,seed=task
    c=study.original.config()
    if experiment=='factor':
        name=f'factor_A{alignment}_R{regularization}'
        data=study.factor_data(seed,0 if alignment=='early' else 128)
        pool=study.pooled(study.factor_data(seed,0 if regularization=='early' else 128))
        parent_path=study.prior.OUT/f'encoder_icm_age000_seed{seed}.pt'
        parent=torch.load(parent_path,map_location='cpu')
        model=study.behavior.Encoder();model.load_state_dict(parent['encoder'])
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(parent['optimizer'])
        rng=torch.Generator().manual_seed(3100000+seed)
        ia=torch.randint(len(data['x']),(c.jepa_steps,c.batch),generator=rng)
        parent_hash=hashlib.sha256(parent_path.read_bytes()).hexdigest()
    else:
        name=f'zoo_{policy}_{alignment}'
        saved=torch.load(study.OUT/'sources'/f'{policy}_seed{seed}.pt',map_location='cpu')
        assert saved['signature']==study.signature()
        data,pool=saved[alignment],saved['pool']
        model,plan=study.original.reliability.make_plan(c,seed,seed,'icm')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        ia=plan['indices'];parent_hash=None
    rng=torch.Generator().manual_seed(3400000+seed)
    ir=torch.randint(len(pool),(c.jepa_steps,2*c.batch),generator=rng)
    return name,c,data,pool,model,optimizer,ia,ir,parent_hash


def attribute(model,optimizer,before,grad_a,grad_r,moments,lr):
    """Exact parameter split with shared Adam denominator; local function JVPs."""
    named=list(model.named_parameters())
    beta1,beta2=optimizer.param_groups[0]['betas'];eps=optimizer.param_groups[0]['eps']
    directions={key:{} for key in ('carry','alignment','regularizer')}
    parameter_error=0.;moment_error=0.
    for name,p in named:
        state=optimizer.state[p];step=int(state['step'])
        denominator=(state['exp_avg_sq']/(1-beta2**step)).sqrt()+eps
        for key in directions:
            directions[key][name]=-lr*moments[key][name]/(1-beta1**step)/denominator
        total=sum(directions[key][name] for key in directions)
        parameter_error=max(parameter_error,float((p.detach()-before[name]-total).abs().max()))
        moment_error=max(moment_error,float((sum(moments[key][name] for key in moments)-state['exp_avg']).abs().max()))
    q=torch.linspace(0,np.log(16),257)
    def evaluate(parameters):
        def forward(log_distance):
            x=study.behavior.sensor(log_distance.exp(),study.original.config())
            return functional_call(model,parameters,(x,))
        return jvp(forward,(q,),(torch.ones_like(q),))
    result={'parameter_reconstruction_error':parameter_error,'moment_reconstruction_error':moment_error,
            'directions':directions,'gradient_alignment':grad_a,'gradient_regularizer':grad_r,
            'curve_before':None,'slope_before':None,'components':{}}
    for key,direction in directions.items():
        (y,slope),(dy,ds)=jvp(evaluate,(before,),(direction,))
        actual_y,actual_slope=evaluate({name:before[name]+direction[name] for name in before})
        result['curve_before']=y.detach().numpy();result['slope_before']=slope.detach().numpy()
        result['components'][key]={'delta_curve_jvp':dy.detach().numpy(),'delta_slope_jvp':ds.detach().numpy(),
                                   'curve_after_component':actual_y.detach().numpy(),
                                   'slope_after_component':actual_slope.detach().numpy()}
    result['after']=study.frame(model)
    return result


def train_task(task):
    torch.set_num_threads(1)
    experiment,policy,alignment,regularization,seed=task
    name,c,data,pool,model,optimizer,ia,ir,parent_hash=load_task(task)
    result_path=study.OUT/f'{name}_seed{seed}.json'
    signature=study.signature()
    if result_path.exists():
        assert json.loads(result_path.read_text())['signature']==signature
        return
    detailed=experiment=='factor' and seed in study.FACTOR_SEEDS[:5]
    initial_metrics,initial_curve=study.original.geometry(model)
    history=[{'step':0,**initial_metrics}]
    frames=[{'step':0,**study.frame(model)}]
    moments=None
    if experiment=='factor':
        moments={key:{} for key in ('carry','alignment','regularizer')}
        for name_p,p in model.named_parameters():
            moments['carry'][name_p]=optimizer.state[p]['exp_avg'].clone()
            moments['alignment'][name_p]=torch.zeros_like(p)
            moments['regularizer'][name_p]=torch.zeros_like(p)
    partial=study.OUT/'partials'/f'{name}_seed{seed}.pt'
    start=0
    if partial.exists():
        old=torch.load(partial,map_location='cpu');assert old['signature']==signature
        model.load_state_dict(old['encoder']);optimizer.load_state_dict(old['optimizer'])
        start,history,frames,moments=old['step'],old['history'],old['frames'],old['moments']
    regularizer=study.behavior.SIGReg1D()
    named=list(model.named_parameters());parameters=[p for _,p in named]
    for step in range(start,c.jepa_steps):
        lr=study.controls.learning_rate(step,c.jepa_steps)
        optimizer.param_groups[0]['lr']=lr
        ix=ia[step];y=model(torch.cat([data['x'][ix],data['xp'][ix]]))
        loss_a=.9*(y[:c.batch]-y[c.batch:]).square().mean()
        loss_r=.1*regularizer(model(pool[ir[step]]))
        ga=torch.autograd.grad(loss_a,parameters)
        gr=torch.autograd.grad(loss_r,parameters)
        snapshot=detailed and step in SNAPSHOTS
        if snapshot:
            before={n:p.detach().clone() for n,p in named}
            opt_before=copy.deepcopy(optimizer.state_dict())
            moments_before=copy.deepcopy(moments)
        optimizer.zero_grad(set_to_none=True)
        beta1=optimizer.param_groups[0]['betas'][0]
        for (n,p),a,r in zip(named,ga,gr):
            p.grad=a+r
            if moments is not None:
                moments['carry'][n].mul_(beta1)
                moments['alignment'][n].mul_(beta1).add_(a,alpha=1-beta1)
                moments['regularizer'][n].mul_(beta1).add_(r,alpha=1-beta1)
        optimizer.step()
        if snapshot:
            a={n:g.detach().clone() for (n,_),g in zip(named,ga)}
            r={n:g.detach().clone() for (n,_),g in zip(named,gr)}
            attribution=attribute(model,optimizer,before,a,r,moments,lr)
            study.atomic_torch(study.OUT/'snapshots'/f'{name}_seed{seed}_update{step}.pt',{
                'signature':signature,'condition':name,'seed':seed,'update':step,'before':before,
                'optimizer_before':opt_before,'moments_before':moments_before,'moments_after':copy.deepcopy(moments),
                'alignment_indices':ia[step],'regularizer_indices':ir[step],
                'attribution':attribution,'loss_alignment_weighted':float(loss_a),'loss_regularizer_weighted':float(loss_r)})
        frame_interval=100 if experiment=='factor' else 500
        if (step+1)%frame_interval==0:
            frames.append({'step':step+1,**study.frame(model)})
        if (step+1)%500==0:
            metrics,_=study.original.geometry(model)
            history.append({'step':step+1,'alignment_weighted':float(loss_a.detach()),
                            'regularizer_weighted':float(loss_r.detach()),**metrics})
            study.atomic_torch(partial,{'signature':signature,'step':step+1,'encoder':model.state_dict(),
                'optimizer':optimizer.state_dict(),'history':history,'frames':frames,'moments':moments})
    model.eval();metrics,curve=study.original.geometry(model)
    study.atomic_torch(study.OUT/f'encoder_{name}_seed{seed}.pt',{'signature':signature,
        'encoder':model.state_dict(),'optimizer':optimizer.state_dict(),'config':vars(c)})
    result={'signature':signature,'condition':name,'experiment':experiment,'policy':policy,
            'alignment_source':alignment,'regularizer_source':regularization,'seed':seed,
            'parent_checkpoint_sha256':parent_hash,'source_hash':study.source_hash(data),
            'regularizer_pool_sha256':hashlib.sha256(pool.numpy().tobytes()).hexdigest(),
            'initial_metrics':initial_metrics,'metrics':metrics,'history':history,'planning':{}}
    arrays={**curve,'initial_curve':initial_curve['curve'],'frame_steps':np.array([f['step'] for f in frames]),
            'frame_distance':frames[0]['distance'],'frame_curves':np.stack([f['curve'] for f in frames]),
            'frame_slopes':np.stack([f['slope'] for f in frames])}
    common=study.prior.matched.shared_predictor_data(study.prior.matched.make_design(size=16384),seed)
    datasets=[('shared',common)]+([('native',data)] if experiment=='zoo' else [])
    for kind,predictor_data in datasets:
        predictor,mean,scale=study.prior.matched.planning.train_predictor(model,predictor_data,common,seed,2000)
        planning_metrics,paths=study.prior.matched.planning_evaluation(model,predictor,mean,scale,seed)
        result['planning'][kind]=planning_metrics
        arrays.update({kind+'_'+key:value for key,value in paths.items()})
        study.atomic_torch(study.OUT/f'predictor_{name}_{kind}_seed{seed}.pt',{
            'predictor':predictor.state_dict(),'mean':mean,'scale':scale,'signature':signature})
    np.savez_compressed(study.OUT/f'{name}_seed{seed}.npz',**arrays)
    study.save_json(result_path,result);partial.unlink(missing_ok=True)
    print(json.dumps({'stage':'fit','condition':name,'seed':seed,'r2':metrics['affine_log_r2'],
                      'folded':metrics['folded'],'goal_error':result['planning']['shared']['goal_error']}),flush=True)


def pool_map(function,tasks,workers):
    with mp.get_context('spawn').Pool(workers) as pool:
        list(pool.imap_unordered(function,tasks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','factor','zoo'],default='all')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();study.OUT.mkdir(parents=True,exist_ok=True)
    path=study.OUT/'MANIFEST.json'
    manifest={'signature':study.signature(),'factor_seeds':study.FACTOR_SEEDS,'zoo_seeds':study.ZOO_SEEDS,
              'zoo_policies':study.ZOO_POLICIES,'planned_tests':54,'config':vars(study.original.config())}
    if path.exists():assert json.loads(path.read_text())['signature']==manifest['signature']
    else:study.save_json(path,manifest)
    if args.stage in ('all','factor'):
        tasks=[('factor','icm',a,r,seed) for seed in study.FACTOR_SEEDS for a in ('early','late') for r in ('early','late')]
        pool_map(train_task,tasks,args.workers)
    if args.stage in ('all','zoo'):
        pool_map(prepare_zoo,[(p,s) for s in study.ZOO_SEEDS for p in study.ZOO_POLICIES],args.workers)
        tasks=[('zoo',p,a,'common',s) for s in study.ZOO_SEEDS for p in study.ZOO_POLICIES for a in ('native','reset')]
        pool_map(train_task,tasks,args.workers)
    if args.stage=='all':study.save_json(study.OUT/'TRAINING_COMPLETE.json',{'signature':study.signature(),'encoders':360,'predictors':640})


if __name__=='__main__':main()
