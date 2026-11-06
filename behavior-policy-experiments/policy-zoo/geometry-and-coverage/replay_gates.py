"""Representative full/resumed encoder and predictor training replays."""
import argparse
import multiprocessing as mp
from pathlib import Path
import torch
import common_gc as study
import run_geometry as runner
from replay_geometry_birth import maximum_error

ACTUAL=study.OUT
TEMP=Path('/tmp/kempner-geometry-coverage-gates')


def job(gate):
    torch.set_num_threads(1)
    if gate=='factor':task=('factor','icm','late','late',200)
    elif gate=='resume':task=('zoo','icm','native','common',8)
    else:task=('zoo','rnd','reset','common',8)
    experiment,policy,a,r,seed=task
    condition=f'factor_A{a}_R{r}' if experiment=='factor' else f'zoo_{policy}_{a}'
    output=TEMP/gate;output.mkdir(parents=True,exist_ok=True);(output/'partials').mkdir(exist_ok=True)
    if not (output/'sources').exists():(output/'sources').symlink_to(ACTUAL/'sources',target_is_directory=True)
    study.OUT=output
    if gate=='resume' and not (output/f'{condition}_seed{seed}.json').exists():
        name,c,data,pool,model,opt,ia,ir,_=runner.load_task(task)
        regularizer=study.behavior.SIGReg1D();initial,_=study.original.geometry(model)
        frames=[{'step':0,**study.frame(model)}]
        named=list(model.named_parameters());parameters=[p for _,p in named]
        for step in range(500):
            opt.param_groups[0]['lr']=study.controls.learning_rate(step,c.jepa_steps)
            y=model(torch.cat([data['x'][ia[step]],data['xp'][ia[step]]]))
            la=.9*(y[:c.batch]-y[c.batch:]).square().mean();lr=.1*regularizer(model(pool[ir[step]]))
            ga=torch.autograd.grad(la,parameters);gr=torch.autograd.grad(lr,parameters)
            opt.zero_grad(set_to_none=True)
            for p,g1,g2 in zip(parameters,ga,gr):p.grad=g1+g2
            opt.step()
        metrics,_=study.original.geometry(model)
        frames.append({'step':500,**study.frame(model)})
        study.atomic_torch(output/'partials'/f'{condition}_seed{seed}.pt',{
            'signature':study.signature(),'step':500,'encoder':model.state_dict(),'optimizer':opt.state_dict(),
            'history':[{'step':0,**initial},{'step':500,'alignment_weighted':float(la.detach()),
                       'regularizer_weighted':float(lr.detach()),**metrics}],
            'frames':frames,'moments':None})
    runner.train_task(task)
    model_name=f'encoder_{condition}_seed{seed}.pt'
    expected=torch.load(ACTUAL/model_name,map_location='cpu');actual=torch.load(output/model_name,map_location='cpu')
    model_error=maximum_error(expected['encoder'],actual['encoder'])
    optimizer_error=maximum_error(expected['optimizer'],actual['optimizer'])
    assert model_error==optimizer_error==0
    prediction_errors={}
    for kind in ['shared']+(['native'] if experiment=='zoo' else []):
        filename=f'predictor_{condition}_{kind}_seed{seed}.pt'
        expected=torch.load(ACTUAL/filename,map_location='cpu');actual=torch.load(output/filename,map_location='cpu')
        prediction_errors[kind]=maximum_error(expected,actual)
        assert prediction_errors[kind]==0
    result={'gate':gate,'condition':condition,'seed':seed,'signature':study.signature(),
            'resumed_at_update':500 if gate=='resume' else 0,'model_error':model_error,
            'optimizer_error':optimizer_error,'predictor_errors':prediction_errors}
    study.save_json(ACTUAL/f'REPLAY_GATE_{gate}.json',result)
    return result


if __name__=='__main__':
    with mp.get_context('spawn').Pool(3) as pool:
        results=list(pool.imap_unordered(job,['factor','resume','zoo_full']))
    study.save_json(ACTUAL/'TRAINING_REPLAY_GATES.json',results)
    print(results)
