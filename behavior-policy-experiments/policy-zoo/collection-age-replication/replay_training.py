"""Full-fit and interrupted-fit replay gates on prespecified fresh seed 200."""
import argparse
import copy
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import torch
import common as study
import run_replication as runner

ACTUAL = study.OUT
TEMP = Path('/tmp/kempner-collection-age-training-replays')


def nested_error(actual, replay):
    if isinstance(actual, torch.Tensor):
        assert actual.shape == replay.shape
        return float((actual-replay).abs().max()) if actual.numel() else 0.
    if isinstance(actual, dict):
        assert actual.keys() == replay.keys()
        return max([nested_error(actual[k],replay[k]) for k in actual] or [0.])
    if isinstance(actual, (list,tuple)):
        assert len(actual) == len(replay)
        return max([nested_error(a,b) for a,b in zip(actual,replay)] or [0.])
    assert actual == replay
    return 0.


def job(kind):
    torch.set_num_threads(1)
    seed = 200
    if kind == 'normal':
        policy,age,continuation = 'icm',0,None
    elif kind == 'resume':
        policy,age,continuation = 'random',0,None
    elif kind == 'ou':
        policy,age,continuation = 'ou',-1,None
    elif kind == 'continue_early':
        policy,age,continuation = 'icm',0,'icm_early_to_early'
    else:
        policy,age,continuation = 'icm',128,'icm_early_to_late'
    condition = continuation or study.label(policy,age)
    expected_path = ACTUAL/f'encoder_{condition}_seed{seed}.pt'
    if not expected_path.exists():
        raise RuntimeError('Reference condition is not finished yet: '+condition)
    output = TEMP/kind
    output.mkdir(parents=True,exist_ok=True)
    (output/'partials').mkdir(exist_ok=True)
    if not (output/'sources').exists():
        (output/'sources').symlink_to(ACTUAL/'sources',target_is_directory=True)
    if continuation:
        parent = ACTUAL/f'encoder_icm_age000_seed{seed}.pt'
        shutil.copyfile(parent,output/parent.name)
    study.OUT = output
    if kind == 'resume' and not study.result_path(condition,seed).exists():
        c = study.original.config()
        data = study.source(policy,age,seed)
        model,plan = study.original.reliability.make_plan(c,seed,seed,'icm')
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        regularizer = study.behavior.SIGReg1D()
        initial,_ = study.original.geometry(model)
        for step in range(1000):
            ix = plan['indices'][step]
            losses = study.controls.update(model,optimizer,regularizer,data['x'][ix],data['xp'][ix],c,
                                           study.controls.learning_rate(step,c.jepa_steps))
        metrics,_ = study.original.geometry(model)
        study.atomic_torch(output/'partials'/f'{condition}_seed{seed}.pt',{
            'signature':study.signature(),'source_hash':study.data_hash(data),'step':1000,
            'history':[{'step':0,**initial},{'step':1000,**losses,**metrics}],
            'encoder':model.state_dict(),'optimizer':optimizer.state_dict(),
        })
    runner.train_condition((policy,age,seed,continuation))
    expected = torch.load(expected_path,map_location='cpu')
    replay = torch.load(output/expected_path.name,map_location='cpu')
    model_error = nested_error(expected['encoder'],replay['encoder'])
    optimizer_error = nested_error(expected['optimizer'],replay['optimizer'])
    assert model_error == optimizer_error == 0
    predictor_errors = {}
    for data_kind in ['shared'] + ([] if policy=='ou' else ['native']):
        name = f'predictor_{condition}_{data_kind}_seed{seed}.pt'
        a = torch.load(ACTUAL/name,map_location='cpu')
        b = torch.load(output/name,map_location='cpu')
        predictor_errors[data_kind] = nested_error(a,b)
        assert predictor_errors[data_kind] == 0
    result = {'gate':kind,'condition':condition,'seed':seed,'signature':study.signature(),
              'model_parameter_error':model_error,'optimizer_state_error':optimizer_error,
              'predictor_errors':predictor_errors,'resumed_at_update':1000 if kind=='resume' else 0}
    study.original.save_json(ACTUAL/f'REPLAY_{kind}.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gates',nargs='+',choices=['normal','resume','ou','continue_early','continue_late'],
                        default=['normal','resume','ou','continue_early','continue_late'])
    args=parser.parse_args()
    with mp.get_context('spawn').Pool(min(3,len(args.gates))) as pool:
        results=list(pool.imap_unordered(job,args.gates))
    print(json.dumps(results))


if __name__=='__main__':
    main()
