"""Audit every fresh condition against stored sources, models and actual rollouts."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import torch

import common as study
import replication_stats as stats


def assert_metrics(actual, expected, label):
    for key,value in actual.items():
        if isinstance(value,(list,float,int,bool)):
            if not np.allclose(value,expected[key],atol=1e-11,rtol=0):
                raise AssertionError((label,key,value,expected[key]))


def verify_source(task):
    policy,seed=task
    torch.set_num_threads(1)
    manifest=json.loads((study.OUT/'source_manifests'/f'{policy}_seed{seed}.json').read_text())
    assert manifest['signature']==study.signature()
    actor=None
    if policy!='random':
        saved=torch.load(study.OUT/'actors'/f'{policy}_seed{seed}.pt',map_location='cpu')
        assert saved['signature']==study.signature()
        actor=study.learned.load(saved['actor'])
        assert study.tensor_hash(actor.state_dict())==manifest['actor_hash']
    c=study.original.config()
    c.burnin,c.horizon=0,192
    full=study.behavior.collect(c,'random' if policy=='random' else 'icm',c.train_episodes,10000+seed,actor)
    cached=torch.load(study.OUT/'sources'/f'{policy}_seed{seed}.pt',map_location='cpu')['full']
    for key in full:
        torch.testing.assert_close(full[key],cached[key],atol=0,rtol=0)
    torch.testing.assert_close(full['xp'][:-256],full['x'][256:],atol=0,rtol=0)
    for age in study.AGES:
        data=study.extract_window(full,age)
        assert study.data_hash(data)==manifest['windows'][str(age)]['sha256']
        assert_metrics(study.exposure(data),manifest['windows'][str(age)]['exposure'],(policy,seed,age))
    torch.manual_seed(10000+seed)
    initial=study.behavior.initial(c.train_episodes,c)
    torch.testing.assert_close(full['x'][:256],study.behavior.sensor(initial,c),atol=0,rtol=0)
    if actor is not None:
        assert study.tensor_hash(actor.state_dict())==manifest['actor_hash']
    return {'policy':policy,'seed':seed,'windows':5}


def verify_condition(path):
    torch.set_num_threads(1)
    row=json.loads(path.read_text())
    condition,seed=row['condition'],row['seed']
    assert row['signature']==study.signature()
    saved=torch.load(study.OUT/f'encoder_{condition}_seed{seed}.pt',map_location='cpu')
    assert saved['signature']==row['signature']
    model=study.behavior.Encoder()
    model.load_state_dict(saved['encoder'])
    model.eval()
    assert study.tensor_hash(model.state_dict())==row['encoder_hash']
    metrics,curve=study.original.geometry(model)
    assert_metrics(metrics,row['metrics'],(condition,seed))
    arrays=np.load(study.OUT/f'{condition}_seed{seed}.npz')
    np.testing.assert_array_equal(curve['curve'],arrays['curve'])
    source=study.source(row['policy'],row['age'],seed)
    assert study.data_hash(source)==row['source_hash']
    if row['continuation']:
        parent_path=study.OUT/f'encoder_icm_age000_seed{seed}.pt'
        assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==row['parent_checkpoint_sha256']
        parent=torch.load(parent_path,map_location='cpu')
        initial=study.behavior.Encoder()
        initial.load_state_dict(parent['encoder'])
    else:
        torch.manual_seed(40000+seed)
        initial=study.behavior.Encoder()
    initial_metrics,initial_curve=study.original.geometry(initial)
    assert_metrics(initial_metrics,row['initial_metrics'],(condition,seed,'initial'))
    np.testing.assert_array_equal(initial_curve['curve'],arrays['initial_curve'])
    for kind,expected in row['planning'].items():
        state=torch.load(study.OUT/f'predictor_{condition}_{kind}_seed{seed}.pt',map_location='cpu')
        assert state['encoder_hash']==row['encoder_hash']
        predictor=study.matched.planning.Predictor()
        predictor.load_state_dict(state['predictor'])
        predictor.eval()
        actual,paths=study.matched.planning_evaluation(model,predictor,state['mean'],state['scale'],seed)
        assert_metrics(actual,expected,(condition,seed,kind))
        for key,value in paths.items():
            np.testing.assert_array_equal(value,arrays[kind+'_'+key])
    assert len(row['planning'])==(1 if row['policy']=='ou' else 2)
    return len(row['planning'])


def verify_statistics(rows,require_complete):
    actual=stats.comparisons(rows,study.SEEDS,study.AGES)
    expected=json.loads((study.OUT/'INFERENCE.json').read_text())
    assert actual==expected
    if require_complete:
        assert actual['complete_seeds']==list(study.SEEDS)
        assert len(actual['tests'])==19
    ps=sorted(r['p'] for r in actual['tests'])
    for row in actual['tests']:
        n=row['wins']+row['losses']
        p=binomtest(row['wins'],n,.5).pvalue if n else 1.
        assert abs(p-row['p'])<1e-14
        index=max(i for i,value in enumerate(ps) if value<=p)
        corrected=min(1.,max((len(ps)-i)*value for i,value in enumerate(ps[:index+1])))
        assert abs(corrected-row['holm_p'])<1e-14
    return len(actual['tests'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    sources=[(policy,seed) for seed in study.SEEDS for policy in study.POLICIES
             if (study.OUT/'source_manifests'/f'{policy}_seed{seed}.json').exists()]
    paths=sorted(study.OUT.glob('*_seed*.json'))
    paths=[p for p in paths if 'condition' in json.loads(p.read_text())]
    if args.require_complete:
        assert len(paths)==360
        assert len(sources)==60
        expected={(study.label(policy,age),seed) for policy in study.POLICIES for age in study.AGES for seed in study.SEEDS}
        expected|={('ou',seed) for seed in study.SEEDS}
        expected|={(name,seed) for name in ('icm_early_to_early','icm_early_to_late') for seed in study.SEEDS}
        actual={(r['condition'],r['seed']) for r in [json.loads(p.read_text()) for p in paths]}
        assert actual==expected
    with mp.get_context('spawn').Pool(args.workers) as pool:
        source_results=list(pool.imap_unordered(verify_source,sources))
        planning_counts=list(pool.imap_unordered(verify_condition,paths))
    rows=[json.loads(p.read_text()) for p in paths]
    tests=verify_statistics(rows,args.require_complete)
    # Both continuation arms must attest to exactly the same saved encoder/Adam parent.
    lookup={(r['condition'],r['seed']):r for r in rows}
    for seed in study.SEEDS:
        a=lookup.get(('icm_early_to_early',seed))
        b=lookup.get(('icm_early_to_late',seed))
        if a and b:
            assert a['parent_checkpoint_sha256']==b['parent_checkpoint_sha256']
    result={'signature':study.signature(),'source_rollouts_recreated':len(source_results),
            'source_windows_verified':5*len(source_results),'geometry_replays':len(paths),
            'physical_planning_replays':sum(planning_counts),'planned_tests_checked':tests,
            'complete':args.require_complete,'errors':0}
    if args.require_complete:
        assert result['physical_planning_replays']==700
    study.save_json(study.OUT/('VERIFICATION.json' if args.require_complete else 'PARTIAL_VERIFICATION.json'),result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
