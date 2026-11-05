"""Replay saved geometry/planning, check source hashes and exact-test p-values."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import torch
import original as o

HERE=Path(__file__).resolve().parent
OUT=HERE/'output'


def close(actual,expected,label):
    for key,value in actual.items():
        if isinstance(value,(list,float,int,bool)):
            assert np.allclose(value,expected[key],rtol=0,atol=1e-11), (label,key,value,expected[key])


def geometry_job(row):
    torch.set_num_threads(1)
    model=o.load_encoder(row['policy'],row['seed'],row['protocol'])
    metrics,_=o.geometry(model)
    close(metrics,row,(row['policy'],row['seed'],row['protocol']))
    return 1


def intervention_job(path):
    torch.set_num_threads(1)
    row=json.loads(path.read_text())
    model=o.behavior.Encoder()
    model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cpu')['encoder'])
    model.eval()
    metrics,curve=o.geometry(model)
    close(metrics,row['metrics'],str(path))
    arrays=np.load(path.with_suffix('.npz'))
    np.testing.assert_array_equal(curve['curve'],arrays['curve'])
    assert row['source_hash']==o.raw_record(row['policy'],row['seed'])['source_stats']['sha256']
    return 1


def planning_job(task):
    path,early=task
    torch.set_num_threads(1)
    row=json.loads(path.read_text())
    if early:
        model=o.behavior.Encoder()
        model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cpu')['encoder'])
        model.eval()
        metrics,_=o.geometry(model)
        close(metrics,row['metrics'],str(path))
        assert row['original_source_hash']==o.raw_record(row['policy'],row['seed'])['source_stats']['sha256']
    else:
        model=o.load_encoder(row['policy'],row['seed'])
    arrays=np.load(path.with_suffix('.npz'))
    for kind,expected in row['planning'].items():
        saved=torch.load(path.with_name(path.stem+'_'+kind+'.pt'),map_location='cpu')
        predictor=o.matched.planning.Predictor()
        predictor.load_state_dict(saved['predictor']);predictor.eval()
        actual,trajectory=o.matched.planning_evaluation(model,predictor,saved['mean'],saved['scale'],row['seed'])
        close(actual,expected,str(path))
        for key,value in trajectory.items():np.testing.assert_array_equal(value,arrays[kind+'_'+key])
    return len(row['planning'])


def tests_check(path):
    def walk(value):
        count=0
        if isinstance(value,list):
            if value and isinstance(value[0],dict) and 'holm_p' in value[0]:
                ps=sorted(r['p'] for r in value)
                for row in value:
                    n=row['wins']+row['losses']
                    expected=binomtest(row['wins'],n,.5).pvalue if n else 1.
                    assert np.isclose(expected,row['p'],atol=1e-14,rtol=0)
                    index=max(i for i,p in enumerate(ps) if p<=row['p'])
                    corrected=min(1.,max((len(ps)-i)*p for i,p in enumerate(ps[:index+1])))
                    assert np.isclose(corrected,row['holm_p'],atol=1e-14,rtol=0)
                    count+=1
            else:
                count+=sum(walk(v) for v in value)
        elif isinstance(value,dict):
            count+=sum(walk(v) for v in value.values())
        return count
    return walk(json.loads(path.read_text()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-early',action='store_true')
    args=parser.parse_args()
    original=json.loads((OUT/'original_geometry.json').read_text())
    controls=sorted(OUT.glob('no_alignment-*_seed*.json'))
    tasks=[(p,False) for p in sorted(OUT.glob('planning-*_seed*.json'))]
    if args.include_early:tasks += [(p,True) for p in sorted(OUT.glob('early-*_seed*.json'))]
    with mp.get_context('spawn').Pool(4) as pool:
        g=sum(pool.imap_unordered(geometry_job,original))
        c=sum(pool.imap_unordered(intervention_job,controls))
        p=sum(pool.imap_unordered(planning_job,tasks))
    stats=tests_check(OUT/'statistics.json')
    if args.include_early:stats+=tests_check(OUT/'early_statistics.json')
    result={'original_geometry_replays':g,'alignment_removal_geometry_replays':c,
            'physical_planning_replays':p,'exact_tests_and_holm_checked':stats,'errors':0}
    o.save_json(OUT/('VERIFICATION.json' if args.include_early else 'ORIGINAL_VERIFICATION.json'),result)
    print(json.dumps(result))


if __name__=='__main__':main()
