"""Original zoo recovery, alignment-removal controls, and physical planning."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import original as o

HERE = Path(__file__).resolve().parent


def code_hash():
    digest = hashlib.sha256()
    for filename in ['original.py', 'run_original.py', 'PROTOCOL.md']:
        digest.update((HERE / filename).read_bytes())
    return digest.hexdigest()


def source_cached(name, seed, output):
    path = output / 'sources' / f'{name}-{seed}.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu')
        if saved['hash'] != o.raw_record(name, seed)['source_stats']['sha256']:
            raise RuntimeError('Cache hash mismatch')
        return saved['data'], saved['hash']
    data, fingerprint = o.original_source(name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'data': data, 'hash': fingerprint}, temporary)
    temporary.replace(path)
    return data, fingerprint


def fit_without_alignment(name, seed, output):
    path = output / f'no_alignment-{name}_seed{seed}.json'
    fingerprint = code_hash()
    if path.exists():
        old = json.loads(path.read_text())
        assert old['code_hash'] == fingerprint
        return
    c = o.config()
    data, source_hash = source_cached(name, seed, output)
    model, plan = o.reliability.make_plan(c, seed, seed, 'icm')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    regularizer = o.behavior.SIGReg1D()
    partial = output / 'partials' / f'{name}-{seed}.pt'
    start, history = 0, []
    if partial.exists():
        old = torch.load(partial, map_location='cpu')
        assert old['code_hash'] == fingerprint
        model.load_state_dict(old['encoder'])
        optimizer.load_state_dict(old['optimizer'])
        start, history = old['step'], old['history']
    for step in range(start, c.jepa_steps):
        optimizer.param_groups[0]['lr'] = o.controls.learning_rate(step, c.jepa_steps)
        ix = plan['indices'][step]
        y = model(torch.cat([data['x'][ix], data['xp'][ix]]))
        alignment = (y[:c.batch] - y[c.batch:]).square().mean()
        penalty = regularizer(y)
        loss = c.sigreg_weight * penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 1000 == 0:
            metrics, _ = o.geometry(model)
            history.append({'step': step+1, 'alignment': float(alignment.detach()),
                            'sigreg': float(penalty.detach()), **metrics})
            partial.parent.mkdir(parents=True, exist_ok=True)
            temporary = partial.with_suffix('.tmp')
            torch.save({'code_hash': fingerprint, 'step': step+1, 'history': history,
                        'encoder': model.state_dict(), 'optimizer': optimizer.state_dict()}, temporary)
            temporary.replace(partial)
    model.eval()
    metrics, curve = o.geometry(model)
    torch.save({'encoder': model.state_dict(), 'config': vars(c), 'code_hash': fingerprint}, path.with_suffix('.pt'))
    np.savez_compressed(path.with_suffix('.npz'), **curve)
    o.save_json(path, {'policy': name, 'seed': seed, 'code_hash': fingerprint,
                      'source_hash': source_hash, 'metrics': metrics, 'history': history})
    partial.unlink(missing_ok=True)
    print(json.dumps({'stage':'no_alignment','policy':name,'seed':seed,
                      'r2':metrics['affine_log_r2'],'folded':metrics['folded']}),flush=True)


def planning(name, seed, output):
    path = output / f'planning-{name}_seed{seed}.json'
    fingerprint = code_hash()
    if path.exists():
        assert json.loads(path.read_text())['code_hash'] == fingerprint
        return
    model = o.load_encoder(name, seed)
    design = o.matched.make_design(size=16384)
    common = o.matched.shared_predictor_data(design, seed)
    datasets = [('shared', common)]
    source_hash = None
    if name != 'ou':
        native, source_hash = source_cached(name, seed, output)
        assert len(native['x']) == len(common['x'])
        datasets.append(('native', native))
    result = {'policy': name, 'seed':seed, 'code_hash':fingerprint,
              'source_hash':source_hash, 'planning':{}}
    paths = {}
    for kind, data in datasets:
        predictor, mean, scale = o.matched.planning.train_predictor(model,data,common,seed,2000)
        metrics, trajectory = o.matched.planning_evaluation(model,predictor,mean,scale,seed)
        result['planning'][kind] = metrics
        paths.update({kind+'_'+key:value for key,value in trajectory.items()})
        torch.save({'predictor':predictor.state_dict(),'mean':mean,'scale':scale},
                   path.with_name(path.stem+'_'+kind+'.pt'))
    np.savez_compressed(path.with_suffix('.npz'), **paths)
    o.save_json(path,result)
    print(json.dumps({'stage':'planning','policy':name,'seed':seed,
                      'shared':result['planning']['shared']['goal_error'],
                      'native':result['planning'].get('native',{}).get('goal_error')}),flush=True)


def job(task):
    stage, name, seed, output = task
    torch.set_num_threads(1)
    if stage == 'intervention':
        fit_without_alignment(name, seed, Path(output))
    else:
        planning(name, seed, Path(output))


def replay_gate(output):
    """Original shared-batch fit must reproduce before new fits are interpreted."""
    torch.set_num_threads(1)
    result = []
    for name in ['icm', 'random']:
        seed = 8
        data, fingerprint = source_cached(name, seed, output)
        c = o.config()
        reference = (o.behavior.collect(c,'ou',c.eval_episodes,20000+seed),
                     o.behavior.collect(c,'ou',c.eval_episodes,30000+seed))
        model, _, _ = o.run_zoo.fit_pure({k:data[k] for k in ['x','xp']},c,seed,reference)
        old = o.load_encoder(name,seed)
        error = max(float((value-old.state_dict()[key]).abs().max()) for key,value in model.state_dict().items())
        assert error == 0, (name,error)
        result.append({'policy':name,'seed':seed,'max_parameter_error':error,'source_hash':fingerprint})
    o.save_json(output/'REPLAY_GATES.json',result)
    print('Both original-loss training gates reproduced exactly.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','geometry','gates','intervention','planning'],default='all')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--output',type=Path,default=HERE/'output')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ['all','geometry']:
        o.export_original_geometry(args.output)
    if args.stage in ['all','gates']:
        replay_gate(args.output)
    tasks=[]
    if args.stage in ['all','intervention']:
        tasks += [('intervention',name,seed,str(args.output)) for seed in o.SEEDS for name in o.INTERVENTION_POLICIES]
    if args.stage in ['all','planning']:
        tasks += [('planning',name,seed,str(args.output)) for seed in range(8,18) for name in o.PLANNING_POLICIES]
    if tasks:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            list(pool.imap_unordered(job,tasks))


if __name__=='__main__':
    main()
