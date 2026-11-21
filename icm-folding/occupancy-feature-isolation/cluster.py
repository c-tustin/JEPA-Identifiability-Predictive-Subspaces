"""Near-cluster position test (see CLUSTER_PROTOCOL.md). Reuses skew.fit with new laws."""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import skew as K
import trace as T

R, torch = K.R, K.torch
WINDOWS = [(1., 1.25), (1.25, 1.5), (1.5, 1.75), (1.75, 2.)]
MASSES = (.5, .75, .9375)
PROTOCOL = HERE/'CLUSTER_PROTOCOL.md'


def name(m, w): return f'cluster-m{m:g}-w{w[0]:g}-{w[1]:g}'


def design():
    ds, c, *_ = K.D.setup(); x, d = K.atoms(ds); near = d < 2; grid = T.GRID.numpy()
    curves = {i: K.f0_curve(i, c) for i in K.INITS}; rows = []
    for m in MASSES:
        for lo, hi in WINDOWS:
            inw = (d >= lo) & (d < hi); w = np.zeros(len(d))
            w[inw] = m/inw.sum(); w[~near] = (1 - m)/(~near).sum()
            np.save(K.OUT/'laws'/f'{name(m, (lo, hi))}.npy', w)
            sk = {str(i): K.skewness(w, np.interp(np.log(d), grid, curves[i])) for i in K.INITS}
            rows.append(dict(name=name(m, (lo, hi)), near_mass=m, window=[lo, hi], atoms_in_window=int(inw.sum()), skew=sk,
                             h_skew={i: K.predict(v) for i, v in sk.items()},
                             h_wall={'131': True, '133': True} if (m == .75 and lo == 1.) else
                                    {'131': False, '133': False} if (m == .75 and lo == 1.75) else {'131': None, '133': None}))
    R.trials.atomic_json(K.OUT/'CLUSTER_PREDICTIONS.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()),
                         code_hash=R.digest(Path(__file__).read_bytes()), laws=rows))
    for r in rows: print(r['name'], r['atoms_in_window'], {k: round(v, 2) for k, v in r['skew'].items()}, r['h_skew'], r['h_wall'])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--design', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    if a.design: return design()
    pred = json.loads((K.OUT/'CLUSTER_PREDICTIONS.json').read_text())
    assert pred['protocol_hash'] == R.digest(PROTOCOL.read_bytes())
    setup = K.D.setup(); x, _ = K.atoms(setup[0])
    jobs = [(r['name'], r['name'], 'sigreg', i) for r in pred['laws'] for i in K.INITS]
    for k, job in enumerate(jobs):
        if k % a.shards == a.shard: K.fit(*job, setup, x)


if __name__ == '__main__':
    main()
