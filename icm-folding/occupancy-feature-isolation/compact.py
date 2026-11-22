"""Compact-cluster test (see COMPACT_PROTOCOL.md). Reuses geometry.fit with a separate output directory."""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import geometry as G

K, R, torch = G.K, G.R, G.torch
PROTOCOL = HERE/'COMPACT_PROTOCOL.md'
LAWS = {'wall': ((1., 1.02), (2., 16.01)), 'wide': ((1., 1.25), (2., 16.01)), 'far': ((8., 9.), (1., 8.))}
CELLS = [(law, m, mp) for m in (.75, .875) for law in LAWS for mp in ('linear', 'tanh1')] + \
        [(law, m, 'tanh4') for m in (.75, .875) for law in ('wall', 'far')]
G.OUT = OUT = K.S.OUT/'compact'


def name(law, m): return f'{law}-m{m:g}'


def law_weights(d, window, tail, m):
    inw = (d >= window[0]) & (d < window[1]); intail = (d >= tail[0]) & (d < tail[1]) & ~inw
    bins = np.floor(d/.25).astype(int); counts = np.bincount(bins[intail], minlength=bins.max() + 1)
    w = np.zeros(len(d)); w[inw] = m/inw.sum()
    t = np.where(intail, 1/np.maximum(counts[bins], 1), 0.); w += (1 - m)*t/t.sum(); return w


def design():
    (OUT/'laws').mkdir(parents=True, exist_ok=True)
    ds, c, *_ = K.D.setup(); x, d = K.atoms(ds); rows = []
    for law, (win, tail) in LAWS.items():
        for m in (.75, .875):
            w = law_weights(d, win, tail, m); np.save(OUT/'laws'/f'{name(law, m)}.npy', w)
            sk = {mp: {str(i): G.oriented_skew(w, G.f0(c, i, mp, x), d) for i in G.INITS} for mp in G.MAPS}
            rows.append(dict(law=name(law, m), window=win, tail=tail, mass=m, ess_window=K.ess(w[(d >= win[0]) & (d < win[1])]),
                             ess_tail=K.ess(w[(d >= tail[0]) & (d < tail[1]) & ~((d >= win[0]) & (d < win[1]))]), skew=sk))
    pred = {'wall-m0.75': {'linear': 'both', 'tanh1': 'both', 'tanh4': 'both'}, 'wall-m0.875': {'linear': 'one', 'tanh1': 'one'},
            'wide-m0.75': {'linear': 'none', 'tanh1': 'none'}, 'wide-m0.875': {'linear': 'none', 'tanh1': 'none'},
            'far-m0.75': {'linear': 'one', 'tanh1': 'one'}}
    R.trials.atomic_json(OUT/'PREDICTIONS.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()),
                         code_hash=R.digest(Path(__file__).read_bytes()), laws=rows, predictions=pred))
    for r in rows: print(r['law'], round(r['ess_window']), round(r['ess_tail']), {k: {i: round(v, 2) for i, v in s.items()} for k, s in r['skew'].items()})


def report():
    pred = json.loads((OUT/'PREDICTIONS.json').read_text()); rows = []
    for law, m, mp in CELLS:
        for i in G.INITS:
            p = OUT/'cells'/f'{name(law, m)}-{mp}-i{i}.json'
            if p.exists():
                o = json.loads(p.read_text()); rows.append(dict(law=name(law, m), input_map=mp, init=i, backward=o['backward_fraction'],
                                                                junction_fold=o['backward_fraction'] > .1, roots=o['roots']))
    verdicts = []
    for law, maps in pred['predictions'].items():
        for mp, want in maps.items():
            f = [r['junction_fold'] for r in rows if r['law'] == law and r['input_map'] == mp]
            if len(f) < 2: continue
            ok = all(f) if want == 'both' else any(f) if want == 'one' else not any(f)
            verdicts.append(dict(law=law, input_map=mp, predicted=want, folds=sum(f), correct=ok))
    R.trials.atomic_json(OUT/'summary.json', dict(rows=rows, verdicts=verdicts, predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes())))
    for r in rows: print(f"{r['input_map']:7s} {r['law']:12s} i{r['init']} bwd {r['backward']:.3f} roots {[round(v, 2) for v in r['roots']]}")
    for v in verdicts: print(v)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--design', action='store_true'); p.add_argument('--report', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    for sub in ('runs', 'cells'): (OUT/sub).mkdir(parents=True, exist_ok=True)
    if a.design: return design()
    if a.report: return report()
    pred = json.loads((OUT/'PREDICTIONS.json').read_text())
    assert pred['protocol_hash'] == R.digest(PROTOCOL.read_bytes())
    setup = K.D.setup(); x, _ = K.atoms(setup[0])
    jobs = [(name(law, m), mp, i) for law, m, mp in CELLS for i in G.INITS]
    for k, job in enumerate(jobs):
        if k % a.shards == a.shard: G.fit(*job, setup, x)


if __name__ == '__main__':
    main()
