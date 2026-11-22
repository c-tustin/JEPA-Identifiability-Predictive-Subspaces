"""Input-geometry test of the fold framework (see GEOMETRY_PROTOCOL.md).

  python3 geometry.py --design            # laws, f0 skewness and locked predictions (no training)
  python3 geometry.py --shard K --shards N
  python3 geometry.py --report
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import skew as K

R, torch, B = K.R, K.torch, K.B
OUT = K.S.OUT/'geometry'
PROTOCOL = HERE/'GEOMETRY_PROTOCOL.md'
MAPS = {'tanh4': lambda v: torch.tanh(4*v), 'linear': lambda v: v, 'tanh1': torch.tanh}
MASSES = (.125, .25, .375, .5, .625, .75, .875, .9375)
INITS = (131, 133)


class MappedEncoder(B.Encoder):
    def __init__(self, input_map):
        super().__init__(); self.input_map = input_map

    def forward(self, x):
        return self.net(MAPS[self.input_map](x)).flatten()


def initial(c, init_seed):
    model, _ = R.trials.study.make_plan(c, 131, init_seed, 'icm'); return model.state_dict()


def f0(c, init_seed, input_map, x):
    m = MappedEncoder(input_map).double(); m.load_state_dict({k: v.double() for k, v in initial(c, init_seed).items()})
    with torch.no_grad(): return m(x.double()).numpy()


def oriented_skew(w, y, d):
    return K.skewness(w, y)*np.sign(np.corrcoef(np.log(d), y)[0, 1])


def name(m): return f'm{m:g}'


def design():
    (OUT/'laws').mkdir(parents=True, exist_ok=True)
    ds, c, *_ = K.D.setup(); x, d = K.atoms(ds); near = d < 2; rows = []
    for m in MASSES:
        w = np.where(near, m/near.sum(), (1 - m)/(~near).sum()); np.save(OUT/'laws'/f'{name(m)}.npy', w)
        for mp in MAPS:
            sk = {str(i): oriented_skew(w, f0(c, i, mp, x), d) for i in INITS}
            rows.append(dict(law=name(m), near_mass=m, input_map=mp, skew=sk,
                             prediction={i: K.predict(abs(v)) for i, v in sk.items()}))
    h = dict(H1=dict(cells=[('tanh4', 'm0.9375')], fold=False, then=[('linear', 'm0.9375'), ('tanh1', 'm0.9375')]),
             H2=dict(no_fold=('tanh4', 'm0.125'), fold=('linear', 'm0.125'), root_below=3.),
             H3=dict(no_fold=[('linear', 'm0.5'), ('tanh1', 'm0.5')]))
    R.trials.atomic_json(OUT/'PREDICTIONS.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()),
                         code_hash=R.digest(Path(__file__).read_bytes()), cells=rows, hypotheses=h))
    for r in rows: print(r['law'], r['input_map'], {k: round(v, 2) for k, v in r['skew'].items()}, r['prediction'])


def fit(law, input_map, init_seed, setup, x):
    ds, c, _, ix, _, _, _, _ = setup
    tag = f'{law}-{input_map}-i{init_seed}'; path = OUT/'runs'/f'{tag}.pt'
    w = torch.from_numpy(np.load(OUT/'laws'/f'{law}.npy'))
    u = torch.rand((4000, 1024), dtype=torch.float64, generator=torch.Generator().manual_seed(9600131))
    cdf = w.cumsum(0); cdf[-1] = 1.; ids = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(w) - 1)
    init = initial(c, init_seed)
    identity = dict(law=law, input_map=input_map, init=init_seed, law_hash=R.digest(w.numpy().tobytes()),
                    initial_hash=R.state_hash(init), alignment_plan_hash=R.digest(ix.numpy().tobytes()),
                    index_hash=R.digest(ids.numpy().tobytes()), code_hash=R.digest(Path(__file__).read_bytes()),
                    predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes()))
    if path.exists():
        assert R.trials.load(path)['identity'] == identity; return
    m = MappedEncoder(input_map); m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']
    for step in range(4000):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        code = m(x[ids[step]]); ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
        loss = .9*(ya[:512] - ya[512:]).square().mean() + .1*reg(code)
        assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step()
    saved = dict(identity=identity, model=m.state_dict(), optimizer=opt.state_dict(), config=vars(c))
    R.trials.save_torch(path, saved)
    check = MappedEncoder(input_map); check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry'] = R.geometry(check, c, 20001); R.trials.save_torch(path, saved)
    R.trials.atomic_json(OUT/'cells'/f'{tag}.json', dict(**identity, **saved['geometry']))
    print(tag, saved['geometry']['backward_fraction'], saved['geometry']['roots'], flush=True)


def jobs():
    # Largest-information cells first.
    order = ['m0.9375', 'm0.125', 'm0.5', 'm0.75', 'm0.25', 'm0.875', 'm0.375', 'm0.625']
    return [(law, mp, i) for law in order for mp in MAPS for i in INITS]


def report():
    pred = json.loads((OUT/'PREDICTIONS.json').read_text()); rows = []
    for r in pred['cells']:
        for i in map(str, INITS):
            p = OUT/'cells'/f"{r['law']}-{r['input_map']}-i{i}.json"
            if not p.exists(): continue
            o = json.loads(p.read_text()); b = o['backward_fraction']; pr = r['prediction'][i]
            rows.append(dict(law=r['law'], near_mass=r['near_mass'], input_map=r['input_map'], init=int(i), skew=r['skew'][i],
                             prediction=pr, backward=b, junction_fold=b > .1, roots=o['roots'],
                             verdict=None if pr is None else (b > .1) == pr))
    R.trials.atomic_json(OUT/'summary.json', dict(rows=rows, predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes())))
    for r in rows:
        print(f"{r['input_map']:7s} {r['law']:8s} i{r['init']} skew {r['skew']:+.2f} pred {str(r['prediction']):5s} "
              f"bwd {r['backward']:.3f} roots {[round(v, 2) for v in r['roots']]} {r['verdict']}")
    s = [r for r in rows if r['verdict'] is not None]
    print('scored', len(s), 'correct', sum(r['verdict'] for r in s))


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
    for k, job in enumerate(jobs()):
        if k % a.shards == a.shard: fit(*job, setup, x)


if __name__ == '__main__':
    main()
