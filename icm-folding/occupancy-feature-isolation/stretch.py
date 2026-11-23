"""Causal stretchability test (see STRETCH_PROTOCOL.md): zero-initialized extra input channel, identical initial code.

  python3 stretch.py --design | --shard K --shards N | --report
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from torch import nn
import geometry as G

K, R, B = G.K, G.R, G.B
OUT = K.S.OUT/'stretch'
PROTOCOL = HERE/'STRETCH_PROTOCOL.md'
CHANNELS = {'near': lambda u: torch.clamp(2*u - 1, min=0), 'far': lambda u: torch.clamp(1 - 2*u, min=0), 'dup': lambda u: torch.tanh(4*u)}
LAWS = ('m0.625', 'm0.75', 'm0.875')
INITS = (131, 133, 135, 137)


class StretchEncoder(nn.Module):
    def __init__(self, channel):
        super().__init__(); self.channel = channel
        self.net = nn.Sequential(nn.Linear(7, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x):
        return self.net(torch.cat([torch.tanh(4*x), CHANNELS[self.channel](x[:, 1:2])], 1)).flatten()


def initial(c, init_seed):
    base = G.initial(c, init_seed); st = {k.replace('net.', 'net.'): v.clone() for k, v in base.items()}
    st['net.0.weight'] = torch.cat([base['net.0.weight'], torch.zeros(48, 1, dtype=base['net.0.weight'].dtype)], 1)
    return st


def check_identical(c, x):
    out = {}
    for i in INITS:
        b = B.Encoder(); b.load_state_dict(G.initial(c, i))
        for ch in CHANNELS:
            m = StretchEncoder(ch); m.load_state_dict(initial(c, i))
            with torch.no_grad(): out[f'{ch}-{i}'] = float((m(x) - b(x)).abs().max())
    return out


def decoupling(ch, init_seed, c):
    m = StretchEncoder(ch).double(); m.load_state_dict({k: v.double() for k, v in initial(c, init_seed).items()}); res = {}
    for label, (a, b) in {'near (d 1 vs 1.9)': (1., 1.9), 'tail (d 4 vs 16)': (4., 16.)}.items():
        J = []
        for dist in (a, b):
            m.zero_grad(); m(B.sensor(torch.tensor([dist], dtype=torch.float64), c).double()).sum().backward()
            J.append(torch.cat([p.grad.flatten() for p in m.parameters()]).clone())
        res[label] = float(1 - (J[0] @ J[1])/(J[0].norm()*J[1].norm()))
    return res


def design():
    (OUT/'laws').mkdir(parents=True, exist_ok=True)
    ds, c, *_ = K.D.setup(); x, _ = K.atoms(ds)
    diff = check_identical(c, x); assert max(diff.values()) == 0., diff
    for law in LAWS: (OUT/'laws'/f'{law}.npy').write_bytes((G.OUT/'laws'/f'{law}.npy').read_bytes())
    kern = {f'{ch}-{i}': decoupling(ch, i, c) for ch in CHANNELS for i in INITS}
    R.trials.atomic_json(OUT/'PREDICTIONS.json', dict(protocol_hash=R.digest(PROTOCOL.read_bytes()), code_hash=R.digest(Path(__file__).read_bytes()),
                         initial_code_max_abs_difference=diff, kernel=kern,
                         predictions={'near': 'no fold in >= 10 of 12', 'far': 'fold in >= 10 of 12', 'dup': 'fold in >= 10 of 12'}))
    for k, v in kern.items(): print(k, {a: round(b, 4) for a, b in v.items()})


def fit(law, ch, init_seed, setup, x):
    ds, c, _, ix, _, _, _, _ = setup
    tag = f'{law}-{ch}-i{init_seed}'; path = OUT/'runs'/f'{tag}.pt'
    w = torch.from_numpy(np.load(OUT/'laws'/f'{law}.npy'))
    u = torch.rand((4000, 1024), dtype=torch.float64, generator=torch.Generator().manual_seed(9600131))
    cdf = w.cumsum(0); cdf[-1] = 1.; ids = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(w) - 1)
    init = initial(c, init_seed)
    identity = dict(law=law, channel=ch, init=init_seed, law_hash=R.digest(w.numpy().tobytes()), initial_hash=R.state_hash(init),
                    alignment_plan_hash=R.digest(ix.numpy().tobytes()), index_hash=R.digest(ids.numpy().tobytes()),
                    code_hash=R.digest(Path(__file__).read_bytes()), predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes()))
    if path.exists():
        assert R.trials.load(path)['identity'] == identity; return
    m = StretchEncoder(ch); m.load_state_dict(init)
    opt = torch.optim.Adam(m.parameters(), lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']
    for step in range(4000):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        code = m(x[ids[step]]); ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
        loss = .9*(ya[:512] - ya[512:]).square().mean() + .1*reg(code)
        assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step()
    saved = dict(identity=identity, model=m.state_dict(), config=vars(c)); R.trials.save_torch(path, saved)
    check = StretchEncoder(ch); check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry'] = R.geometry(check, c, 20001); R.trials.save_torch(path, saved)
    # end-state statistics on the law
    wn = w.numpy(); _, d = K.atoms(ds)
    with torch.no_grad(): y = check(x).double().numpy()
    mu = wn @ y; sd = np.sqrt(wn @ (y - mu)**2); z = (y - mu)/sd*np.sign(np.corrcoef(np.log(d), y)[0, 1])
    near = d < 2; wn_ = wn[near]/wn[near].sum()
    stats = dict(sd=float(sd), skew=float(wn @ z**3), near_spread=float(np.sqrt(wn_ @ (z[near] - wn_ @ z[near])**2)))
    R.trials.atomic_json(OUT/'cells'/f'{tag}.json', dict(**identity, **saved['geometry'], **stats))
    print(tag, saved['geometry']['backward_fraction'], stats, flush=True)


def report():
    pred = json.loads((OUT/'PREDICTIONS.json').read_text()); rows = []
    for ch in CHANNELS:
        for law in LAWS:
            for i in INITS:
                p = OUT/'cells'/f'{law}-{ch}-i{i}.json'
                if p.exists():
                    o = json.loads(p.read_text())
                    rows.append(dict(channel=ch, law=law, init=i, backward=o['backward_fraction'], fold=o['backward_fraction'] > .1,
                                     skew=o['skew'], sd=o['sd'], near_spread=o['near_spread']))
    verdict = {}
    for ch in CHANNELS:
        f = [r['fold'] for r in rows if r['channel'] == ch]
        ok = (len(f) - sum(f) >= 10) if ch == 'near' else (sum(f) >= 10)
        verdict[ch] = dict(cells=len(f), folds=sum(f), correct=ok if len(f) == 12 else None)
    R.trials.atomic_json(OUT/'summary.json', dict(rows=rows, verdict=verdict, predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes())))
    for r in rows: print(f"{r['channel']:5s} {r['law']:7s} i{r['init']} bwd {r['backward']:.3f} skew {r['skew']:+.2f} sd {r['sd']:.2f} spike spread {r['near_spread']:.2f}")
    print(verdict)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--design', action='store_true'); p.add_argument('--report', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    for sub in ('runs', 'cells'): (OUT/sub).mkdir(parents=True, exist_ok=True)
    if a.design: return design()
    if a.report: return report()
    pred = json.loads((OUT/'PREDICTIONS.json').read_text()); assert pred['protocol_hash'] == R.digest(PROTOCOL.read_bytes())
    setup = K.D.setup(); x, _ = K.atoms(setup[0])
    jobs = [(law, ch, i) for ch in CHANNELS for law in LAWS for i in INITS]
    for k, job in enumerate(jobs):
        if k % a.shards == a.shard: fit(*job, setup, x)


if __name__ == '__main__':
    main()
