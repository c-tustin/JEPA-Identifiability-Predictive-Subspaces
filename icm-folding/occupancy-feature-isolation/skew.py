"""Skewness interventions and objective decomposition (see SKEW_PROTOCOL.md).

  python3 skew.py --design          # build laws + predictions, hash them (no training)
  python3 skew.py --shard K --shards N
  python3 skew.py --report
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dose as D
import trace as T

S, R, torch, np, B, N = D.S, D.R, D.torch, D.np, D.B, D.N
OUT = S.OUT/'skew'
PROTOCOL = HERE/'SKEW_PROTOCOL.md'
TARGETS = [(.25, -.8), (.25, .7), (.5, .1), (.5, 1.1), (.6, .45), (.6, 1.4), (.75, 0.), (.75, 1.2), (.75, 2.),
           (.875, .3), (.875, 1.8), (.875, 3.1), (.9375, .3), (.9375, 2.15), (.9375, 3.), (.9375, 4.2),
           (.96875, 2.4), (.96875, 3.), (.96875, 5.5)]
INITS = (131, 133)


def f0_curve(init_seed, c):
    model, _ = R.trials.study.make_plan(c, 131, init_seed, 'icm')
    return T.curve(model, c)


def skewness(w, y):
    mu = w @ y; s = np.sqrt(w @ (y - mu)**2); return float(w @ ((y - mu)/s)**3)


def atoms(ds):
    x = torch.cat([ds[130]['x'], ds[136]['x']]); d = (1/x[:, 1]).double().numpy()
    order = np.argsort(d, kind='stable'); return x[torch.from_numpy(order)], d[order]


def tilted(mass, ln, lf, y, near):
    w = np.where(near, ln*(y - y[near].mean()), lf*(y - y[~near].mean()))
    w = np.exp(w - np.where(near, w[near].max(), w[~near].max()))
    w[near] *= mass/w[near].sum(); w[~near] *= (1 - mass)/w[~near].sum(); return w


def ess(w): return float(w.sum()**2/(w**2).sum())


def solve(mass, target, y, near):
    """Smallest-|lambda| tilt reaching the target skewness: far tilt, else near tilt, else both toward the junction."""
    modes = dict(far=lambda l: (0., l), near=lambda l: (l, 0.), both=lambda l: (l, -l))
    for which, lam_of in modes.items():
        f = lambda l: skewness(tilted(mass, *lam_of(l), y, near), y)
        lams = np.linspace(-200, 200, 4001); vals = np.array([f(l) - target for l in lams])
        zero = np.argsort(np.abs(lams))
        for i in zero:
            j = i + 1 if i + 1 < len(lams) else i - 1
            if np.sign(vals[i]) != np.sign(vals[j]):
                lo, hi = sorted([lams[i], lams[j]])
                for _ in range(80):
                    mid = (lo + hi)/2
                    if np.sign(f(mid) - target) == np.sign(f(lo) - target): lo = mid
                    else: hi = mid
                return lam_of((lo + hi)/2)
    raise ValueError(f'target skew {target} unreachable at mass {mass}')


def predict(g):
    if .75 <= g <= 2.6: return True
    if g <= .45 or g >= 3.8: return False
    return None


def design():
    ds, c, *_ = D.setup(); x, d = atoms(ds); near = d < 2
    grid = T.GRID.numpy(); curves = {i: f0_curve(i, c) for i in INITS}
    y131 = np.interp(np.log(d), grid, curves[131]); laws = []
    for mass, target in TARGETS:
        ln, lf = solve(mass, target, y131, near); w = tilted(mass, ln, lf, y131, near)
        row = dict(name=f'm{mass:g}-g{target:g}', near_mass=mass, target_skew=target, lambda_near=ln, lambda_far=lf,
                   ess_near=ess(w[near]), ess_far=ess(w[~near]),
                   mass_below={str(t): float(w[d < t].sum()) for t in (1.5, 2, 2.5, 3, 4, 6)},
                   skew={str(i): skewness(w, np.interp(np.log(d), grid, curves[i])) for i in INITS})
        row['prediction'] = {str(i): predict(row['skew'][str(i)]) for i in INITS}
        assert row['ess_near'] >= 200 and row['ess_far'] >= 200, row
        assert abs(row['skew']['131'] - target) < 1e-6
        laws.append(row); np.save(OUT/'laws'/f"{row['name']}.npy", w)
    native = {a: tilted(a, 0, 0, y131, near) for a in (.5, .75, .9375)}
    for a, w in native.items():
        np.save(OUT/'laws'/f'native-m{a:g}.npy', w)
    pred = dict(protocol_hash=R.digest(PROTOCOL.read_bytes()), code_hash=R.digest(Path(__file__).read_bytes()),
                atoms_hash=R.digest(x.numpy().tobytes()), part_a=laws,
                part_b=dict(even_sigreg='no junction fold in any of the five laws',
                            variance_skew10='junction fold at native m=0.75',
                            infonce_skew10='junction fold at native m=0.75',
                            controls='plain variance control and plain raw InfoNCE: no junction fold on m0.5-g1.1 and m0.9375-g2.1'),
                native_skew={f'{a:g}': skewness(w, y131) for a, w in native.items()},
                law_hashes={p.name: R.digest(p.read_bytes()) for p in sorted((OUT/'laws').glob('*.npy'))})
    R.trials.atomic_json(OUT/'PREDICTIONS.json', pred)
    print(json.dumps([(r['name'], round(r['skew']['131'], 3), round(r['skew']['133'], 3), r['prediction'],
                       round(r['ess_far']), {k: round(v, 3) for k, v in r['mass_below'].items()}) for r in laws], indent=0))


def code_skew(code):
    z = (code - code.mean())/code.std(unbiased=False).clamp_min(1e-6)
    return z.pow(3).mean()


def even_sigreg(reg, code):
    ph = code.reshape(-1, 1)*reg.frequency
    return code.numel()*(((ph.cos().mean(0) - reg.target).square())*reg.weight).sum()


def fit(name, law, method, init_seed, setup, x):
    ds, c, _, ix, _, mean, std, critic_state = setup
    path = OUT/'runs'/f'{name}-{method}-i{init_seed}.pt'
    w = torch.from_numpy(np.load(OUT/'laws'/f'{law}.npy'))
    u = torch.rand((4000, 1024), dtype=torch.float64, generator=torch.Generator().manual_seed(9600131))
    cdf = w.cumsum(0); cdf[-1] = 1.; ids = torch.searchsorted(cdf, u.contiguous()).clamp_max(len(w) - 1)
    model, _ = R.trials.study.make_plan(c, 131, init_seed, 'icm'); initial = model.state_dict()
    identity = dict(law=law, method=method, init=init_seed, law_hash=R.digest(w.numpy().tobytes()),
                    initial_hash=R.state_hash(initial), alignment_plan_hash=R.digest(ix.numpy().tobytes()),
                    index_hash=R.digest(ids.numpy().tobytes()), code_hash=R.digest(Path(__file__).read_bytes()),
                    predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes()))
    if path.exists():
        assert R.trials.load(path)['identity'] == identity; return
    m = B.Encoder(); m.load_state_dict(initial); critic = N.critic(); critic.load_state_dict(critic_state)
    infonce = method.startswith('infonce')
    params = list(m.parameters()) + (list(critic.parameters()) if infonce else [])
    opt = torch.optim.Adam(params, lr=.001); reg = B.SIGReg1D(); data = ds[131]['data']
    kappa = float(method.split('skew')[1]) if 'skew' in method else 0.
    for step in range(4000):
        opt.param_groups[0]['lr'] = R.trials.study.controls.learning_rate(step, 4000)
        raw = x[ids[step, :256] if infonce else ids[step]]; code = m(raw)
        if infonce:
            loss = N.objective((raw - mean)/std, code, critic)
        else:
            ya = m(torch.cat([data['x'][ix[step]], data['xp'][ix[step]]]))
            alignment = (ya[:512] - ya[512:]).square().mean()
            if method == 'sigreg': loss = .9*alignment + .1*reg(code)
            elif method == 'even_sigreg': loss = .9*alignment + .1*even_sigreg(reg, code)
            else:
                variance = .5*(torch.relu(1 - torch.sqrt(code[:512].var() + 1e-4)) +
                               torch.relu(1 - torch.sqrt(code[512:].var() + 1e-4)))
                loss = 25*alignment + 25*variance
        if kappa: loss = loss + kappa*code_skew(code).square()
        assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step()
    saved = dict(identity=identity, model=m.state_dict(), optimizer=opt.state_dict(),
                 critic=critic.state_dict() if infonce else None, config=vars(c))
    R.trials.save_torch(path, saved)
    check = B.Encoder(); check.load_state_dict(R.trials.load(path)['model'])
    saved['geometry'] = R.geometry(check, c, 20001); R.trials.save_torch(path, saved)
    R.trials.atomic_json(OUT/'cells'/f'{name}-{method}-i{init_seed}.json', dict(name=name, **identity, **saved['geometry']))
    print(name, method, init_seed, saved['geometry']['backward_fraction'], saved['geometry']['roots'], flush=True)


def jobs():
    out = [(f'm{m:g}-g{g:g}', f'm{m:g}-g{g:g}', 'sigreg', i) for m, g in TARGETS for i in INITS]
    laws = ['native-m0.5', 'native-m0.75', 'native-m0.9375', 'm0.5-g1.1', 'm0.9375-g2.15']
    for method in ['even_sigreg', 'variance_skew1', 'variance_skew10', 'infonce_skew1', 'infonce_skew10']:
        out += [(law, law, method, 131) for law in laws]
    out += [(law, law, method, 131) for law in ['m0.5-g1.1', 'm0.9375-g2.15'] for method in ['variance', 'infonce']]
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--design', action='store_true')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    a = p.parse_args(); torch.set_num_threads(1)
    for sub in ('laws', 'runs', 'cells'): (OUT/sub).mkdir(parents=True, exist_ok=True)
    if a.design: return design()
    pred = json.loads((OUT/'PREDICTIONS.json').read_text())
    assert pred['protocol_hash'] == R.digest(PROTOCOL.read_bytes())
    setup = D.setup(); x, _ = atoms(setup[0])
    for k, job in enumerate(jobs()):
        if k % a.shards == a.shard: fit(*job, setup, x)


if __name__ == '__main__':
    main()
