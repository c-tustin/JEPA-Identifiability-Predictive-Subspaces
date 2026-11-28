"""SIGReg aliasing with a 2-d state (depth d, lateral offset b). See PROTOCOL.md.

From the workspace root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 icm-folding/multistate-sigreg/run.py [--workers 10]
    python3 icm-folding/multistate-sigreg/run.py --report-only
"""
import argparse
import importlib.util
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('multidim_run', HERE.parent/'multidim-sigreg/run.py')
MD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MD)
R, B, trials = MD.R, MD.B, MD.trials
OUT = HERE/'output'
PROTOCOL_SHA = '23d65eba7765dd45ec9f4f8bb1da80759acc6e206db6bc1216428878e0c6a24b'

TRIANGLE = torch.tensor([[0., 1.], [-.8660254, -.5], [.8660254, -.5]])
DMIN, DMAX, BMAX, MOVE = 1., 16., 3., .25
CONDITIONS = {'balanced': (.1, .25, .25, .2, .2), 'skew-d': (.1, .31, .19, .2, .2), 'skew-db': (.1, .31, .19, .24, .16)}
SEEDS = tuple(range(5))
DIMS = (1, 2, 3, 4, 8)
ARMS = ('ordinary', 'weighted')
STEPS, BATCH, EPISODES, HORIZON, BURNIN = 4000, 512, 256, 64, 3000


def sensor(d, b):
    v = TRIANGLE.to(d.dtype)
    x = (v[None, :, 0]+b[:, None])/d[:, None]
    y = v[None, :, 1]/d[:, None]
    return torch.stack([x, y], -1).flatten(1)


def reflect(v, lo, hi):
    v = torch.where(v < lo, 2*lo-v, v)
    return torch.where(v > hi, 2*hi-v, v)


def move(d, b, a):
    dd = torch.where(a == 1, -MOVE, torch.where(a == 2, MOVE, 0.))
    db = torch.where(a == 3, -MOVE, torch.where(a == 4, MOVE, 0.))
    return reflect(d+dd, DMIN, DMAX), reflect(b+db, -BMAX, BMAX)


def collect(probs, seed):
    g = torch.Generator().manual_seed(seed)
    d = DMIN+(DMAX-DMIN)*torch.rand(EPISODES, generator=g, dtype=torch.float64)
    b = -BMAX+2*BMAX*torch.rand(EPISODES, generator=g, dtype=torch.float64)
    p = torch.tensor(probs, dtype=torch.float64)
    rec = {k: [] for k in ('d', 'b', 'dp', 'bp', 'action')}
    for t in range(BURNIN+HORIZON):
        a = torch.multinomial(p.expand(EPISODES, -1), 1, generator=g)[:, 0]
        dp, bp = move(d, b, a)
        if t >= BURNIN:
            for k, v in zip(rec, (d, b, dp, bp, a)):
                rec[k].append(v)
        d, b = dp, bp
    rec = {k: torch.cat(v) for k, v in rec.items()}  # time-major: episode = index % EPISODES
    return dict(x=sensor(rec['d'], rec['b']).float(), xp=sensor(rec['dp'], rec['bp']).float(), action=rec['action'],
                d=rec['d'], b=rec['b'], dp=rec['dp'], bp=rec['bp'])


def dataset(condition, seed):
    path = OUT/'data'/f'{condition}-{seed}.pt'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if condition == 'shared':
            data = collect((.2,)*5, 650000+seed)
        else:
            data = collect(CONDITIONS[condition], 600000+1000*list(CONDITIONS).index(condition)+seed)
        torch.save(data, path)
    return torch.load(path, weights_only=False)


def train(condition, seed, dim, arm):
    path = OUT/'runs'/f'{condition}-{arm}-d{dim}-{seed}.pt'
    if path.exists():
        return trials.load(path)
    data = dataset(condition, seed)
    n = len(data['x'])
    torch.manual_seed(40000+seed)
    model = MD.EncoderD(dim)
    indices = torch.randint(n, (STEPS, BATCH), generator=torch.Generator().manual_seed(700000+seed))
    reg_indices = torch.randint(n, (STEPS, BATCH), generator=torch.Generator().manual_seed(800000+seed))
    ed, eb = torch.cat([data['d'], data['dp']]), torch.cat([data['b'], data['bp']])
    bins = (ed.log()/np.log(16)*16).long().clamp(0, 15)*12+((eb+BMAX)/(2*BMAX)*12).long().clamp(0, 11)
    mass = torch.bincount(bins, minlength=16*12).float()/len(bins)
    inverse = 1/mass[bins]
    reg = B.SIGReg1D()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    dirgen = torch.Generator().manual_seed(3000000+seed)
    t0, history = time.time(), []
    for step in range(STEPS):
        opt.param_groups[0]['lr'] = trials.study.controls.learning_rate(step, STEPS)
        ix, ri = indices[step], reg_indices[step]
        z = model(torch.cat([data['x'][ix], data['xp'][ix]]))
        alignment = (z[:BATCH]-z[BATCH:]).square().mean()
        directions = None
        if dim > 1:
            directions = torch.randn(MD.DIRECTIONS, dim, generator=dirgen)
            directions = directions/directions.norm(dim=1, keepdim=True)
        rz = model(torch.cat([data['x'][ri], data['xp'][ri]]))
        weights = inverse[torch.cat([ri, ri+n])] if arm == 'weighted' else None
        penalty = MD.sigreg(rz, reg, weights, directions)
        loss = .9*alignment+.1*penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step+1) % 500 == 0:
            history.append(dict(step=step+1, alignment=float(alignment), regularizer=float(penalty)))
    saved = dict(condition=condition, seed=seed, dim=dim, arm=arm, model=model.state_dict(), history=history,
                 seconds=time.time()-t0, protocol_sha256=PROTOCOL_SHA)
    path.parent.mkdir(parents=True, exist_ok=True)
    trials.save_torch(path, saved)
    return saved


@torch.no_grad()
def aliasing(model, data):
    dg = torch.exp(torch.linspace(0, np.log(16), 41, dtype=torch.float64))
    bg = torch.linspace(-BMAX, BMAX, 25, dtype=torch.float64)
    D, Bb = torch.meshgrid(dg, bg, indexing='ij')
    d, b = D.flatten(), Bb.flatten()
    f = model(sensor(d, b))
    local = torch.full((len(d),), np.inf, dtype=torch.float64)
    for dd, db in ((MOVE, 0), (-MOVE, 0), (0, MOVE), (0, -MOVE)):
        nd, nb = d+dd, b+db
        ok = (nd >= DMIN-1e-9) & (nd <= DMAX+1e-9) & (nb >= -BMAX-1e-9) & (nb <= BMAX+1e-9)
        disp = (model(sensor(nd.clamp(DMIN, DMAX), nb.clamp(-BMAX, BMAX)))-f).norm(dim=1)
        local = torch.where(ok, torch.minimum(local, disp), local)
    dist = torch.cdist(f, f)
    far = ((d[:, None]-d[None, :]).abs()+(b[:, None]-b[None, :]).abs()) > 1.0
    ratio = torch.where(far, dist, torch.full_like(dist, np.inf)).min(1).values/local.clamp_min(1e-300)
    aliased = (ratio < 1).numpy()
    ed, eb = torch.cat([data['d'], data['dp']]).numpy(), torch.cat([data['b'], data['bp']]).numpy()
    i = np.clip(np.rint(np.log(ed)/np.log(16)*40).astype(int), 0, 40)
    j = np.clip(np.rint((eb+BMAX)/MOVE).astype(int), 0, 24)
    return dict(aliased_native_mass=float(aliased[i*25+j].mean()), aliased_grid_fraction=float(aliased.mean()),
                min_far_ratio=float(ratio.min()), mass_below_2=float((ed < 2).mean()))


class PredictorD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+5, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, dim))

    def forward(self, y, a):
        return y+self.net(torch.cat([y, F.one_hot(a, 5).float()], -1))


def bank(seed, count=256, horizon=8):
    seqs = []
    for i in range(-horizon, horizon+1):
        for j in range(-horizon, horizon+1):
            if abs(i)+abs(j) <= horizon:
                s = [1 if i < 0 else 2]*abs(i)+[3 if j < 0 else 4]*abs(j)
                seqs.append(s+[0]*(horizon-len(s)))
    rand = torch.randint(0, 5, (count-len(seqs), horizon), generator=torch.Generator().manual_seed(1500000+seed))
    return torch.cat([torch.tensor(seqs), rand])


def goals(seed, near, cases=64):
    g = torch.Generator().manual_seed((1700000 if near else 1600000)+seed)
    lo, hi = (1., 3.5) if near else (3., 14.)
    d0 = lo+(hi-lo)*torch.rand(cases, generator=g, dtype=torch.float64)
    b0 = -2.5+5*torch.rand(cases, generator=g, dtype=torch.float64)
    kd, kb = torch.zeros(cases, dtype=torch.long), torch.zeros(cases, dtype=torch.long)
    bad = torch.ones(cases, dtype=torch.bool)
    while bad.any():
        kd[bad] = torch.randint(-6, 7, (int(bad.sum()),), generator=g)
        kb[bad] = torch.randint(-6, 7, (int(bad.sum()),), generator=g)
        gd, gb = d0+MOVE*kd, b0+MOVE*kb
        bad = ((kd == 0) & (kb == 0)) | (gd < DMIN) | (gd > DMAX) | (gb < -BMAX) | (gb > BMAX)
    gd, gb = d0+MOVE*kd, b0+MOVE*kb
    od, ob = d0.clone(), b0.clone()
    for _ in range(16):  # greedy known-dynamics oracle: depth first, then offset
        a = torch.where(gd-od > MOVE/2, 2, torch.where(od-gd > MOVE/2, 1, torch.where(gb-ob > MOVE/2, 4, torch.where(ob-gb > MOVE/2, 3, 0))))
        od, ob = move(od, ob, a)
    assert float(((od-gd).abs()+(ob-gb).abs()).max()) < 1e-6
    return d0, b0, gd, gb


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, o))


def evaluate(model, data, shared, seed, dim):
    for p in model.parameters():
        p.requires_grad_(False)  # frozen encoder
    enc = lambda x: model(x)
    episode = torch.arange(len(data['x'])) % EPISODES
    mask = episode < int(.8*EPISODES)
    tr = {k: v[mask] for k, v in data.items()}
    te = {k: v[~mask] for k, v in data.items()}
    goal_sets = dict(standard=goals(seed, False), near=goals(seed, True))
    acts = bank(seed)
    reps = []
    for r in range(R.REPS):
        s = seed+100000*r
        rep = {}
        with torch.no_grad():
            code = enc(tr['x'])
            center, scale = code.mean(0), code.std(0, unbiased=False).clamp_min(.05)
            t = torch.tanh(4*tr['xp'])
            tm, ts = t.mean(0), t.std(0, unbiased=False).clamp_min(.05)
            feat = lambda dd: torch.cat([(enc(dd['x'])-center)/scale, F.one_hot(dd['action'], 5).float()], 1)
            tgt = lambda dd: (torch.tanh(4*dd['xp'])-tm)/ts
        torch.manual_seed(5100000+s)
        net = R.fit(mlp(dim+5, 6), feat(tr), tgt(tr), 5200000+s)
        with torch.no_grad():
            rep['next_obs'] = float(F.mse_loss(net(feat(te)), tgt(te)))
            rep['next_obs_shared'] = float(F.mse_loss(net(feat(shared)), tgt(shared)))
            state = lambda dd: torch.stack([dd['d'].log(), dd['b']], 1).float()
            z = lambda dd: (enc(dd['x'])-center)/scale
        torch.manual_seed(5300000+s)
        ro = R.fit(mlp(dim, 2), z(tr), state(tr), 5400000+s)
        with torch.no_grad():
            err = (ro(z(te))-state(te)).square().mean(0)
            rep['readout_logd'], rep['readout_b'] = float(err[0]), float(err[1])
        rep['planning'] = {}
        for kind, pairs in (('shared', shared), ('native', data)):
            torch.manual_seed(1200000+s)
            pred = PredictorD(dim)
            opt = torch.optim.Adam(pred.parameters(), lr=.001)
            with torch.no_grad():
                cal = enc(shared['x'])
                mean, sc = cal.mean(0), cal.std(0, unbiased=False).clamp_min(1e-6)
                y, yp = (enc(pairs['x'])-mean)/sc, (enc(pairs['xp'])-mean)/sc
            gen = torch.Generator().manual_seed(1300000+s)
            for _ in range(1000):
                ix = torch.randint(len(y), (256,), generator=gen)
                loss = (pred(y[ix], pairs['action'][ix])-yp[ix]).square().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            pred.eval()
            rep['planning'][kind] = {}
            for gname, (d0, b0, gd, gb) in goal_sets.items():
                with torch.no_grad():
                    d, b = d0.clone(), b0.clone()
                    gy = (enc(sensor(gd, gb).float())-mean)/sc
                    for _ in range(16):
                        yy = (enc(sensor(d, b).float())-mean)/sc
                        count = len(acts)
                        st = yy[:, None, :].expand(-1, count, -1).reshape(-1, dim)
                        rg = gy.repeat_interleave(count, 0)
                        stage = torch.zeros(len(st))
                        for t_ in range(acts.shape[1]):
                            st = pred(st, acts[:, t_].repeat(len(yy)))
                            stage += (st-rg).square().sum(-1)
                        cost = (st.reshape(len(yy), count, dim)-gy[:, None, :]).square().sum(-1)+.01*stage.reshape(len(yy), count)
                        cost += 1e-8*(acts != 0).float().sum(1)[None]
                        d, b = move(d, b, acts[cost.argmin(1), 0])
                    e = torch.sqrt((d-gd).square()+(b-gb).square())
                    ok = ((d-gd).abs() <= MOVE+1e-5) & ((b-gb).abs() <= MOVE+1e-5)
                rep['planning'][kind][gname] = dict(error=float(e.mean()), success=float(ok.float().mean()))
        reps.append(rep)
    return reps


def job(spec):
    torch.set_num_threads(1)
    condition, seed, dim, arm = spec
    out = OUT/'cells'/f'{condition}-{arm}-d{dim}-{seed}.json'
    if out.exists():
        return json.loads(out.read_text())
    t0 = time.time()
    saved = train(condition, seed, dim, arm)
    model = MD.EncoderD(dim)
    model.load_state_dict(saved['model'])
    model.eval()
    data, shared = dataset(condition, seed), dataset('shared', seed)
    result = dict(condition=condition, seed=seed, dim=dim, arm=arm, train_seconds=saved['seconds'], protocol_sha256=PROTOCOL_SHA,
                  **aliasing(model.double(), data))
    result['reps'] = evaluate(model.float(), data, shared, seed, dim)
    result['seconds'] = time.time()-t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f'{condition}-{arm}-d{dim}-{seed} done {result["seconds"]:.0f}s aliased={result["aliased_native_mass"]:.3f} ratio={result["min_far_ratio"]:.2f}', flush=True)
    return result


KEYS = {'next_obs': ('next_obs',), 'readout_logd': ('readout_logd',), 'readout_b': ('readout_b',),
        'standard_shared': ('planning', 'shared', 'standard', 'error'), 'near_shared': ('planning', 'shared', 'near', 'error'),
        'standard_native': ('planning', 'native', 'standard', 'error'), 'near_native': ('planning', 'native', 'near', 'error'),
        'standard_success': ('planning', 'shared', 'standard', 'success'), 'near_success': ('planning', 'shared', 'near', 'success')}


def report():
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    cells = {(r['condition'], r['arm'], r['dim'], r['seed']): r for r in (json.loads(p.read_text()) for p in (OUT/'cells').glob('*.json'))}
    m = lambda r, k: float(np.mean([R.get(rep, KEYS[k]) for rep in r['reps']]))
    al = lambda r: r['aliased_native_mass'] > .01
    by = {}
    for c in CONDITIONS:
        for a in ARMS:
            for d in DIMS:
                rs = [cells[(c, a, d, s)] for s in SEEDS]
                by[f'{c}/{a}/d{d}'] = dict(aliased=sum(al(r) for r in rs), mean_aliased_mass=float(np.mean([r['aliased_native_mass'] for r in rs])),
                                           min_far_ratio_median=float(np.median([r['min_far_ratio'] for r in rs])),
                                           **{k: float(np.mean([m(r, k) for r in rs])) for k in KEYS})
    occ = {c: float(np.mean([cells[(c, 'ordinary', 1, s)]['mass_below_2'] for s in SEEDS])) for c in CONDITIONS}
    s1 = sum(al(cells[(c, a, 1, s)]) for c in CONDITIONS for a in ARMS for s in SEEDS)
    s2_skew, s2_bal = by['skew-d/ordinary/d2']['aliased'], by['balanced/ordinary/d2']['aliased']
    s3 = sum(by[f'{c}/ordinary/d{d}']['aliased'] for c in CONDITIONS for d in DIMS if d >= 4)
    s4 = sum(cells[('skew-d', 'weighted', 2, s)]['aliased_native_mass'] < cells[('skew-d', 'ordinary', 2, s)]['aliased_native_mass'] for s in SEEDS)
    pairs = [(cells[(c, 'ordinary', d, s)], cells[(c, 'weighted', d, s)]) for c in ('skew-d', 'skew-db') for d in DIMS if d >= 3 for s in SEEDS]
    s5 = sum(m(w, 'standard_shared') < m(o, 'standard_shared') for o, w in pairs)
    pool = [cells[(c, 'ordinary', d, s)] for c in CONDITIONS for d in (2, 3) for s in SEEDS]
    rho = spearmanr([r['aliased_native_mass'] for r in pool], [m(r, 'next_obs') for r in pool])
    preds = dict(S1=dict(count=s1, of=len(CONDITIONS)*len(ARMS)*len(SEEDS), held=s1 == len(CONDITIONS)*len(ARMS)*len(SEEDS)),
                 S2=dict(skew_d=s2_skew, balanced=s2_bal, held=s2_skew >= 3 and s2_skew > s2_bal),
                 S3=dict(aliased_fits=s3, held=s3 == 0),
                 S4=dict(count=s4, of=len(SEEDS), held=s4 >= 4),
                 S5=dict(count=s5, of=len(pairs), held=s5 >= .7*len(pairs)),
                 S6=dict(rho=float(rho.statistic) if np.isfinite(rho.statistic) else None, p=float(rho.pvalue) if np.isfinite(rho.pvalue) else None,
                         held=bool(np.isfinite(rho.statistic) and rho.statistic > .3)))
    checks = dict(cells=len(cells), expected=len(CONDITIONS)*len(ARMS)*len(DIMS)*len(SEEDS), mass_below_2=occ,
                  protocol_unchanged=R.sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA)
    summary = dict(checks=checks, predictions=preds, by_group=by,
                   cells={f'{c}/{a}/d{d}/{s}': dict(aliased_native_mass=r['aliased_native_mass'], min_far_ratio=r['min_far_ratio'], **{k: m(r, k) for k in KEYS})
                          for (c, a, d, s), r in cells.items()})
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    for row, c in enumerate(CONDITIONS):
        for ax, k, lab in zip(axes[row], ('aliased', 'next_obs', 'near_shared', 'standard_shared'),
                              ('aliased native mass', 'next-obs MSE (native held-out)', 'planning error, near goals', 'planning error, standard goals')):
            for a, col, off in (('ordinary', 'C3', -.12), ('weighted', 'C0', .12)):
                vals = [[(cells[(c, a, d, s)]['aliased_native_mass'] if k == 'aliased' else m(cells[(c, a, d, s)], k)) for s in SEEDS] for d in DIMS]
                for i, v in enumerate(vals):
                    ax.scatter(np.full(len(v), i)+off, v, s=12, color=col, alpha=.6)
                ax.plot(range(len(DIMS)), [np.median(v) for v in vals], color=col, label=a)
            ax.set_xticks(range(len(DIMS))); ax.set_xticklabels([f'D={d}' for d in DIMS])
            ax.set_title(f'{c}: {lab}', fontsize=9)
            if k == 'next_obs':
                ax.set_yscale('log')
    axes[0, 0].legend()
    fig.suptitle('2-d state (depth, lateral offset): SIGReg by embedding dimension and occupancy (dots = seeds)')
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT/'by_dimension.png', dpi=150)

    # D = 2 embeddings of the state grid: colour = log d, marker size shows b.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    dg = torch.exp(torch.linspace(0, np.log(16), 41, dtype=torch.float64))
    bg = torch.linspace(-BMAX, BMAX, 13, dtype=torch.float64)
    for row, a in enumerate(ARMS):
        for col, c in enumerate(CONDITIONS):
            model = MD.EncoderD(2); model.load_state_dict(trials.load(OUT/'runs'/f'{c}-{a}-d2-0.pt')['model']); model.double().eval()
            ax = axes[row, col]
            with torch.no_grad():
                for bv in bg:   # lines of constant offset b, coloured along depth
                    f = model(sensor(dg, torch.full_like(dg, float(bv)))).numpy()
                    ax.plot(f[:, 0], f[:, 1], color=plt.cm.coolwarm((float(bv)+BMAX)/(2*BMAX)), lw=.8)
                for dv in dg[::5]:  # lines of constant depth
                    f = model(sensor(torch.full_like(bg, float(dv)), bg)).numpy()
                    ax.plot(f[:, 0], f[:, 1], color=plt.cm.viridis(float(np.log(dv)/np.log(16))), lw=.8)
            ax.set_title(f'{a}, {c}, seed 0: aliased {cells[(c, a, 2, 0)]["aliased_native_mass"]:.2f}', fontsize=9)
            ax.set_aspect('equal', 'datalim')
    fig.suptitle('D = 2 embeddings of the (d, b) grid: red/blue = constant-offset lines, viridis = constant-depth lines')
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT/'d2_embeddings.png', dpi=150)
    print(json.dumps(dict(checks=checks, predictions=preds), indent=1))
    for k, v in by.items():
        print(k, v['aliased'], ' '.join(f'{kk}={v[kk]:.4f}' for kk in ('mean_aliased_mass', 'min_far_ratio_median', 'next_obs', 'near_shared', 'standard_shared', 'readout_b')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    assert R.sha(HERE/'PROTOCOL.md') == PROTOCOL_SHA, 'PROTOCOL.md changed after lock'
    if not a.report_only:
        for c in list(CONDITIONS)+['shared']:
            for s in SEEDS:
                dataset(c, s)
        specs = [(c, s, d, a_) for d in sorted(DIMS, reverse=True) for c in CONDITIONS for s in SEEDS for a_ in ARMS]
        with Pool(a.workers) as pool:
            for _ in pool.imap_unordered(job, specs):
                pass
    report()


if __name__ == '__main__':
    main()
