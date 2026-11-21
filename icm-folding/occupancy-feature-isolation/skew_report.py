"""Score the pre-registered skewness predictions and draw the mechanism figure."""
import json
import pickle
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skew as K
import trace as T
import feature_bank as FB

OUT = K.OUT
R, torch, B = K.R, K.torch, K.B


def sigreg_force(ze, z, w):
    reg = B.SIGReg1D().double(); t = reg.frequency.numpy(); wt = reg.weight.numpy(); phi = reg.target.numpy()
    C = w @ np.cos(np.outer(z, t)); S = w @ np.sin(np.outer(z, t))
    return -((2*(C - phi)*(-t*np.sin(np.outer(ze, t))) + 2*S*(t*np.cos(np.outer(ze, t))))@wt)


def cell(law, method, init):
    p = OUT/'cells'/f'{law}-{method}-i{init}.json'
    if not p.exists(): return None
    r = json.loads(p.read_text())
    b = r['backward_fraction']
    return dict(backward=b, junction_fold=b > .1, tiny=0 < b <= .1, roots=[round(x, 3) for x in r['roots']])


def retrospective(c):
    """Skewness in init-131 f0 for every earlier audited fit with the same initializer."""
    grid = T.GRID.numpy(); f0 = K.f0_curve(131, c); rows = []
    for r in FB.cells():
        d, w = r['law']; w = w/w.sum(); y = np.interp(np.log(d), grid, f0)
        rows.append(dict(study=r['study'], name=r['name'], skew=K.skewness(w, y), near_mass=float(w[d < 2].sum()),
                         backward=r['backward'], junction_fold=r['backward'] > .1))
    return rows


def main():
    pred = json.loads((OUT/'PREDICTIONS.json').read_text())
    assert pred['protocol_hash'] == R.digest(K.PROTOCOL.read_bytes())
    ds, c, *_ = K.D.setup()
    A = []
    for law in pred['part_a']:
        for init in map(str, K.INITS):
            o = cell(law['name'], 'sigreg', init); p = law['prediction'][init]
            A.append(dict(law=law['name'], init=int(init), near_mass=law['near_mass'], skew=law['skew'][init],
                          mass_below=law['mass_below'], prediction=p, **(o or {}),
                          verdict=None if o is None or p is None else (o['junction_fold'] == p)))
    Bp = []
    for method in ['even_sigreg', 'variance_skew1', 'variance_skew10', 'infonce_skew1', 'infonce_skew10', 'variance', 'infonce']:
        for law in ['native-m0.5', 'native-m0.75', 'native-m0.9375', 'm0.5-g1.1', 'm0.9375-g2.15']:
            o = cell(law, method, 131)
            if o is not None: Bp.append(dict(method=method, law=law, **o))
    retro = retrospective(c)
    summary = dict(part_a=A, part_b=Bp, predictions_hash=R.digest((OUT/'PREDICTIONS.json').read_bytes()),
                   scored=sum(a['verdict'] is not None for a in A), correct=sum(a['verdict'] is True for a in A),
                   wrong=[a for a in A if a['verdict'] is False], retrospective=retro)
    R.trials.atomic_json(OUT/'summary.json', summary)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    # (a) traces
    for a, col in zip(['0.5', '0.75', '0.9375'], ['#4C78A8', '#E45756', '#54A24B']):
        z = np.load(HERE/f'output/trace/n130-f130-a{a}.npz'); dd = np.exp(z['grid'])
        for s, ls in [(100, ':'), (300, '--'), (4000, '-')]:
            ax[0].plot(dd, z['curves'][list(z['steps']).index(s)], ls, color=col, label=f'm={a}, step {s}')
    ax[0].set_xscale('log'); ax[0].set_ylim(-1.5, 4); ax[0].axvline(2, c='k', lw=.5)
    ax[0].set_xlabel('distance'); ax[0].set_ylabel('code'); ax[0].set_title('(a) code curves: monotone / fold / ejection')
    ax[0].legend(fontsize=6, ncol=3)
    # (b) exact force sign structure
    from scipy.stats import skewnorm
    zz = np.linspace(-4, 7, 1101)
    for a in [2, 6]:
        x = skewnorm.rvs(a, size=100000, random_state=0); x = (x - x.mean())/x.std()
        f = sigreg_force(zz, x, np.full(len(x), 1/len(x)))
        ax[1].plot(zz, f/np.abs(f).max(), label=f'skew {np.mean(x**3):.2f}')
    ax[1].axhline(0, c='k', lw=.5); ax[1].axvspan(1.1, 3.45, color='#E45756', alpha=.12, label='pull-back band')
    ax[1].set_xlabel('standardized code z'); ax[1].set_ylabel('SIGReg force (normalized)')
    ax[1].set_title('(b) exact SIGReg force on a skewed code law'); ax[1].legend(fontsize=7)
    # (c) skew vs near mass, all fits
    rm = np.array([r['near_mass'] for r in retro]); rs = np.array([r['skew'] for r in retro]); rf = np.array([r['junction_fold'] for r in retro])
    ax[2].scatter(rm[~rf], rs[~rf], s=12, facecolors='none', edgecolors='#888', label='earlier: no junction fold')
    ax[2].scatter(rm[rf], rs[rf], s=12, c='#E45756', alpha=.5, label='earlier: junction fold')
    cl = json.loads((OUT/'CLUSTER_PREDICTIONS.json').read_text())
    C = []
    for law in cl['laws']:
        for init in map(str, K.INITS):
            o = cell(law['name'], 'sigreg', init)
            C.append(dict(law=law['name'], init=int(init), near_mass=law['near_mass'], window=law['window'],
                          skew=law['skew'][init], h_skew=law['h_skew'][init], h_wall=law['h_wall'][init], **o))
    summary['cluster'] = C; R.trials.atomic_json(OUT/'summary.json', summary)
    a131 = [a for a in A + C if a['init'] == 131 and 'junction_fold' in a]
    for a in a131:
        ax[2].scatter(a['near_mass'], a['skew'], s=90, marker='*', c='#E45756' if a['junction_fold'] else 'white', edgecolors='k', zorder=5)
    ax[2].scatter([], [], s=90, marker='*', c='#E45756', edgecolors='k', label='new intervention: fold')
    ax[2].scatter([], [], s=90, marker='*', c='white', edgecolors='k', label='new intervention: no fold')
    ax[2].axhspan(.75, 2.6, color='#E45756', alpha=.08)
    ax[2].set_xlabel('mass below distance 2'); ax[2].set_ylabel('skewness of f0(d) under P')
    ax[2].set_title('(c) skewness vs near mass; shaded = pre-registered fold band'); ax[2].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(OUT/'skew_mechanism.png', dpi=130)
    print(json.dumps(dict(scored=summary['scored'], correct=summary['correct'], wrong=summary['wrong']), indent=1))
    for a in A: print(a['law'], a['init'], round(a['skew'], 2), a['prediction'], a.get('backward'), a.get('roots'), a['verdict'])
    for b in Bp: print(b['method'], b['law'], round(b['backward'], 4), b['roots'])


if __name__ == '__main__':
    main()
