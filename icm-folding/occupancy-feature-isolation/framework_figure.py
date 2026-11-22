"""Draw the fold-framework figure (FRAMEWORK.md) from saved traces, audited cells and the geometry test."""
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
import skew_report as SR
import feature_bank as FB

OUT = K.S.OUT
COL = {'0.5': '#2a78d6', '0.75': '#eb6834', '0.9375': '#1baf7a'}
MAPCOL = {'tanh4': '#2a78d6', 'linear': '#eb6834', 'tanh1': '#1baf7a'}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6',
                     'axes.edgecolor': '#888', 'font.size': 9})


def standardized(curve, grid, d, w):
    y = np.interp(np.log(d), grid, curve); mu = w @ y; sd = np.sqrt(w @ (y - mu)**2); return (y - mu)/sd, sd


def two_lump():
    rows = []
    for m in np.linspace(.5, .985, 98):
        r = 1 - m; z = np.array([-np.sqrt(r/m), np.sqrt(m/r)]); F = SR.sigreg_force(z, z, np.array([m, r]))
        rows.append(dict(m=m, tail_z=z[1], skew=(2*m - 1)/np.sqrt(m*r), tail_over_cluster=abs(F[1])/abs(F[0])))
    return rows


def main():
    ds, c, *_ = K.D.setup(); _, d = K.atoms(ds); near = d < 2
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.4)); ax = ax.ravel()
    # (a) initial standardized code law: cluster + tail
    for a in COL:
        z = np.load(HERE/f'output/trace/n130-f130-a{a}.npz'); w = np.load(OUT/'skew'/'laws'/f'native-m{a}.npy')
        zz, _ = standardized(z['curves'][0], z['grid'], d, w)
        h, e = np.histogram(zz, bins=np.linspace(-2, 7, 91), weights=w)
        ax[0].step(e[:-1], h, where='post', color=COL[a], lw=2, label=f'near mass {a}')
    ax[0].axvspan(1.1, 3.45, color='#eb6834', alpha=.08)
    ax[0].text(2.27, .55, 'SIGReg pull-back\nband', ha='center', color='#555', fontsize=8)
    ax[0].set_yscale('log'); ax[0].set_ylim(1e-3, 1); ax[0].set_xlabel('standardized code at initialization')
    ax[0].set_ylabel('probability per bin'); ax[0].set_title('(a) Start: a spike (near states) plus a tail (far states)', loc='left')
    ax[0].legend(frameon=False)
    # (b) the race: same occupancy (m = 0.75), two input maps
    for mp, col, lab in [('tanh4', '#2a78d6', 'tanh(4x)'), ('linear', '#eb6834', 'x')]:
        r = json.loads((OUT/'geometry'/'race'/f'm0.75-{mp}-i131.json').read_text())['rows']
        st = [q['step'] for q in r if q['step'] <= 2000]; q = [q for q in r if q['step'] <= 2000]
        ax[1].plot(st, [v['far_end'] for v in q], color=col, lw=2, label=f'{lab}: far end of tail')
        ax[1].plot(st, [v['far_median'] for v in q], color=col, lw=1.2, ls='--', label=f'{lab}: tail median')
        ax[1].plot(st, [v['near_spread'] for v in q], color=col, lw=1.2, ls=':', label=f'{lab}: spread of near cluster')
    ax[1].set_xlabel('training step'); ax[1].set_ylabel('standardized code units')
    ax[1].set_title('(b) Same occupancy (m = 0.75), two encoders', loc='left'); ax[1].legend(frameon=False, ncol=2, fontsize=7)
    # (c) two-lump idealization vs all audited cells
    tl = two_lump(); m = np.array([r['m'] for r in tl])
    rows = [(float(1 - (w/w.sum())[dd >= 2].sum()), r_['backward'] > .1) for r_ in FB.cells() for dd, w in [r_['law']]]
    for p in sorted((OUT/'skew'/'cells').glob('*-sigreg-i131.json')):
        j = json.loads(p.read_text()); w = np.load(OUT/'skew'/'laws'/f"{j['law']}.npy")
        rows.append((float(w[near].sum()), j['backward_fraction'] > .1))
    nm = np.array([r[0] for r in rows]); fo = np.array([r[1] for r in rows])
    bins = np.array([0, .45, .55, .65, .72, .78, .84, .89, .915, .93, .945, 1.])
    centers, frac, n = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (nm >= lo) & (nm < hi)
        if s.sum(): centers.append(nm[s].mean()); frac.append(fo[s].mean()); n.append(s.sum())
    ax[2].plot(centers, frac, 'o-', color='#2a78d6', ms=7, lw=2)
    for x_, y_, k in zip(centers, frac, n): ax[2].annotate(f'{k}', (x_, y_), textcoords='offset points', xytext=(7, -3), fontsize=7, color='#555')
    ax[2].axvspan(.4, .6, color='#2a78d6', alpha=.06); ax[2].axvspan(.92, 1, color='#1baf7a', alpha=.08)
    ax[2].text(.5, 1.12, 'balanced\n(monotone)', ha='center', fontsize=8, color='#555')
    ax[2].text(.76, 1.12, 'skewed, tail in reach\n(fold)', ha='center', fontsize=8, color='#555')
    ax[2].text(.96, 1.12, 'tail out\nof reach\n(eject)', ha='center', fontsize=8, color='#555')
    ax[2].set_xlim(.4, 1); ax[2].set_ylim(0, 1.3); ax[2].set_xlabel('near mass m'); ax[2].set_ylabel('fraction of fits that fold')
    ax[2].set_title('(c) Baseline encoder: 307 audited fits (labels = n)', loc='left')
    # (d) geometry test
    sp = OUT/'geometry'/'summary.json'
    if sp.exists():
        g = json.loads(sp.read_text())['rows']
        for mp, lab in [('tanh4', 'tanh(4x) (baseline)'), ('linear', 'x'), ('tanh1', 'tanh(x)')]:
            for i, mk in [(131, 'o'), (133, 's')]:
                r = sorted([r for r in g if r['input_map'] == mp and r['init'] == i], key=lambda r: r['near_mass'])
                ax[3].plot([q['near_mass'] for q in r], [q['backward'] for q in r], marker=mk, color=MAPCOL[mp], lw=1.5,
                           ls='-' if i == 131 else '--', ms=6, label=f'{lab}, init {i}')
        ax[3].axhline(.1, color='#555', lw=1, ls=':'); ax[3].text(.13, .11, 'fold threshold', fontsize=7, color='#555')
        ax[3].set_xlabel('near mass m'); ax[3].set_ylabel('backward fraction')
        ax[3].set_title('(d) Same occupancies, three input maps (x and tanh(x) overlap at 0)', loc='left'); ax[3].legend(frameon=False, fontsize=7)
    plt.tight_layout(); plt.savefig(HERE/'output'/'framework.png', dpi=140)
    R = K.R; R.trials.atomic_json(HERE/'output'/'framework.json', dict(two_lump=tl, near_mass_bins=dict(centers=centers, fold_fraction=frac, n=[int(k) for k in n])))


if __name__ == '__main__':
    main()


def phase_figure():
    import matplotlib.patches as mp_
    rows = json.loads((HERE/'output'/'phase.json').read_text())
    maps = ['linear', 'tanh1', 'tanh1.5x', 'tanh2x', 'tanh2.5x', 'tanh3x', 'tanh4', 'tanh6x', 'tanh8x']
    labels = ['x', 'tanh(x)', 'tanh(1.5x)', 'tanh(2x)', 'tanh(2.5x)', 'tanh(3x)', 'tanh(4x)\nbaseline', 'tanh(6x)', 'tanh(8x)']
    laws = ['m0.5', 'm0.625', 'm0.75', 'm0.875', 'm0.9375']
    col = {'fold': '#eb6834', 'inflated': '#1baf7a', 'stretched': '#86b6ef'}
    txt = {'fold': 'fold', 'inflated': 'inflated', 'stretched': 'stretched'}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for r in rows:
        i, j = maps.index(r['input_map']), laws.index(r['law'])
        ax.add_patch(mp_.Rectangle((i + .03, j + .03), .94, .94, color=col[r['state']], lw=0))
        ax.text(i + .5, j + .5, txt[r['state']], ha='center', va='center', fontsize=7.5, color='#1a1a19')
    ax.set_xlim(0, len(maps)); ax.set_ylim(0, len(laws)); ax.grid(False)
    ax.set_xticks(np.arange(len(maps)) + .5, labels, fontsize=8); ax.set_yticks(np.arange(len(laws)) + .5, [l[1:] for l in laws])
    ax.set_xlabel('encoder input map (squashing of the near-wall region increases →)'); ax.set_ylabel('near mass m')
    ax.set_title('End state after 4,000 updates, initialization 131 (exploratory map; fold vs no-fold in the 0.75 and 0.875 rows repeats at 2–4 initializations)', loc='left', fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout(); plt.savefig(HERE/'output'/'phase.png', dpi=140)


if __name__ == '__main__' and 'phase' in sys.argv:
    phase_figure()
