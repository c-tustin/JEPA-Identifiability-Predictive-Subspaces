"""Build the fold explainer page (fold-explainer.html) from saved fits and figures' data.

  python3 explainer/build.py
Embeds: initial histograms and excess landscapes for near mass 0.5–0.975 (baseline encoder, init 131), final code
curves, race replays and the phase map. Requires race.py, phase.py and the geometry/kernel-test fits to have run.
"""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
import geometry as G
import kernel_test  # noqa: F401  (registers the tanh(kx) input maps)
import force_landscape as FL

K = G.K
ROOT = HERE.parent/'output'


def main():
    ds, c, *_ = K.D.setup(); x, d = K.atoms(ds); near = d < 2
    y = np.interp(np.log(d), K.T.GRID.numpy(), K.f0_curve(131, c))
    U = np.round(np.arange(-3, 7.001, .05), 2); out = {'u': U.tolist(), 'masses': [], 'hist_edges': [-2, 7, .1]}
    for m in np.round(np.arange(.5, .976, .025), 3):
        w = K.tilted(m, 0, 0, y, near); mu = w @ y; z = (y - mu)/np.sqrt(w @ (y - mu)**2)
        b = np.clip(np.rint((z + 4)/.01).astype(int), 0, 1200); wb = np.bincount(b, weights=w, minlength=1201)
        zb = -4 + .01*np.arange(1201); k = wb > 0
        slope = -np.gradient(FL.excess(FL.U, zb[k], wb[k]), FL.U); zs = [q for q in FL.zeros(FL.U, slope) if q > 0]
        edges = np.arange(-2, 7.01, .1); fw = w[~near]/w[~near].sum()
        out['masses'].append(dict(m=float(m), excess=np.round(FL.excess(U, zb[k], wb[k]), 5).tolist(),
                                  band=[round(zs[0], 2), round(zs[1], 2)] if len(zs) > 1 else None,
                                  hist_near=np.round(np.histogram(z[near], edges, weights=w[near])[0], 5).tolist(),
                                  hist_far=np.round(np.histogram(z[~near], edges, weights=w[~near])[0], 5).tolist(),
                                  tail_end=round(float(z[~near].max()), 2),
                                  tail_median=round(float(np.interp(.5, np.cumsum(fw), z[~near])), 2), skew=round(float(w @ z**3), 2)))
    D = torch.exp(torch.linspace(0, np.log(16), 161, dtype=torch.float64)); out['curve_d'] = D.numpy().round(3).tolist(); out['curves'] = {}
    for mp in ('tanh4', 'linear', 'tanh8x'):
        for law in ('m0.5', 'm0.75', 'm0.9375'):
            p = next(q for q in (ROOT/s/'runs'/f'{law}-{mp}-i131.pt' for s in ('geometry', 'kernel-test', 'phase')) if q.exists())
            m = G.MappedEncoder(mp); m.load_state_dict(K.R.trials.load(p)['model'])
            with torch.no_grad(): out['curves'][f'{mp}|{law}'] = m(K.B.sensor(D, c).float()).double().numpy().round(4).tolist()
    out['race'] = {mp: [[q['step'], round(q['far_end'], 3), round(q['far_median'], 3), round(q['near_spread'], 3), round(q['backward'], 3)]
                        for q in json.loads((ROOT/'geometry'/'race'/f'm0.75-{mp}-i131.json').read_text())['rows'] if q['step'] <= 2000]
                   for mp in ('tanh4', 'linear')}
    out['phase'] = [{k: r[k] for k in ('law', 'near_mass', 'input_map', 'k', 'state', 'backward')} for r in json.loads((ROOT/'phase.json').read_text())]
    html = (HERE/'template.html').read_text().replace('__DATA__', json.dumps(out, separators=(',', ':')))
    (HERE/'fold-explainer.html').write_text(html); print('wrote', HERE/'fold-explainer.html', len(html)//1024, 'KB')


if __name__ == '__main__':
    main()
