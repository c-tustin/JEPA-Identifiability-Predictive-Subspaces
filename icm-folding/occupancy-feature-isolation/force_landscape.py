"""SIGReg's per-sample force is the negative slope of a smoothed density-excess landscape.

With frequencies on [0, 3] and Gaussian weights, SIGReg1D is N·MMD² with kernel ≈ √(2π)·exp(-u²/2). For a code law Q,
the gradient with respect to one code u is proportional to the slope of

    E(u) = (Q * φ)(u) − φ_√2(u),

the code histogram blurred by a unit Gaussian, minus the N(0, 2) density (= N(0,1) blurred by the same kernel).
Codes slide downhill on E. Its extrema are the force zero-crossings. This script checks the identity against the exact
SIGReg1D gradient on skewed laws and on the three native initial code laws, and saves the landscapes.
"""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from scipy.stats import norm, skewnorm
import skew as K
import skew_report as SR

U = np.linspace(-4, 8, 1201)


def excess(u, z, w):
    return (w[None, :]*norm.pdf(u[:, None] - z[None, :])).sum(1) - norm.pdf(u, scale=np.sqrt(2))


def zeros(u, f): return [float(v) for v in u[np.flatnonzero(np.diff(np.sign(f)))]]


def check(z, w):
    F = SR.sigreg_force(U, z, w); E = excess(U, z, w); slope = -np.gradient(E, U)
    return dict(corr=float(np.corrcoef(F, slope)[0, 1]), force_zeros=zeros(U, F), landscape_extrema=zeros(U, slope),
                excess=E.tolist())


def main():
    out = {}
    for a in (2, 6):
        x = skewnorm.rvs(a, size=40000, random_state=0); x = (x - x.mean())/x.std()
        out[f'skewnorm-{a}'] = check(x, np.full(len(x), 1/len(x)))
    ds, c, *_ = K.D.setup(); _, d = K.atoms(ds); grid = K.T.GRID.numpy(); f0 = K.f0_curve(131, c)
    for m in ('0.5', '0.75', '0.9375'):
        w = np.load(K.OUT/'laws'/f'native-m{m}.npy'); y = np.interp(np.log(d), grid, f0)
        mu = w @ y; z = (y - mu)/np.sqrt(w @ (y - mu)**2)
        b = np.clip(np.rint((z + 4)/.01).astype(int), 0, 1200); wb = np.bincount(b, weights=w, minlength=1201); zb = -4 + .01*np.arange(1201)
        keep = wb > 0; out[f'native-m{m}'] = check(zb[keep], wb[keep])
    K.R.trials.atomic_json(HERE/'output'/'force_landscape.json', dict(u=U.tolist(), laws=out))
    for k, v in out.items(): print(k, 'corr', round(v['corr'], 5), 'force zeros', np.round(v['force_zeros'], 2), 'extrema', np.round(v['landscape_extrema'], 2))


if __name__ == '__main__':
    main()
