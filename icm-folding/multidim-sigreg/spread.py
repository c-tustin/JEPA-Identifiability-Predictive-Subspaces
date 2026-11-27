"""Exploratory (not pre-registered): where does the occupancy cost go when aliasing disappears?

Reads the trained checkpoints and writes output/exploratory_spread.json and paired planning counts.
From the workspace root: python3 icm-folding/multidim-sigreg/spread.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('multidim_run', HERE/'run.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def main():
    out = {}
    summary = json.loads((M.OUT/'summary.json').read_text())['cells']
    for arm in M.ARMS:
        for D in M.DIMS:
            regs, inball, ratio, maxnorm = [], [], [], []
            for s in M.SOURCES:
                sv = M.trials.load(M.OUT/'runs'/f'{arm}-d{D}-{s}.pt')
                m = M.load_model(sv)
                src = M.trials.load(M.R.SOURCES/f'icm-{s}.pt')
                c = M.B.Config(**src['config'])
                regs.append(sv['history'][-1]['regularizer'])
                with torch.no_grad():
                    z = m(src['data']['x'].double())
                    inball.append(float((z.norm(dim=1) < 3*np.sqrt(D)).double().mean()))

                    def step(lo, hi):
                        d = torch.exp(torch.linspace(np.log(lo), np.log(hi), 200, dtype=torch.float64))
                        return float((m(M.B.sensor(d+0.25, c))-m(M.B.sensor(d, c))).norm(dim=1).mean())
                    ratio.append(step(8, 15.7)/step(1, 2))
                    g = m(M.B.sensor(torch.exp(torch.linspace(0, np.log(16), 500, dtype=torch.float64)), c))
                    maxnorm.append(float(g.norm(dim=1).max()))
            out[f'{arm}/d{D}'] = dict(final_regularizer_median=float(np.median(regs)),
                                      native_fraction_within_3sqrtD=float(np.mean(inball)),
                                      far_over_near_move_displacement_median=float(np.median(ratio)),
                                      max_code_norm_median=float(np.median(maxnorm)))
    paired = {}
    for D in M.DIMS:
        for key in ('standard_shared', 'near_shared', 'next_obs'):
            o = [summary[f'ordinary/d{D}/{s}'][key] for s in M.SOURCES]
            w = [summary[f'weighted/d{D}/{s}'][key] for s in M.SOURCES]
            paired[f'd{D}/{key}'] = dict(weighted_better=int(sum(b < a for a, b in zip(o, w))), of=len(o))
    out['paired_weighted_better'] = paired
    (M.OUT/'exploratory_spread.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
