"""Check the implicit-root prediction using refined roots, not grid positions."""

import csv
import json

import numpy as np
import torch

from run_mechanism import OUT, ScaledEncoder, trials


def derivative(model, c, q):
    point = torch.tensor([q], dtype=torch.float64, requires_grad=True)
    output = model(trials.behavior.sensor(point.exp(), c))
    return float(torch.autograd.grad(output.sum(), point)[0])


def refined_roots(model, c, grid, slope):
    found = []
    for k in np.flatnonzero(slope[:-1] * slope[1:] < 0):
        low, high = float(grid[k]), float(grid[k+1])
        left, right = derivative(model,c,low), derivative(model,c,high)
        if left*right >= 0:
            continue
        for _ in range(32):
            middle = (low+high)/2
            value = derivative(model,c,middle)
            if left*value <= 0:
                high = middle
            else:
                low, left = middle, value
        found.append((low+high)/2)
    return found


def main():
    torch.set_num_threads(1)
    rows = []
    for path in sorted((OUT/"runs").glob("*.pt")):
        run = trials.load(path)
        if not run["detailed"]:
            continue
        c = trials.behavior.Config(**run["config"])
        frames = {f["step"]:f for f in run["frames"]}
        model = ScaledEncoder(run["gain"]).double()
        for step,snapshot in run["snapshots"].items():
            frame = frames[step]
            before = {k:v.double() for k,v in snapshot["before"].items()}
            after = {k:v.double() for k,v in snapshot["after"].items()}
            model.load_state_dict(before)
            initial = refined_roots(model,c,run["log_distance"],frame["slope_before"])
            model.load_state_dict(after)
            final = refined_roots(model,c,run["log_distance"],frame["slope_after"])
            for root in initial:
                row = dict(seed=run["seed"],gain=run["gain"],condition=run["condition"],step=step,
                           root_before_log=root,matched=False,reason="",actual_log_shift=None,
                           predicted_split_shift=None,predicted_actual_direction_shift=None,
                           split_prediction_error=None,actual_direction_prediction_error=None)
                nearby = [q for q in final if abs(q-root)<.05]
                if len(nearby)!=1:
                    row["reason"] = "No unique local after-root"
                    rows.append(row)
                    continue
                model.load_state_dict(before)
                point = torch.tensor([root],dtype=torch.float64,requires_grad=True)
                y = model(trials.behavior.sensor(point.exp(),c))
                slope = torch.autograd.grad(y.sum(),point,create_graph=True)[0]
                curvature = float(torch.autograd.grad(slope,point,retain_graph=True)[0])
                if abs(curvature)<1e-3*max(float(np.std(frame["curve_before"])),1e-8):
                    row["reason"] = "Near-degenerate root"
                    rows.append(row)
                    continue
                named = list(model.named_parameters())
                # The output bias has exactly zero spatial derivative, so it
                # is correctly absent from this mixed-derivative graph.
                raw_mixed = torch.autograd.grad(slope,[p for _,p in named],allow_unused=True)
                mixed = [torch.zeros_like(p) if j is None else j for (_,p),j in zip(named,raw_mixed)]
                a = sum(float((j*snapshot["direction_a"][name].double()).sum()) for (name,_),j in zip(named,mixed))
                r = sum(float((j*snapshot["direction_r"][name].double()).sum()) for (name,_),j in zip(named,mixed))
                actual_direction = sum(float((j*(after[name]-before[name])).sum()) for (name,_),j in zip(named,mixed))
                observed = nearby[0]-root
                predicted, predicted_actual = -(a+r)/curvature, -actual_direction/curvature
                row.update(matched=True,reason="Refined simple roots",actual_log_shift=observed,
                    predicted_split_shift=predicted,predicted_actual_direction_shift=predicted_actual,
                    split_prediction_error=predicted-observed,
                    actual_direction_prediction_error=predicted_actual-observed)
                rows.append(row)
    matched = [r for r in rows if r["matched"]]
    def quantiles(key):
        return np.quantile([abs(r[key]) for r in matched],[.5,.9,.99,1]).tolist()
    result = dict(refined_matched_roots=len(matched),unmatched_or_degenerate=len(rows)-len(matched),
        bisection_iterations=32,split_prediction_error_quantiles=quantiles("split_prediction_error"),
        actual_direction_prediction_error_quantiles=quantiles("actual_direction_prediction_error"),
        note="Held checkpoint weights, double-precision root refinement and exact curvature. This validates a local approximation, not a global convergence theorem.")
    with (OUT/"refined-root-motion.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    trials.atomic_json(OUT/"refined-root-motion-check.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
