"""Locate the first in-range reversal in the chosen seed-131/gain-4 example.

Replay the unchanged path from initialization through update 250. Inspect
every update, so an earlier transient root cannot hide between saved frames.
At the first endpoint sign change, apply each actual Adam numerator component
separately, with the shared denominator. Retain both counterfactual outcomes.
"""

import copy
import json

import numpy as np
import torch

from run_mechanism import (
    OUT, ORIGINAL, ScaledEncoder, encoder_gain, frozen_parameters,
    adam_directions, trials,
)
from verify_root_motion import refined_roots, derivative


def grid_slope(model, c, n=2049):
    q = torch.linspace(0,np.log(16),n,requires_grad=True)
    y = model(trials.behavior.sensor(q.exp(),c))
    return q.detach().numpy(), torch.autograd.grad(y.sum(),q)[0].detach().numpy()


def outcome(state, gain, c):
    model = ScaledEncoder(gain).double()
    model.load_state_dict({k:v.double() for k,v in state.items()})
    q = torch.linspace(0,np.log(16),2049,dtype=torch.float64,requires_grad=True)
    y = model(trials.behavior.sensor(q.exp(),c))
    slope = torch.autograd.grad(y.sum(),q)[0].detach().numpy()
    points = refined_roots(model,c,q.detach().numpy(),slope)
    return dict(near_endpoint_slope=derivative(model,c,0.),
                far_endpoint_slope=derivative(model,c,float(np.log(16))),
                turning_distances=[float(np.exp(x)) for x in points])


def main():
    torch.set_num_threads(1)
    seed,gain = 131,4
    run = trials.load(OUT/"runs"/"ordinary-g4-131.pt")
    c = trials.behavior.Config(**run["config"])
    source = trials.load(ORIGINAL/"sources"/f"icm-{seed}.pt")["data"]
    ou = trials.study.controls.ou_data(c,seed)
    with encoder_gain(gain):
        model,plan = trials.study.make_plan(c,seed,seed,"icm")
    plan["masks"] = torch.zeros_like(plan["masks"])
    reg_plan = copy.copy(plan)
    reg_plan["indices"] = torch.randint(len(source["x"]),plan["indices"].shape,
        generator=torch.Generator().manual_seed(1900000+seed))
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    named = list(model.named_parameters())
    moment_a = {name:torch.zeros_like(p) for name,p in named}
    moment_r = {name:torch.zeros_like(p) for name,p in named}
    regularizer = trials.behavior.SIGReg1D()
    event, trace = None, []
    for step in range(250):
        if step == 100:
            target100 = run["snapshots"][100]["before"]
            assert all(torch.equal(p,target100[k]) for k,p in model.state_dict().items())
        lr = trials.study.controls.learning_rate(step,c.jepa_steps)
        optimizer.param_groups[0]["lr"] = lr
        x,xp = trials.study.controls.batch_at(source,ou,plan,step)
        y = model(torch.cat([x,xp]))
        alignment = (y[:len(x)]-y[len(x):]).square().mean()
        rx,rxp = trials.study.controls.batch_at(source,ou,reg_plan,step)
        penalty = regularizer(model(torch.cat([rx,rxp])))
        ga = torch.autograd.grad(.9*alignment,[p for _,p in named])
        gr = torch.autograd.grad(.1*penalty,[p for _,p in named])
        before = frozen_parameters(model)
        before_optimizer = copy.deepcopy(optimizer.state_dict()) if event is None else None
        q,bs = grid_slope(model,c)
        optimizer.zero_grad(set_to_none=True)
        for (name,p),a,r in zip(named,ga,gr):
            p.grad = a+r
            moment_a[name].mul_(.9).add_(a,alpha=.1)
            moment_r[name].mul_(.9).add_(r,alpha=.1)
        optimizer.step()
        _,ass = grid_slope(model,c)
        before_roots = bool(np.any(bs[:-1]*bs[1:]<0))
        after_roots = bool(np.any(ass[:-1]*ass[1:]<0))
        trace.append(dict(step=step,before_has_root=before_roots,after_has_root=after_roots,
                          before_far_slope=float(bs[-1]),after_far_slope=float(ass[-1])))
        if event is not None or before_roots or not after_roots:
            continue
        da,dr,_ = adam_directions(optimizer,named,moment_a,moment_r,step+1,lr)
        after = frozen_parameters(model)
        a_only = {k:before[k]+da[k] for k in before}
        r_only = {k:before[k]+dr[k] for k in before}
        event = dict(seed=seed,gain=gain,update_before=step,update_after=step+1,
            before=outcome(before,gain,c),after=outcome(after,gain,c),
            alignment_moment_only=outcome(a_only,gain,c),sigreg_moment_only=outcome(r_only,gain,c),
            note="One-step parameter interventions at the actual before-state and shared Adam denominator. Not independent optimizers or alternate full training histories.")
        trials.save_torch(OUT/"birth-event.pt",dict(
            before=before,after=after,alignment_only=a_only,sigreg_only=r_only,
            optimizer_before=before_optimizer,direction_a=da,direction_r=dr,
            event=event,log_distance=q,before_slope=bs,after_slope=ass))
    # The replay must finish at the independently saved before-update-250 state.
    target = run["snapshots"][250]["before"]
    error = max(float((p-target[k]).abs().max()) for k,p in model.state_dict().items())
    assert error == 0.0
    if event is None:
        raise RuntimeError("No first in-range root in the fixed inspection interval")
    event["replayed_update250_parameter_error"] = error
    event["all_updates_inspected_from_initialization"] = 250
    event["derivative_grid_points"] = 2049
    trials.atomic_json(OUT/"birth-event.json",event)
    trials.atomic_json(OUT/"birth-interval-trace.json",trace)
    print(json.dumps(event,indent=2))


if __name__ == "__main__":
    main()
