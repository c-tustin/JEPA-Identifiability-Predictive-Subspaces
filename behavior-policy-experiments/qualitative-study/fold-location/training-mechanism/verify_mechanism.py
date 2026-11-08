"""Independent checkpoint, sampled-loss, Adam-step and mixed-derivative audit."""

import copy
import hashlib
import json

import numpy as np
import torch

from run_mechanism import (
    HERE, OUT, ORIGINAL, DIAGNOSTIC_SEEDS, GAINS, ScaledEncoder, encoder_gain, trials,
)


def model_slope(model, q, c):
    q = q.detach().clone().requires_grad_()
    curve = model(trials.behavior.sensor(q.exp(),c))
    slope = torch.autograd.grad(curve.sum(),q)[0]
    return curve.detach(), slope.detach()


def main():
    torch.set_num_threads(1)
    settings = json.loads((OUT/"settings.json").read_text())
    assert hashlib.sha256((HERE/"run_mechanism.py").read_bytes()).hexdigest() == settings["code_sha256"]
    assert hashlib.sha256((HERE/"PROTOCOL.md").read_bytes()).hexdigest() == settings["protocol_sha256"]
    parent = json.loads((HERE.parent/"output"/"settings.json").read_text())
    repo = HERE.parents[3]
    for relative,expected in parent["source_sha256"].items():
        assert hashlib.sha256((repo/relative).read_bytes()).hexdigest() == expected
    c = trials.behavior.Config(**settings["config"])
    max_parameter_error, max_gradient_error, max_curve_error = 0.,0.,0.
    max_adam_error, max_jvp_error = 0.,0.
    snapshots, numerical_probes, models = 0,0,0
    for seed in settings["seeds"]:
        source_file = ORIGINAL/"sources"/f"icm-{seed}.pt"
        assert hashlib.sha256(source_file.read_bytes()).hexdigest() == parent["icm_data_sha256"][str(seed)]
        source = trials.load(source_file)["data"]
        ou = trials.study.controls.ou_data(c,seed)
        balanced = trials.replay_pairs(source,seed,"log")
        for gain in GAINS:
            with encoder_gain(gain):
                _,plan = trials.study.make_plan(c,seed,seed,"icm")
            plan["masks"] = torch.zeros_like(plan["masks"])
            reg_plan = copy.copy(plan)
            reg_plan["indices"] = torch.randint(len(source["x"]),plan["indices"].shape,
                generator=torch.Generator().manual_seed(1900000+seed))
            conditions = ("ordinary","balanced") if seed in DIAGNOSTIC_SEEDS else ("balanced",)
            for condition in conditions:
                run = trials.load(OUT/"runs"/f"{condition}-g{gain}-{seed}.pt")
                assert run["steps"] == 4000 and run["config"] == settings["config"]
                model = ScaledEncoder(gain)
                model.load_state_dict(run["encoder"])
                d = trials.representation(model,c,seed)
                error = float(np.max(abs(d["embedding"]-run["diagnostic"]["embedding"])))
                max_curve_error = max(max_curve_error,error)
                assert error < 1e-6 and d["folded"] == run["diagnostic"]["folded"]
                models += 1
                if condition == "ordinary":
                    folder = ORIGINAL if gain == 4 else HERE.parent/"output"/f"gain-{gain}"
                    reference = trials.load(folder/"encoders"/f"icm-{seed}.pt")["encoder"]
                    error = max(float((p-reference[k]).abs().max()) for k,p in run["encoder"].items())
                    max_parameter_error = max(max_parameter_error,error)
                    assert error < 1e-6
                frames = {f["step"]:f for f in run["frames"]}
                for step,snapshot in run["snapshots"].items():
                    model = ScaledEncoder(gain)
                    model.load_state_dict(snapshot["before"])
                    named = list(model.named_parameters())
                    x,xp = trials.study.controls.batch_at(source,ou,plan,step)
                    y = model(torch.cat([x,xp]))
                    alignment = (y[:len(x)]-y[len(x):]).square().mean()
                    rx,rxp = trials.study.controls.batch_at(balanced if condition=="balanced" else source,ou,reg_plan,step)
                    penalty = trials.behavior.SIGReg1D()(model(torch.cat([rx,rxp])))
                    gradients = (torch.autograd.grad(.9*alignment,[p for _,p in named]),
                                 torch.autograd.grad(.1*penalty,[p for _,p in named]))
                    for tag,gradient in zip(("a","r"),gradients):
                        for (name,_),g in zip(named,gradient):
                            error = float((g-snapshot[f"gradient_{tag}"][name]).abs().max())
                            max_gradient_error = max(max_gradient_error,error)
                            torch.testing.assert_close(g,snapshot[f"gradient_{tag}"][name],atol=1e-5,rtol=1e-5)
                    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
                    optimizer.load_state_dict(snapshot["optimizer_before"])
                    for (name,p),ga,gr in zip(named,*gradients):
                        p.grad = ga+gr
                    optimizer.step()
                    for name,p in named:
                        error = float((p-snapshot["after"][name]).abs().max())
                        max_adam_error = max(max_adam_error,error)
                        assert error < 1e-7
                    q = torch.tensor(run["log_distance"])
                    for which in ("before","after"):
                        model.load_state_dict(snapshot[which])
                        _,slope = model_slope(model,q,c)
                        np.testing.assert_allclose(slope.numpy(),frames[step][f"slope_{which}"],atol=1e-5,rtol=1e-4)
                    snapshots += 1
                    if step not in (0,250,1000):
                        continue
                    double_model = ScaledEncoder(gain).double()
                    double_model.load_state_dict({k:v.double() for k,v in snapshot["before"].items()})
                    for tag in ("a","r"):
                        epsilon = .001
                        slopes = []
                        for sign in (-1,1):
                            state = {k:v.double()+sign*epsilon*snapshot[f"direction_{tag}"][k].double() for k,v in snapshot["before"].items()}
                            double_model.load_state_dict(state)
                            _,slope = model_slope(double_model,q.double(),c)
                            slopes.append(slope)
                        numeric = ((slopes[1]-slopes[0])/(2*epsilon)).numpy()
                        saved = frames[step][f"delta_slope_{tag}"]
                        error = float(np.max(abs(numeric-saved)))
                        max_jvp_error = max(max_jvp_error,error)
                        np.testing.assert_allclose(numeric,saved,atol=1e-4,rtol=1e-3)
                        numerical_probes += 1
    assert models == 39 and snapshots == 144 and numerical_probes == 108
    trials.atomic_json(OUT/"verification.json",dict(
        passed=True,models=models,replayed_snapshots=snapshots,numerical_mixed_derivative_checks=numerical_probes,
        max_ordinary_reproduction_parameter_error=max_parameter_error,
        max_recomputed_sampled_gradient_error=max_gradient_error,
        max_recreated_curve_error=max_curve_error,max_replayed_adam_parameter_error=max_adam_error,
        max_numerical_mixed_derivative_error=max_jvp_error,
        source_and_protocol_hashes_unchanged=True))
    print("All 39 models, 144 sampled-loss/Adam snapshots and 108 mixed derivatives verified")


if __name__ == "__main__":
    main()
