"""Independently replay every continuation and check every saved checkpoint."""

import copy
import json

import numpy as np
import torch

from run_continuation import ARMS, HERE, OUT, ORIGINAL, PARENT_OUT, SEEDS, STEPS, ScaledEncoder, digest, trials


def main():
    torch.set_num_threads(1)
    settings = json.loads((OUT/"settings.json").read_text())
    assert settings["code_sha256"] == digest(HERE/"run_continuation.py")
    assert settings["protocol_sha256"] == digest(HERE/"PROTOCOL.md")
    rows = []
    checked_states = 0
    max_curve_error, max_slope_error = 0., 0.
    for seed in SEEDS:
        assert settings["sources"][str(seed)] == digest(ORIGINAL/"sources"/f"icm-{seed}.pt")
        assert settings["original_checkpoints"][str(seed)] == digest(PARENT_OUT/"runs"/f"ordinary-g4-{seed}.pt")
        start = trials.load(OUT/"starts"/f"seed-{seed}.pt")
        original = trials.load(PARENT_OUT/"runs"/f"ordinary-g4-{seed}.pt")
        assert all(torch.equal(v, original["encoder"][k]) for k, v in start["encoder"].items())
        assert start["optimizer"]["param_groups"][0]["lr"] == 0.
        source = trials.load(ORIGINAL/"sources"/f"icm-{seed}.pt")["data"]
        config = trials.behavior.Config(**start["config"])
        for arm, (lr, balance) in ARMS.items():
            saved = trials.load(OUT/"runs"/f"{arm}-{seed}.pt")
            model = ScaledEncoder(4)
            model.load_state_dict(start["encoder"])
            optimizer = torch.optim.Adam(model.parameters())
            # Adam mutates loaded moment tensors. Each independent replay must
            # get its own copy of the original, unchanged optimizer state.
            optimizer.load_state_dict(copy.deepcopy(start["optimizer"]))
            optimizer.param_groups[0]["lr"] = lr
            regularizer = trials.behavior.SIGReg1D()
            reg_source = trials.replay_pairs(source, seed, "log") if balance else source
            alignment_rng = torch.Generator().manual_seed(2100000+seed)
            reg_rng = torch.Generator().manual_seed(2200000+seed)
            maximum = 0.
            for step in range(STEPS+1):
                if step % 500 == 0:
                    expected = saved["saved_states"][step]
                    error = max(float((v-expected[k]).abs().max()) for k,v in model.state_dict().items())
                    maximum = max(maximum, error)
                    assert error == 0., (seed, arm, step, error)
                    checked_states += 1
                    q = torch.linspace(0, np.log(16), 1025, requires_grad=True)
                    y = model(trials.behavior.sensor(q.exp(), config))
                    slope = torch.autograd.grad(y.sum(), q)[0]
                    frame = saved["frames"][step//500]
                    curve_error = float(np.max(np.abs(y.detach().numpy()-frame["curve"])))
                    slope_error = float(np.max(np.abs(slope.detach().numpy()-frame["slope"])))
                    max_curve_error = max(max_curve_error, curve_error)
                    max_slope_error = max(max_slope_error, slope_error)
                    assert curve_error == slope_error == 0.
                if step == STEPS:
                    break
                ix = torch.randint(len(source["x"]), (512,), generator=alignment_rng)
                ri = torch.randint(len(reg_source["x"]), (512,), generator=reg_rng)
                y = model(torch.cat([source["x"][ix], source["xp"][ix]]))
                alignment = (y[:512]-y[512:]).square().mean()
                penalty = regularizer(model(torch.cat([reg_source["x"][ri], reg_source["xp"][ri]])))
                optimizer.zero_grad(set_to_none=True)
                (.9*alignment+.1*penalty).backward()
                optimizer.step()
            assert torch.equal(alignment_rng.get_state(), saved["alignment_rng"])
            assert torch.equal(reg_rng.get_state(), saved["reg_rng"])
            # Full optimizer moments, not merely final model outputs.
            for key, state in optimizer.state_dict()["state"].items():
                for field, value in state.items():
                    assert torch.equal(value, saved["optimizer"]["state"][key][field]), (seed, arm, field)
            rows.append(dict(seed=seed, arm=arm, replayed_updates=STEPS, max_parameter_difference=maximum))
            print(f"Verified {arm} seed {seed}: 8,000 replayed updates", flush=True)
    result = dict(passed=True, runs=rows, checkpoint_states_checked=checked_states,
                  total_replayed_updates=len(rows)*STEPS,
                  max_curve_difference=max_curve_error, max_slope_difference=max_slope_error,
                  scope="Exact replay of all continuation weights, optimizer moments and RNG states; all saved curves and slopes. This does not establish convergence.")
    trials.atomic_json(OUT/"verification.json", result)


if __name__ == "__main__":
    main()
