"""Verify all removal curves/scores and replay two difficult-case fits."""

import hashlib
import json

import numpy as np
import torch

from run_crossed import HERE, LEVELS, OUT, EVAL_SEED, behavior, controls, reliability, geometry, load, reference, sha
from report_crossed import fold_flag


def mask(source, data_seed, arm):
    reflected = (source["action"] == 0) & (source["x"][:, 1] > 1 / 1.25)
    if arm == "remove_reflection":
        return reflected
    remove = torch.zeros_like(reflected)
    generator = torch.Generator().manual_seed(2400000 + data_seed)
    permutation = torch.randperm(len(remove), generator=generator)
    remove[permutation[:int(reflected.sum())]] = True
    return remove


def main():
    torch.set_num_threads(1)
    settings = json.loads((OUT / "reflection-control-settings.json").read_text())
    assert settings["code_sha256"] == sha(HERE / "run_reflection_control.py")
    assert settings["protocol_sha256"] == sha(HERE / "REFLECTION_PROTOCOL.md")
    c = behavior.Config()
    common = reference(c, EVAL_SEED)
    summary = json.loads((OUT / "reflection-control-summary.json").read_text())
    checked = 0
    max_curve_error = 0.
    fold_counts = {"remove_reflection": 0, "remove_random": 0}
    for d in LEVELS:
        source = load(OUT / "sources" / f"icm-{d}.pt")
        expected_count = int(mask(source, d, "remove_reflection").sum())
        for i in LEVELS:
            for arm in fold_counts:
                saved = load(OUT / "reflection-controls" / f"{arm}-data{d}-init{i}.pt")
                remove = mask(source, d, arm)
                assert int(remove.sum()) == saved["removed_pairs"] == expected_count
                assert hashlib.sha256(remove.numpy().tobytes()).hexdigest() == saved["mask_sha256"]
                model = behavior.Encoder()
                model.load_state_dict(saved["encoder"])
                recreated = geometry(model, c)
                error = max(float(np.max(np.abs(recreated[k] - saved["final"][k]))) for k in ("curve", "slope"))
                assert error == 0.
                max_curve_error = max(max_curve_error, error)
                metric, _ = controls.evaluate(model, common, c, EVAL_SEED)
                assert abs(metric["common_grid_r2"] - saved["metrics"]["common_grid_r2"]) < 1e-12
                assert metric["common_grid_r2"] == summary["matrices"][arm][LEVELS.index(d)][LEVELS.index(i)]
                fold_counts[arm] += int(fold_flag(model, c))
                checked += 1
    for arm in fold_counts:
        assert fold_counts[arm] == summary["folds"][arm]
    gate = load(OUT / "reflection-controls" / "all_pairs-data8-init8.pt")
    baseline = load(OUT / "crossed" / "icm-data8-init8.pt")
    assert all(torch.equal(v, baseline["encoder"][k]) for k, v in gate["encoder"].items())
    errors = []
    # Chosen for verification after looking at the matrices, not an inference sample.
    for arm in fold_counts:
        source = load(OUT / "sources" / "icm-9.pt")
        remove = mask(source, 9, arm)
        model, plan = reliability.make_plan(c, 9, 10, "icm")
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        regularizer = behavior.SIGReg1D()
        for step in range(c.jepa_steps):
            ix = plan["indices"][step]
            y = model(torch.cat([source["x"][ix], source["xp"][ix]]))
            alignment = ((y[:512] - y[512:]).square() * (~remove[ix]).float()).mean()
            penalty = regularizer(y)
            optimizer.zero_grad(set_to_none=True)
            (.9 * alignment + .1 * penalty).backward()
            optimizer.param_groups[0]["lr"] = controls.learning_rate(step, c.jepa_steps)
            optimizer.step()
        saved = load(OUT / "reflection-controls" / f"{arm}-data9-init10.pt")
        error = max(float((v - saved["encoder"][k]).abs().max()) for k, v in model.state_dict().items())
        assert error == 0.
        errors.append(error)
        print(f"Replayed {arm}, data 9, init 10", flush=True)
    result = dict(passed=True, checked_models=checked, fold_counts=fold_counts,
                  max_curve_or_slope_difference=max_curve_error, all_pairs_gate_max_difference=0.,
                  full_replays=2, full_replay_parameter_errors=errors,
                  scope="All 50 scores/curves/fold flags/masks, original all-pairs gate, and two full 4,000-update control replays. Not all 50 training histories replayed.")
    (OUT / "reflection-control-verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
