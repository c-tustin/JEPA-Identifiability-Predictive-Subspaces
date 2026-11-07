"""Recreate saved curves and check their derivatives from model checkpoints."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_location import HERE, ORIGINAL, OUT, ScaledEncoder, trials


def main():
    torch.set_num_threads(1)
    settings = json.loads((OUT / "settings.json").read_text())
    c = trials.behavior.Config(**settings["config"])
    repo = HERE.parents[2]
    for relative, expected in settings["source_sha256"].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((HERE/"run_location.py").read_bytes()).hexdigest() == settings["code_sha256"]
    assert hashlib.sha256((HERE/"PROTOCOL.md").read_bytes()).hexdigest() == settings["protocol_sha256"]
    assert hashlib.sha256((ORIGINAL/"settings.json").read_bytes()).hexdigest() == settings["original_settings_sha256"]
    curve_error, slope_error, numerical_error = 0., 0., 0.
    models, initials = 0, 0
    for seed in settings["seeds"]:
        source = ORIGINAL / "sources" / f"icm-{seed}.pt"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == settings["icm_data_sha256"][str(seed)]
        for gain in settings["gains"]:
            arms = ("initial", "icm", "ou", "balanced_log") if gain == 4 else ("initial", "icm", "ou")
            for arm in arms:
                p = trials.load(OUT/"profiles"/f"{arm}-gain-{gain}-{seed}.pt")
                torch.manual_seed(40000+seed)
                model = ScaledEncoder(gain)
                if arm != "initial":
                    folder = ORIGINAL if gain == 4 else OUT/f"gain-{gain}"
                    saved = trials.load(folder/"encoders"/f"{arm}-{seed}.pt")
                    assert saved["steps"] == c.jepa_steps
                    assert saved["seed"] == seed and saved["arm"] == arm
                    model.load_state_dict(saved["encoder"])
                    models += 1
                else:
                    initials += 1
                model.eval()
                recreated = trials.representation(model,c,seed)
                difference = float(np.max(abs(recreated["embedding"]-p["embedding"])))
                curve_error = max(curve_error,difference)
                assert difference < 1e-6
                assert recreated["folded"] == p["folded"]
                logd = torch.linspace(0,np.log(16),1024).requires_grad_()
                raw = model(trials.behavior.sensor(logd.exp(),c))
                # Recover the saved affine display factor without using the
                # derivative-decomposition implementation being checked.
                raw_values = raw.detach().numpy()
                factor = np.cov(raw_values,p["embedding"],bias=True)[0,1] / np.var(raw_values,dtype=np.float64)
                derivative = torch.autograd.grad(raw.sum(),logd)[0].numpy()*factor
                difference = float(np.max(abs(derivative-p["derivative_log_distance"])))
                slope_error = max(slope_error,difference)
                np.testing.assert_allclose(derivative,p["derivative_log_distance"],atol=1e-4,rtol=1e-4)
                selected = np.linspace(10,1010,17,dtype=int)
                # Small initial outputs lose precision when subtracting two
                # float32 predictions. Check finite differences in float64,
                # retaining exactly the checkpoint's parameter values.
                numeric_model = copy.deepcopy(model).double()
                points = logd.detach()[selected].double()
                epsilon = .0001
                with torch.no_grad():
                    high = numeric_model(trials.behavior.sensor((points+epsilon).exp(),c))
                    low = numeric_model(trials.behavior.sensor((points-epsilon).exp(),c))
                numeric = ((high-low)/(2*epsilon)).numpy()*factor
                numerical_error = max(numerical_error,float(np.max(abs(numeric-derivative[selected]))))
                np.testing.assert_allclose(numeric,derivative[selected],atol=1e-4,rtol=1e-4)
    original = trials.load(ORIGINAL/"encoders"/"icm-131.pt")["encoder"]
    baseline = trials.load(OUT/"baseline-check"/"encoders"/"icm-131.pt")["encoder"]
    for name,value in original.items():
        assert torch.equal(value,baseline[name])
    assert models == 70 and initials == 30
    trials.atomic_json(OUT/"verification.json",dict(
        passed=True,trained_models=models,initial_models=initials,
        max_recreated_curve_error=curve_error,
        max_recomputed_derivative_error=slope_error,
        max_finite_difference_error=numerical_error,
        original_gain4_retraining_parameter_difference=0.,
        source_and_protocol_hashes_unchanged=True))
    print("Audited 70 trained models and 30 initial models; curves and derivatives pass")


if __name__ == "__main__":
    main()
