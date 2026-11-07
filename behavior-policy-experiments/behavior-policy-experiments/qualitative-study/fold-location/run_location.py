"""Matched input-scale intervention and exact derivative diagnostics."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_trials as trials
import numpy as np
import torch

BASE_ENCODER = trials.behavior.Encoder
GAINS = (2, 4, 8)
ARMS = ("icm", "ou")
OUT = HERE / "output"
ORIGINAL = HERE.parent / "output"


class ScaledEncoder(BASE_ENCODER):
    """Same trainable network; change only the fixed input gain."""

    def __init__(self, gain=4):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        return self.net(torch.tanh(self.gain * x)).flatten()


@contextmanager
def encoder_gain(gain):
    """Use the unchanged trial trainer, restoring its factory on exit."""
    previous = trials.behavior.Encoder
    trials.behavior.Encoder = lambda: ScaledEncoder(gain)
    try:
        yield
    finally:
        trials.behavior.Encoder = previous


def profile(model, c, seed):
    """Physical labels and gradients enter only this saved-model diagnostic."""
    with torch.no_grad():
        base = trials.representation(model, c, seed)
        gen = torch.Generator().manual_seed(1800000 + seed)
        calibration = torch.exp(torch.rand(4096, generator=gen) * np.log(16))
        raw = model(trials.behavior.sensor(calibration, c))
        scale = max(float(raw.std(unbiased=False)), 1e-8)
        sign = 1 if float(torch.cov(torch.stack([raw, calibration.log()]))[0,1]) >= 0 else -1
    logd = torch.linspace(0, np.log(16), 1024).requires_grad_()
    x = trials.behavior.sensor(logd.exp(), c)
    u = torch.tanh(model.gain * x)
    h1 = model.net[1](model.net[0](u))
    h2 = model.net[3](model.net[2](h1))
    output = model.net[4](h2).flatten()
    direct = torch.autograd.grad(output.sum(), logd, retain_graph=True)[0]
    terms = []
    for j in range(h2.shape[1]):
        derivative = torch.autograd.grad(h2[:,j].sum(), logd, retain_graph=True)[0]
        terms.append(derivative * model.net[4].weight[0,j])
    terms = torch.stack(terms,1) * sign / scale
    derivative = direct * sign / scale
    torch.testing.assert_close(terms.sum(1), derivative, atol=1e-5, rtol=1e-4)
    cancellation = terms.sum(1).abs() / terms.abs().sum(1).clamp_min(1e-12)
    d = logd.detach().exp().numpy()
    slope = derivative.detach().numpy()
    ix = np.flatnonzero(slope[:-1] * slope[1:] < 0)
    turns = []
    grid = logd.detach().numpy()
    for k in ix:
        fraction = -slope[k] / (slope[k+1] - slope[k])
        turns.append(float(np.exp(grid[k] + fraction * (grid[k+1]-grid[k]))))
    peak = int(np.argmax(abs(slope)))
    tail = np.flatnonzero(abs(slope[peak+1:]) <= abs(slope[peak]) / 2)
    half = None if base["folded"] or not len(tail) else float(d[peak+1+tail[0]])
    return dict(**base, gain=model.gain, derivative_log_distance=slope,
        cancellation_ratio=cancellation.detach().numpy(),
        last_layer_derivative_terms=terms.detach().numpy(),
        input_saturated=(u.detach().abs()>.95).float().mean(1).numpy(),
        hidden1_saturated=(h1.detach().abs()>.95).float().mean(1).numpy(),
        hidden2_saturated=(h2.detach().abs()>.95).float().mean(1).numpy(),
        turning_distances=turns, peak_sensitivity_distance=float(d[peak]),
        far_half_sensitivity_distance=half,
        derivative_sum_max_error=float((terms.sum(1)-derivative).abs().max().detach()))


def save_profile(model, c, seed, arm, gain, initial=False):
    tag = "initial" if initial else arm
    path = OUT / "profiles" / f"{tag}-gain-{gain}-{seed}.pt"
    if not path.exists():
        trials.save_torch(path, profile(model.eval(), c, seed))


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ORIGINAL / "settings.json").read_text())
    c = trials.behavior.Config(**settings["config"])
    settings = dict(config=settings["config"], seeds=settings["seeds"], gains=GAINS,
        arms=ARMS, exploratory=True,
        protocol_sha256=hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_settings_sha256=hashlib.sha256((ORIGINAL/"settings.json").read_bytes()).hexdigest())
    settings["source_sha256"] = {
        str(Path(module.__file__).resolve().relative_to(HERE.parents[2])):
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (trials, trials.behavior, trials.study, trials.study.controls,
                       sys.modules[BASE_ENCODER.__module__],
                       sys.modules[trials.behavior.SIGReg1D.__module__])
    }
    settings["icm_data_sha256"] = {
        str(seed): hashlib.sha256((ORIGINAL/"sources"/f"icm-{seed}.pt").read_bytes()).hexdigest()
        for seed in settings["seeds"]
    }
    # JSON round-trip makes tuple/list comparison stable when resuming.
    settings = json.loads(json.dumps(settings))
    metadata = OUT / "settings.json"
    if metadata.exists() and json.loads(metadata.read_text()) != settings:
        raise RuntimeError("Gain-study settings changed; use a new output folder")
    trials.atomic_json(metadata, settings)
    # One full reproduction guards the factory substitution and training path.
    if not (OUT/"baseline-reproduction.json").exists():
        seed = 131
        data = trials.load(ORIGINAL/"sources"/f"icm-{seed}.pt")["data"]
        ou = trials.study.controls.ou_data(c, seed)
        with encoder_gain(4):
            model, saved = trials.train_encoder(c, seed, "icm", data, ou, OUT/"baseline-check")
        original = trials.load(ORIGINAL/"encoders"/f"icm-{seed}.pt")
        errors = [float((v-original["encoder"][k]).abs().max()) for k,v in saved["encoder"].items()]
        if max(errors) > 1e-7:
            raise RuntimeError(f"Baseline training did not reproduce: {max(errors)}")
        trials.atomic_json(OUT/"baseline-reproduction.json",dict(seed=seed,passed=True,max_parameter_difference=max(errors)))
    for seed in settings["seeds"]:
        data = trials.load(ORIGINAL/"sources"/f"icm-{seed}.pt")["data"]
        ou = trials.study.controls.ou_data(c, seed)
        for gain in GAINS:
            torch.manual_seed(40000+seed)
            save_profile(ScaledEncoder(gain),c,seed,"initial",gain,initial=True)
            for arm in (*ARMS, "balanced_log") if gain == 4 else ARMS:
                if gain == 4:
                    model = ScaledEncoder(gain)
                    model.load_state_dict(trials.load(ORIGINAL/"encoders"/f"{arm}-{seed}.pt")["encoder"])
                else:
                    with encoder_gain(gain):
                        model,_ = trials.train_encoder(c, seed, arm, data, ou, OUT/f"gain-{gain}")
                save_profile(model,c,seed,arm,gain)
                print(f"Profile saved: seed={seed}, gain={gain}, arm={arm}",flush=True)
        trials.atomic_json(OUT/f"completed-{seed}.json",dict(seed=seed,completed=True))
    print("All gain controls and derivative profiles complete",flush=True)


if __name__ == "__main__":
    main()
