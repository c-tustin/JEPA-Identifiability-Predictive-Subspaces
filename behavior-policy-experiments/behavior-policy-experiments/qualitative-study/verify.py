"""Audit saved scientific outputs independently of the plotting code."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_trials import (
    ARMS,
    ROOT,
    atomic_json,
    behavior,
    distances,
    load,
    representation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    out = parser.parse_args().output.resolve()
    settings = json.loads((out / "settings.json").read_text())
    c = behavior.Config(**settings["config"])
    seeds = settings["seeds"]
    torch.set_num_threads(1)
    for path, expected in settings["source_sha256"].items():
        # A clone has different absolute paths; match the repository-relative suffix.
        original = Path(path)
        suffix = original.parts[original.parts.index("behavior-policy-experiments") :]
        current = ROOT.parent.parent.joinpath(*suffix)
        assert hashlib.sha256(current.read_bytes()).hexdigest() == expected, current
    assert (
        hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()
        == settings["protocol_sha256"]
    )
    counts = dict(
        encoders=0,
        predictors=0,
        planning_results=0,
        held_out_goals=0,
        screening_goals=0,
    )
    max_path_difference, max_error_difference, max_curve_difference = 0.0, 0.0, 0.0
    for seed in seeds:
        assert (out / f"completed-{seed}.json").exists()
        for source in ("icm", "physical-common"):
            saved = load(out / "sources" / f"{source}-{seed}.pt")
            data = saved["data"] if source == "icm" else saved
            assert len(data["x"]) == c.train_episodes * c.horizon
            assert bool(((data["action"] >= 0) & (data["action"] <= 2)).all())
            expected = behavior.move(distances(data["x"]), data["action"], c)
            np.testing.assert_allclose(
                expected.numpy(), distances(data["xp"]).numpy(), atol=1e-5, rtol=0
            )
        keys = [
            f"{arm}-{kind}-{seed}"
            for arm in ARMS
            for kind in (("shared",) if arm == "ou" else ("shared", "native"))
        ]
        keys += [f"remove-{region}-{seed}" for region in ("all", *range(8))]
        common_goals = {}
        for key in keys:
            counts["predictors"] += int((out / "predictors" / f"{key}.pt").exists())
            saved = load(out / "planning" / f"{key}.pt")
            coverage = saved["data_coverage"]["counts"]
            assert sum(coverage) == c.train_episodes * c.horizon
            if key.startswith("remove-") and not key.startswith("remove-all-"):
                region = int(key.split("-")[1])
                assert sum(coverage[4 * region : 4 * (region + 1)]) == 0
            counts["planning_results"] += 1
            for split in ("screen", "test"):
                result = saved[split]
                expected_n = settings[
                    f"{'screen' if split == 'screen' else 'test'}_cases"
                ]
                assert len(result["error"]) == expected_n
                counts[
                    "screening_goals" if split == "screen" else "held_out_goals"
                ] += expected_n
                if split not in common_goals:
                    common_goals[split] = result
                for field in ("start", "goal", "oracle_path"):
                    np.testing.assert_array_equal(
                        result[field], common_goals[split][field]
                    )
                distance = torch.tensor(result["start"])
                paths = [distance.numpy().copy()]
                assert np.isin(result["actions"], [0, 1, 2]).all()
                for action in result["actions"]:
                    distance = behavior.move(distance, torch.tensor(action), c)
                    paths.append(distance.numpy().copy())
                difference = float(np.max(abs(np.stack(paths) - result["path"])))
                max_path_difference = max(max_path_difference, difference)
                assert difference < 1e-6
                error = np.abs(distance.numpy() - result["goal"])
                difference = float(np.max(abs(error - result["error"])))
                max_error_difference = max(max_error_difference, difference)
                assert difference < 1e-6
                assert abs(float(error.mean()) - result["mean_error"]) < 1e-6
                assert np.max(abs(result["oracle_path"][-1] - result["goal"])) < 1e-4
        for arm in ARMS:
            checkpoint = load(out / "encoders" / f"{arm}-{seed}.pt")
            assert checkpoint["config"] == asdict(c)
            assert checkpoint["steps"] == c.jepa_steps
            encoder = behavior.Encoder()
            encoder.load_state_dict(checkpoint["encoder"])
            with torch.no_grad():
                curve = representation(encoder.eval(), c, seed)
            saved_curve = load(out / "curves" / f"{arm}-{seed}.pt")
            difference = float(
                np.max(abs(curve["embedding"] - saved_curve["embedding"]))
            )
            max_curve_difference = max(max_curve_difference, difference)
            assert difference < 1e-6
            assert curve["folded"] == saved_curve["folded"]
            counts["encoders"] += 1
        assert not (out / "planning" / f"ou-native-{seed}.pt").exists()
    assert counts["encoders"] == len(seeds) * 5
    assert counts["predictors"] == counts["planning_results"] == len(seeds) * 18
    atomic_json(
        out / "verification.json",
        dict(
            passed=True,
            quick=settings["quick"],
            seeds=seeds,
            counts=counts,
            max_replayed_path_difference=max_path_difference,
            max_recomputed_goal_error_difference=max_error_difference,
            max_recomputed_curve_difference=max_curve_difference,
            checks=[
                "Protocol and training-source hashes unchanged",
                "Physical training pairs agree with their saved action labels",
                "Region removals keep the pair budget and exclude their specified bins",
                "All requested encoders and predictors saved",
                "Same goals and oracle paths across methods within seed",
                "Physical actions replay to saved paths and terminal errors",
                "Every goal reachable by oracle",
                "Representation curves reproduce from encoder checkpoints",
                "No fabricated OU-native action data",
            ],
        ),
    )
    print(json.dumps(counts, indent=2))
    print("All output audits passed")


if __name__ == "__main__":
    main()
