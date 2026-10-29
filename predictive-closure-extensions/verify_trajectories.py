"""Replay saved executed actions/noise; check substep positivity without replanning."""

import json
from pathlib import Path
import numpy as np


def main():
    output = Path(__file__).parent / "output"
    reports = []
    for filename in ["drift.json", "drift-weighted.json"]:
        data = json.loads((output / filename).read_text())
        minimum, maximum, error = np.inf, -np.inf, 0.0
        for row in data["planning"]:
            rng = np.random.default_rng(row["seed"] * 10000 + row["goal_index"])
            state = np.array(row["start"])
            for step, action in enumerate(row["actions"]):
                for _ in range(64):
                    drift = 0.3 * np.sin(state) + 0.25 * action
                    diffusion = 0.05 * (1 + 0.1 * np.cos(state))
                    state = (
                        state
                        + drift / 64
                        + diffusion * rng.normal(size=state.shape) / 8
                    )
                    minimum = min(minimum, float(state))
                    maximum = max(maximum, float(state))
                error = max(error, abs(float(state) - row["path"][step + 1]))
        assert minimum > 0, (filename, minimum)
        assert error < 1e-12, (filename, error)
        reports.append(
            dict(
                file=filename,
                rollouts=len(data["planning"]),
                executed_substep_minimum=minimum,
                executed_substep_maximum=maximum,
                maximum_decision_state_replay_error=error,
                reflection=False,
            )
        )
    (output / "trajectory-verification.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
