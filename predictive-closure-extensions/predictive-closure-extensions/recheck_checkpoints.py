"""Recompute all saved evaluation metrics from this study's trusted checkpoints."""

import json
from pathlib import Path
import torch
from run_learned import Encoder, network, evaluate, probe_metrics


def main():
    torch.set_num_threads(1)
    out = Path(__file__).parent / "output"
    rows = json.loads((out / "learned.json").read_text())
    maximum = 0.0
    for row in rows:
        kind, seed = row["kind"], row["seed"]
        checkpoint = torch.load(out / f"{kind}-{seed}.pt", map_location="cpu")
        encoder, predictor = Encoder(), network()
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])
        metrics = evaluate(encoder, predictor, kind, seed + 50000000)
        metrics.update(probe_metrics(encoder, seed + 60000000))
        for key, value in metrics.items():
            if isinstance(value, list):
                error = max(abs(a - b) for a, b in zip(value, row[key]))
            else:
                error = abs(value - row[key])
            maximum = max(maximum, error)
            assert error < 1e-6, (kind, seed, key, error)
    result = {"checkpoints": len(rows), "maximum_absolute_error": maximum}
    (out / "checkpoint-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
