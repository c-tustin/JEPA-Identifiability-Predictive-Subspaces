"""Add durable update-boundary checkpoints to the pinned official traffic trainer.

Training equations stay in upstream run(). Checked source insertions restore
state, skip finished batches, and save after complete updates/evaluations.
This cannot attach to a trainer that was already started without these hooks.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile

from launch import verify_source


class Paused(Exception):
    """Training stopped after a durable checkpoint; exit status is 75."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_file(path, writer):
    """Flush bytes before replacing a file, then flush the directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".saving-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RestartState:
    def __init__(self, directory, every=10, stop_after=None):
        if every <= 0:
            raise ValueError("Checkpoint interval must be positive")
        self.directory = Path(directory)
        self.every = every
        self.stop_after = stop_after
        self.pause_requested = False
        self.adapter_hash = digest(__file__)

    def request_pause(self, *_):
        # A signal must not serialize tensors halfway through an optimizer step.
        self.pause_requested = True

    def provenance(self):
        return {"adapter": "restart_traffic.py", "sha256": self.adapter_hash,
                "checkpoint_boundary": "after optimizer update and scheduled evaluation"}

    def restore(self, model, optimizer, generator, sigreg, config, source_hashes, runtime):
        import numpy as np
        import torch
        pointer = self.directory / "latest.json"
        if not pointer.exists():
            return None
        receipt = json.loads(pointer.read_text())
        path = self.directory / receipt["file"]
        if digest(path) != receipt["sha256"]:
            raise RuntimeError("Restart checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state["config"] != asdict(config) or state["source_hashes"] != source_hashes
                or state["adapter_sha256"] != self.adapter_hash or state["runtime"] != runtime):
            raise RuntimeError("Restart config, code or runtime differs; refusing silent changes")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sigreg.load_state_dict(state["sigreg"])
        generator.set_state(state["data_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(json.dumps({"event": "resumed", "step": state["step"]}), flush=True)
        return state

    def save(self, model, optimizer, generator, sigreg, config, step, curve,
             evaluations, raw_examples, source_hashes, runtime):
        import numpy as np
        import torch
        state = dict(config=asdict(config), step=step, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), sigreg=sigreg.state_dict(),
                     data_rng=generator.get_state(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     curve=curve, evaluations=evaluations, raw_examples=raw_examples,
                     source_hashes=source_hashes, adapter_sha256=self.adapter_hash, runtime=runtime)
        path = self.directory / f"step-{step:07d}.pt"
        atomic_file(path, lambda handle: torch.save(state, handle))
        receipt = dict(file=path.name, sha256=digest(path), step=step)
        atomic_file(self.directory / "latest.json",
                    lambda handle: handle.write((json.dumps(receipt, indent=2) + "\n").encode()))
        # Keep the latest two generated snapshots, rather than filling the disk.
        for old in sorted(self.directory.glob("step-*.pt"))[:-2]:
            old.unlink()
        print(json.dumps({"event": "restart_checkpoint", **receipt}), flush=True)

    def boundary(self, model, optimizer, generator, sigreg, config, step, curve,
                 evaluations, raw_examples, source_hashes, runtime):
        should_pause = self.pause_requested or (self.stop_after is not None and step >= self.stop_after)
        if step % self.every == 0 or step == config.steps or should_pause:
            self.save(model, optimizer, generator, sigreg, config, step, curve,
                      evaluations, raw_examples, source_hashes, runtime)
        if should_pause:
            raise Paused


def adapted_run(base, restart):
    """Reject changed upstream text rather than silently inserting in the wrong place."""
    source = inspect.getsource(base.run)

    def insert(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("Pinned trainer insertion point changed")
        source = source.replace(old, new, 1)

    arguments = "model, optimizer, generator, sigreg, config"
    metadata = "current_source_hashes(), runtime_contract(device)"
    insert("    step_index = 0\n", "    step_index = 0\n"
           f"    restored = _restart.restore({arguments}, {metadata})\n"
           "    if restored is not None:\n"
           "        step_index = restored['step']\n"
           "        curve, evaluations = restored['curve'], restored['evaluations']\n"
           "        raw_examples = restored['raw_examples']\n"
           "    else:\n"
           f"        _restart.save({arguments}, step_index, curve, evaluations, raw_examples, {metadata})\n")
    insert("        for batch_index in range(config.steps_per_epoch):\n",
           "        for batch_index in range(config.steps_per_epoch):\n"
           "            if epoch_index * config.steps_per_epoch + batch_index < step_index:\n"
           "                continue\n")
    insert("                model.train()\n    if step_index != config.steps",
           "                model.train()\n"
           f"            _restart.boundary({arguments}, step_index, curve, evaluations, raw_examples, {metadata})\n"
           "    if step_index != config.steps")
    insert("    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)\n",
           "    checkpoint['restart_adapter'] = _restart.provenance()\n"
           "    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)\n")
    namespace = dict(vars(base), _restart=restart)
    exec(compile(source, "<pinned-traffic-with-restart-hooks>", "exec"), namespace)
    return namespace["run"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--stop-after", type=int, help="Pause after this completed update")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    verify_source(upstream)
    sys.path.insert(0, str(upstream))
    from jepa.traffic import train_jepa as base
    config = base.Config.from_mapping(json.loads(args.config.read_text()))
    restart = RestartState(Path(config.output_checkpoint).parent / "restart",
                           args.checkpoint_every, args.stop_after)
    for number in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(number, restart.request_pause)
    try:
        adapted_run(base, restart)(config)
    except Paused:
        print("Paused after a durable checkpoint; rerun the same command to resume.", flush=True)
        raise SystemExit(75)


if __name__ == "__main__":
    main()
