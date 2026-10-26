"""Run the official CPU timing check, then full-budget traffic experiments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs-local")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    state = {"scientific_result": False, "profile": "Intel Mac CPU FP32; Torch 2.2.2",
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    common = [sys.executable, str(ROOT / "launch.py"), "--upstream", str(args.upstream.resolve()),
              "--output", str(output), "--device", "cpu", "--cpu-fp32", "--local-intel",
              "--study", "traffic"]
    stages = [
        ("smoke", ["--mode", "smoke", "--seeds", "303", "--values", "1", "--smoke-timeout", "1800"]),
        ("full_endpoints", ["--mode", "full", "--seeds", "303", "--values", "0", "1", "--skip-plot"]),
        ("full_traffic", ["--mode", "full", "--seeds", "303", "304", "305"]),
    ]

    def save():
        temporary = output / "local-process.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "local-process.json")

    for label, options in stages:
        command = common + options
        with (output / f"{label}-launcher.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            state.update(stage=label, status="running", child_pid=process.pid, command=command,
                         returncode=None,
                         stage_started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            save()
            code = process.wait()
        state.update(status="completed" if code == 0 else "failed", returncode=code)
        save()
        if code:
            raise SystemExit(code)
    state["scientific_result"] = True
    state["completed_scope"] = "Traffic-light grid only: 5 q values x 3 seeds; other paper studies pending"
    save()


if __name__ == "__main__":
    main()
