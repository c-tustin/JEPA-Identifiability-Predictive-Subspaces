"""Pinned original-paper launcher. Importable with only the Python standard library."""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PIN = "2964fa8f737fa43871d47928edc9aac16fa9298f"
SEEDS = (303, 304, 305)
STUDIES = ("traffic", "predictability", "subset", "reconstruction", "side_input")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(upstream):
    manifest = read(ROOT / "source-manifest.json")
    for row in manifest["files"]:
        data = (upstream / row["path"]).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != row["git_blob_sha1"]:
            raise RuntimeError("Pinned source differs: " + row["path"])
    return {row["path"]: sha256(upstream / row["path"]) for row in manifest["files"]}


def official_runner(upstream):
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("official_reproduction_run", upstream / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_records(official, data, output, seeds=SEEDS, studies=STUDIES, device="cuda:0", cpu_fp32=False):
    records = []
    requested = set(studies)
    if requested & {"reconstruction", "side_input"}:
        requested.add("predictability")
    for seed in seeds:
        selected = set(requested)
        if seed != SEEDS[0]:
            selected.discard("reconstruction")
        records.extend(official.jobs(selected, (.1,), seed, data, output / f"seed{seed}", device))
    if cpu_fp32:
        for row in records:
            for _, config in row["stages"]:
                config["device"] = "cpu"
                config["bf16"] = False
    return records


def resource_audit(upstream, output):
    audit = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                 cpu_count=os.cpu_count(), disk_free_bytes=shutil.disk_usage(output).free,
                 source_commit=PIN, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    audit["packages"] = {}
    for name in ("torch", "numpy", "matplotlib"):
        try:
            audit["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            audit["packages"][name] = None
    try:
        import torch
        audit["cuda_available"] = torch.cuda.is_available()
        audit["torch_cuda_build"] = torch.version.cuda
        audit["gpu"] = [dict(index=i, name=torch.cuda.get_device_name(i),
                             memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                        for i in range(torch.cuda.device_count())]
    except Exception as error:
        audit["torch_audit_error"] = repr(error)
    try:
        audit["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                            timeout=10, check=False).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        audit["nvidia_smi"] = None
    if Path("/proc/meminfo").exists():
        audit["memory_info"] = Path("/proc/meminfo").read_text()
    audit["source_sha256"] = verify_source(upstream)
    write(output / "runtime-audit.json", audit)
    return audit


def validate_runtime(audit, device, cpu_fp32, local_intel=False):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Recorded environment requires Python 3.12")
    expected = {"torch": "2.11.0", "numpy": "2.4.3", "matplotlib": "3.10.8"}
    if local_intel:
        if not cpu_fp32 or device != "cpu":
            raise ValueError("--local-intel requires --device cpu --cpu-fp32")
        if platform.system() != "Darwin" or platform.machine() != "x86_64":
            raise ValueError("--local-intel is the recorded Intel macOS runtime variant")
        expected = {"torch": "2.2.2", "numpy": "1.26.4", "matplotlib": "3.9.0"}
    for name, version in expected.items():
        actual = audit["packages"][name]
        if actual is None or actual.split("+")[0] != version:
            raise RuntimeError(f"{name} must be {version}; found {actual}; use isolated bootstrap.sh")
    if cpu_fp32:
        if device != "cpu":
            raise ValueError("--cpu-fp32 requires --device cpu")
    elif not device.startswith("cuda:"):
        raise ValueError("Upstream bf16 needs CUDA; CPU requires explicit --cpu-fp32")
    elif audit.get("torch_cuda_build") != "12.8":
        raise RuntimeError("Code-faithful profile expects the upstream CUDA 12.8 wheel")
    from jepa.common import resolve_device
    resolve_device(device, bf16=not cpu_fp32)


def save_plan(records, output):
    by_seed = {}
    for row in records:
        by_seed.setdefault(row["seed"], []).append(row)
        for index, (module, config) in enumerate(row["stages"]):
            destination = output / "planned-configs" / row["tag"] / f"stage{index}.json"
            write(destination, config)
    plan = []
    for row in records:
        plan.append({**{k: v for k, v in row.items() if k != "stages"},
                     "stages": [dict(module=module, config=config,
                                    semantic_config_sha256=hashlib.sha256(
                                        json.dumps(config, sort_keys=True).encode()).hexdigest())
                                for module, config in row["stages"]]})
    write(output / "launcher-plan.json", plan)
    write(output / "budget.json", dict(encoders=len(records),
          counts={name: sum(r["study"] == name for r in records) for name in STUDIES},
          seeds={str(seed): len(rows) for seed, rows in by_seed.items()}))
    return by_seed


def smoke(official, records, upstream, data, output, timeout):
    choices = []
    for study in STUDIES:
        candidates = [r for r in records if r["study"] == study and r["seed"] == SEEDS[0]]
        if not candidates:
            continue
        if study == "traffic":
            candidates = [r for r in candidates if r["x"] == 1.]
        elif study == "subset":
            candidates = [r for r in candidates if r["x"] == 16]
        else:
            candidates = [r for r in candidates if r["x"] == .85]
            if study == "reconstruction":
                candidates = [r for r in candidates if r["beta"] == .135]
        choices.append(candidates[0])
    timing = []
    for row in choices:
        directory = output / "smoke-only" / row["tag"]
        directory.mkdir(parents=True, exist_ok=False)
        module, original = row["stages"][0]
        config = copy.deepcopy(original)
        config.update(run_id=row["tag"] + "_SMOKE_ONLY", warmup_steps=1,
                      log_every=1, evaluation_every=4,
                      output_checkpoint=str(directory / "smoke.pt"),
                      output_json=str(directory / "smoke.json"))
        if row["study"] in {"traffic", "subset"}:
            config.update(epochs=1, sequences_per_epoch=config["batch_size"] * 4)
        else:
            config.update(steps=4, schedule_steps=4)
            from jepa.four_image.materialize_corpus import materialize
            from jepa.four_image.feed import DictionaryFeedConfig
            config["training_corpus"] = str(directory / "smoke-corpus.pt")
            start = time.perf_counter()
            materialize(data_root=str(data), output_corpus=Path(config["training_corpus"]),
                        output_json=directory / "smoke-corpus.json",
                        feed_config=DictionaryFeedConfig(**config["feed"]),
                        corpus_seed=config["train_sequence_seed"], epoch_count=1,
                        sequences_per_epoch=config["batch_size"] * 4)
            write(directory / "corpus-timing.json", dict(wall_seconds=time.perf_counter() - start))
        importlib.import_module(module).Config.from_mapping(config)
        path = directory / "config.json"
        write(path, config)
        start = time.perf_counter()
        status, detail = "completed", None
        try:
            with (directory / "training.log").open("x") as handle:
                subprocess.run([sys.executable, "-m", module, "--config", str(path)],
                               cwd=upstream, stdout=handle, stderr=subprocess.STDOUT,
                               timeout=timeout, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            status, detail = "failed_or_timed_out", repr(error)
        elapsed = time.perf_counter() - start
        timing.append(dict(study=row["study"], status=status, detail=detail,
                           training_process_wall_seconds=elapsed,
                           workload="4 updates; includes startup, loading, evaluation and checkpoint I/O",
                           scientific_result=False, config_sha256=sha256(path)))
        write(output / "smoke-timing.json", timing)
        print(json.dumps(timing[-1]), flush=True)
        if status != "completed":
            raise RuntimeError("Smoke did not complete; inspect its log before launching full runs")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=ROOT / "upstream")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=["plan", "audit", "fetch-data", "smoke", "full"], default="plan")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-fp32", action="store_true")
    parser.add_argument("--local-intel", action="store_true",
                        help="Explicit Intel Mac variant: Torch 2.2.2, NumPy 1.26.4, Matplotlib 3.9.0; CPU FP32 only")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--study", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--values", nargs="+", type=float,
                        help="Run only these q/pD/N grid values; partial grids are not a complete reproduction")
    parser.add_argument("--skip-plot", action="store_true",
                        help="For partial grids: the upstream plotter requires a complete sweep")
    parser.add_argument("--smoke-timeout", type=float, default=180.)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Optimization seeds must be distinct")
    if args.smoke_timeout <= 0:
        parser.error("Smoke timeout must be positive")
    upstream, output = args.upstream.resolve(), args.output.resolve()
    data = args.data_root.resolve() if args.data_root else upstream / "data/cifar-10-batches-py"
    verify_source(upstream)
    official = official_runner(upstream)
    records = build_records(official, data, output, args.seeds, args.study, args.device, args.cpu_fp32)
    if args.values is not None:
        records = [row for row in records if row["x"] in args.values]
    if not records:
        raise ValueError("No jobs selected")
    output.mkdir(parents=True, exist_ok=True)
    by_seed = save_plan(records, output)
    if args.mode == "plan":
        print(f"Prepared {len(records)} encoder settings; no Torch import or training.")
        return
    if args.mode == "fetch-data":
        from download_data import download, verify
        if data.name != "cifar-10-batches-py":
            raise ValueError("Download destination must end in cifar-10-batches-py")
        download(data.parent)
        verify(data)
        write(output / "data-verification.json", dict(status="verified", source_commit=PIN,
              files=read(ROOT / "source-manifest.json")["cifar10"]["extracted_sha256"]))
        return
    audit = resource_audit(upstream, output)
    if args.mode == "audit":
        print(json.dumps({k: audit[k] for k in ["cpu_count", "disk_free_bytes", "packages"]}))
        return
    validate_runtime(audit, args.device, args.cpu_fp32, args.local_intel)
    from download_data import verify
    verify(data)
    if args.mode == "smoke":
        smoke(official, records, upstream, data, output, args.smoke_timeout)
        return
    journal = output / "full-run-timing.jsonl"
    for seed, rows in by_seed.items():
        seed_output = output / f"seed{seed}"
        official.save_plan(rows, seed_output)
        for row in rows:
            importlib.import_module(row["stages"][0][0]).Config.from_mapping(row["stages"][0][1])
            started = time.perf_counter()
            with journal.open("a") as handle:
                handle.write(json.dumps(dict(event="start", tag=row["tag"])) + "\n")
            official.execute([row], seed_output)
            with journal.open("a") as handle:
                handle.write(json.dumps(dict(event="completed", tag=row["tag"],
                                             wall_seconds=time.perf_counter() - started)) + "\n")
        if not args.skip_plot:
            from plot import plot
            plot(seed_output)
    print("Selected official training/probe stages completed. Appendix C is not implemented by this launcher.")


if __name__ == "__main__":
    main()
