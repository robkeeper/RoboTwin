"""Two-GPU, fixed-scene evaluation with persistent progress and provenance."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rlinf_eval.cli import source_fingerprints, write_json

ROOT = Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reproduction_script(output: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_PATH="$bundle/sources/RLinf"
export PYTHONPATH="$bundle/sources/RoboTwin:$REPO_PATH"
replay_output=${3:-"${bundle}_replay_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"}
exec /opt/venv/openpi/bin/python -m rlinf_eval.campaign \\
  --output "$replay_output" --gpu-7272 "${1:-3}" --gpu-18180 "${2:-6}" \\
  --seeds "$bundle/success_seeds.json" --runtime-source "$bundle/runtime" \\
  --reference "$bundle"
"""
    (output / "reproduce.sh").write_text(script)


def finalize(output: Path, comparison: dict, mapping: dict) -> None:
    report = {
        "complete": True,
        "gpu_mapping": mapping,
        "episodes_per_checkpoint": 100,
        "finished_at": timestamp(),
        "results": {},
    }
    for checkpoint, summary in comparison.items():
        run_dir = output / "evaluation" / checkpoint
        measurements = [
            json.loads(s)
            for s in (run_dir / "gpu_usage.jsonl").read_text().splitlines()
        ]
        values = [
            int(r["gpu"].split(",")[2].strip()) for r in measurements if r.get("gpu")
        ]
        report["results"][checkpoint] = {
            k: summary[k] for k in ["success", "total", "success_rate"]
        }
        report["results"][checkpoint]["sampled_gpu_peak_mib"] = (
            max(values) if values else None
        )
    write_json(output / "report.json", report)
    checksums = {
        str(p.relative_to(output)): {"size": p.stat().st_size, "sha256": file_digest(p)}
        for p in output.rglob("*")
        if p.is_file()
        and p.name
        not in {
            "artifact_checksums.json",
            "model_load.lock",
            "driver.log",
            "status.json",
        }
    }
    write_json(output / "artifact_checksums.json", checksums)


def copy_runtime(output: Path, previous: Path | None) -> None:
    """Copy actual task assets and import-time object metadata into the bundle."""
    target = output / "runtime"
    if (target / "ready.json").exists():
        return
    source_assets = previous / "assets" if previous else Path("/data/robotwin_assets")
    source_sapien = (
        previous / "sapien300b1"
        if previous
        else Path("/workspace/ctr_backup/sapien300b1")
    )
    for relative in ["objects/015_laptop", "embodiments/aloha-agilex"]:
        shutil.copytree(
            source_assets / relative, target / "assets" / relative, dirs_exist_ok=True
        )
    metadata = list((source_assets / "objects").glob("*/*.json"))
    metadata += [
        source_assets / "objects/same.json",
        source_assets / "objects/objaverse/list.json",
    ]
    for path in metadata:
        destination = target / "assets" / path.relative_to(source_assets)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
    shutil.copytree(source_sapien, target / "sapien300b1", dirs_exist_ok=True)
    source_cache = (
        previous / "openpi-cache"
        if previous
        else Path("/data/robotwin_ckpt/openpi-cache")
    )
    shutil.copytree(source_cache, target / "openpi-cache", dirs_exist_ok=True)
    files = {
        str(p.relative_to(output)): {"size": p.stat().st_size, "sha256": file_digest(p)}
        for p in target.rglob("*")
        if p.is_file()
    }
    write_json(target / "ready.json", {"created_at": timestamp(), "files": files})


def capture_models(output: Path, reference: Path | None = None) -> None:
    """Hash the existing weights and archive all accompanying small metadata."""
    manifests = {}
    expected = (
        json.loads((reference / "repro/checkpoints.json").read_text())
        if reference
        else None
    )
    for checkpoint in ["7272", "18180"]:
        source = Path("/data/robotwin_ckpt/rlinf") / checkpoint
        destination = output / "repro" / "checkpoints" / checkpoint
        files = {}
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            before = path.stat()
            digest = file_digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(f"Checkpoint changed while hashing: {path}")
            if expected and digest != expected[checkpoint][str(relative)]["sha256"]:
                raise ValueError(f"Checkpoint differs from the reference run: {path}")
            files[str(relative)] = {
                "path": str(path),
                "size": after.st_size,
                "sha256": digest,
            }
            if path.name != "model.safetensors":
                saved = destination / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, saved)
        manifests[checkpoint] = files
    write_json(output / "repro/checkpoints.json", manifests)
    result = subprocess.run(
        ["uv", "pip", "freeze", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=True,
    )
    (output / "repro/requirements.freeze.txt").write_text(result.stdout)
    write_json(
        output / "repro/environment.json",
        {
            "recorded_at": timestamp(),
            "python": sys.version,
            "ray_resources": {"CPU": 8, "object_store_memory": 1073741824},
            "gpu_mapping": {
                "7272": int(os.environ["CAMPAIGN_GPU_7272"]),
                "18180": int(os.environ["CAMPAIGN_GPU_18180"]),
            },
            "source_sha256": source_fingerprints(),
            "source_root": str(ROOT),
            "rlinf_root": os.environ["REPO_PATH"],
        },
    )


def screening_plan(gpu_a: int, gpu_b: int) -> list[dict]:
    """Allocate four independent screening processes per reserved GPU."""
    return [
        {
            "label": f"scan_gpu{gpu}" + (f"_{index}" if index else ""),
            "gpu": gpu,
            "start": base + index * 10000,
            "count": count,
        }
        for gpu, base in [(gpu_a, 100000), (gpu_b, 200000)]
        for index, count in enumerate([13, 13, 12, 12])
    ]


def merge_shards(paths: list[Path], output: Path, expected: int = 100) -> dict:
    """Merge independently screened ranges without duplicates or protocol drift."""
    shards = [json.loads(p.read_text()) for p in paths]
    for shard in shards:
        if not shard["complete"]:
            raise ValueError("Cannot merge an incomplete expert-screening shard")
    common = ["source_sha256", "task_config", "instruction_type", "sapien_runtime"]
    for key in common:
        if any(s["protocol"][key] != shards[0]["protocol"][key] for s in shards):
            raise ValueError(f"Different screening protocol: {key}")
    instructions = {}
    for shard in shards:
        for seed in shard["open_laptop"]["success_seeds"]:
            if str(seed) in instructions:
                raise ValueError(f"Duplicate success seed: {seed}")
            instructions[str(seed)] = shard["open_laptop"]["instructions"][str(seed)]
    seeds = sorted(int(seed) for seed in instructions)
    if len(seeds) != expected:
        raise ValueError(f"Expected {expected} distinct scenes, got {len(seeds)}")
    manifest = {
        "schema": "robotwin_rlinf_merged_success_seeds/v1",
        "complete": True,
        "created_at": timestamp(),
        "protocol": {
            **{key: shards[0]["protocol"][key] for key in common},
            "selection": "Union of disjoint expert-screened ranges; quotas recorded per shard",
            "shards": [
                {"file": str(p), "sha256": file_digest(p), "protocol": s["protocol"]}
                for p, s in zip(paths, shards)
            ],
        },
        "open_laptop": {"success_seeds": seeds, "instructions": instructions},
    }
    write_json(output, manifest)
    return manifest


def child_environment(gpu: int, output: Path, label: str) -> dict[str, str]:
    env = os.environ.copy()
    ray_key = hashlib.sha256(str(output).encode()).hexdigest()[:6]
    env.update(
        {
            "GPU_LIST": str(gpu),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "RAY_ADDRESS": "local",
            "RAY_OVERRIDE_RESOURCES": '{"CPU":8,"object_store_memory":1073741824}',
            "RAY_TMPDIR": f"/tmp/wol_{ray_key}_{label}",
            "ROBOTWIN_ASSETS_PATH": str(output / "runtime/assets"),
            "SAPIEN_RUNTIME": str(output / "runtime/sapien300b1"),
            "OPENPI_DATA_HOME": str(output / "runtime/openpi-cache"),
            "MODEL_LOAD_LOCK": str(output / "model_load.lock"),
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "JAX_PLATFORMS": "cpu",
        }
    )
    env["PYTHONPATH"] = f"{env['SAPIEN_RUNTIME']}:{ROOT}:{env['REPO_PATH']}"
    return env


def launch(
    label: str, command: list[str], env: dict[str, str], output: Path
) -> subprocess.Popen:
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{label}.log").open("a") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        logs / f"{label}.process.json",
        {
            "pid": process.pid,
            "command": command,
            "started_at": timestamp(),
            "gpu": env["GPU_LIST"],
            "environment": {
                k: env[k]
                for k in [
                    "REPO_PATH",
                    "PYTHONPATH",
                    "GPU_LIST",
                    "CUDA_VISIBLE_DEVICES",
                    "RAY_ADDRESS",
                    "RAY_OVERRIDE_RESOURCES",
                    "RAY_TMPDIR",
                    "SAPIEN_RUNTIME",
                    "ROBOTWIN_ASSETS_PATH",
                    "MODEL_LOAD_LOCK",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                ]
            },
        },
    )
    return process


def wait_tasks(
    tasks: dict[str, subprocess.Popen], output: Path, stage: str, gpu_mapping: dict
) -> None:
    while True:
        progress = {}
        for label, process in tasks.items():
            item = {"pid": process.pid, "exit_code": process.poll()}
            if stage == "screening":
                path = output / f"{label}.json"
                if path.exists():
                    manifest = json.loads(path.read_text())
                    item.update(
                        selected=len(manifest["open_laptop"]["success_seeds"]),
                        attempts=len(manifest["attempts"]),
                        target=manifest["protocol"]["count"],
                    )
            else:
                path = output / "evaluation" / label / "episodes.jsonl"
                records = (
                    [
                        json.loads(s)
                        for s in path.read_text().splitlines(keepends=True)
                        if s.endswith("\n")
                    ]
                    if path.exists()
                    else []
                )
                item.update(
                    completed=len(records),
                    success=sum(r["success"] for r in records),
                    target=100,
                )
            progress[label] = item
        state = {
            "stage": stage,
            "updated_at": timestamp(),
            "gpu_mapping": gpu_mapping,
            "progress": progress,
        }
        write_json(output / "status.json", state)
        print(json.dumps(state), flush=True)
        if any(p.poll() not in (None, 0) for p in tasks.values()):
            raise RuntimeError(f"A {stage} child failed; see logs and status.json")
        if all(p.poll() is not None for p in tasks.values()):
            return
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-7272", type=int, required=True)
    parser.add_argument("--gpu-18180", type=int, required=True)
    parser.add_argument(
        "--seeds", type=Path, help="Reuse an already frozen 100-scene manifest"
    )
    parser.add_argument(
        "--runtime-source", type=Path, help="Reuse archived assets and SAPIEN on replay"
    )
    parser.add_argument(
        "--reference", type=Path, help="Require weights to match an earlier campaign"
    )
    args = parser.parse_args()
    if args.gpu_7272 == args.gpu_18180:
        parser.error("This campaign requires two different reserved GPUs")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "repro").mkdir(exist_ok=True)
    write_reproduction_script(output)
    os.environ.update(
        CAMPAIGN_GPU_7272=str(args.gpu_7272), CAMPAIGN_GPU_18180=str(args.gpu_18180)
    )
    mapping = {"7272": args.gpu_7272, "18180": args.gpu_18180}
    active = {}

    def stop_signal(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        write_json(
            output / "status.json",
            {
                "stage": "snapshotting_runtime",
                "updated_at": timestamp(),
                "gpu_mapping": mapping,
            },
        )
        copy_runtime(output, args.runtime_source)
        test_env = child_environment(args.gpu_7272, output, "unit_tests")
        test = launch(
            "unit_tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(ROOT / "tests"),
                "-p",
                "test_*adapter.py",
                "-v",
            ],
            test_env,
            output,
        )
        active["unit_tests"] = test
        if test.wait():
            raise RuntimeError("Preflight tests failed; see logs/unit_tests.log")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            provenance = pool.submit(capture_models, output, args.reference)
            if args.seeds:
                manifest = json.loads(args.seeds.read_text())
                if (
                    not manifest["complete"]
                    or len(manifest["open_laptop"]["success_seeds"]) != 100
                ):
                    raise ValueError(
                        "Replay requires exactly 100 complete success seeds"
                    )
                write_json(output / "success_seeds.json", manifest)
            else:
                tasks = {}
                paths = []
                plan = screening_plan(args.gpu_7272, args.gpu_18180)
                write_json(output / "repro/screening_plan.json", {"shards": plan})
                for shard in plan:
                    gpu, start, label = shard["gpu"], shard["start"], shard["label"]
                    path = output / f"{label}.json"
                    paths.append(path)
                    if path.exists() and json.loads(path.read_text()).get("complete"):
                        continue
                    command = [
                        "bash",
                        str(ROOT / "rlinf_eval/run.sh"),
                        "scan",
                        "--start",
                        str(start),
                        "--count",
                        str(shard["count"]),
                        "--output",
                        str(path),
                    ]
                    tasks[label] = launch(
                        label, command, child_environment(gpu, output, label), output
                    )
                active.update(tasks)
                wait_tasks(tasks, output, "screening", mapping)
                merge_shards(paths, output / "success_seeds.json")
            provenance.result()
        tasks = {}
        for checkpoint, gpu in mapping.items():
            command = [
                "bash",
                str(ROOT / "rlinf_eval/run.sh"),
                "run",
                "--checkpoint",
                checkpoint,
                "--episodes",
                "100",
                "--seeds",
                str(output / "success_seeds.json"),
                "--output",
                str(output / "evaluation"),
            ]
            tasks[checkpoint] = launch(
                checkpoint, command, child_environment(gpu, output, checkpoint), output
            )
        active.update(tasks)
        wait_tasks(tasks, output, "evaluation", mapping)
        comparison = {
            c: json.loads((output / "evaluation" / c / "summary.json").read_text())
            for c in mapping
        }
        for checkpoint, summary in comparison.items():
            if not summary["complete"] or summary["total"] != 100:
                raise RuntimeError(f"Incomplete final result: {checkpoint}")
        first = {r["seed"]: r for r in comparison["7272"]["episodes"]}
        second = {r["seed"]: r for r in comparison["18180"]["episodes"]}
        if set(first) != set(second) or any(
            first[s]["instruction"] != second[s]["instruction"] for s in first
        ):
            raise RuntimeError("Paired seed or instruction coverage differs")
        write_json(output / "comparison.json", comparison)
        write_json(
            output / "paired_results.json",
            {
                str(s): {
                    c: table[s] for c, table in [("7272", first), ("18180", second)]
                }
                for s in sorted(first)
            },
        )
        finalize(output, comparison, mapping)
        write_json(
            output / "status.json",
            {
                "stage": "complete",
                "complete": True,
                "updated_at": timestamp(),
                "gpu_mapping": mapping,
                "results": {
                    c: {
                        k: comparison[c][k]
                        for k in ["success", "total", "success_rate"]
                    }
                    for c in mapping
                },
            },
        )
    except BaseException as error:
        write_json(output / "failure.json", {"time": timestamp(), "error": repr(error)})
        raise
    finally:
        for process in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
