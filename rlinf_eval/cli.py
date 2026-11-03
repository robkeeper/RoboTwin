"""Prepare paired RLinf evaluation using expert-validated fixed scenes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rlinf_eval.monitor import record_gpu

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/workspace/ctr_backup/open_laptop_eval")


def write_json(path: Path, value: dict) -> None:
    """Atomically replace a small manifest or summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprints() -> dict:
    """Identify source independently of Git metadata, which is not synchronized."""
    paths = [
        "envs/open_laptop.py",
        "envs/_base_task.py",
        "envs/_GLOBAL_CONFIGS.py",
        "envs/robot/robot.py",
        "envs/robot/planner.py",
        "robotwin/envs/vector_env.py",
        "rlinf_eval/open_laptop.yaml",
        "envs/utils/create_actor.py",
        "envs/utils/rand_create_cluttered_actor.py",
        "envs/utils/transforms.py",
        "description/utils/generate_episode_instructions.py",
        "description/task_instruction/open_laptop.json",
    ]
    return {name: sha256(ROOT / name) for name in paths if (ROOT / name).is_file()}


def is_known_expert_failure(error: Exception) -> bool:
    """Recognize a native expert failure to produce a grasp target."""
    return (
        isinstance(error, AssertionError)
        and str(error) == "target_pose cannot be None for move action."
    )


def scan(args: argparse.Namespace) -> None:
    """Screen experts once, recording fixed seeds and instructions for both models."""
    import yaml
    from envs.utils.create_actor import UnStableError

    from robotwin.envs.vector_env import new_task, prepare_task_config, task_instruction

    raw = yaml.safe_load((ROOT / "rlinf_eval/open_laptop.yaml").read_text())
    task_config = prepare_task_config(raw["env"]["eval"]["task_config"])
    fingerprints = source_fingerprints()
    settings = {
        "start": args.start,
        "count": args.count,
        "max_attempts": args.max_attempts,
        "source_sha256": fingerprints,
        "task_config": task_config,
        "instruction_type": "seen",
        "sapien_runtime": os.environ["SAPIEN_RUNTIME"],
    }
    if args.output.exists():
        manifest = json.loads(args.output.read_text())
        if manifest["protocol"] != settings:
            raise ValueError(
                "Existing seed manifest has a different protocol; choose a new output"
            )
        if manifest["complete"]:
            raise ValueError(
                "Seed manifest is already complete; reuse it for evaluation"
            )
    else:
        manifest = {
            "schema": "robotwin_rlinf_success_seeds/v1",
            "complete": False,
            "protocol": settings,
            "attempts": [],
            "open_laptop": {"success_seeds": [], "instructions": {}},
        }
    selected = manifest["open_laptop"]
    next_seed = (
        manifest["attempts"][-1]["seed"] + 1 if manifest["attempts"] else args.start
    )
    while (
        len(selected["success_seeds"]) < args.count
        and next_seed < args.start + args.max_attempts
    ):
        task = new_task("open_laptop")
        record = {"seed": next_seed, "success": False}
        try:
            task.setup_demo(
                now_ep_num=next_seed, seed=next_seed, is_test=True, **task_config
            )
            task.play_once()
            record["success"] = bool(task.plan_success and task.check_success())
            if record["success"]:
                instruction = task_instruction(task, next_seed, "seen")
                selected["success_seeds"].append(next_seed)
                selected["instructions"][str(next_seed)] = instruction
        except UnStableError as error:
            record["reason"] = str(error)
        except AssertionError as error:
            if not is_known_expert_failure(error):
                raise
            record["reason"] = "expert_no_valid_grasp_pose"
            record["exception"] = str(error)
        finally:
            task.close_env(clear_cache=True)
            del task
            gc.collect()
        manifest["attempts"].append(record)
        manifest["complete"] = len(selected["success_seeds"]) == args.count
        write_json(args.output, manifest)
        print(
            f"seed={next_seed} expert_success={record['success']} selected={len(selected['success_seeds'])}/{args.count}",
            flush=True,
        )
        next_seed += 1
    if not manifest["complete"]:
        raise RuntimeError(
            f"Only {len(selected['success_seeds'])} valid seeds found; manifest is incomplete"
        )


def run_one(args: argparse.Namespace, checkpoint: str, output_root: Path) -> None:
    """Materialize configuration and launch one standalone RLinf evaluation."""
    from omegaconf import OmegaConf

    manifest = json.loads(args.seeds.read_text())
    if not manifest.get("complete"):
        raise ValueError("Use a complete expert-screened seed manifest")
    if manifest["protocol"]["source_sha256"] != source_fingerprints():
        raise ValueError(
            "Source changed since expert screening; regenerate the seed manifest"
        )
    available = manifest["open_laptop"]["success_seeds"]
    if len(set(available)) != len(available) or len(available) < args.episodes:
        raise ValueError(
            "Seed manifest must contain enough distinct expert-validated seeds"
        )
    run_dir = output_root / checkpoint
    run_dir.mkdir(parents=True, exist_ok=False)
    seeds = available[: args.episodes]
    subset = {
        "open_laptop": {
            "success_seeds": seeds,
            "instructions": {
                str(s): manifest["open_laptop"]["instructions"][str(s)] for s in seeds
            },
        }
    }
    seed_path = run_dir / "seeds.json"
    write_json(seed_path, subset)
    checkpoint_path = Path("/data/robotwin_ckpt/rlinf") / checkpoint
    model = OmegaConf.load(checkpoint_path / "rlinf_model.yaml")
    model.model_path = str(checkpoint_path)
    model.openpi_data.norm_stats_path = str(
        checkpoint_path / "physical-intelligence/robotwin/norm_stats.json"
    )
    for path in [
        checkpoint_path / "model.safetensors",
        Path(model.openpi_data.norm_stats_path),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = OmegaConf.load(ROOT / "rlinf_eval/open_laptop.yaml")
    cfg.rollout.model = model
    cfg.rollout.model_load_lock = os.environ.get("MODEL_LOAD_LOCK")
    cfg.rollout.runtime_report_path = str(run_dir / "model_runtime.json")
    cfg.cluster.component_placement = {"env, rollout": str(args.gpu)}
    cfg.env.eval.rollout_epoch = args.episodes
    cfg.env.eval.assets_path = str(ROOT)
    cfg.env.eval.seeds_path = str(seed_path)
    cfg.env.eval.task_config.episode_manifest_path = str(seed_path)
    cfg.env.eval.task_config.episode_log_path = str(run_dir / "episodes.jsonl")
    cfg.runner.logger.log_path = str(run_dir)
    cfg.runner.logger.experiment_name = f"open_laptop_{checkpoint}"
    cfg.env.eval.video_cfg.video_base_dir = str(run_dir / "video")
    OmegaConf.save(cfg, run_dir / "config.yaml")
    command = [
        sys.executable,
        "-m",
        "rlinf_eval.eval_agent",
        "--config-path",
        str(run_dir),
        "--config-name",
        "config",
    ]
    versions = {}
    for package in ("torch", "sapien", "mplib", "ray"):
        versions[package] = importlib.metadata.version(package)
    write_json(
        run_dir / "run.json",
        {
            "checkpoint": checkpoint,
            "gpu": args.gpu,
            "episodes": args.episodes,
            "command": command,
            "source_sha256": source_fingerprints(),
            "evaluation_source_sha256": {
                str(p.relative_to(ROOT)): sha256(p)
                for p in sorted((ROOT / "rlinf_eval").iterdir())
                if p.is_file()
            },
            "seed_manifest_sha256": sha256(args.seeds),
            "package_versions": versions,
            "sapien_runtime": os.environ["SAPIEN_RUNTIME"],
            "assets_path": os.environ.get("ROBOTWIN_ASSETS_PATH"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "render_device": os.environ.get("ROBOTWIN_RENDER_DEVICE"),
            "environment": {
                name: os.environ.get(name)
                for name in [
                    "RAY_ADDRESS",
                    "RAY_OVERRIDE_RESOURCES",
                    "RAY_TMPDIR",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "JAX_PLATFORMS",
                    "OPENPI_DATA_HOME",
                    "VK_ICD_FILENAMES",
                    "MODEL_LOAD_LOCK",
                ]
            },
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "exit_code": None,
        },
    )
    with (
        record_gpu(run_dir / "gpu_usage.jsonl"),
        (run_dir / "console.log").open("w") as log,
    ):
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    metadata = json.loads((run_dir / "run.json").read_text())
    metadata["exit_code"] = code
    metadata["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    write_json(run_dir / "run.json", metadata)
    if code:
        raise subprocess.CalledProcessError(code, command)
    summarize(run_dir)


def summarize(run_dir: Path) -> dict:
    """Require every requested scene exactly once before reporting success rate."""
    expected = json.loads((run_dir / "seeds.json").read_text())["open_laptop"][
        "success_seeds"
    ]
    metadata = json.loads((run_dir / "run.json").read_text())
    records = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    seen = [record["seed"] for record in records]
    complete = (
        metadata["exit_code"] == 0
        and len(seen) == len(expected)
        and set(seen) == set(expected)
    )
    successes = sum(record["success"] for record in records)
    result = {
        "complete": complete,
        "success": successes,
        "total": len(records),
        "expected": len(expected),
        "success_rate": successes / len(records) if records else None,
        "episodes": records,
    }
    write_json(run_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not complete:
        raise RuntimeError(
            "Incomplete or duplicate episode coverage; do not use as a final success rate"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser(
        "scan", help="Run expert screening; uses one GPU for rendering"
    )
    scan_parser.add_argument("--start", type=int, default=100000)
    scan_parser.add_argument("--count", type=int, default=100)
    scan_parser.add_argument("--max-attempts", type=int, default=2000)
    scan_parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "success_seeds.json"
    )
    run_parser = commands.add_parser(
        "run", help="Run one checkpoint, or both sequentially"
    )
    run_parser.add_argument(
        "--checkpoint", choices=["7272", "18180", "both"], default="both"
    )
    run_parser.add_argument(
        "--seeds", type=Path, default=OUTPUT_ROOT / "success_seeds.json"
    )
    run_parser.add_argument("--episodes", type=int, default=4)
    run_parser.add_argument("--output", type=Path)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "scan":
        if args.count < 1 or args.max_attempts < args.count or args.start < 0:
            parser.error("Require count > 0, max-attempts >= count and start >= 0")
        with record_gpu(args.output.with_suffix(".gpu_usage.jsonl")):
            scan(args)
    elif args.command == "run":
        if args.episodes < 1:
            parser.error("episodes must be positive")
        args.gpu = int(os.environ["GPU_LIST"])
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        output = (args.output or OUTPUT_ROOT / timestamp).resolve()
        checkpoints = (
            ["7272", "18180"] if args.checkpoint == "both" else [args.checkpoint]
        )
        summaries = {}
        for checkpoint in checkpoints:
            run_one(args, checkpoint, output)
            summaries[checkpoint] = json.loads(
                (output / checkpoint / "summary.json").read_text()
            )
        comparison_name = (
            "comparison.json"
            if args.checkpoint == "both"
            else f"comparison_{args.checkpoint}.json"
        )
        write_json(output / comparison_name, summaries)
    else:
        summarize(args.run_dir)


if __name__ == "__main__":
    main()
