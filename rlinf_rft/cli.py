"""Prepare, inspect and run the open-laptop PPO validation in explicit stages."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rlinf_rft.common import (
    BASE_MODEL,
    EVAL_CONFIG,
    ROOT,
    TRAIN_CONFIG,
    disjoint_seeds,
    file_hash,
    gpu_ids,
    load_manifest,
    parse_gpu_report,
    select_gpus,
    select_seeds,
    training_seed_count,
    validate_train_config,
    write_json,
)


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def compose_template(name: str):
    from hydra import compose, initialize_config_dir

    repo = Path(os.environ.get("REPO_PATH", "/workspace/RLinf"))
    directory = repo / (
        "evaluations/robotwin" if name == EVAL_CONFIG else "examples/embodiment/config"
    )
    os.environ["EMBODIED_PATH"] = str(repo / "examples/embodiment")
    with initialize_config_dir(version_base="1.1", config_dir=str(directory.resolve())):
        return compose(config_name=name)


def configure_train(gpus: list[int]):
    from omegaconf import OmegaConf

    cfg = compose_template(TRAIN_CONFIG)
    cfg.cluster.component_placement = {"actor, env, rollout": ",".join(map(str, gpus))}
    cfg.env.train.total_num_envs = 2 * len(gpus)
    cfg.actor.global_batch_size = 4 * len(gpus)
    validate_train_config(OmegaConf.to_container(cfg, resolve=True), len(gpus))
    return cfg


def save_config(cfg, path: Path) -> None:
    from omegaconf import OmegaConf

    snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    snapshot.defaults = ["_self_", {"override hydra/job_logging": "stdout"}]
    snapshot.hydra = {"run": {"dir": "."}, "output_subdir": None}
    OmegaConf.save(snapshot, path)


def model_files(checkpoint: Path) -> dict:
    """Check the existing converted SFT export without instantiating a model."""
    import yaml
    from safetensors import safe_open

    metadata = yaml.safe_load((checkpoint / "rlinf_model.yaml").read_text())
    if metadata["openpi"]["config_name"] != "pi05_aloha_robotwin":
        raise ValueError("Expected a RoboTwin Pi0.5 SFT checkpoint")
    if metadata["num_action_chunks"] != 50 or metadata["action_dim"] != 14:
        raise ValueError(
            "Checkpoint must use T50 and 14-dimensional environment actions"
        )
    if metadata["openpi_data"]["adapt_to_pi"] is not False:
        raise ValueError("Expected the existing SFT adapt_to_pi=false convention")
    norm = checkpoint / "physical-intelligence/robotwin/norm_stats.json"
    statistics = json.loads(norm.read_text())["norm_stats"]
    if not all(key in statistics for key in ("state", "actions")):
        raise ValueError("Missing state/actions normalization statistics")
    with safe_open(
        checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        names = list(weights.keys())
        if not any(name.startswith("llm.") for name in names):
            raise ValueError("Expected converted RLinf OpenPI weights with llm.* keys")
        if any(name.startswith("value_head.") for name in names):
            raise ValueError("Expected SFT weights without a PPO value head")
    return {
        "checkpoint": str(checkpoint),
        "tensor_count": len(names),
        "norm_stats_sha256": file_hash(norm),
        "metadata_sha256": file_hash(checkpoint / "rlinf_model.yaml"),
    }


def resources(args: argparse.Namespace) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows = parse_gpu_report(subprocess.check_output(command, text=True, timeout=15))
    report = {"recorded_at": timestamp(), "gpus": rows}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        report["host_memory_kib"] = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in meminfo.read_text().splitlines()
            if line.startswith(
                ("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")
            )
        }
    if args.peak_mib:
        report["measured_peak_mib"] = args.peak_mib
        report["reserve_mib"] = 4096
        report["eligible_gpus"] = select_gpus(
            rows, peak_mib=args.peak_mib, max_gpus=args.max_gpus
        )
    print(json.dumps(report, indent=2))


def check_config(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    gpus = gpu_ids(args.gpus)
    cfg = configure_train(gpus)
    evaluation = compose_template(EVAL_CONFIG)
    result = {
        "config_valid": True,
        "gpus": gpus,
        "environments": cfg.env.train.total_num_envs,
        "global_batch_size": cfg.actor.global_batch_size,
        "chunks_per_episode": cfg.env.train.max_episode_steps
        // cfg.actor.model.num_action_chunks,
        "model": model_files(args.checkpoint) if args.checkpoint else None,
        "train": OmegaConf.to_container(cfg, resolve=True),
        "eval": OmegaConf.to_container(evaluation, resolve=True),
    }
    print(json.dumps(result, indent=2))


def prepare(args: argparse.Namespace) -> None:
    """Freeze non-overlapping manifests and composed templates for one run."""
    gpus = gpu_ids(args.gpus)
    train_manifest = load_manifest(args.train_seeds, require_complete=True)
    eval_manifest = load_manifest(args.eval_seeds, require_complete=True)
    demo_seeds = {int(value) for value in args.demo_seeds.read_text().split()}
    disjoint_seeds(train_manifest, eval_manifest, demo_seeds)
    count = training_seed_count(
        len(train_manifest["open_laptop"]["success_seeds"]), len(gpus)
    )
    train = select_seeds(train_manifest, count)
    evaluation = select_seeds(eval_manifest, 8)
    model_report = model_files(args.checkpoint)
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError(
            "Write training artifacts outside the synchronized RoboTwin tree"
        )
    repo = Path(os.environ.get("REPO_PATH", "/workspace/RLinf")).resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Write training artifacts outside the synchronized RLinf tree")
    cfg = configure_train(gpus)
    if getattr(args, "gradient_checkpointing", None) is not None:
        from omegaconf import OmegaConf

        cfg.actor.fsdp_config.gradient_checkpointing = args.gradient_checkpointing
        OmegaConf.update(
            cfg.actor.fsdp_config,
            "gradient_checkpointing_use_reentrant",
            False,
            force_add=True,
        )
    cfg.runner.logger.log_path = str(output / "train")
    cfg.env.train.assets_path = str(ROOT)
    cfg.env.train.seeds_path = str(output / "train_seeds.json")
    cfg.env.train.task_config.episode_manifest_path = str(output / "train_seeds.json")
    cfg.env.train.task_config.episode_log_path = str(
        output / "train" / "episodes.jsonl"
    )
    cfg.actor.model.model_path = str(args.checkpoint)
    cfg.rollout.model.model_path = str(args.checkpoint)
    cfg.actor.model.openpi_data.norm_stats_path = str(
        args.checkpoint / "physical-intelligence/robotwin/norm_stats.json"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "train_seeds.json", train)
    write_json(output / "eval_seeds.json", evaluation)
    save_config(cfg, output / "train_config.yaml")
    sources = list((ROOT / "rlinf_rft").glob("*.py")) + list(
        (ROOT / "rlinf_rft").glob("*.sh")
    )
    sources += [ROOT / "robotwin/envs/vector_env.py", ROOT / "envs/open_laptop.py"]
    sources += [
        repo / "examples/embodiment/config" / f"{TRAIN_CONFIG}.yaml",
        repo / "examples/embodiment/config/env/robotwin_open_laptop.yaml",
        repo / "evaluations/robotwin" / f"{EVAL_CONFIG}.yaml",
    ]
    write_json(
        output / "run.json",
        {
            "prepared_at": timestamp(),
            "gpus": gpus,
            "eval_gpu": getattr(args, "eval_gpu", None)
            if getattr(args, "eval_gpu", None) is not None
            else gpus[0],
            "training_config_path": str(output / "train_config.yaml"),
            "training_config_sha256": file_hash(output / "train_config.yaml"),
            "training_seed_count": count,
            "unused_training_seed_count": len(
                train_manifest["open_laptop"]["success_seeds"]
            )
            - count,
            "training_manifest_sha256": file_hash(args.train_seeds),
            "evaluation_manifest_sha256": file_hash(args.eval_seeds),
            "demonstration_seeds_sha256": file_hash(args.demo_seeds),
            "model": model_report,
            "model_path": str(args.checkpoint),
            "model_weights_sha256": file_hash(args.checkpoint / "model.safetensors"),
            "source_sha256": {str(path): file_hash(path) for path in sources},
            "norm_stats_path": str(
                args.checkpoint / "physical-intelligence/robotwin/norm_stats.json"
            ),
        },
    )
    print(
        json.dumps(
            {"prepared": str(output), "gpus": gpus, "training_scenes": count}, indent=2
        )
    )


def run_agent(config: Path, log_dir: Path, gpus: list[int], mode: str) -> None:
    from rlinf_rft.monitor import record_resources

    if gpu_ids(os.environ.get("GPU_LIST", "")) != gpus:
        raise ValueError(f"GPU_LIST must match the frozen GPU list {gpus}")
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-B",
        "-m",
        "rlinf_rft.agent",
        "--config-path",
        str(config.parent),
        "--config-name",
        config.stem,
    ]
    env = os.environ.copy()
    key = file_hash(config)[:10]
    env["RAY_TMPDIR"] = f"/tmp/wrft_{key}"
    env["RFT_MODE"] = mode
    env["RFT_OUTPUT"] = str(log_dir)
    metadata = {
        "command": command,
        "gpus": gpus,
        "started_at": timestamp(),
        "config_sha256": file_hash(config),
        "exit_code": None,
    }
    write_json(log_dir / "process.json", metadata)
    with record_resources(log_dir, gpus), (log_dir / "console.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        metadata["pid"] = process.pid
        write_json(log_dir / "process.json", metadata)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    metadata.update(exit_code=code, finished_at=timestamp())
    write_json(log_dir / "process.json", metadata)
    if code:
        raise subprocess.CalledProcessError(code, command)


def assert_sources(metadata: dict) -> None:
    for path, expected in metadata["source_sha256"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f"Source changed after configuration freeze: {path}")
    if (
        "training_config_path" in metadata
        and file_hash(Path(metadata["training_config_path"]))
        != metadata["training_config_sha256"]
    ):
        raise ValueError("Training config changed after configuration freeze")
    model = Path(metadata["model_path"])
    if file_hash(model / "model.safetensors") != metadata["model_weights_sha256"]:
        raise ValueError("SFT weights changed after configuration freeze")
    if (
        file_hash(Path(metadata["norm_stats_path"]))
        != metadata["model"]["norm_stats_sha256"]
    ):
        raise ValueError("Normalization statistics changed after configuration freeze")


def configure_long_training(
    cfg, metadata: dict, rounds: int, save_interval: int, output: Path
):
    """Start an independent run from the original SFT, with fresh optimizer state."""
    if cfg.runner.max_steps != 3 or cfg.runner.max_epochs != 3:
        raise ValueError("Long training requires the unchanged three-round recipe")
    if rounds < 1 or save_interval < 1:
        raise ValueError("Require positive training rounds and checkpoint interval")
    cfg = copy.deepcopy(cfg)
    cfg.runner.max_epochs = cfg.runner.max_steps = rounds
    cfg.runner.resume_dir = None
    cfg.runner.ckpt_path = None
    cfg.runner.save_interval = save_interval
    cfg.runner.logger.log_path = str(output / "long_train")
    cfg.actor.model.model_path = metadata["model_path"]
    cfg.rollout.model.model_path = metadata["model_path"]
    cfg.env.train.task_config.episode_log_path = str(
        output / "long_train/episodes.jsonl"
    )
    return cfg


def train(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    output = args.run_dir.resolve()
    metadata = json.loads((output / "run.json").read_text())
    assert_sources(metadata)
    config = output / "train_config.yaml"
    gpus = metadata["gpus"]
    long_rounds = getattr(args, "long_rounds", None)
    label = "probe" if args.probe else "long_train" if long_rounds else "train"
    if args.probe:
        cfg = OmegaConf.load(config)
        cfg.runner.max_epochs = cfg.runner.max_steps = 1
        cfg.runner.save_interval = -1
        cfg.runner.logger.log_path = str(output / label)
        cfg.env.train.task_config.episode_log_path = str(
            output / label / "episodes.jsonl"
        )
        cfg.actor.optim.lr = cfg.actor.optim.value_lr = 0.0
        config = output / "probe_config.yaml"
        save_config(cfg, config)
    else:
        if not args.peak_mib:
            raise ValueError(
                "Formal training requires --peak-mib from the PPO resource probe"
            )
        rows = parse_gpu_report(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=15,
            )
        )
        # Keep the frozen selection; do not re-rank it against unrelated GPUs.
        eligible = {
            row["index"] for row in rows if row["free_mib"] >= args.peak_mib + 4096
        }
        if any(gpu not in eligible for gpu in gpus):
            raise ValueError(
                "Selected GPU no longer has measured PPO peak + 4 GiB free"
            )
        if long_rounds:
            from rlinf_rft.validation import check_validation

            gate = check_validation(output)
            cfg = configure_long_training(
                OmegaConf.load(config),
                metadata,
                long_rounds,
                args.long_save_interval,
                output,
            )
            config = output / "long_train_config.yaml"
            save_config(cfg, config)
            write_json(
                output / "long_training_initialization.json",
                {
                    "model_initialization": metadata["model_path"],
                    "model_weights_sha256": metadata["model_weights_sha256"],
                    "validation_checkpoint": gate["checkpoint"],
                    "resume_dir": None,
                    "fresh_optimizer": True,
                    "initial_global_step": 0,
                    "max_steps": long_rounds,
                    "save_interval": args.long_save_interval,
                },
            )
    run_agent(config, output / label, gpus, "train")


def evaluate(args: argparse.Namespace) -> None:
    output = args.run_dir.resolve()
    metadata = json.loads((output / "run.json").read_text())
    assert_sources(metadata)
    cfg = compose_template(EVAL_CONFIG)
    gpu = metadata["eval_gpu"]
    checkpoint = args.checkpoint or Path(metadata["model_path"])
    log_dir = output / f"eval_{args.label}"
    cfg.cluster.component_placement = {"env, rollout": str(gpu)}
    cfg.runner.logger.log_path = str(log_dir)
    cfg.rollout.model.model_path = str(checkpoint)
    cfg.rollout.model.openpi_data.norm_stats_path = metadata["norm_stats_path"]
    cfg.env.eval.assets_path = str(ROOT)
    cfg.env.eval.seeds_path = str(output / "eval_seeds.json")
    cfg.env.eval.task_config.episode_manifest_path = str(output / "eval_seeds.json")
    cfg.env.eval.task_config.episode_log_path = str(log_dir / "episodes.jsonl")
    config = output / f"eval_{args.label}_config.yaml"
    save_config(cfg, config)
    run_agent(config, log_dir, [gpu], "eval")
    records = [
        json.loads(line)
        for line in (log_dir / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    expected = load_manifest(output / "eval_seeds.json")["open_laptop"]["success_seeds"]
    if len(records) != len(expected) or {r["seed"] for r in records} != set(expected):
        raise RuntimeError("Evaluation did not cover each frozen scene exactly once")
    manifest = load_manifest(output / "eval_seeds.json")["open_laptop"]
    if any(
        r["instruction"] != manifest["instructions"][str(r["seed"])] for r in records
    ):
        raise RuntimeError("Evaluation instructions differ from the frozen manifest")
    write_json(
        log_dir / "summary.json",
        {
            "complete": True,
            "success": sum(r["success"] for r in records),
            "total": len(records),
            "success_rate": sum(r["success"] for r in records) / len(records),
            "checkpoint": str(checkpoint),
            "episodes": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resource = commands.add_parser(
        "resources", help="Read GPU and host memory; never launches CUDA work"
    )
    resource.add_argument("--peak-mib", type=int)
    resource.add_argument("--max-gpus", type=int, default=6)
    check = commands.add_parser(
        "check-config", help="CPU-only Hydra/model-header checks"
    )
    check.add_argument("--gpus", default="0,3,6")
    check.add_argument("--checkpoint", type=Path)
    screening = commands.add_parser(
        "scan", help="Run the existing expert scene screening"
    )
    screening.add_argument("--start", type=int, default=300000)
    screening.add_argument("--count", type=int, default=56)
    screening.add_argument("--max-attempts", type=int, default=1000)
    screening.add_argument("--output", type=Path, required=True)
    prepared = commands.add_parser(
        "prepare", help="Freeze CPU-side run configuration and scene manifests"
    )
    prepared.add_argument("--gpus", required=True)
    prepared.add_argument("--checkpoint", type=Path, default=BASE_MODEL)
    prepared.add_argument("--train-seeds", type=Path, required=True)
    prepared.add_argument("--eval-seeds", type=Path, required=True)
    prepared.add_argument("--demo-seeds", type=Path, required=True)
    prepared.add_argument("--output", type=Path, required=True)
    prepared.add_argument("--eval-gpu", type=int)
    prepared.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    training = commands.add_parser(
        "train",
        help="Resource probe, three-round validation or an independent long run",
    )
    training.add_argument("--run-dir", type=Path, required=True)
    stage = training.add_mutually_exclusive_group()
    stage.add_argument("--probe", action="store_true")
    stage.add_argument("--long-rounds", type=int)
    training.add_argument("--long-save-interval", type=int, default=10)
    training.add_argument("--peak-mib", type=int)
    evaluation = commands.add_parser(
        "eval", help="Eight fixed scenes on the same physical GPU"
    )
    evaluation.add_argument("--run-dir", type=Path, required=True)
    evaluation.add_argument(
        "--label", choices=["before", "after", "after_long"], required=True
    )
    evaluation.add_argument("--checkpoint", type=Path)
    verification = commands.add_parser(
        "verify", help="CPU comparison of SFT and saved PPO parameters"
    )
    verification.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train" and (
        (args.long_rounds is not None and args.long_rounds < 1)
        or args.long_save_interval < 1
    ):
        parser.error("Training rounds and checkpoint interval must be positive")
    if args.command == "prepare" and args.eval_gpu is not None and args.eval_gpu < 0:
        parser.error("Evaluation GPU must be nonnegative")
    if args.command == "scan":
        from rlinf_eval.cli import scan
        from rlinf_rft.monitor import record_resources

        selected = gpu_ids(os.environ.get("GPU_LIST", ""))
        if (
            len(selected) != 1
            or args.start < 0
            or args.count < 1
            or args.max_attempts < args.count
        ):
            parser.error(
                "Screening requires one GPU, positive count and sufficient attempts"
            )
        with record_resources(args.output.parent / "screening_resources", selected):
            scan(args)
    elif args.command == "verify":
        from rlinf_rft.verify import verify

        verify(args.run_dir)
    else:
        {
            "resources": resources,
            "check-config": check_config,
            "prepare": prepare,
            "train": train,
            "eval": evaluate,
        }[args.command](args)


if __name__ == "__main__":
    main()
