"""Attach to an existing probe and continue without repeated stability delays."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from rlinf_rft.cli import save_config, timestamp
from rlinf_rft.common import file_hash, gpu_ids, validate_train_config, write_json
from rlinf_rft.workflow import Workflow, checkpoint_path, gpu_report, probe_peak


def derive_training_run(original: Path, output: Path, gpus: list[int]) -> Path:
    """Reuse frozen data/model provenance for the same-size training placement."""
    from omegaconf import OmegaConf

    metadata = json.loads((original / "run.json").read_text())
    if len(gpus) != len(metadata["gpus"]):
        raise ValueError("The validated FSDP world size must be preserved")
    cfg = OmegaConf.load(original / "train_config.yaml")
    cfg.cluster.component_placement = {"actor, env, rollout": ",".join(map(str, gpus))}
    cfg.runner.logger.log_path = str(output / "train")
    cfg.env.train.seeds_path = str(output / "train_seeds.json")
    cfg.env.train.task_config.episode_manifest_path = str(output / "train_seeds.json")
    cfg.env.train.task_config.episode_log_path = str(output / "train/episodes.jsonl")
    validate_train_config(OmegaConf.to_container(cfg, resolve=True), len(gpus))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("train_seeds.json", "eval_seeds.json"):
        shutil.copyfile(original / name, output / name)
    save_config(cfg, output / "train_config.yaml")
    metadata.update(
        prepared_at=timestamp(),
        gpus=gpus,
        derived_from=str(original),
        resource_probe_dir=str(original / "probe"),
        training_config_path=str(output / "train_config.yaml"),
        training_config_sha256=file_hash(output / "train_config.yaml"),
    )
    write_json(output / "run.json", metadata)
    return output


class ImmediateWorkflow(Workflow):
    def __init__(self, root: Path, training_gpus: list[int] | None = None):
        self.root = root.resolve()
        self.status_path = self.root / "status.json"
        self.status = json.loads(self.status_path.read_text())
        plan = json.loads((self.root / "plan.json").read_text())
        self.sources = plan["source_sha256"]
        self.args = SimpleNamespace(**plan)
        self.args.stable_checks = 1
        self.args.poll_seconds = 5
        self.training_gpus = training_gpus
        self.unchanged_sources()
        if self.status["phase"] not in (
            "probe_running",
            "probe_passed",
            "waiting_validation_resources",
            "checking_validation_resources",
        ):
            raise ValueError(
                "Only an existing probe or its post-probe wait can be attached"
            )
        old_pid = self.status["pid"]
        command_file = Path(f"/proc/{old_pid}/cmdline")
        command = command_file.read_bytes().split(b"\0")
        if (
            b"rlinf_rft.workflow" not in command
            and str(Path(__file__).resolve()).encode() not in command
        ):
            raise ValueError(
                "Refusing to stop a process other than the original workflow controller"
            )
        # Stop only the controller. Its running probe CLI and Ray workers continue.
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(100):
            try:
                command = command_file.read_bytes()
            except FileNotFoundError:
                break
            if not command:
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("Original workflow controller did not stop")
        self.status["pid"] = os.getpid()
        plan.setdefault("execution_policy_changes", []).append(
            {
                "at": timestamp(),
                "stable_checks": 1,
                "poll_seconds": 5,
                "reason": "User requested immediate validation without repeated idle waits",
                "controller": str(Path(__file__).resolve()),
                "controller_sha256": file_hash(Path(__file__)),
                "replaced_controller_pid": old_pid,
                "requested_training_gpus": training_gpus,
            }
        )
        write_json(self.root / "plan.json", plan)
        self.event(
            "probe_running",
            "已接续当前 probe，模型和 worker 保持运行；后续仅检查一次资源，不再等待三次稳定采样。",
        )

    def execute(self):
        run_dir = self.root / "run"
        missing_since = None
        while True:
            self.unchanged_sources()
            process = json.loads((run_dir / "probe/process.json").read_text())
            if process["exit_code"] is not None:
                if process["exit_code"] != 0:
                    raise RuntimeError("Current probe failed; inspect probe.log")
                break
            if not Path(f"/proc/{process['pid']}").exists():
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since > 30:
                    raise RuntimeError(
                        "Probe exited without a completed process report"
                    )
            else:
                missing_since = None
            time.sleep(5)
        gpus = self.training_gpus or self.status["gpus"]
        peak = probe_peak(run_dir / "probe")
        if any(
            row["used_mib"] + row["free_mib"] < peak + 4096
            for row in gpu_report()
            if row["index"] in gpus
        ):
            raise ValueError(
                "Measured peak plus 4 GiB exceeds the selected GPU capacity"
            )
        self.status["measured_peak_mib"] = peak
        self.event(
            "probe_passed",
            f"完整模型 probe 通过，保守峰值 {peak / 1024:.2f} GiB；立即进入三轮验证。",
            peak_mib=peak,
        )
        self.wait("checking_validation_resources", peak + 4096, gpus)
        if self.training_gpus:
            probe_run = run_dir
            run_dir = derive_training_run(
                probe_run, self.root / ("run_" + "_".join(map(str, gpus))), gpus
            )
            self.status.update(
                gpus=gpus,
                training_run_dir=str(run_dir),
                resource_probe_dir=str(probe_run / "probe"),
            )
            self.event(
                "training_placement",
                f"三轮及正式训练固定使用 GPU {gpus}，沿用同规模 probe，保留峰值 + 4 GiB 余量。",
            )
        self.event("validation_training", "开始 3 轮 PPO，每轮保存 checkpoint。")
        self.run(
            "validation_train",
            ["train", "--run-dir", str(run_dir), "--peak-mib", str(peak)],
            gpus,
        )
        self.event(
            "parameter_verification", "检查三轮参数更新、冻结范围、奖励和 checkpoint。"
        )
        self.run("verify", ["verify", "--run-dir", str(run_dir)])
        self.wait("checking_reload_resources", 22 * 1024, [3])
        self.event(
            "reload_evaluation",
            "在原评测 GPU 3 上重载第 3 轮 checkpoint，执行固定 8 场评测。",
        )
        self.run(
            "reload_eval",
            [
                "eval",
                "--run-dir",
                str(run_dir),
                "--label",
                "after",
                "--checkpoint",
                str(checkpoint_path(run_dir, "train", 3)),
            ],
            [3],
        )
        from rlinf_rft.validation import check_validation

        check_validation(run_dir)
        self.event(
            "validation_passed",
            "三轮验收全部通过，准备从 7272 SFT 开始独立 300 轮训练。",
        )
        self.wait("checking_long_training_resources", peak + 4096, gpus)
        self.event(
            "long_training",
            "从原始 7272 SFT 权重和全新优化器开始 300 轮，每 10 轮保存 checkpoint。",
        )
        self.run(
            "long_train",
            [
                "train",
                "--run-dir",
                str(run_dir),
                "--peak-mib",
                str(peak),
                "--long-rounds",
                "300",
                "--long-save-interval",
                str(self.args.long_save_interval),
            ],
            gpus,
        )
        final = checkpoint_path(run_dir, "long_train", 300)
        if not (final / "actor/model_state_dict/full_weights.pt").is_file():
            raise FileNotFoundError(f"Final checkpoint missing: {final}")
        self.event(
            "complete",
            f"独立的 300 轮训练完成，checkpoint：{final}",
            checkpoint=str(final),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument(
        "--training-gpus",
        help="Same-size physical GPU set, fixed before formal training",
    )
    args = parser.parse_args()
    gpus = gpu_ids(args.training_gpus) if args.training_gpus else None
    workflow = ImmediateWorkflow(args.job_dir, gpus)
    try:
        workflow.execute()
    except BaseException as error:
        workflow.event("failed", f"流程停止：{error}；查看阶段日志。")
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
