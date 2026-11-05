"""Wait for three GPUs, validate PPO, then start a fresh 300-round SFT-initialized run."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from rlinf_rft.common import ROOT, file_hash, parse_gpu_report, write_json
from rlinf_rft.cli import timestamp

CAMPAIGN = Path("/workspace/ctr_backup/open_laptop_eval/full100_20261103_gpu3_6")


def gpu_report() -> list[dict]:
    return parse_gpu_report(
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


def choose_gpus(
    rows: list[dict],
    minimum: int,
    maximum_utilization: int,
    fixed: list[int] | None = None,
) -> list[int] | None:
    eligible = [
        row
        for row in rows
        if row["free_mib"] >= minimum and row["utilization"] <= maximum_utilization
    ]
    if fixed is not None:
        return fixed if set(fixed) <= {row["index"] for row in eligible} else None
    eligible.sort(key=lambda row: (-row["free_mib"], row["utilization"], row["index"]))
    return sorted(row["index"] for row in eligible[:3]) if len(eligible) >= 3 else None


def source_hashes() -> dict:
    repo = Path(os.environ.get("REPO_PATH", "/workspace/RLinf"))
    files = list((ROOT / "rlinf_rft").glob("*.py")) + list(
        (ROOT / "rlinf_rft").glob("*.sh")
    )
    files += [ROOT / "robotwin/envs/vector_env.py", ROOT / "envs/open_laptop.py"]
    files += [
        repo / "examples/embodiment/config/robotwin_open_laptop_ppo_openpi_pi05.yaml",
        repo / "examples/embodiment/config/env/robotwin_open_laptop.yaml",
        repo / "evaluations/robotwin/robotwin_open_laptop_openpi_pi05_eval.yaml",
    ]
    return {str(path): file_hash(path) for path in files}


def probe_peak(probe_dir: Path) -> int:
    summary = json.loads((probe_dir / "resource_summary.json").read_text())
    by_device = {}
    for path in (probe_dir / "workers").glob("*.json"):
        worker = json.loads(path.read_text())
        if not worker["ready"]:
            raise ValueError(f"Probe worker never became ready: {path}")
        uuid = worker["uuid"]
        by_device[uuid] = by_device.get(uuid, 0) + max(
            worker["peak_reserved_mib"], worker["peak_allocated_mib"]
        )
    peak = max(summary["sampled_worker_peak_mib"], max(by_device.values(), default=0))
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("Probe did not produce a usable memory measurement")
    return math.ceil(peak)


def checkpoint_path(run_dir: Path, label: str, step: int) -> Path:
    return (
        run_dir
        / label
        / "robotwin_open_laptop_ppo_openpi_pi05/checkpoints"
        / f"global_step_{step}"
    )


class Workflow:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.status_path = self.root / "status.json"
        self.status = {
            "schema": "robotwin_ppo_workflow/v1",
            "run_id": str(self.root),
            "phase": "starting",
            "pid": os.getpid(),
            "events": [],
        }
        self.sources = source_hashes()
        write_json(
            self.root / "plan.json",
            {
                **vars(args),
                "output": str(self.root),
                "smoke_report": str(args.smoke_report),
                "source_sha256": self.sources,
            },
        )
        self.event(
            "starting",
            "自动流程启动：3 轮验收通过后，从原始 7272 SFT 权重新开 300 轮训练。",
        )

    def event(self, phase: str, message: str, **details):
        event = {
            "id": len(self.status["events"]) + 1,
            "time": timestamp(),
            "phase": phase,
            "message": message,
            **details,
        }
        self.status.update(phase=phase, updated_at=event["time"])
        self.status["events"].append(event)
        write_json(self.status_path, self.status)
        print(f"[{event['time']}] {message}", flush=True)

    def unchanged_sources(self):
        if source_hashes() != self.sources:
            raise ValueError("Source changed while the workflow was waiting/running")

    def wait(
        self, phase: str, minimum: int, fixed: list[int] | None = None
    ) -> list[int]:
        self.event(
            phase,
            f"等待 GPU：每卡空余至少 {minimum / 1024:.1f} GiB，连续 {self.args.stable_checks} 次满足条件。",
            fixed_gpus=fixed,
        )
        previous = None
        stable = 0
        while True:
            self.unchanged_sources()
            rows = gpu_report()
            available_kib = next(
                int(line.split()[1])
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            choice = choose_gpus(rows, minimum, self.args.max_utilization, fixed)
            host_ok = available_kib >= self.args.host_free_gib * 1024**2
            disk_ok = (
                shutil.disk_usage(self.root).free >= self.args.disk_free_gib * 1024**3
            )
            stable = (
                stable + 1
                if choice is not None and choice == previous and host_ok and disk_ok
                else int(choice is not None and host_ok and disk_ok)
            )
            previous = choice
            self.status.update(
                updated_at=timestamp(),
                resources={
                    "gpus": rows,
                    "host_available_kib": available_kib,
                    "disk_free_bytes": shutil.disk_usage(self.root).free,
                    "stable_checks": stable,
                    "candidate_gpus": choice,
                },
            )
            write_json(self.status_path, self.status)
            if stable >= self.args.stable_checks:
                return choice
            time.sleep(self.args.poll_seconds)

    def run(self, label: str, command: list[str], gpus: list[int] | None = None):
        self.unchanged_sources()
        environment = os.environ.copy()
        environment["GPU_LIST"] = ",".join(map(str, gpus or []))
        with (self.root / f"{label}.log").open("a") as log:
            subprocess.run(
                ["bash", str(ROOT / "rlinf_rft/run.sh"), *command],
                env=environment,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def execute(self):
        smoke = json.loads(self.args.smoke_report.read_text())
        if (
            smoke.get("kind") != "low_memory_smoke"
            or smoke.get("passed") is not True
            or smoke.get("source_sha256") != self.sources
        ):
            raise ValueError(
                "Require a passing low-memory smoke report for the exact current sources"
            )
        gpus = self.wait("waiting_resources", self.args.min_free_mib)
        self.status["gpus"] = gpus
        run_dir = self.root / "run"
        self.event(
            "preparing", f"已选 GPU {gpus}，准备模型、54 个训练场景及固定 8 场评测。"
        )
        self.run(
            "prepare",
            [
                "prepare",
                "--gpus",
                ",".join(map(str, gpus)),
                "--checkpoint",
                "/data/robotwin_ckpt/rlinf/7272",
                "--train-seeds",
                "/workspace/open_laptop_ppo/scenes/train56.json",
                "--eval-seeds",
                str(CAMPAIGN / "success_seeds.json"),
                "--demo-seeds",
                "/data/RoboTwin/dataset/open_laptop/aloha-agilex_clean_50/seed.txt",
                "--eval-gpu",
                "3",
                "--gradient-checkpointing",
                "--output",
                str(run_dir),
            ],
        )
        self.event("probe_running", "开始完整模型 probe：1 轮，学习率为 0。")
        self.run("probe", ["train", "--run-dir", str(run_dir), "--probe"], gpus)
        peak = probe_peak(run_dir / "probe")
        self.status["measured_peak_mib"] = peak
        if any(
            row["used_mib"] + row["free_mib"] < peak + 4096
            for row in gpu_report()
            if row["index"] in gpus
        ):
            raise ValueError(
                "Measured peak plus 4 GiB exceeds the selected GPU capacity"
            )
        self.event(
            "probe_passed",
            f"完整模型 probe 通过；保守显存峰值 {peak / 1024:.2f} GiB，接下来验证 3 轮 PPO。",
            peak_mib=peak,
        )
        self.wait("waiting_validation_resources", peak + 4096, gpus)
        self.event("validation_training", "开始 3 轮 PPO；每轮保存 checkpoint。")
        self.run(
            "validation_train",
            ["train", "--run-dir", str(run_dir), "--peak-mib", str(peak)],
            gpus,
        )
        self.event(
            "parameter_verification",
            "三轮训练完成，检查参数更新、冻结 VLM、奖励和 checkpoint。",
        )
        self.run("verify", ["verify", "--run-dir", str(run_dir)])
        self.wait("waiting_reload_evaluation", 22 * 1024, [3])
        self.event(
            "reload_evaluation",
            "在原评测 GPU 3 上重载第 3 轮 checkpoint，评测固定 8 场。",
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
            "3 轮验证全部通过：参数、冻结范围、reward/return、TensorBoard 和 checkpoint 重载正常。",
        )
        self.wait("waiting_long_training_resources", peak + 4096, gpus)
        self.event(
            "long_training",
            "三轮验收已通过；从原始 7272 SFT 权重和全新优化器开始 300 轮训练。",
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
            raise FileNotFoundError(f"Final full checkpoint missing: {final}")
        self.event(
            "complete",
            f"独立的 300 轮训练完成。最终 checkpoint：{final}",
            checkpoint=str(final),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/workspace/open_laptop_ppo/auto_7272")
    )
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=44 * 1024)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--host-free-gib", type=int, default=64)
    parser.add_argument("--disk-free-gib", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--long-save-interval", type=int, default=10)
    args = parser.parse_args()
    if (
        min(
            args.min_free_mib,
            args.host_free_gib,
            args.disk_free_gib,
            args.poll_seconds,
            args.stable_checks,
            args.long_save_interval,
        )
        <= 0
        or not 0 <= args.max_utilization <= 100
    ):
        parser.error("Require positive limits/intervals and utilization in [0, 100]")
    for tree in (ROOT, Path(os.environ.get("REPO_PATH", "/workspace/RLinf")).resolve()):
        if args.output.resolve() == tree or tree in args.output.resolve().parents:
            parser.error("Workflow artifacts must be outside synchronized source trees")
    workflow = Workflow(args)
    try:
        workflow.execute()
    except BaseException as error:
        workflow.event(
            "failed", f"流程停止：{error}；查看 {workflow.root} 下的阶段日志。"
        )
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
