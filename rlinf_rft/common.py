"""CPU-only manifest, configuration and resource checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = "robotwin_open_laptop_ppo_openpi_pi05"
EVAL_CONFIG = "robotwin_open_laptop_openpi_pi05_eval"
BASE_MODEL = Path("/data/robotwin_ckpt/rlinf/7272")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def gpu_ids(value: str) -> list[int]:
    """Parse physical indices without allowing duplicate or excessive ranks."""
    pieces = value.split(",")
    if not all(piece.strip().isdigit() for piece in pieces):
        raise ValueError("GPUs must be comma-separated nonnegative physical indices")
    result = [int(piece.strip()) for piece in pieces]
    if not 1 <= len(result) <= 6 or len(set(result)) != len(result):
        raise ValueError("Select between one and six distinct GPUs")
    return result


def load_manifest(path: Path, *, require_complete: bool = False) -> dict:
    manifest = json.loads(path.read_text())
    if require_complete and manifest.get("complete") is not True:
        raise ValueError(f"Expert screening is incomplete: {path}")
    task = manifest["open_laptop"]
    seeds = task["success_seeds"]
    instructions = task["instructions"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Expected nonempty distinct seeds: {path}")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError(f"Invalid seed in {path}")
    if any(
        not isinstance(instructions.get(str(s)), str)
        or not instructions[str(s)].strip()
        for s in seeds
    ):
        raise ValueError(f"Missing fixed instruction in {path}")
    return manifest


def select_seeds(manifest: dict, count: int) -> dict:
    task = manifest["open_laptop"]
    seeds = sorted(task["success_seeds"])[:count]
    if count < 1 or len(seeds) != count:
        raise ValueError(f"Expected {count} scenes, found {len(seeds)}")
    return {
        "open_laptop": {
            "success_seeds": seeds,
            "instructions": {str(s): task["instructions"][str(s)] for s in seeds},
        }
    }


def disjoint_seeds(train: dict, evaluation: dict, demo_seeds: set[int]) -> None:
    train_seeds = set(train["open_laptop"]["success_seeds"])
    overlap = train_seeds & (
        set(evaluation["open_laptop"]["success_seeds"]) | demo_seeds
    )
    if overlap:
        raise ValueError(
            f"Training scenes overlap demonstrations/evaluation: {sorted(overlap)}"
        )


def training_seed_count(available: int, ranks: int) -> int:
    # RLinf partitions by rank and then by per-rank environment groups.
    count = available // (2 * ranks) * (2 * ranks)
    if count < 6 * ranks:
        raise ValueError(
            "Need at least six distinct training scenes per GPU for three rounds"
        )
    return count


def validate_train_config(cfg: dict, ranks: int) -> None:
    actor = cfg["actor"]
    env = cfg["env"]["train"]
    model = actor["model"]
    if env["total_num_envs"] != 2 * ranks or actor["global_batch_size"] != 4 * ranks:
        raise ValueError("Require total_num_envs=2N and global_batch_size=4N")
    if actor["micro_batch_size"] != 1:
        raise ValueError("The validation recipe requires micro_batch_size=1")
    if env["max_steps_per_rollout_epoch"] % model["num_action_chunks"]:
        raise ValueError("Rollout steps must be divisible by the action chunk")
    samples = (
        env["total_num_envs"]
        * env["rollout_epoch"]
        * env["max_steps_per_rollout_epoch"]
        // model["num_action_chunks"]
    )
    if samples % actor["global_batch_size"]:
        raise ValueError("Rollout samples must be divisible by the global batch")
    if env["auto_reset"] or env["ignore_terminations"]:
        raise ValueError("Training must retain success terminations without auto-reset")
    if (
        not env["use_custom_reward"]
        or not env["use_rel_reward"]
        or env["reward_coef"] != 1
    ):
        raise ValueError("Require a single +1 termination reward")
    if model["openpi"]["task"] != "rl" or not model["add_value_head"]:
        raise ValueError("PPO requires the RL policy and value head")
    if model["openpi_data"]["adapt_to_pi"] or model["precision"] != "fp32":
        raise ValueError("Preserve the SFT transform and FP32 parameter contract")


def parse_gpu_report(output: str) -> list[dict]:
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 6:
            raise ValueError(f"Unexpected nvidia-smi row: {row}")
        index, uuid, total, used, free, utilization = [value.strip() for value in row]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization": int(utilization),
            }
        )
    return rows


def select_gpus(
    rows: list[dict], *, peak_mib: int, max_gpus: int = 6, reserve_mib: int = 4096
) -> list[int]:
    """Admit shared GPUs by capacity; zero utilization is not a requirement."""
    if peak_mib <= 0 or not 1 <= max_gpus <= 6:
        raise ValueError("Require a positive measured peak and at most six GPUs")
    candidates = [row for row in rows if row["free_mib"] >= peak_mib + reserve_mib]
    candidates.sort(
        key=lambda row: (row["utilization"], -row["free_mib"], row["index"])
    )
    return [row["index"] for row in candidates[:max_gpus]]
