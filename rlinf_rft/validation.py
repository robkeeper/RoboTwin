"""Gate long training on the completed three-round validation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from rlinf_rft.common import file_hash, load_manifest, write_json


def check_validation(run_dir: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = run_dir.resolve()
    verification = json.loads((run_dir / "parameter_verification.json").read_text())
    evaluation = json.loads((run_dir / "eval_after/summary.json").read_text())
    process = json.loads((run_dir / "train/process.json").read_text())
    checkpoint = (
        run_dir / "train/robotwin_open_laptop_ppo_openpi_pi05/checkpoints/global_step_3"
    )
    if (
        not verification["passed"]
        or process["exit_code"] != 0
        or not verification["changed_trainable_parameters"]
        or not verification["changed_value_head_parameters"]
        or verification["changed_frozen_parameters"]
        or verification["nonfinite_parameters"]
        or verification["training_metric_files"] != 3
        or verification["successful_training_episodes"] < 1
        or Path(verification["checkpoint"]).resolve() != checkpoint
    ):
        raise ValueError("Three-round parameter/reward verification did not pass")
    weights = Path(verification["checkpoint_weights_file"]).resolve()
    if (
        checkpoint not in weights.parents
        or file_hash(weights) != verification["checkpoint_weights_sha256"]
    ):
        raise ValueError("Validated checkpoint weights have changed")
    resume = checkpoint / "actor/dcp_checkpoint"
    if not (resume / ".metadata").is_file() or not list(resume.glob("*.distcp")):
        raise ValueError(
            "Missing sharded model/optimizer checkpoint required for continuation"
        )
    expected = load_manifest(run_dir / "eval_seeds.json")["open_laptop"]
    records = evaluation["episodes"]
    if (
        not evaluation["complete"]
        or evaluation["total"] != 8
        or len(records) != 8
        or Path(evaluation["checkpoint"]).resolve() != checkpoint
        or sorted(record["seed"] for record in records)
        != sorted(expected["success_seeds"])
        or any(
            record["instruction"] != expected["instructions"][str(record["seed"])]
            for record in records
        )
    ):
        raise ValueError(
            "Checkpoint reload evaluation is incomplete or uses different scenes"
        )
    for path in (run_dir / "train/metrics").glob("*.json"):
        metrics = json.loads(path.read_text())["metrics"]
        if not all(
            math.isfinite(float(value))
            for name, value in metrics.items()
            if any(term in name for term in ("loss", "grad_norm", "approx_kl"))
        ):
            raise ValueError(f"Non-finite training metrics: {path}")
    events = EventAccumulator(
        str(run_dir / "train/tensorboard"), size_guidance={"scalars": 0}
    )
    events.Reload()
    tags = (
        "env/return",
        "env/reward",
        "env/success_once",
        "rollout/rewards",
        "rollout/returns_mean",
        "train/actor/policy_loss",
        "train/critic/value_loss",
    )
    for tag in tags:
        values = events.Scalars(tag)
        if len(values) < 3 or not all(math.isfinite(value.value) for value in values):
            raise ValueError(f"Missing or non-finite TensorBoard metric: {tag}")
        if tag in ("env/return", "env/reward", "env/success_once", "rollout/rewards"):
            if any(not -1e-6 <= value.value <= 1.000001 for value in values):
                raise ValueError(f"Sparse reward metric outside [0, 1]: {tag}")
    result = {
        "passed": True,
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": verification["checkpoint_weights_sha256"],
        "tensorboard_tags": list(tags),
        "reload_evaluation_success": evaluation["success"],
        "reload_evaluation_total": 8,
        "success_rate_improvement_required": False,
    }
    write_json(run_dir / "validation_gate.json", result)
    return result
