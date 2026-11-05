"""Compare complete saved PPO tensors with the SFT and initial value head."""

from __future__ import annotations

import json
from pathlib import Path

from rlinf_rft.common import BASE_MODEL, file_hash, write_json


def verify(run_dir: Path) -> None:
    import torch
    from safetensors import safe_open

    from rlinf.models.embodiment.openpi.checkpoint import (
        _normalize_state_dict,
        resolve_full_weights,
    )
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict
    from rlinf_rft.workers import tensor_hash

    run_dir = run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    initial = json.loads((run_dir / "train/initial_parameters.json").read_text())
    process = json.loads((run_dir / "train/process.json").read_text())
    checkpoint = (
        run_dir / "train/robotwin_open_laptop_ppo_openpi_pi05/checkpoints/global_step_3"
    )
    weights = resolve_full_weights(checkpoint)
    if process["exit_code"] != 0 or weights is None:
        raise ValueError(
            "Require a completed training process and global_step_3 weights"
        )
    for step in (1, 2, 3):
        if resolve_full_weights(checkpoint.parent / f"global_step_{step}") is None:
            raise ValueError(f"Missing checkpoint at PPO iteration {step}")
    state = _normalize_state_dict(
        as_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        )
    )
    parameters = initial["parameters"]
    missing = sorted(set(parameters) - set(state))
    if missing:
        raise ValueError(f"Saved checkpoint is missing parameters: {missing[:8]}")
    changed_trainable = []
    changed_frozen = []
    nonfinite = []
    checkpoint_model = Path(run.get("model_path", BASE_MODEL))
    with safe_open(
        checkpoint_model / "model.safetensors", framework="pt", device="cpu"
    ) as base:
        for name, description in parameters.items():
            tensor = state[name]
            if tuple(tensor.shape) != tuple(description["shape"]):
                raise ValueError(f"Parameter shape changed: {name}")
            if name.startswith("value_head."):
                continue
            original = base.get_tensor(name).reshape(-1)
            flat = tensor.reshape(-1)
            equal = True
            finite = True
            for start in range(0, flat.numel(), 1024 * 1024):
                part = flat[start : start + 1024 * 1024]
                finite = finite and bool(torch.isfinite(part).all())
                equal = equal and torch.equal(
                    part, original[start : start + part.numel()]
                )
            if not finite:
                nonfinite.append(name)
            if not equal:
                (
                    changed_trainable if description["trainable"] else changed_frozen
                ).append(name)
    changed_heads = [
        name
        for name, digest in initial["initial_value_head_sha256"].items()
        if tensor_hash(state[name]) != digest
    ]
    for name in initial["initial_value_head_sha256"]:
        if not torch.isfinite(state[name]).all():
            nonfinite.append(name)
    metric_files = sorted((run_dir / "train/metrics").glob("*.json"))
    rewards = [
        json.loads(line)
        for line in (run_dir / "train/episodes.jsonl").read_text().splitlines()
        if line
    ]
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_weights_file": str(weights),
        "checkpoint_weights_sha256": file_hash(weights),
        "parameter_count": len(parameters),
        "changed_trainable_parameters": changed_trainable,
        "changed_value_head_parameters": changed_heads,
        "changed_frozen_parameters": changed_frozen,
        "nonfinite_parameters": nonfinite,
        "training_metric_files": len(metric_files),
        "completed_training_episodes": len(rewards),
        "successful_training_episodes": sum(record["success"] for record in rewards),
    }
    result["passed"] = bool(
        changed_trainable
        and changed_heads
        and not changed_frozen
        and not nonfinite
        and len(metric_files) == 3
        and any(record["success"] for record in rewards)
    )
    write_json(run_dir / "parameter_verification.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(
            "PPO parameter/reward verification failed; inspect the saved report"
        )
