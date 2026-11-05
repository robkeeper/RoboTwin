"""Exercise the adapter and PPO plumbing using a tiny policy, without loading Pi0.5."""

from __future__ import annotations

import argparse
import io
import json
import os
import traceback
import unittest
from contextlib import nullcontext
from pathlib import Path

from rlinf_rft.common import ROOT, gpu_ids, load_manifest, write_json
from rlinf_rft.workflow import gpu_report, source_hashes


def native_environment(manifest_path: Path) -> tuple[list[float], dict]:
    import numpy as np
    from omegaconf import OmegaConf
    from robotwin.envs.vector_env import SubEnv, prepare_task_config, update_obs

    from rlinf_rft.cli import compose_template
    from rlinf_rft.common import EVAL_CONFIG

    manifest = load_manifest(manifest_path, require_complete=True)["open_laptop"]
    seed = sorted(manifest["success_seeds"])[0]
    cfg = compose_template(EVAL_CONFIG)
    task = prepare_task_config(
        OmegaConf.to_container(cfg.env.eval.task_config, resolve=True)
    )
    task["episode_manifest_path"] = str(manifest_path)
    task["episode_log_path"] = None
    env = SubEnv(0, task, seed, "seen")
    try:
        env.reset()
        observation = update_obs(env.task)
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite 14-dimensional native observation")
        for name in ("full_image", "left_wrist_image", "right_wrist_image"):
            image = np.asarray(observation[name])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Missing native RGB camera: {name}")
        if observation["instruction"] != manifest["instructions"][str(seed)]:
            raise ValueError("Native instruction differs from the fixed manifest")
        result = env.step(np.repeat(state[None], 50, axis=0))
        executed = int(result["info"]["executed_action_count"])
        if not 1 <= executed <= 50:
            raise ValueError(
                "Native T50 action execution did not advance the simulator"
            )
        env.reset(seed)
        reset_observation = update_obs(env.task)
        if reset_observation["instruction"] != observation["instruction"]:
            raise ValueError("Reset changed the fixed instruction")
        return state.tolist(), {
            "seed": seed,
            "executed_actions": executed,
            "cameras": 3,
            "instruction": observation["instruction"],
            "reset_passed": True,
        }
    finally:
        env.close()


def tiny_update(output: Path, device: str, state: list[float]) -> dict:
    import torch
    from omegaconf import OmegaConf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
    from rlinf.utils.metric_logger import MetricLogger

    torch.manual_seed(20261103)
    model = torch.nn.Sequential(
        torch.nn.Linear(14, 64), torch.nn.Tanh(), torch.nn.Linear(64, 15)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    inputs = torch.as_tensor(state, device=device).float().repeat(4, 1)
    inputs = inputs + 0.01 * torch.randn_like(inputs)
    initial = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        old = model(inputs)
        actions = old[:, :14] + 0.1 * torch.randn_like(old[:, :14])
        old_logprobs = (
            torch.distributions.Normal(old[:, :14], 0.1).log_prob(actions).sum(-1)
        )
    current = model(inputs)
    logprobs = (
        torch.distributions.Normal(current[:, :14], 0.1).log_prob(actions).sum(-1)
    )
    mask = torch.tensor([True, True, False, True], device=device)
    returns = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
    advantages = returns - old[:, 14]
    advantages = (advantages - advantages[mask].mean()) / advantages[mask].std(
        unbiased=False
    ).clamp_min(1e-6)
    loss, metrics = compute_ppo_actor_critic_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        values=current[:, 14],
        prev_values=old[:, 14],
        returns=returns,
        advantages=advantages,
        loss_mask=mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        value_clip=0.2,
        huber_delta=10.0,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Tiny PPO loss is non-finite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError("Tiny PPO gradient is non-finite")
    optimizer.step()
    if not any(
        not torch.equal(value.detach().cpu(), initial[name])
        for name, value in model.state_dict().items()
    ):
        raise AssertionError("Tiny PPO did not update its parameters")
    checkpoint = output / "tiny_checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
    )
    restored = torch.nn.Sequential(
        torch.nn.Linear(14, 64), torch.nn.Tanh(), torch.nn.Linear(64, 15)
    ).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=5e-6)
    restored_optimizer.load_state_dict(saved["optimizer"])
    if not torch.equal(restored(inputs), model(inputs)) or not restored_optimizer.state:
        raise AssertionError("Tiny checkpoint model/optimizer roundtrip failed")
    logger = MetricLogger(
        OmegaConf.create(
            {
                "runner": {
                    "logger": {
                        "log_path": str(output),
                        "experiment_name": "low_memory_smoke",
                        "logger_backends": ["tensorboard"],
                    }
                }
            }
        )
    )
    scalars = {
        "env/return": 1.0,
        "env/reward": 1.0 / 700,
        "rollout/rewards": float(returns[mask].mean()),
        "rollout/returns_mean": float(returns[mask].mean()),
        "train/actor/policy_loss": float(metrics["actor/policy_loss"]),
        "train/critic/value_loss": float(metrics["critic/value_loss"]),
    }
    logger.log(scalars, 0)
    logger.finish()
    events = EventAccumulator(str(output / "tensorboard"))
    events.Reload()
    if not set(scalars) <= set(events.Tags()["scalars"]):
        raise AssertionError("TensorBoard reward/return/loss events were not saved")
    return {
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "parameters_updated": True,
        "checkpoint_reloaded": True,
        "tensorboard_tags": list(scalars),
        "metrics_are_synthetic": True,
    }


def main():
    import torch

    from rlinf_rft.monitor import record_resources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--budget-mib", type=int, default=4096)
    parser.add_argument(
        "--train-seeds",
        type=Path,
        default=Path("/workspace/open_laptop_ppo/scenes/train56.json"),
    )
    args = parser.parse_args()
    if args.budget_mib <= 0:
        parser.error("Memory budget must be positive")
    output = args.output.resolve()
    for tree in (ROOT, Path(os.environ.get("REPO_PATH", "/workspace/RLinf")).resolve()):
        if output == tree or tree in output.parents:
            parser.error("Smoke artifacts must be outside source trees")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "low_memory_smoke" if args.device == "cuda" else "cpu_smoke",
        "passed": False,
        "full_pi05_model_tested": False,
        "source_sha256": source_hashes(),
    }
    try:
        selected = (
            gpu_ids(os.environ.get("GPU_LIST", "")) if args.device == "cuda" else []
        )
        if args.device == "cuda":
            if len(selected) != 1 or not torch.cuda.is_available():
                raise ValueError("CUDA smoke requires exactly one visible GPU")
            row = next(row for row in gpu_report() if row["index"] == selected[0])
            if row["free_mib"] < args.budget_mib + 4096:
                raise ValueError("GPU lacks the smoke budget plus 4 GiB headroom")
            torch.cuda.set_per_process_memory_fraction(
                min(args.budget_mib / 2, 2048)
                * 1024**2
                / torch.cuda.get_device_properties(0).total_memory
            )
            write_json(output / "workers/smoke_0.json", {"pid": os.getpid()})
        log = io.StringIO()
        tests = unittest.defaultTestLoader.discover(
            str(ROOT / "tests"), pattern="test_rlinf_rft.py"
        )
        result = unittest.TextTestRunner(stream=log).run(tests)
        (output / "boundary_tests.log").write_text(log.getvalue())
        if not result.wasSuccessful() or result.skipped:
            raise AssertionError("Boundary/configuration tests failed or were skipped")
        report["boundary_tests_passed"] = result.testsRun
        from rlinf_rft.preflight import validate_worker_bindings

        report["worker_preflight"] = validate_worker_bindings()
        with record_resources(output, selected) if selected else nullcontext():
            state, environment = (
                native_environment(args.train_seeds)
                if selected
                else ([0.0] * 14, {"skipped": True})
            )
            report["native_environment"] = environment
            report["tiny_policy"] = tiny_update(output, args.device, state)
        if selected:
            summary = json.loads((output / "resource_summary.json").read_text())
            peak = max(
                summary["sampled_worker_peak_mib"],
                torch.cuda.max_memory_reserved() / 1024**2,
            )
            report["peak_mib"] = peak
            report["budget_mib"] = args.budget_mib
            if not 0 < peak <= args.budget_mib:
                raise MemoryError(
                    f"Smoke memory measurement {peak:.1f} MiB is outside the budget"
                )
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        traceback.print_exc()
        raise
    finally:
        write_json(output / "report.json", report)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "source_sha256"},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
