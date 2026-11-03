"""RLinf observation and action adapter over RoboTwin's native task API.

Environments execute serially because task setup seeds process-global RNGs.
The initial evaluation configuration deliberately uses one environment.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def prepare_task_config(overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve cameras and robot files using this checkout's configuration."""
    config_dir = ROOT / "env_cfg" / "task_config"
    args = yaml.safe_load((config_dir / "demo_clean.yml").read_text())
    for key, value in deepcopy(overrides).items():
        if isinstance(value, dict) and isinstance(args.get(key), dict):
            args[key].update(value)
        else:
            args[key] = value
    assets_root = Path(os.environ.get("ASSETS_PATH", str(ROOT)))
    assets_directory = Path(
        os.environ.get("ROBOTWIN_ASSETS_PATH", str(assets_root / "assets"))
    )
    robots = yaml.safe_load((config_dir / "_embodiment_config.yml").read_text())
    cameras = yaml.safe_load((config_dir / "_camera_config.yml").read_text())
    embodiment = args["embodiment"]
    if len(embodiment) not in (1, 3):
        raise ValueError("embodiment must have one entry or [left, right, distance]")
    for side, name in zip(
        ("left", "right"),
        (embodiment[0], embodiment[-2] if len(embodiment) == 3 else embodiment[0]),
    ):
        robot_dir = assets_directory / Path(robots[name]["file_path"]).relative_to(
            "assets"
        )
        args[f"{side}_robot_file"] = str(robot_dir)
        args[f"{side}_embodiment_config"] = yaml.safe_load(
            (robot_dir / "config.yml").read_text()
        )
    args["dual_arm_embodied"] = len(embodiment) == 1
    if len(embodiment) == 3:
        args["embodiment_dis"] = embodiment[2]
    args["embodiment_name"] = (
        embodiment[0] if len(embodiment) == 1 else f"{embodiment[0]}_{embodiment[1]}"
    )
    camera = cameras[args["camera"]["head_camera_type"]]
    args.update(
        head_camera_h=camera["h"],
        head_camera_w=camera["w"],
        eval_mode=True,
        eval_video_log=False,
        render_freq=0,
        collect_data=False,
        save_data=False,
        action_dim=14,
    )
    args.setdefault("action_execution_mode", "take_action")
    return args


def new_task(task_name: str):
    """Instantiate a native task without creating its scene yet."""
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def task_instruction(task, seed: int, instruction_type: str) -> str:
    """Generate one instruction deterministically from initialized task metadata."""
    from description.utils.generate_episode_instructions import (
        generate_episode_descriptions,
    )

    descriptions = generate_episode_descriptions(
        task.task_name, [task.get_info()], 1, seed=seed
    )
    return descriptions[0][instruction_type][0]


def update_obs(task) -> dict[str, Any]:
    """Convert the native observation to the keys consumed by RLinf."""
    raw = task.get_obs()
    cameras = raw["observation"]
    return {
        "full_image": cameras["head_camera"]["rgb"],
        "left_wrist_image": cameras.get("left_camera", {}).get("rgb"),
        "right_wrist_image": cameras.get("right_camera", {}).get("rgb"),
        "state": np.asarray(raw["joint_action"]["vector"], dtype=np.float32),
        "instruction": task.get_instruction(),
    }


class SubEnv:
    """A fixed-seed task; setup errors never substitute a different scene."""

    def __init__(
        self, env_id: int, args: dict[str, Any], seed: int, instruction_type: str
    ):
        self.env_id = env_id
        self.args = deepcopy(args)
        self.env_seed = int(seed)
        self.instruction_type = instruction_type
        self.task = None
        self.result_written = False
        self.mode = args["action_execution_mode"]
        if self.mode not in ("take_action", "chunk"):
            raise ValueError(f"Unsupported action_execution_mode: {self.mode}")
        manifest = args.get("episode_manifest_path")
        self.instructions = {}
        if manifest:
            self.instructions = json.loads(Path(manifest).read_text())[
                args["task_name"]
            ]["instructions"]

    def reset(self, seed: int | None = None) -> None:
        self.close()
        if seed is not None:
            self.env_seed = int(seed)
        self.task = new_task(self.args["task_name"])
        if self.mode == "chunk" and not hasattr(self.task, "gen_sparse_reward_data"):
            raise ValueError(
                "This task has no chunk controller; use take_action on fork/main"
            )
        try:
            self.task.setup_demo(
                now_ep_num=self.env_seed, seed=self.env_seed, is_test=True, **self.args
            )
            self.task.step_lim = int(self.args["step_lim"])
            if self.args.get("episode_manifest_path"):
                instruction = self.instructions[str(self.env_seed)]
            else:
                instruction = task_instruction(
                    self.task, self.env_seed, self.instruction_type
                )
            self.task.set_instruction(instruction)
            self.result_written = False
        except Exception:
            self.close()
            raise

    def step(self, actions: np.ndarray, full_obs: bool = False) -> dict[str, Any]:
        if self.task is None:
            self.reset()
        actions = np.asarray(actions)
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                f"Expected finite actions [horizon, 14], got {actions.shape}"
            )
        if not len(actions):
            raise ValueError("Action horizon must be positive")
        observations = []
        if self.mode == "chunk":
            if full_obs:
                raise ValueError(
                    "Full observations require take_action mode in this adapter"
                )
            before = self.task.take_action_cnt
            reward, terminated, truncated, info = self.task.gen_sparse_reward_data(
                actions
            )
            executed = self.task.take_action_cnt - before
        else:
            executed = 0
            for action in actions:
                if (
                    self.task.eval_success
                    or self.task.take_action_cnt >= self.task.step_lim
                ):
                    break
                self.task.take_action(action)
                executed += 1
                if full_obs:
                    observations.append(update_obs(self.task))
            success = bool(self.task.eval_success)
            reward = np.array([float(success)], dtype=np.float32)
            terminated = np.array([success], dtype=np.int32)
            truncated = np.array(
                [self.task.take_action_cnt >= self.task.step_lim and not success],
                dtype=np.int32,
            )
            info = {"success": success}
        info.update(executed_action_count=executed, seed=self.env_seed)
        final_obs = observations[-1] if observations else update_obs(self.task)
        if full_obs:
            observations.extend([final_obs] * (len(actions) - len(observations)))
        if (
            bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any())
        ) and not self.result_written:
            output = self.args.get("episode_log_path")
            if output:
                record = {
                    "seed": self.env_seed,
                    "env_id": self.env_id,
                    "success": bool(info["success"]),
                    "steps": self.task.take_action_cnt,
                    "instruction": self.task.get_instruction(),
                }
                with Path(output).open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            self.result_written = True
        return {
            "obs": observations if full_obs else final_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def close(self, clear_cache: bool = True) -> None:
        if self.task is not None:
            try:
                self.task.close_env(clear_cache=clear_cache)
            finally:
                self.task = None
                gc.collect()


class VectorEnv:
    """Serial native tasks with RLinf's vector observation contract."""

    def __init__(
        self,
        task_config: dict[str, Any],
        n_envs: int,
        env_seeds=None,
        instruction_type="seen",
    ):
        self.args = prepare_task_config(task_config)
        self.n_envs = n_envs
        seeds = list(range(n_envs)) if env_seeds is None else list(env_seeds)
        if n_envs < 1 or len(seeds) != n_envs:
            raise ValueError("env_seeds must contain one seed per environment")
        self.envs = [
            SubEnv(i, self.args, seed, instruction_type) for i, seed in enumerate(seeds)
        ]

    def reset(self, env_idx=None, env_seeds=None) -> None:
        indices = (
            list(range(self.n_envs))
            if env_idx is None
            else np.asarray(env_idx).reshape(-1).tolist()
        )
        if env_seeds is not None and len(env_seeds) != len(indices):
            raise ValueError("env_seeds must match the environments being reset")
        for offset, index in enumerate(indices):
            self.envs[index].reset(None if env_seeds is None else env_seeds[offset])

    def get_obs(self) -> list[dict[str, Any]]:
        return [update_obs(env.task) for env in self.envs]

    def step(self, actions):
        return self._step(actions, full_obs=False)

    def step_with_full_obs(self, actions):
        return self._step(actions, full_obs=True)

    def _step(self, actions, full_obs: bool):
        if len(actions) != self.n_envs:
            raise ValueError("Action batch must match n_envs")
        results = [
            env.step(action, full_obs) for env, action in zip(self.envs, actions)
        ]
        obs = [r["obs"] for r in results]
        if full_obs:
            obs = [list(step) for step in zip(*obs)]
        return (
            obs,
            [r["reward"] for r in results],
            [r["terminated"] for r in results],
            [r["truncated"] for r in results],
            [r["info"] for r in results],
        )

    def close(self, clear_cache: bool = True) -> None:
        for env in self.envs:
            env.close(clear_cache)
