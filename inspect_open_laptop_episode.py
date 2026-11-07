"""Record one SFT episode through the unchanged RLinf training environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def boundary_checks(output: Path) -> list[dict]:
    """Exercise the real wrapper at its simulator edge without a GPU/model."""
    import torch
    from omegaconf import OmegaConf

    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

    class CPUEnv(RoboTwinEnv):
        @property
        def device(self):
            return torch.device("cpu")

    class NativeResult:
        def __init__(self):
            self.success = False
            self.timeout = False
            self.count = 50

        def step(self, actions):
            return (
                [{}],
                [np.array([float(self.success)])],
                [np.array([self.success])],
                [np.array([self.timeout])],
                [{"success": self.success, "executed_action_count": self.count}],
            )

    env = CPUEnv.__new__(CPUEnv)
    env.cfg = OmegaConf.create(
        {"reward_coef": 1, "max_episode_steps": 700, "is_eval": False}
    )
    env.num_envs = 1
    env.record_metrics = True
    env.use_rel_reward = True
    env.use_custom_reward = True
    env.skip_intermediate_renders = True
    env.ignore_terminations = False
    env.auto_reset = False
    env.prev_step_reward = torch.zeros(1)
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    env._init_metrics()
    env.venv = NativeResult()
    env._extract_obs_image = lambda obs: {}
    results = []
    cases = [
        ("ordinary_failure", 0, False, False, 50, True),
        ("success_mid_chunk", 0, True, False, 17, True),
        ("success_repeated", 50, True, False, 0, False),
        ("timeout_failure", 650, False, True, 50, True),
        ("success_on_action_700", 650, True, False, 50, True),
        ("success_then_padded_to_700", 650, True, False, 0, False),
        ("reset_failure", 0, False, False, 50, True),
    ]
    for name, elapsed, success, timeout, count, reset in cases:
        if reset:
            env._reset_metrics()
        env._elapsed_steps.fill_(elapsed)
        env.venv.success, env.venv.timeout, env.venv.count = success, timeout, count
        _, reward, term, trunc, infos = env.chunk_step(np.zeros((1, 50, 14)))
        results.append(
            {
                "case": name,
                "reward_sum": float(reward.sum()),
                "reward_slots": np.flatnonzero(array(reward)[0]).tolist(),
                "terminated": bool(term.any()),
                "truncated": bool(trunc.any()),
                "valid_actions": int(infos[-1]["valid_action_mask"].sum()),
            }
        )
    save_json(output / "wrapper_boundary_checks.json", results)
    return results


class Movie:
    def __init__(self, path: Path, frame, fps: int):
        import imageio_ffmpeg

        h, w = frame.shape[:2]
        self.stream = imageio_ffmpeg.write_frames(
            str(path),
            (w, h),
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=2,
            ffmpeg_log_level="error",
        )
        self.stream.send(None)

    def add(self, frame):
        self.stream.send(np.ascontiguousarray(frame, dtype=np.uint8))

    def close(self):
        self.stream.close()


def task_state(task) -> tuple[np.ndarray, dict]:
    state = np.array(
        task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState(),
        dtype=np.float32,
    )
    limits = np.asarray(task.laptop.get_qlimits())[0]
    hinge = float(task.laptop.get_qpos()[0])
    contact = task.laptop.get_contact_point(1)
    tcp = (
        task.robot.get_left_tcp_pose()
        if task.arm_tag == "left"
        else task.robot.get_right_tcp_pose()
    )
    distance = float(np.linalg.norm(np.asarray(tcp[:3]) - np.asarray(contact[:3])))
    return state, {
        "native_step": int(task.take_action_cnt),
        "native_success": bool(task.eval_success),
        "hinge_qpos": hinge,
        "hinge_progress": float((hinge - limits[0]) / (limits[1] - limits[0])),
        "tcp_contact_distance": distance,
        "success_geometry": bool(
            hinge >= limits[0] + 0.4 * (limits[1] - limits[0]) and distance < 0.1
        ),
    }


def physical_state(task) -> np.ndarray:
    robot = task.robot
    values = (
        robot.get_left_arm_real_jointState() + robot.get_right_arm_real_jointState()
    )
    for index, side in [(6, "left"), (13, "right")]:
        joint, factor, offset = getattr(robot, f"{side}_gripper")[0]
        entity = getattr(robot, f"{side}_entity")
        qpos = entity.get_qpos()[entity.get_active_joints().index(joint)]
        low, high = getattr(robot, f"{side}_gripper_scale")
        values[index] = ((qpos - offset) / factor - low) / (high - low)
    return np.asarray(values, dtype=np.float32)


def plot_trace(output: Path, values: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"L joint {i + 1}" for i in range(6)] + ["L gripper"]
    labels += [f"R joint {i + 1}" for i in range(6)] + ["R gripper"]
    x = np.arange(1, len(values["state_after"]) + 1)
    fig, axes = plt.subplots(7, 2, figsize=(15, 16), sharex=True)
    for i, ax in enumerate(axes.T.flat):
        ax.plot(x, values["state_after"][:, i], label="RL state (drive target)", lw=1)
        ax.plot(x, values["state_actual"][:, i], label="physical qpos", lw=0.8)
        executed = values["executed"].astype(bool)
        ax.plot(
            x,
            np.where(executed, values["env_actions"][:, i], np.nan),
            label="executed target",
            lw=0.8,
            alpha=0.8,
        )
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("RL rollout action slot (padding retains last state)")
    axes[-1, 1].set_xlabel("RL rollout action slot")
    fig.suptitle("7272 SFT: raw 14-D proprioception and absolute joint targets")
    fig.tight_layout()
    fig.savefig(output / "proprio_and_actions.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(x, values["hinge_progress"], label="hinge progress")
    axes[0].axhline(0.4, ls="--", color="r", label="success threshold")
    axes[0].legend()
    axes[0].set_ylabel("hinge fraction")
    axes[1].plot(x, values["tcp_contact_distance"])
    axes[1].axhline(0.1, ls="--", color="r")
    axes[1].set_ylabel("TCP-contact (m)")
    axes[2].step(x, values["reward"], label="RLinf reward")
    axes[2].step(x, np.cumsum(values["reward"]), label="return")
    axes[2].legend()
    for key in ["executed", "terminated", "truncated", "native_success"]:
        axes[3].step(x, values[key], label=key, alpha=0.7)
    axes[3].legend(ncol=4)
    axes[3].set_xlabel("RL rollout action slot")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "reward_and_task_state.png", dpi=150)
    plt.close(fig)

    model_state = values["model_state"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for i in range(14):
        axes[0].plot(np.arange(len(model_state)), model_state[:, i], label=labels[i])
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_ylabel("normalized state")
    axes[1].imshow(
        model_state.T, aspect="auto", interpolation="nearest", cmap="coolwarm"
    )
    axes[1].set_ylabel("model state dimension (32)")
    axes[1].set_xlabel("T50 policy input index")
    fig.suptitle(
        "Actual transformed model state; Pi0.5 also tokenizes state into prompt"
    )
    fig.tight_layout()
    fig.savefig(output / "model_input_state.png", dpi=150)
    plt.close(fig)


def record(args, output: Path, boundaries: list[dict]) -> None:
    import cv2
    import torch
    from omegaconf import OmegaConf

    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.load(args.config)
    if str(cfg.actor.model.model_path) != "/data/robotwin_ckpt/rlinf/7272":
        raise ValueError("Diagnostic must load original SFT 7272")
    manifest_path = Path(cfg.env.train.seeds_path)
    manifest = json.loads(manifest_path.read_text())
    task_manifest = manifest["open_laptop"]
    seed = sorted(task_manifest["success_seeds"])[0] if args.seed is None else args.seed
    if seed not in task_manifest["success_seeds"]:
        raise ValueError("Scene must be in the actual training set")
    single = {
        "open_laptop": {
            "success_seeds": [seed],
            "instructions": {str(seed): task_manifest["instructions"][str(seed)]},
        }
    }
    save_json(output / "scene.json", single)
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
    env_cfg.total_num_envs = 1
    env_cfg.seeds_path = str(output / "scene.json")
    env_cfg.task_config.episode_manifest_path = env_cfg.seeds_path
    env_cfg.task_config.episode_log_path = str(output / "native_episode.jsonl")
    OmegaConf.save(env_cfg, output / "diagnostic_env_config.yaml")
    save_json(
        output / "source_fingerprints.json",
        {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path(args.config),
                manifest_path,
                Path(cfg.actor.model.openpi_data.norm_stats_path),
                Path("/workspace/RLinf/rlinf/envs/sim/robotwin/robotwin_env.py"),
                Path("/workspace/RoboTwin/robotwin/envs/vector_env.py"),
                Path("/workspace/RoboTwin/envs/open_laptop.py"),
            ]
        },
    )
    print(f"Loading SFT 7272; mode={args.mode}; seed={seed}", flush=True)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    model = get_model(cfg.actor.model, torch_dtype=torch.float32).to("cuda").eval()
    env = None
    movies = {}
    model_inputs = []
    chunks = []
    slot_rows = []
    states = []
    actual_states = []
    actions_all = []
    model_actions = []
    prev_values = []
    prev_logprobs = []
    captured = {}
    original_prefix = model.build_prefix_cache

    def capture_prefix(observation, *pos, **kw):
        captured["input"] = {
            "state": array(observation.state)[0],
            "tokenized_prompt": array(observation.tokenized_prompt)[0],
            "tokenized_prompt_mask": array(observation.tokenized_prompt_mask)[0],
            "images": {k: array(v)[0] for k, v in observation.images.items()},
            "image_masks": {
                k: array(v).tolist() for k, v in observation.image_masks.items()
            },
        }
        return original_prefix(observation, *pos, **kw)

    model.build_prefix_cache = capture_prefix
    camera_keys = ["head_camera", "left_camera", "right_camera"]
    frame_buffer = []
    native_rows = []
    current_chunk = 0
    try:
        env = RoboTwinEnv(env_cfg, 1, 0, 1, None)
        obs, _ = env.reset(env_seeds=[seed])
        task = env.venv.envs[0].task
        raw = task.get_obs()
        last_frames = {k: raw["observation"][k]["rgb"].copy() for k in camera_keys}
        initial_state, initial_task_state = task_state(task)
        save_json(
            output / "initial_state.json",
            {
                **initial_task_state,
                "state": initial_state.tolist(),
                "physical_state": physical_state(task).tolist(),
            },
        )
        original_action = task.take_action

        def capture_action(action, *pos, **kw):
            result = original_action(action, *pos, **kw)
            native_obs = task.get_obs()
            frames = {
                k: native_obs["observation"][k]["rgb"].copy() for k in camera_keys
            }
            state, info = task_state(task)
            frame_buffer.append(frames)
            native_rows.append(
                {
                    **info,
                    "chunk": current_chunk,
                    "state": state,
                    "physical_state": physical_state(task),
                    "action": np.asarray(action).copy(),
                }
            )
            return result

        task.take_action = capture_action
        # Reset/scene construction can seed process-global RNGs. Set the policy
        # RNG after reset, like independent env and rollout worker processes.
        random.seed(args.rng_seed)
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
        running_return = 0.0
        for current_chunk in range(700 // 50):
            frame_buffer.clear()
            native_rows.clear()
            raw_policy_state = array(obs["states"])[0]
            with torch.inference_mode():
                policy_actions, result = model.predict_action_batch(obs, mode=args.mode)
            model_inputs.append(captured["input"])
            model_actions.append(array(result["model_actions"])[0])
            if result["prev_values"] is not None:
                prev_values.append(array(result["prev_values"])[0])
                prev_logprobs.append(array(result["prev_logprobs"])[0])
            commands = prepare_actions(
                policy_actions,
                env_type="robotwin",
                model_type="openpi",
                num_action_chunks=50,
                action_dim=14,
                env_cfg=env_cfg,
            )
            before = int(task.take_action_cnt)
            observations, rewards, term, trunc, infos = env.chunk_step(commands)
            obs = observations[-1]
            reward, term, trunc = array(rewards)[0], array(term)[0], array(trunc)[0]
            valid = array(infos[-1]["valid_action_mask"])[0]
            count = int(task.take_action_cnt) - before
            if count != len(frame_buffer) or count != int(valid.sum()):
                raise AssertionError(
                    "Native executed steps differ from actual RLinf valid-action mask"
                )
            final_state, final_task_state = task_state(task)
            final_physical_state = physical_state(task)
            input_images = {}
            for key, value in captured["input"]["images"].items():
                if value.shape[0] == 3:
                    value = value.transpose(1, 2, 0)
                input_images[key] = np.clip((value + 1) * 127.5, 0, 255).astype(
                    np.uint8
                )
            chunks.append(
                {
                    "chunk": current_chunk,
                    "native_actions": count,
                    "reward_sum": float(reward.sum()),
                    "terminated": bool(term.any()),
                    "truncated": bool(trunc.any()),
                    "native_success": bool(task.eval_success),
                    "raw_policy_state": raw_policy_state.tolist(),
                    "input_image_masks": captured["input"]["image_masks"],
                    "input_shapes": {
                        k: list(v.shape) for k, v in captured["input"]["images"].items()
                    },
                }
            )
            for slot in range(50):
                executed = slot < count
                if executed:
                    last_frames = frame_buffer[slot]
                    row = native_rows[slot]
                    state = row["state"]
                else:
                    row = final_task_state
                    state = final_state
                running_return += float(reward[slot])
                global_slot = current_chunk * 50 + slot + 1
                item = {
                    "rollout_slot": global_slot,
                    "chunk": current_chunk,
                    "chunk_slot": slot,
                    "executed": bool(executed),
                    "reward": float(reward[slot]),
                    "return": running_return,
                    "terminated": bool(term[slot]),
                    "truncated": bool(trunc[slot]),
                    **{k: row[k] for k in final_task_state},
                }
                slot_rows.append(item)
                states.append(state.copy())
                actual_states.append(
                    row["physical_state"].copy()
                    if executed
                    else final_physical_state.copy()
                )
                actions_all.append(commands[0, slot].copy())
                for key, frame in last_frames.items():
                    if key not in movies:
                        movies[key] = Movie(output / f"rgb_{key}.mp4", frame, args.fps)
                    movies[key].add(frame)
                for key, frame in input_images.items():
                    name = "input_" + key
                    if name not in movies:
                        movies[name] = Movie(output / f"{name}.mp4", frame, args.fps)
                    movies[name].add(frame)
                canvas = np.zeros((360, 960, 3), dtype=np.uint8)
                for index, key in enumerate(camera_keys):
                    frame = cv2.resize(last_frames[key], (320, 240))
                    canvas[:240, 320 * index : 320 * (index + 1)] = frame
                    cv2.putText(
                        canvas,
                        key,
                        (320 * index + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 0),
                        1,
                    )
                text = [
                    f"SFT 7272 | mode={args.mode} | scene={seed} | chunk={current_chunk + 1}/14",
                    f"RL slot={global_slot}/700 | native step={row['native_step']} | {'EXECUTED' if executed else 'PADDED (no action)'}",
                    f"reward={reward[slot]:+.1f}  return={running_return:.1f}  success={int(row['native_success'])}  term={int(term[slot])} trunc={int(trunc[slot])}",
                    f"hinge={row['hinge_progress']:.3f} (>=0.4), TCP distance={row['tcp_contact_distance']:.3f}m (<0.1)",
                    f"{args.fps} fps visualization; policy observes every 50 slots; RGB above is extra diagnostic rendering",
                ]
                for line, label in enumerate(text):
                    cv2.putText(
                        canvas,
                        label,
                        (8, 260 + line * 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.49,
                        (245, 245, 245),
                        1,
                    )
                if "composite" not in movies:
                    movies["composite"] = Movie(
                        output / "episode_reward.mp4", canvas, args.fps
                    )
                movies["composite"].add(canvas)
            save_json(output / "chunks.json", chunks)
            print(
                f"chunk {current_chunk + 1}/14: executed={count}, success={task.eval_success}, reward={reward.sum()}, return={running_return}",
                flush=True,
            )
        values = {k: np.array([row[k] for row in slot_rows]) for k in slot_rows[0]}
        values.update(
            state_after=np.stack(states),
            state_actual=np.stack(actual_states),
            env_actions=np.stack(actions_all),
            model_actions=np.stack(model_actions),
            model_state=np.stack([v["state"] for v in model_inputs]),
            policy_raw_state=np.array([v["raw_policy_state"] for v in chunks]),
            tokenized_prompt=np.stack([v["tokenized_prompt"] for v in model_inputs]),
            tokenized_prompt_mask=np.stack(
                [v["tokenized_prompt_mask"] for v in model_inputs]
            ),
        )
        if prev_values:
            values.update(
                prev_values=np.stack(prev_values), prev_logprobs=np.stack(prev_logprobs)
            )
        np.savez_compressed(output / "trajectory.npz", **values)
        np.savez_compressed(
            output / "model_rgb.npz",
            **{
                k: np.stack([v["images"][k] for v in model_inputs])
                for k in model_inputs[0]["images"]
            },
        )
        with (output / "steps.jsonl").open("w") as stream:
            for row in slot_rows:
                stream.write(json.dumps(row) + "\n")
        with (output / "state_actions.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                list(slot_rows[0])
                + [f"state_{i}" for i in range(14)]
                + [f"action_{i}" for i in range(14)]
                + [f"physical_state_{i}" for i in range(14)]
            )
            for row, state, action, real_state in zip(
                slot_rows, states, actions_all, actual_states
            ):
                writer.writerow(
                    list(row.values())
                    + state.tolist()
                    + action.tolist()
                    + real_state.tolist()
                )
        plot_trace(output, values)
        finite = all(
            np.isfinite(v).all()
            for v in values.values()
            if np.issubdtype(v.dtype, np.number)
        )
        save_json(
            output / "report.json",
            {
                "complete": True,
                "model": str(cfg.actor.model.model_path),
                "mode": args.mode,
                "seed": seed,
                "rng_seed": args.rng_seed,
                "instruction": task.get_instruction(),
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "rlinf_commit": "5d284646d8d2c570d5fd4af7e42e8cf234aeb16e",
                "robotwin_commit": "cdac830bab83f50e486ed172b64215114d120b07",
                "source_edits": "New standalone diagnostic only; core sources unchanged",
                "success": bool(task.eval_success),
                "native_actions": int(task.take_action_cnt),
                "return": running_return,
                "reward_pulses": int(np.count_nonzero(values["reward"])),
                "all_numeric_arrays_finite": bool(finite),
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "boundary_checks": boundaries,
                "model_inputs": {
                    k: list(v.shape)
                    for k, v in values.items()
                    if k
                    in [
                        "model_state",
                        "tokenized_prompt",
                        "model_actions",
                        "policy_raw_state",
                    ]
                },
                "image_shapes": chunks[0]["input_shapes"],
                "video_fps": args.fps,
                "video_timebase": "One RL action slot per frame; not real-time physics",
                "diagnostic_changes": [
                    "single environment instead of two per rank",
                    "extra render/read-only state capture after each native action",
                    "fresh policy RNG; no optimizer or weight updates",
                ],
                "reward_semantics": "Actual RLinf reward: pulse at T50 last slot; native success can occur earlier; padded slots are flagged separately",
                "state_semantics": "RL state14 uses arm drive targets and commanded gripper values, not actual qpos; separate physical_state14 uses actual arm/finger qpos (finger inverse coupling+scale, unclipped); model normalizes/pads state32 and tokenizes state into prompt",
            },
        )
    finally:
        for movie in movies.values():
            movie.close()
        if env is not None:
            env.venv.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rng-seed", type=int, default=20261103)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--boundaries-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    boundaries = boundary_checks(args.output)
    if not args.boundaries_only:
        record(args, args.output, boundaries)
    print(f"Diagnostic complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
