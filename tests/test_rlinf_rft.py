"""PPO configuration and sparse-reward boundaries, without GPU execution."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlinf_rft.common import (
    disjoint_seeds,
    gpu_ids,
    load_manifest,
    parse_gpu_report,
    select_gpus,
    select_seeds,
    training_seed_count,
    validate_train_config,
)


def manifest(seeds):
    return {
        "complete": True,
        "open_laptop": {
            "success_seeds": seeds,
            "instructions": {str(seed): f"Open the laptop {seed}." for seed in seeds},
        },
    }


class ManifestTest(unittest.TestCase):
    def test_selection_freezes_sorted_seeds_and_matching_instructions(self):
        selected = select_seeds(manifest([9, 2, 7]), 2)["open_laptop"]
        self.assertEqual(selected["success_seeds"], [2, 7])
        self.assertEqual(set(selected["instructions"]), {"2", "7"})

    def test_overlap_with_either_evaluation_or_demonstrations_is_rejected(self):
        for evaluation, demos in (
            (manifest([8, 9]), {1, 2}),
            (manifest([1, 2]), {8, 9}),
        ):
            with self.assertRaisesRegex(ValueError, "overlap"):
                disjoint_seeds(manifest([8, 10]), evaluation, demos)
        disjoint_seeds(manifest([300000, 300001]), manifest([100000]), {0, 1, 2})

    def test_missing_instruction_duplicate_seed_and_incomplete_scan(self):
        for broken in (
            manifest([1, 1]),
            {
                "complete": True,
                "open_laptop": {"success_seeds": [1], "instructions": {}},
            },
            {**manifest([1]), "complete": False},
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    load_manifest(path, require_complete=True)

    def test_every_gpu_count_has_three_distinct_rounds_per_environment(self):
        for ranks in range(1, 7):
            count = training_seed_count(56, ranks)
            self.assertEqual(count % (2 * ranks), 0)
            self.assertGreaterEqual(count // ranks, 6)
        with self.assertRaises(ValueError):
            training_seed_count(8, 3)


class ResourceAndConfigTest(unittest.TestCase):
    def test_physical_gpu_indices_and_limits(self):
        self.assertEqual(gpu_ids("0,3,6"), [0, 3, 6])
        for value in ("", "0,0", "-1", "0-3", "0,1,2,3,4,5,6"):
            with self.assertRaises(ValueError):
                gpu_ids(value)

    def test_busy_gpu_can_pass_capacity_check(self):
        rows = parse_gpu_report(
            "0, GPU-a, 49140, 3000, 45000, 50\n3, GPU-b, 49140, 20, 48000, 0\n6, GPU-c, 49140, 25000, 23000, 100"
        )
        self.assertEqual(select_gpus(rows, peak_mib=30000), [3, 0])
        self.assertEqual(select_gpus(rows, peak_mib=30000, max_gpus=1), [3])

    def test_exact_headroom_threshold(self):
        rows = parse_gpu_report("1, GPU-a, 49140, 25000, 24096, 99")
        self.assertEqual(select_gpus(rows, peak_mib=20000), [1])
        self.assertEqual(select_gpus(rows, peak_mib=20001), [])

    def test_ppo_sample_and_batch_divisibility_for_all_supported_sizes(self):
        for ranks in range(1, 7):
            cfg = {
                "actor": {
                    "micro_batch_size": 1,
                    "global_batch_size": 4 * ranks,
                    "model": {
                        "num_action_chunks": 50,
                        "precision": "fp32",
                        "add_value_head": True,
                        "openpi": {"task": "rl"},
                        "openpi_data": {"adapt_to_pi": False},
                    },
                },
                "env": {
                    "train": {
                        "total_num_envs": 2 * ranks,
                        "rollout_epoch": 1,
                        "max_steps_per_rollout_epoch": 700,
                        "auto_reset": False,
                        "ignore_terminations": False,
                        "use_custom_reward": True,
                        "use_rel_reward": True,
                        "reward_coef": 1,
                    }
                },
            }
            validate_train_config(cfg, ranks)
            cfg["env"]["train"]["auto_reset"] = True
            with self.assertRaises(ValueError):
                validate_train_config(cfg, ranks)


try:
    import numpy as np
    from robotwin.envs.vector_env import SubEnv
except ImportError:
    np = None


@unittest.skipIf(np is None, "Native adapter tests need NumPy and PyYAML")
class SparseNativeBoundaryTest(unittest.TestCase):
    class Task:
        def __init__(self, success_after=None):
            self.success_after = success_after
            self.take_action_cnt = 0
            self.eval_success = False
            self.step_lim = 3

        def take_action(self, action):
            self.take_action_cnt += 1
            self.eval_success = self.take_action_cnt == self.success_after

        def get_instruction(self):
            return "Open the laptop."

        def setup_demo(self, **kwargs):
            self.take_action_cnt = 0
            self.eval_success = False

        def set_instruction(self, instruction):
            pass

        def close_env(self, **kwargs):
            pass

    def environment(self, task):
        env = SubEnv(
            0,
            {
                "action_execution_mode": "take_action",
                "task_name": "open_laptop",
                "step_lim": 3,
            },
            300000,
            "seen",
        )
        env.task = task
        return env

    @patch("robotwin.envs.vector_env.update_obs", return_value={})
    def test_failure_then_timeout_never_gives_success_reward(self, _):
        env = self.environment(self.Task())
        normal = env.step(np.zeros((1, 14)))
        self.assertEqual(normal["reward"].item(), 0)
        self.assertFalse(normal["terminated"].item())
        self.assertFalse(normal["truncated"].item())
        timeout = env.step(np.zeros((50, 14)))
        self.assertEqual(timeout["reward"].item(), 0)
        self.assertFalse(timeout["terminated"].item())
        self.assertTrue(timeout["truncated"].item())

    @patch("robotwin.envs.vector_env.update_obs", return_value={})
    def test_mid_chunk_and_last_step_success_take_precedence_over_timeout(self, _):
        for success_after in (1, 2, 3):
            env = self.environment(self.Task(success_after))
            result = env.step(np.zeros((50, 14)))
            self.assertEqual(result["reward"].item(), 1)
            self.assertTrue(result["terminated"].item())
            self.assertFalse(result["truncated"].item())
            self.assertEqual(result["info"]["executed_action_count"], success_after)
            again = env.step(np.zeros((50, 14)))
            self.assertTrue(again["terminated"].item())
            self.assertEqual(again["info"]["executed_action_count"], 0)
            self.assertEqual(env.task.take_action_cnt, success_after)

    @patch("robotwin.envs.vector_env.update_obs", return_value={})
    def test_reset_clears_terminal_state_and_preserves_requested_seed(self, _):
        env = self.environment(self.Task(2))
        env.step(np.zeros((50, 14)))
        with (
            patch("robotwin.envs.vector_env.new_task", return_value=self.Task(2)),
            patch(
                "robotwin.envs.vector_env.task_instruction",
                return_value="Open the laptop.",
            ),
        ):
            env.reset(300001)
        self.assertFalse(env.task.eval_success)
        self.assertEqual(env.env_seed, 300001)
        result = env.step(np.zeros((50, 14)))
        self.assertTrue(result["terminated"].item())


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "RLinf tensor checks run in the OpenPI CPU environment",
)
class RLinfSparseRewardTest(unittest.TestCase):
    def test_success_pulse_chunk_aggregation_and_reset(self):
        import torch
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

        env = object.__new__(RoboTwinEnv)
        env.cfg = SimpleNamespace(reward_coef=1.0)
        env.num_envs = 1
        env.use_rel_reward = True
        env.record_metrics = False
        env.prev_step_reward = torch.zeros(1)
        rewards = []
        for success in (False, False, True, True, True):
            terminated = torch.tensor([success])
            step = env._calc_step_reward(terminated)
            chunks = env._cal_chunk_rewards(step, 50, terminated, {})
            rewards.append(chunks.sum().item())
        self.assertEqual(rewards, [0, 0, 1, 0, 0])
        env._reset_metrics()
        reward = env._calc_step_reward(torch.tensor([True]))
        self.assertEqual(reward.item(), 1)

    def test_success_chunk_is_retained_and_following_chunks_are_masked(self):
        import torch
        from rlinf.utils.metric_utils import compute_loss_mask

        dones = torch.zeros(4, 1, 50, dtype=torch.bool)
        dones[1:, 0, -1] = True
        mask, _ = compute_loss_mask(dones)
        chunk_mask = mask.any(dim=-1).flatten().tolist()
        self.assertEqual(chunk_mask, [True, False, False])


@unittest.skipUnless(
    importlib.util.find_spec("hydra"),
    "Hydra composition checks run in the OpenPI CPU environment",
)
class ConfigurationRoundTripTest(unittest.TestCase):
    def test_prepare_and_reload_the_exact_training_snapshot(self):
        import argparse
        from omegaconf import OmegaConf
        from hydra import compose, initialize_config_dir

        from rlinf_rft.cli import prepare

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            train = temporary / "training.json"
            train.write_text(json.dumps(manifest(list(range(300000, 300056)))))
            evaluation = temporary / "evaluation.json"
            evaluation.write_text(json.dumps(manifest(list(range(100000, 100100)))))
            demos = temporary / "seed.txt"
            demos.write_text("0 1 2 3 4")
            checkpoint = temporary / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"test-checkpoint")
            args = argparse.Namespace(
                gpus="0,3,6",
                train_seeds=train,
                eval_seeds=evaluation,
                demo_seeds=demos,
                checkpoint=checkpoint,
                output=temporary / "run",
            )
            with patch("rlinf_rft.cli.model_files", return_value={"tensor_count": 667}):
                prepare(args)
            with initialize_config_dir(version_base="1.1", config_dir=str(args.output)):
                cfg = compose(config_name="train_config")
            validate_train_config(OmegaConf.to_container(cfg, resolve=True), 3)
            self.assertEqual(
                cfg.cluster.component_placement["actor, env, rollout"], "0,3,6"
            )
            self.assertEqual(
                cfg.env.train.seeds_path, str(args.output / "train_seeds.json")
            )
            self.assertEqual(cfg.actor.model.model_path, str(checkpoint))
            self.assertEqual(
                len(
                    load_manifest(args.output / "eval_seeds.json")["open_laptop"][
                        "success_seeds"
                    ]
                ),
                8,
            )
            frozen = json.loads((args.output / "run.json").read_text())
            self.assertEqual(frozen["training_seed_count"], 54)
            self.assertEqual(frozen["unused_training_seed_count"], 2)

    def test_all_card_counts_compose_with_the_same_sparse_reward_contract(self):
        from omegaconf import OmegaConf
        from rlinf_rft.cli import configure_train, compose_template

        for ranks in range(1, 7):
            cfg = configure_train(list(range(ranks)))
            validate_train_config(OmegaConf.to_container(cfg, resolve=True), ranks)
        evaluation = compose_template("robotwin_open_laptop_openpi_pi05_eval")
        self.assertTrue(evaluation.env.eval.is_eval)
        self.assertEqual(evaluation.env.eval.rollout_epoch, 8)
        self.assertEqual(evaluation.rollout.model.openpi_data.adapt_to_pi, False)


if __name__ == "__main__":
    unittest.main()
