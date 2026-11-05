"""Check terminal notifications and the gate before a long training run."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rlinf_rft.cli import configure_long_training
from rlinf_rft.common import file_hash, write_json
from rlinf_rft.validation import check_validation
from rlinf_rft.workflow import choose_gpus


@unittest.skipUnless(
    importlib.util.find_spec("omegaconf"),
    "OmegaConf is required for configuration tests",
)
class FreshTrainingConfigTest(unittest.TestCase):
    def test_formal_run_starts_from_sft_with_no_resume(self):
        from omegaconf import OmegaConf

        base = OmegaConf.create(
            {
                "runner": {
                    "max_steps": 3,
                    "max_epochs": 3,
                    "resume_dir": "/old/step3",
                    "ckpt_path": "/old/step3",
                    "save_interval": 1,
                    "logger": {"log_path": "/validation"},
                },
                "actor": {
                    "model": {"model_path": "/old/step3"},
                    "global_batch_size": 12,
                    "micro_batch_size": 1,
                    "optim": {"lr": 5e-6, "value_lr": 1e-4},
                },
                "rollout": {"model": {"model_path": "/old/step3"}},
                "env": {
                    "train": {
                        "task_config": {
                            "episode_log_path": "/validation/episodes.jsonl"
                        }
                    }
                },
            }
        )
        cfg = configure_long_training(
            base,
            {"model_path": "/data/robotwin_ckpt/rlinf/7272"},
            300,
            10,
            Path("/outputs/run"),
        )
        self.assertEqual(
            (cfg.runner.max_steps, cfg.runner.max_epochs, cfg.runner.save_interval),
            (300, 300, 10),
        )
        self.assertIsNone(cfg.runner.resume_dir)
        self.assertIsNone(cfg.runner.ckpt_path)
        self.assertEqual(cfg.actor.model.model_path, "/data/robotwin_ckpt/rlinf/7272")
        self.assertEqual(cfg.rollout.model.model_path, cfg.actor.model.model_path)
        self.assertEqual(cfg.actor.global_batch_size, 12)
        self.assertEqual(cfg.actor.micro_batch_size, 1)
        self.assertEqual(cfg.actor.optim.lr, 5e-6)
        self.assertEqual(cfg.actor.optim.value_lr, 1e-4)
        self.assertEqual(base.runner.max_steps, 3)
        self.assertEqual(base.runner.resume_dir, "/old/step3")
        self.assertEqual(
            [
                step
                for step in range(1, cfg.runner.max_steps + 1)
                if step % cfg.runner.save_interval == 0
            ],
            list(range(10, 301, 10)),
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("hydra"),
    "RLinf runtime dependencies are required for worker binding tests",
)
class WorkerBindingTest(unittest.TestCase):
    def test_all_real_worker_classes_bind_without_reserved_method_collisions(self):
        from rlinf_rft.preflight import validate_worker_bindings

        result = validate_worker_bindings()
        self.assertTrue(result["worker_group_bindings_valid"])
        self.assertEqual(result["workers"], ["PPOActor", "PPOEnv", "PPORollout"])


class WatchTest(unittest.TestCase):
    def run_watch(self, directory, phase):
        path = Path(directory) / "status.json"
        write_json(
            path,
            {
                "run_id": "test",
                "phase": phase,
                "events": [
                    {
                        "id": 1,
                        "time": "2026-11-04",
                        "phase": "probe_passed",
                        "message": "完整模型 probe 通过",
                    }
                ],
            },
        )
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "rlinf_rft.watch",
                "--local-status",
                str(path),
                "--until",
                "probe",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_probe_is_printed_even_when_training_has_already_started(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_watch(directory, "long_training")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("完整模型 probe 通过", result.stdout)

    def test_later_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_watch(directory, "failed")
        self.assertEqual(result.returncode, 1)


class GPUSelectionTest(unittest.TestCase):
    def test_requires_three_cards_meeting_both_limits(self):
        rows = [
            {"index": index, "free_mib": free, "utilization": use}
            for index, free, use in [
                (0, 48000, 0),
                (1, 47000, 2),
                (2, 46000, 99),
                (3, 10000, 0),
            ]
        ]
        self.assertIsNone(choose_gpus(rows, 45056, 10))
        rows[2]["utilization"] = 0
        self.assertEqual(choose_gpus(rows, 45056, 10), [0, 1, 2])

    def test_frozen_cards_are_not_replaced_by_other_free_cards(self):
        rows = [
            {"index": index, "free_mib": 48000, "utilization": 0} for index in (0, 1, 2)
        ]
        self.assertIsNone(choose_gpus(rows, 45056, 10, [3, 4, 5]))


@unittest.skipUnless(
    importlib.util.find_spec("tensorboard"),
    "TensorBoard is required for artifact gate tests",
)
class ValidationGateTest(unittest.TestCase):
    def setUp(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = (
            self.root
            / "train/robotwin_open_laptop_ppo_openpi_pi05/checkpoints/global_step_3"
        )
        weights = self.checkpoint / "actor/model_state_dict/full_weights.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"external checkpoint fixture")
        resume = self.checkpoint / "actor/dcp_checkpoint"
        resume.mkdir()
        (resume / ".metadata").write_bytes(b"metadata fixture")
        (resume / "__0_0.distcp").write_bytes(b"sharded training state fixture")
        self.verification = {
            "passed": True,
            "checkpoint": str(self.checkpoint),
            "checkpoint_weights_file": str(weights),
            "checkpoint_weights_sha256": file_hash(weights),
            "changed_trainable_parameters": ["action_in_proj.weight"],
            "changed_value_head_parameters": ["value_head.weight"],
            "changed_frozen_parameters": [],
            "nonfinite_parameters": [],
            "training_metric_files": 3,
            "successful_training_episodes": 1,
        }
        write_json(self.root / "parameter_verification.json", self.verification)
        write_json(self.root / "train/process.json", {"exit_code": 0})
        manifest = {
            "success_seeds": list(range(8)),
            "instructions": {str(seed): f"instruction {seed}" for seed in range(8)},
        }
        write_json(self.root / "eval_seeds.json", {"open_laptop": manifest})
        self.evaluation = {
            "complete": True,
            "checkpoint": str(self.checkpoint),
            "total": 8,
            "success": 0,
            "episodes": [
                {
                    "seed": seed,
                    "instruction": manifest["instructions"][str(seed)],
                    "success": False,
                }
                for seed in range(8)
            ],
        }
        write_json(self.root / "eval_after/summary.json", self.evaluation)
        for step in (35, 70, 105):
            write_json(
                self.root / f"train/metrics/optimizer_step_{step}.json",
                {
                    "metrics": {
                        "actor/policy_loss": 0.1,
                        "actor/grad_norm": 1.0,
                        "actor/approx_kl": 0.0,
                    }
                },
            )
        writer = EventFileWriter(str(self.root / "train/tensorboard"))
        for step in range(3):
            tags = [
                "env/return",
                "env/reward",
                "env/success_once",
                "rollout/rewards",
                "rollout/returns_mean",
                "train/actor/policy_loss",
                "train/critic/value_loss",
            ]
            writer.add_event(
                Event(
                    wall_time=step + 1,
                    step=step,
                    summary=Summary(
                        value=[Summary.Value(tag=tag, simple_value=0.5) for tag in tags]
                    ),
                )
            )
        writer.close()

    def test_complete_validation_does_not_require_eval_improvement(self):
        self.assertTrue(check_validation(self.root)["passed"])

    def test_changed_frozen_weights_block_long_training(self):
        self.verification["changed_frozen_parameters"] = ["img.weight"]
        write_json(self.root / "parameter_verification.json", self.verification)
        with self.assertRaises(ValueError):
            check_validation(self.root)

    def test_missing_optimizer_state_blocks_long_training(self):
        (self.checkpoint / "actor/dcp_checkpoint/.metadata").unlink()
        with self.assertRaises(ValueError):
            check_validation(self.root)

    def test_incomplete_reload_evaluation_blocks_long_training(self):
        self.evaluation["complete"] = False
        write_json(self.root / "eval_after/summary.json", self.evaluation)
        with self.assertRaises(ValueError):
            check_validation(self.root)


if __name__ == "__main__":
    unittest.main()
