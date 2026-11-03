"""Native task boundary checks; no SAPIEN or model loading is required."""

import unittest
from unittest.mock import patch

import numpy as np

from robotwin.envs.vector_env import SubEnv


class NativeTask:
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


class NativeControllerTest(unittest.TestCase):
    def env(self, task):
        env = SubEnv(
            0,
            {"action_execution_mode": "take_action", "task_name": "open_laptop"},
            100000,
            "seen",
        )
        env.task = task
        return env

    @patch("robotwin.envs.vector_env.update_obs", return_value={"state": np.zeros(14)})
    def test_success_stops_execution_and_pads_full_observations(self, _):
        task = NativeTask(success_after=2)
        result = self.env(task).step(np.zeros((50, 14)), full_obs=True)
        self.assertEqual(task.take_action_cnt, 2)
        self.assertEqual(result["info"]["executed_action_count"], 2)
        self.assertTrue(result["info"]["success"])
        self.assertEqual(len(result["obs"]), 50)
        self.assertFalse(result["truncated"][0])

    @patch("robotwin.envs.vector_env.update_obs", return_value={})
    def test_budget_stops_at_limit_inside_a_chunk(self, _):
        task = NativeTask()
        result = self.env(task).step(np.zeros((50, 14)))
        self.assertEqual(task.take_action_cnt, 3)
        self.assertEqual(result["info"]["executed_action_count"], 3)
        self.assertTrue(result["truncated"][0])
        self.assertFalse(result["info"]["success"])

    def test_setup_error_does_not_replace_requested_seed(self):
        seeds = []

        class FailingTask:
            def setup_demo(self, **kwargs):
                seeds.append(kwargs["seed"])
                raise RuntimeError("bad scene")

            def close_env(self, **kwargs):
                pass

        env = self.env(None)
        env.args["step_lim"] = 700
        with patch("robotwin.envs.vector_env.new_task", return_value=FailingTask()):
            with self.assertRaisesRegex(RuntimeError, "bad scene"):
                env.reset(100003)
        self.assertEqual(seeds, [100003])


if __name__ == "__main__":
    unittest.main()
