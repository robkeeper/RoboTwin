"""Regression for full-robot state forwarding at the mplib boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from envs.robot.planner import MplibPlanner, MplibWrapperPlanner


class PlannerStateTest(unittest.TestCase):
    def test_planning_preserves_both_arms_and_grippers(self):
        planner = object.__new__(MplibWrapperPlanner)
        state = np.linspace(-0.1, 0.1, 38)
        pose = SimpleNamespace(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
        with patch.object(
            MplibPlanner, "plan_path", return_value={"status": "Fail"}
        ) as native:
            planner.plan_path(state, pose, arms_tag="left")
        forwarded = native.call_args.args[0]
        self.assertEqual(forwarded.shape, (38,))
        np.testing.assert_allclose(forwarded, state, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
