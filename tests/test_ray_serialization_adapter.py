"""Regression for Ray actor export when Torch's backend proxy is loaded."""

import unittest

import ray
import ray.cloudpickle
import torch
from rlinf_eval.rollout_worker import SeededRolloutWorker


class RaySerializationTest(unittest.TestCase):
    def test_modified_actor_can_serialize_with_cudnn_proxy_loaded(self):
        _ = torch.backends.cudnn.allow_tf32
        actor = ray.remote(SeededRolloutWorker)
        data = ray.cloudpickle.dumps(actor.__ray_metadata__.modified_class)
        restored = ray.cloudpickle.loads(data)
        self.assertTrue(callable(restored.init_worker))


if __name__ == "__main__":
    unittest.main()
