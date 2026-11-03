"""Importable rollout worker and runtime reporting for Ray actors."""

import fcntl
import json
import random
from pathlib import Path

import numpy as np
import ray
import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def runtime_report(model, seed: int) -> dict:
    """Read backend settings on the worker, outside the serialized method body."""
    return {
        "seed": seed,
        "torch_version": torch.__version__,
        "device": str(next(model.parameters()).device),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "ray_cluster_resources": ray.cluster_resources(),
    }


class SeededRolloutWorker(MultiStepRolloutWorker):
    """Use the same sampling seed and serialize only CPU-heavy weight loading."""

    def init_worker(self):
        lock_path = self.cfg.rollout.get("model_load_lock")
        if lock_path:
            with Path(lock_path).open("a") as lock:
                self.log_info("Waiting for the shared model-loading lock")
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.log_info("Acquired model-loading lock")
                super().init_worker()
        else:
            super().init_worker()
        seed = int(self.cfg.rollout.seed) + self._rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        report_path = self.cfg.rollout.get("runtime_report_path")
        if report_path:
            Path(report_path).write_text(
                json.dumps(runtime_report(self.hf_model, seed), indent=2) + "\n"
            )
