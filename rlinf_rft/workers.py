"""Runtime instrumentation; PPO and FSDP operations remain in RLinf."""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf_rft.common import write_json


def output_dir() -> Path:
    return Path(os.environ["RFT_OUTPUT"])


def tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def record_worker(worker, role: str, *, ready: bool = False) -> None:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    write_json(
        output_dir() / "workers" / f"{role}_{worker._rank}.json",
        {
            "role": role,
            "rank": worker._rank,
            "pid": os.getpid(),
            "ready": ready,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "render_device": os.environ.get("ROBOTWIN_RENDER_DEVICE"),
            "device": str(torch.cuda.current_device()),
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "torch_version": torch.__version__,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    )


@contextmanager
def model_load_lock():
    # Only model construction/loading is serialized, never FSDP collectives.
    with (output_dir() / "model_load.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class PPOActor(EmbodiedFSDPActor):
    def init_worker(self):
        record_worker(self, "actor")
        seed_process(int(self.cfg.actor.seed))
        super().init_worker()
        record_worker(self, "actor", ready=True)

    def model_provider_func(self):
        with model_load_lock():
            model = super().model_provider_func()
            if self._rank == 0:
                parameters = {
                    name: {
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "trainable": param.requires_grad,
                    }
                    for name, param in model.named_parameters()
                }
                heads = {
                    name: tensor_hash(param)
                    for name, param in model.named_parameters()
                    if name.startswith("value_head.")
                }
                write_json(
                    output_dir() / "initial_parameters.json",
                    {"parameters": parameters, "initial_value_head_sha256": heads},
                )
            return model

    def run_training(self):
        metrics = super().run_training()
        for name, value in metrics.items():
            if any(
                term in name for term in ("loss", "grad_norm", "approx_kl")
            ) and not math.isfinite(float(value)):
                raise FloatingPointError(f"Non-finite PPO metric {name}: {value}")
        if self._rank == 0:
            write_json(
                output_dir()
                / "metrics"
                / f"optimizer_step_{self.optimizer_steps}.json",
                {
                    "optimizer_steps": self.optimizer_steps,
                    "metrics": {name: float(value) for name, value in metrics.items()},
                },
            )
        record_worker(self, "actor", ready=True)
        return metrics


class PPORollout(MultiStepRolloutWorker):
    def init_worker(self):
        record_worker(self, "rollout")
        with model_load_lock():
            super().init_worker()
        seed_process(int(self.cfg.rollout.seed) + self._rank)
        record_worker(self, "rollout", ready=True)


class PPOEnv(EnvWorker):
    def release_environments(self):
        for env in self.env_list + self.eval_env_list:
            venv = getattr(env, "venv", None)
            if venv is not None:
                venv.close()
        self.env_list.clear()
        self.eval_env_list.clear()

    def init_worker(self):
        os.environ["ROBOTWIN_RENDER_DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        record_worker(self, "env")
        super().init_worker()
        record_worker(self, "env", ready=True)
