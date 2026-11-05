"""Select instrumented workers while using RLinf's stock runners."""

import os

import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf_rft.preflight import validate_worker_bindings
from rlinf_rft.workers import PPOActor, PPOEnv, PPORollout


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    validate_worker_bindings()
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    groups = []
    env = None
    try:
        rollout = PPORollout.create_group(cfg)
        groups.append(rollout)
        rollout.launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env = PPOEnv.create_group(cfg)
        groups.append(env)
        env.launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        if os.environ["RFT_MODE"] == "eval":
            if not cfg.runner.only_eval:
                raise ValueError("Evaluation requires runner.only_eval=true")
            runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
        else:
            if cfg.runner.only_eval:
                raise ValueError("Training requires runner.only_eval=false")
            actor = PPOActor.create_group(cfg)
            groups.append(actor)
            actor.launch(
                cluster,
                name=cfg.actor.group_name,
                placement_strategy=placement.get_strategy("actor"),
            )
            runner = EmbodiedRunner(cfg=cfg, actor=actor, rollout=rollout, env=env)
        runner.init_workers()
        runner.run()
    finally:
        errors = []
        if env is not None and hasattr(env, "release_environments"):
            try:
                env.release_environments().wait()
            except Exception as error:
                errors.append(str(error))
        for group in reversed(groups):
            try:
                group._close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError(f"Failed to close this run's worker groups: {errors}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
