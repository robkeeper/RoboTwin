"""RLinf's embodied eval runner with an explicit rollout RNG seed."""

import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf_eval.rollout_worker import SeededRolloutWorker


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = SeededRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
