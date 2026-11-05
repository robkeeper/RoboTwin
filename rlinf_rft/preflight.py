"""Check worker method binding before creating Ray actors or loading models."""


def validate_worker_bindings() -> dict:
    from rlinf.scheduler.worker.worker_group import WorkerGroup
    from rlinf_rft.workers import PPOActor, PPOEnv, PPORollout

    names = []
    for worker in (PPOActor, PPOEnv, PPORollout):
        group = WorkerGroup(worker, (), {})
        group._attach_cls_func()
        names.append(worker.__name__)
    return {"worker_group_bindings_valid": True, "workers": names}
