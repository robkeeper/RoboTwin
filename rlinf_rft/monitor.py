"""Sample whole-GPU usage and this run's worker memory independently."""

from __future__ import annotations

import csv
import json
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rlinf_rft.common import parse_gpu_report, write_json


def owned_pids(output: Path) -> set[int]:
    pids = set()
    for path in (output / "workers").glob("*.json"):
        try:
            pids.add(int(json.loads(path.read_text())["pid"]))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    pending = list(pids)
    while pending:
        pid = pending.pop()
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            continue
        for value in children:
            child = int(value)
            if child not in pids:
                pids.add(child)
                pending.append(child)
    return pids


@contextmanager
def record_resources(output: Path, gpus: list[int]):
    output.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    peak_by_uuid = {}
    errors = []

    def sample():
        with (output / "resources.jsonl").open("a") as stream:
            while not stopped.is_set():
                try:
                    gpu_text = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    rows = [
                        row
                        for row in parse_gpu_report(gpu_text)
                        if row["index"] in gpus
                    ]
                    app_text = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    pids = owned_pids(output)
                    usage = {row["uuid"]: 0 for row in rows}
                    for app in csv.reader(app_text.splitlines()):
                        uuid, pid, used = [value.strip() for value in app]
                        if uuid in usage and int(pid) in pids:
                            usage[uuid] += int(used)
                    for uuid, used in usage.items():
                        peak_by_uuid[uuid] = max(peak_by_uuid.get(uuid, 0), used)
                    memory = Path("/proc/meminfo").read_text()
                    available = next(
                        int(line.split()[1])
                        for line in memory.splitlines()
                        if line.startswith("MemAvailable:")
                    )
                    record = {
                        "gpus": rows,
                        "worker_mib_by_uuid": usage,
                        "host_available_kib": available,
                    }
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    errors.append(str(error))
                    record = {"error": str(error)}
                record["time"] = datetime.now(timezone.utc).isoformat()
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                stopped.wait(1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=12)
        write_json(
            output / "resource_summary.json",
            {
                "sampled_worker_peak_mib_by_uuid": peak_by_uuid,
                "sampled_worker_peak_mib": max(peak_by_uuid.values(), default=0),
                "sampling_errors": errors,
                "note": "One-second process-memory samples can miss transient peaks; require at least 4096 MiB additional headroom.",
            },
        )
