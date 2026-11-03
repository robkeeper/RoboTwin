"""Record reserved-GPU usage while a screening or evaluation process runs."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def record_gpu(output: Path):
    """Sample the reserved physical GPU without creating CUDA contexts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()

    def sample():
        with output.open("a") as stream:
            while not stopped.is_set():
                command = [
                    "nvidia-smi",
                    "-i",
                    os.environ["GPU_LIST"],
                    "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
                try:
                    result = subprocess.run(
                        command, capture_output=True, text=True, timeout=5
                    )
                    record = {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "returncode": result.returncode,
                        "gpu": result.stdout.strip(),
                        "error": result.stderr.strip(),
                    }
                except (OSError, subprocess.TimeoutExpired) as error:
                    record = {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "error": str(error),
                    }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                stopped.wait(1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=6)
