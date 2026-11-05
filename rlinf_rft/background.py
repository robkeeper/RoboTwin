"""Run expert screening independently of SSH or an assistant polling loop."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rlinf_rft.common import ROOT, write_json


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_suffix(".job.json"),
        output.with_suffix(".log"),
        output.with_suffix(".job.lock"),
    )


def progress(output: Path) -> dict:
    if not output.exists():
        return {"complete": False, "accepted": 0, "attempts": 0}
    manifest = json.loads(output.read_text())
    attempts = manifest["attempts"]
    return {
        "complete": manifest["complete"],
        "accepted": len(manifest["open_laptop"]["success_seeds"]),
        "attempts": len(attempts),
        "last_seed": attempts[-1]["seed"] if attempts else None,
    }


def target_process(pid: int, output: Path) -> bool:
    """Only recognize the exact old screening command and output path."""
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return False
    module = ["-m", "rlinf_rft.cli", "scan"]
    if not any(arguments[i : i + 3] == module for i in range(len(arguments) - 2)):
        return False
    try:
        actual_output = arguments[arguments.index("--output") + 1]
    except (ValueError, IndexError):
        return False
    return Path(actual_output).resolve() == output.resolve()


def handoff(pid: int, output: Path) -> None:
    if not target_process(pid, output):
        raise ValueError(
            f"PID {pid} is not the expected screening process; refusing to stop it"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while target_process(pid, output):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Original screening process has not stopped; refusing a duplicate launch"
            )
        time.sleep(0.2)


def scan_command(args) -> list[str]:
    return [
        "bash",
        str(ROOT / "rlinf_rft/run.sh"),
        "scan",
        "--start",
        str(args.start),
        "--count",
        str(args.count),
        "--max-attempts",
        str(args.max_attempts),
        "--output",
        str(args.output),
    ]


def launch(args) -> None:
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status, log_path, lock_path = paths(args.output)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Screening already has a background supervisor"
            ) from error
        if args.replace_pid is not None:
            handoff(args.replace_pid, args.output)
        current = progress(args.output)
        if current["complete"]:
            print(json.dumps({"state": "already_complete", **current}, indent=2))
            return
        command = [
            sys.executable,
            "-u",
            "-B",
            "-m",
            "rlinf_rft.background",
            "worker",
            "--gpu",
            str(args.gpu),
            "--start",
            str(args.start),
            "--count",
            str(args.count),
            "--max-attempts",
            str(args.max_attempts),
            "--output",
            str(args.output),
            "--lock-fd",
            str(lock.fileno()),
        ]
        write_json(
            status,
            {
                "state": "starting",
                "started_at": now(),
                "gpu": args.gpu,
                "target": args.count,
                "max_attempts": args.max_attempts,
                "output": str(args.output),
                "log": str(log_path),
                "resumed_progress": current,
            },
        )
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        print(
            json.dumps(
                {
                    "state": "launched",
                    "supervisor_pid": process.pid,
                    "status": str(status),
                    "log": str(log_path),
                    "resumed_progress": current,
                },
                indent=2,
            )
        )


def worker(args) -> None:
    args.output = args.output.resolve()
    status_path, log_path, _ = paths(args.output)
    status = json.loads(status_path.read_text())
    status.update(state="running", supervisor_pid=os.getpid(), updated_at=now())
    write_json(status_path, status)
    environment = os.environ.copy()
    environment["GPU_LIST"] = str(args.gpu)
    command = scan_command(args)
    code = 1
    try:
        print(f"[{now()}] Starting/resuming screening on GPU {args.gpu}", flush=True)
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL
        )
        status.update(screening_pid=process.pid, command=command)
        write_json(status_path, status)
        code = process.wait()
        current = progress(args.output)
        complete = (
            code == 0 and current["complete"] and current["accepted"] == args.count
        )
        status.update(
            state="complete" if complete else "failed", exit_code=code, progress=current
        )
    except BaseException as error:
        status.update(state="failed", exit_code=code, error=str(error))
        traceback.print_exc()
    finally:
        status.update(finished_at=now(), updated_at=now(), log=str(log_path))
        write_json(status_path, status)
        os.close(args.lock_fd)
    print(json.dumps(status, indent=2), flush=True)
    if status["state"] != "complete":
        raise SystemExit(code or 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--gpu", type=int, required=True)
        command.add_argument("--start", type=int, default=300000)
        command.add_argument("--count", type=int, default=56)
        command.add_argument("--max-attempts", type=int, default=1000)
        command.add_argument("--output", type=Path, required=True)
        if name == "launch":
            command.add_argument("--replace-pid", type=int)
        else:
            command.add_argument("--lock-fd", type=int, required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        path, _, _ = paths(args.output.resolve())
        state = (
            json.loads(path.read_text()) if path.exists() else {"state": "not_launched"}
        )
        state["progress"] = progress(args.output.resolve())
        print(json.dumps(state, indent=2))
    else:
        if (
            args.gpu < 0
            or args.start < 0
            or args.count < 1
            or args.max_attempts < args.count
        ):
            parser.error(
                "Require a nonnegative GPU/start, positive count and sufficient attempts"
            )
        {"launch": launch, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
