"""Read workflow status over SSH and print each event once."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

DEFAULT_STATUS = "/data0/wangyihan/open_laptop_ppo/auto_7272/status.json"


def read_status(host: str, path: str) -> dict:
    quoted = shlex.quote(path)
    command = (
        f"if test -f {quoted}; then cat -- {quoted}; "
        'else printf \'%s\\n\' \'{"phase":"not_started","events":[]}\'; fi'
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="9981")
    parser.add_argument("--status", default=DEFAULT_STATUS)
    parser.add_argument(
        "--local-status", type=Path, help="Read a local status file instead of SSH"
    )
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument(
        "--until", choices=("probe", "validation", "complete"), default="complete"
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or not args.host or args.host.startswith("-"):
        parser.error("Require a positive interval and a valid SSH host")
    if not args.status.startswith("/"):
        parser.error("Remote status path must be absolute")
    print(f"监视状态：{args.local_status or f'{args.host}:{args.status}'}", flush=True)
    print("仅运行普通脚本；Ctrl+C 停止监视，不会停止服务器训练。", flush=True)
    seen = set()
    previous = None
    error_reported = False
    target = {
        "probe": "probe_passed",
        "validation": "validation_passed",
        "complete": "complete",
    }[args.until]
    try:
        while True:
            try:
                status = (
                    json.loads(args.local_status.read_text())
                    if args.local_status
                    else read_status(args.host, args.status)
                )
                if error_reported:
                    print("连接已恢复。", flush=True)
                error_reported = False
                for event in status.get("events", []):
                    key = (status.get("run_id"), event["id"])
                    if key not in seen:
                        print(f"[{event['time']}] {event['message']}", flush=True)
                        seen.add(key)
                phase = status.get("phase", "unknown")
                if phase != previous:
                    print(f"当前阶段：{phase}", flush=True)
                    previous = phase
                if phase == "failed":
                    raise SystemExit(1)
                reached = {event["phase"] for event in status.get("events", [])}
                if phase == "complete" or target in reached or args.once:
                    return
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                if not error_reported:
                    print(f"暂时无法读取状态，稍后重试：{error}", flush=True)
                    error_reported = True
                if args.once:
                    raise SystemExit(2) from error
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止本地监视。", flush=True)


if __name__ == "__main__":
    main()
