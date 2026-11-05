#!/usr/bin/env python3
"""Print remote PPO progress without using an AI service."""

import os
import sys
from pathlib import Path

codex_prefix = Path("/opt/anaconda3/envs/codex")
if Path(sys.prefix).resolve() != codex_prefix.resolve():
    codex_python = codex_prefix / "bin/python"
    if not codex_python.is_file():
        raise SystemExit(f"Missing codex Conda interpreter: {codex_python}")
    os.execv(
        str(codex_python),
        [str(codex_python), "-u", "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlinf_rft.watch import main  # noqa: E402

if __name__ == "__main__":
    print(f"监视环境：codex ({sys.prefix})", flush=True)
    main()
