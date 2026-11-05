"""The controller handoff must preserve the running probe process."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from rlinf_rft.common import write_json


@unittest.skipUnless(
    Path("/proc").is_dir(), "Controller handoff uses Linux process metadata"
)
class ControllerHandoffTest(unittest.TestCase):
    def test_only_controller_exits_and_probe_child_survives(self):
        from resume_open_laptop_ppo import ImmediateWorkflow
        from rlinf_rft.workflow import source_hashes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_file = root / "child.pid"
            code = "import subprocess,sys,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[2]).write_text(str(child.pid)); time.sleep(60)"
            parent = subprocess.Popen(
                [sys.executable, "-c", code, "rlinf_rft.workflow", str(child_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child = None
            try:
                deadline = time.monotonic() + 5
                while not child_file.exists():
                    if time.monotonic() >= deadline:
                        self.fail("Test controller did not launch its child")
                    time.sleep(0.02)
                child = int(child_file.read_text())
                write_json(
                    root / "status.json",
                    {
                        "pid": parent.pid,
                        "phase": "probe_running",
                        "events": [],
                        "run_id": "test",
                    },
                )
                write_json(root / "plan.json", {"source_sha256": source_hashes()})
                controller = ImmediateWorkflow(root)
                parent.wait(timeout=5)
                self.assertEqual(parent.returncode, -signal.SIGTERM)
                self.assertTrue(Path(f"/proc/{child}/cmdline").read_bytes())
                self.assertEqual(controller.args.stable_checks, 1)
                self.assertEqual(controller.status["pid"], os.getpid())
            finally:
                if parent.poll() is None:
                    parent.terminate()
                    parent.wait(timeout=5)
                if child is not None:
                    try:
                        os.kill(child, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
