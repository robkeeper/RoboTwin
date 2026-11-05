"""Background screening can resume without an assistant polling its progress."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlinf_rft.background import handoff, paths, progress, scan_command, target_process


class BackgroundScreeningTest(unittest.TestCase):
    def test_missing_manifest_and_persisted_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "train56.json"
            self.assertEqual(progress(output)["accepted"], 0)
            output.write_text(
                json.dumps(
                    {
                        "complete": False,
                        "attempts": [{"seed": 300000}],
                        "open_laptop": {"success_seeds": [300000]},
                    }
                )
            )
            self.assertEqual(
                progress(output),
                {"complete": False, "accepted": 1, "attempts": 1, "last_seed": 300000},
            )

    def test_job_files_do_not_replace_the_manifest(self):
        output = Path("/tmp/scenes/train56.json")
        status, log, lock = paths(output)
        self.assertEqual(status.name, "train56.job.json")
        self.assertEqual(log.name, "train56.log")
        self.assertEqual(lock.name, "train56.job.lock")
        self.assertNotIn(output, (status, log, lock))

    @patch("rlinf_rft.background.os.kill")
    def test_handoff_refuses_an_unrelated_process(self, kill):
        with patch("rlinf_rft.background.target_process", return_value=False):
            with self.assertRaises(ValueError):
                handoff(123, Path("/tmp/train56.json"))
        kill.assert_not_called()

    def test_exact_module_and_output_are_required_for_handoff(self):
        data = b"python\0-B\0-m\0rlinf_rft.cli\0scan\0--output\0/tmp/train56.json\0"
        with patch("rlinf_rft.background.Path.read_bytes", return_value=data):
            self.assertTrue(target_process(123, Path("/tmp/train56.json")))
            self.assertFalse(target_process(123, Path("/tmp/other.json")))
        with patch(
            "rlinf_rft.background.Path.read_bytes", return_value=b"python\0other.py\0"
        ):
            self.assertFalse(target_process(123, Path("/tmp/train56.json")))

    def test_original_limits_are_passed_to_the_existing_screening_loop(self):
        args = argparse.Namespace(
            start=300000, count=56, max_attempts=1000, output=Path("/tmp/train56.json")
        )
        command = scan_command(args)
        self.assertEqual(command[command.index("--count") + 1], "56")
        self.assertEqual(command[command.index("--max-attempts") + 1], "1000")
        self.assertEqual(command[-1], "/tmp/train56.json")


if __name__ == "__main__":
    unittest.main()
