"""Boundary checks for paired seed selection and protocol identity."""

import json
import tempfile
import unittest
from pathlib import Path

from rlinf_eval.campaign import merge_shards, screening_plan
from rlinf_eval.cli import is_known_expert_failure


class SeedManifestTest(unittest.TestCase):
    def test_parallel_screening_has_exactly_100_nonoverlapping_scenes(self):
        plan = screening_plan(3, 6)
        self.assertEqual(len(plan), 8)
        for gpu in [3, 6]:
            self.assertEqual(sum(p["count"] for p in plan if p["gpu"] == gpu), 50)
        ranges = [set(range(p["start"], p["start"] + 2000)) for p in plan]
        self.assertEqual(len(set.union(*ranges)), 8 * 2000)

    def test_missing_expert_grasp_is_a_scene_failure(self):
        self.assertTrue(
            is_known_expert_failure(
                AssertionError("target_pose cannot be None for move action.")
            )
        )

    def test_unrelated_errors_are_not_silently_filtered(self):
        self.assertFalse(is_known_expert_failure(AssertionError("unexpected shape")))
        self.assertFalse(
            is_known_expert_failure(
                FileNotFoundError("target_pose cannot be None for move action.")
            )
        )

    def shard(self, path, seed, fingerprint="same"):
        value = {
            "complete": True,
            "protocol": {
                "source_sha256": {"task": fingerprint},
                "task_config": {"task_name": "open_laptop"},
                "instruction_type": "seen",
                "sapien_runtime": "frozen",
            },
            "open_laptop": {
                "success_seeds": [seed],
                "instructions": {str(seed): f"instruction {seed}"},
            },
        }
        path.write_text(json.dumps(value))
        return path

    def test_merge_preserves_instruction_identity_and_sorted_scene_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                self.shard(root / "a.json", 200001),
                self.shard(root / "b.json", 100001),
            ]
            result = merge_shards(files, root / "merged.json", expected=2)
            self.assertEqual(result["open_laptop"]["success_seeds"], [100001, 200001])
            self.assertEqual(
                result["open_laptop"]["instructions"]["200001"], "instruction 200001"
            )

    def test_duplicate_scenes_cannot_count_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                self.shard(root / "a.json", 100001),
                self.shard(root / "b.json", 100001),
            ]
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                merge_shards(files, root / "merged.json", expected=2)

    def test_different_source_versions_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                self.shard(root / "a.json", 100001),
                self.shard(root / "b.json", 200001, "changed"),
            ]
            with self.assertRaisesRegex(ValueError, "Different screening protocol"):
                merge_shards(files, root / "merged.json", expected=2)


if __name__ == "__main__":
    unittest.main()
