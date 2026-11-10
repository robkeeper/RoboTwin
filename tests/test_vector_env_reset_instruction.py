import threading
import unittest
from unittest.mock import patch

from robotwin.envs import vector_env


class FakeTask:
    def __init__(self):
        self.events = []
        self.unstable_seeds = set()
        self.seed = None
        self.instruction = None

    def setup_demo(self, now_ep_num, seed, is_test=False, instruction=None, **args):
        self.events.append(("setup", seed, is_test, instruction))
        self.seed = seed
        self.instruction = instruction
        if seed in self.unstable_seeds:
            raise vector_env.UnStableError("unstable scene")

    def get_info(self):
        self.events.append(("info", self.seed))
        return {"{A}": f"base{self.seed}", "{a}": "right" if self.seed % 2 else "left"}

    def set_instruction(self, instruction):
        self.instruction = instruction

    def close_env(self, clear_cache=False):
        self.events.append(("close", self.seed, clear_cache))
        self.seed = None


class LiveInfoTask(FakeTask):
    instruction_info_from_live_reset = True


def generate_descriptions(task_name, episodes, max_descriptions, seed):
    (info,) = episodes
    text = f"{info['{A}']} {info['{a}']} {seed}"
    return [{"seen": [f"seen {text}"], "unseen": [f"unseen {text}"]}]


class ResetInstructionTest(unittest.TestCase):
    def make_env(self, task_class, instruction_type="seen", **args):
        tasks = []

        def make_task(task_name):
            task = task_class()
            tasks.append(task)
            return task

        task_factory = patch.object(vector_env, "class_decorator", side_effect=make_task)
        generator = patch.object(
            vector_env, "generate_episode_descriptions", side_effect=generate_descriptions
        )
        with task_factory:
            env = vector_env.SubEnv(
                0,
                "task",
                {"step_lim": 700, **args},
                env_seed=2,
                instruction_type=instruction_type,
                global_lock=threading.Lock(),
            )
            env.setup_task()
        generate = generator.start()
        self.addCleanup(generator.stop)
        return env, tasks, generate

    def test_prompt_follows_consecutive_live_resets(self):
        env, tasks, _ = self.make_env(LiveInfoTask, instruction="shared value")
        self.assertFalse(hasattr(env, "episode_info_list"))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(env.task.events, [])

        for seed, expected in [(2, "seen base2 left 2"), (3, "seen base3 right 3")]:
            with self.subTest(seed=seed):
                env.reset(seed)
                self.assertEqual(env.get_instruction(), expected)
                self.assertEqual(env.task.instruction, expected)
                self.assertEqual(
                    env.task.events[-2:], [("setup", seed, False, None), ("info", seed)]
                )
                self.assertEqual(env.args["instruction"], "shared value")

    def test_unstable_seed_uses_successful_trial_for_unseen_instruction(self):
        env, _, generate = self.make_env(LiveInfoTask, instruction_type="unseen")
        env.task.unstable_seeds.add(2)
        env.reset(2)

        self.assertEqual(
            env.task.events[1:],
            [
                ("setup", 2, False, None),
                ("close", 2, True),
                ("setup", 3, False, None),
                ("info", 3),
            ],
        )
        generate.assert_called_once_with(
            "task", [{"{A}": "base3", "{a}": "right"}], 1, 3
        )
        self.assertEqual(env.task.instruction, "unseen base3 right 3")
        self.assertEqual(env.get_instruction(), env.task.instruction)
        self.assertNotIn("instruction", env.args)

    def test_legacy_task_reads_metadata_only_on_temporary_instance(self):
        env, tasks, _ = self.make_env(FakeTask)
        temporary_task = tasks[-1]
        self.assertIsNot(temporary_task, env.task)
        self.assertEqual(
            temporary_task.events,
            [("setup", 2, True, None), ("info", 2), ("close", 2, False)],
        )

        env.reset(3)
        self.assertEqual(env.task.events[1:], [("setup", 3, False, "seen base2 left 3")])
        self.assertEqual(env.get_instruction(), "seen base2 left 3")

    def test_instruction_failure_closes_scene_and_discards_previous_prompt(self):
        for failing_stage, error in [
            ("metadata", RuntimeError("invalid metadata")),
            ("generator", SystemExit("missing description")),
        ]:
            with self.subTest(failing_stage=failing_stage):
                env, _, generate = self.make_env(LiveInfoTask)
                env.reset(2)
                with patch.object(env.task, "get_info", wraps=env.task.get_info) as get_info:
                    failing_call = get_info if failing_stage == "metadata" else generate
                    failing_call.side_effect = error
                    with self.assertRaises(type(error)):
                        env.reset(3)
                self.assertEqual(env.task.events[-1], ("close", 3, True))
                self.assertIsNone(env.get_instruction())
                self.assertIsNone(env.task.instruction)


if __name__ == "__main__":
    unittest.main()
