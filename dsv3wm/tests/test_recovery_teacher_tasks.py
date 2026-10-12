import copy
import importlib.util
import random
import unittest
from pathlib import Path

from dsv3wm.data import make_record
from dsv3wm.domains import rules_for
from dsv3wm.recovery_data import _task

spec = importlib.util.spec_from_file_location('recovery_teacher_tasks', Path(__file__).parents[1] / 'scripts/build_recovery_teacher_tasks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryTeacherTaskTests(unittest.TestCase):
    def sources(self):
        train = [make_record('register', {'x': x}, ['add1'] * 4,
                             rules_for('register', limit=40), split='train', seed=1)
                 for x in range(-16, 16)]
        for row in train:
            row['provenance']['cohort'] = 'recovery_train'
        rng = random.Random(7)
        original = [_task(row, rng, 2, 4)[0] for row in train[:16]]
        return train, original

    def test_fixed_originals_and_next_parents_are_preserved(self):
        train, original = self.sources()
        before = copy.deepcopy((train, original))
        tasks = module.build(train, original)
        self.assertEqual(tasks[:16], original)
        self.assertEqual([task['parent_world_id'] for task in tasks], [row['parent_world_id'] for row in train])
        self.assertEqual(len({task['parent_world_id'] for task in tasks}), 32)
        self.assertEqual((train, original), before)
        self.assertEqual(tasks, module.build(train, original))
        for task in tasks:
            self.assertNotIn('labels', task)
            self.assertNotIn('optimal_moves', task)
            self.assertNotIn('plan', task)

    def test_nontraining_and_forged_source_are_rejected(self):
        train, original = self.sources()
        train[31]['split'] = 'dev'
        with self.assertRaises(PermissionError):
            module.build(train, original)
        train, original = self.sources()
        original[0]['s0']['x'] += 1
        with self.assertRaises(ValueError):
            module.build(train, original)


if __name__ == '__main__':
    unittest.main()
