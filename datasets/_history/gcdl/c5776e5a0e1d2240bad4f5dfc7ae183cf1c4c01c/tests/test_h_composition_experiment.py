"""Hand witnesses and once-only custody; no selected DEV pilot measurements."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.generators import h_composition_experiment as h
from goal_conditioned_decision_data.pipelines.common import canonical


class HandWitnessTests(unittest.TestCase):
    def setUp(self):
        # Developmental seed outside the frozen confirmation range.
        self.g, self.x = h.make_bundles(111, 0)

    def test_three_register_noncommuting_hand_endpoints(self):
        expected = [{'a': 42, 'b': 2, 'c': 42}, {'a': 21, 'b': 4, 'c': 21},
                    {'a': 42, 'b': 2, 'c': 2}]
        for row, end in zip(self.g, expected):
            oracle, labels = h._dual(row['input'])
            self.assertEqual(labels['best_candidate_ids'], row['expected_best'])
            winner = next(c for c in oracle['candidates'] if c['id'] in row['expected_best'])
            self.assertEqual(winner['outcomes'][0]['terminal_state'], {'registers': end})
            self.assertEqual(winner['outcomes'][0]['cost'], 3)
            self.assertEqual(len(row['input']['candidates']), 11)

    def test_previously_degenerate_last_index_keeps_all_actions_functional(self):
        rows, _ = h.make_bundles(112, 31)
        for row in rows:
            initial = row['input']['initial_state']['registers']
            self.assertTrue(all(initial[k] != v for k, v in row['input']['goal']['values'].items()))
            oracle, labels = h._dual(row['input'])
            self.assertEqual(labels['best_candidate_ids'], row['expected_best'])
            winner = next(c for c in oracle['candidates'] if c['id'] in row['expected_best'])
            states = winner['outcomes'][0]['trajectory']
            self.assertTrue(all(a != b for a, b in zip(states, states[1:])))

    def test_constraint_conjunction_and_intermediate_violation(self):
        for row in self.x:
            oracle, labels = h._dual(row['input'])
            self.assertEqual(labels['best_candidate_ids'], row['expected_best'])
            self.assertEqual(len(row['input']['candidates']), 9)
            if row['variant'] in (1, 3):
                p = next(c for c in oracle['candidates'] if c['id'] == row['role_ids']['P'])
                self.assertEqual(p['outcomes'][0]['terminal_state']['registers']['a'], 10)
                self.assertEqual([v['step'] for v in p['outcomes'][0]['constraint_violations']], [1, 2])
                self.assertEqual(p['feasible_probability']['numerator'], 0)
        self.assertEqual(len(self.x[3]['expected_best']), 2)

    def test_every_goal_atom_is_necessary(self):
        for row in self.g + self.x:
            original = set(row['expected_best'])
            for omitted in row['input']['goal']['values']:
                task = deepcopy(row['input']); del task['goal']['values'][omitted]
                _, labels = h._dual(task)
                self.assertTrue(labels['best_candidate_ids'])
                self.assertFalse(original & set(labels['best_candidate_ids']))

    def test_ablated_tie_is_retained(self):
        task = deepcopy(self.x[0]['input']); del task['goal']['values']['a']
        _, labels = h._dual(task)
        self.assertEqual(labels['best_candidate_ids'], sorted(self.x[0]['role_ids'][k]
                         for k in ('F', 'partial_b_maximum')))

    def test_pairing_changes_only_goal_or_constraints(self):
        for rows, key in ((self.g, 'goal'), (self.x, 'constraints')):
            residuals = []
            for row in rows:
                task = deepcopy(row['input']); del task[key]
                residuals.append(canonical(task))
            self.assertEqual(len(set(residuals)), 1)
            self.assertEqual(len({canonical(r['input'][key]) for r in rows}), len(rows))

    def test_goal_blind_and_terminal_only_controls(self):
        for name, chosen in h._blind_choices(self.g[0]['input']).items():
            self.assertLessEqual(sum(chosen in r['expected_best'] for r in self.g), 1, name)
        correct = 0
        for row in self.x:
            oracle, _ = h._dual(row['input'])
            correct += bool(set(h._terminal_only_best(row['input'], oracle)) & set(row['expected_best']))
        self.assertEqual(correct, 2)

    def test_no_unknown_or_invalid_index(self):
        for seed, index in ((-1, 0), (True, 0), (1, -1), (1, 32), (1, True)):
            with self.assertRaises(ValueError):
                h.make_bundles(seed, index)


class FreezeCustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'configs').mkdir()
        self.original_spec = (h.ROOT / h.SPEC_PATH).read_bytes()
        (self.root / h.SPEC_PATH).write_bytes(self.original_spec)
        self.root_patch = patch.object(h, 'ROOT', self.root); self.root_patch.start()
        self.code = {'witness.py': 'a' * 64}
        self.code_patch = patch.object(h, 'source_hashes', return_value=self.code); self.code_patch.start()

    def tearDown(self):
        self.code_patch.stop(); self.root_patch.stop(); self.tmp.cleanup()

    def test_freeze_commits_inputs_without_oracle_measurement(self):
        with patch.object(h, '_dual', side_effect=AssertionError('oracle must not run')):
            receipt = h.freeze()
        self.assertEqual(len(receipt['input_commitments']), 32)
        self.assertEqual(len(receipt['input_commitments'][0]['goal_inputs']), 3)
        self.assertEqual(len(receipt['input_commitments'][0]['constraint_inputs']), 4)
        self.assertFalse(receipt['publication'])
        self.assertFalse((self.root / h.STARTED_PATH).exists())
        self.assertEqual(h.freeze(), receipt)

    def test_source_change_blocks_measurement_before_attempt(self):
        h.freeze(); self.code['witness.py'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'source/specification changed'):
            h.measure()
        self.assertFalse((self.root / h.STARTED_PATH).exists())

    def test_spec_change_blocks_measurement(self):
        h.freeze()
        spec = json.loads(self.original_spec); spec['interpretation'] += ' changed'
        (self.root / h.SPEC_PATH).write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'source/specification changed'):
            h.measure()

    def test_interruption_cannot_rerun_once_only_attempt(self):
        h.freeze()
        started = self.root / h.STARTED_PATH; started.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'no adaptive rerun'):
            h.measure()
        self.assertEqual(started.read_text(), '{}\n')

    def test_changed_frozen_input_is_failure_without_replacing_seeds(self):
        h.freeze()
        original = h.make_bundles
        def changed(seed, index):
            g, x = original(seed, index)
            g[0]['input']['initial_state']['registers']['a'] += 1
            return g, x
        with patch.object(h, 'make_bundles', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'input commitment mismatch'):
                h.measure()
        self.assertTrue((self.root / h.STARTED_PATH).exists())
        with self.assertRaisesRegex(ValueError, 'no adaptive rerun'):
            h.measure()

    def test_training_or_publication_cannot_be_enabled(self):
        for key in ('publication', 'model_training', 'replacement_seeds'):
            spec = json.loads(self.original_spec); spec[key] = True
            (self.root / h.SPEC_PATH).write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'unsupported frozen'):
                h.freeze()
            self.assertFalse((self.root / h.FREEZE_PATH).exists())


if __name__ == '__main__':
    unittest.main()
