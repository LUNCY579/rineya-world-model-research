"""Independent preregistration challenges; selected confirmation seeds unrun.

Semantic tests use developmental seed 113 only. Completed-result custody tests
use synthetic metadata and cached seed-113 inputs in temporary directories;
selected-seed construction, oracles and baseline generation never execute.
No production/profile/runtime/sealed/remote changes.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.generators import h_composition_experiment as h
from goal_conditioned_decision_data.pipelines.common import canonical, digest, file_hash


class IndependentHCompositionMechanisms(unittest.TestCase):
    def setUp(self):
        # Distinct from root's developmental witnesses and every reserved seed.
        self.g, self.x = h.make_bundles(113, 17)

    def roles(self, row, identifiers):
        inverse = {identifier: role for role, identifier in row['role_ids'].items()}
        return frozenset(inverse[identifier] for identifier in identifiers)

    def test_exact_constraint_rankings_include_infeasible_tiers_and_ties(self):
        expected = (
            (('P',), ('R',), ('Q', 'Q_reverse'), ('partial_a',),
             ('F', 'partial_b_maximum'), ('partial_b_conjunction',), ('N',)),
            (('R',), ('Q', 'Q_reverse'), ('partial_a',), ('partial_b_maximum',),
             ('partial_b_conjunction',), ('N',), ('P',), ('F',)),
            (('P',), ('Q', 'Q_reverse'), ('partial_a',), ('F',),
             ('partial_b_conjunction',), ('N',), ('R',), ('partial_b_maximum',)),
            (('Q', 'Q_reverse'), ('partial_a',), ('partial_b_conjunction',),
             ('N',), ('P',), ('R',), ('F', 'partial_b_maximum')),
        )
        for row, tiers in zip(self.x, expected):
            with self.subTest(variant=row['variant']):
                _, labels = h._dual(row['input'])
                self.assertEqual([self.roles(row, item['candidate_ids']) for item in labels['ranking']],
                                 [frozenset(tier) for tier in tiers])
                self.assertEqual(self.roles(row, labels['best_candidate_ids']), frozenset(tiers[0]))

    def test_each_constraint_is_indispensable_under_the_conjunction(self):
        both = self.x[3]
        _, full = h._dual(both['input'])
        self.assertEqual(self.roles(both, full['best_candidate_ids']), frozenset(('Q', 'Q_reverse')))
        for removed, winner in (('maximum_values', 'P'), ('protected_registers', 'R')):
            task = deepcopy(both['input']); del task['constraints'][removed]
            _, labels = h._dual(task)
            self.assertEqual(self.roles(both, labels['best_candidate_ids']), frozenset((winner,)))

    def test_initial_constraints_hold_but_intermediate_violation_cannot_be_repaired(self):
        for row in self.x:
            oracle, _ = h._dual(row['input'])
            for candidate in oracle['candidates']:
                self.assertTrue(candidate['outcomes'][0]['legal'])
                self.assertFalse(any(v['step'] == 0 for v in candidate['outcomes'][0]['constraint_violations']))
        for variant in (1, 3):
            row = self.x[variant]; oracle, _ = h._dual(row['input'])
            p = next(c for c in oracle['candidates'] if c['id'] == row['role_ids']['P'])
            self.assertEqual([v['step'] for v in p['outcomes'][0]['constraint_violations']], [1, 2])
            self.assertEqual(p['outcomes'][0]['terminal_state']['registers']['a'], row['input']['goal']['values']['a'])
            self.assertEqual(p['feasible_probability'], {'numerator': 0, 'denominator': 1})
            self.assertEqual(p['outcomes'][0]['cost'], 3)

    def test_atom_ablation_exact_winners_and_negative_control_tie(self):
        drop_a = (('F', 'partial_b_maximum'), ('partial_b_maximum',),
                  ('F',), ('partial_b_conjunction',))
        for row, a_roles in zip(self.x, drop_a):
            for omitted, expected in (('a', a_roles), ('b', ('partial_a',))):
                with self.subTest(variant=row['variant'], omitted=omitted):
                    task = deepcopy(row['input']); del task['goal']['values'][omitted]
                    _, labels = h._dual(task)
                    self.assertEqual(self.roles(row, labels['best_candidate_ids']), frozenset(expected))
                    self.assertFalse(set(labels['best_candidate_ids']) & set(row['expected_best']))
        for row in self.g:
            for omitted in row['input']['goal']['values']:
                task = deepcopy(row['input']); del task['goal']['values'][omitted]
                oracle, labels = h._dual(task)
                self.assertEqual(len(labels['best_candidate_ids']), 1)
                self.assertFalse(set(labels['best_candidate_ids']) & set(row['expected_best']))
                winner = next(c for c in oracle['candidates'] if c['id'] in labels['best_candidate_ids'])
                self.assertEqual(winner['outcomes'][0]['utility'], [1, 2, -2])

    def test_noncommuting_order_and_each_winning_action_are_necessary(self):
        initial = self.g[0]['input']['initial_state']['registers']
        for row in self.g:
            task = row['input']; oracle, labels = h._dual(task)
            self.assertTrue(all(initial[k] != v for k, v in task['goal']['values'].items()))
            winner = next(c for c in task['candidates'] if c['id'] in labels['best_candidate_ids'])
            self.assertEqual(sorted(a['op'] for a in winner['plan']), ['copy', 'multiply', 'swap'])
            self.assertEqual(len(winner['plan']), 3)
            for deleted in range(3):
                removed = deepcopy(task)
                removed['candidates'] = [{'id': 'developmental-action-deletion',
                                         'plan': [deepcopy(a) for i, a in enumerate(winner['plan']) if i != deleted]}]
                removal_oracle, _ = h._dual(removed)
                self.assertFalse(removal_oracle['candidates'][0]['outcomes'][0]['goal_satisfied'])
        ids = self.g[0]['role_ids']
        plans = {c['id']: c['plan'] for c in self.g[0]['input']['candidates']}
        self.assertEqual(plans[ids['P']][0], plans[ids['Q']][1])
        self.assertEqual(plans[ids['P']][1], plans[ids['Q']][0])
        self.assertEqual(plans[ids['P']][2], plans[ids['Q']][2])

    def test_partial_plan_deduplication_and_pairing_exclude_expected_metadata(self):
        for rows, axis, count in ((self.g, 'goal', 11), (self.x, 'constraints', 9)):
            residuals = []
            for row in rows:
                task = deepcopy(row['input']); task.pop(axis)
                residuals.append(canonical(task))
                self.assertEqual(len(task['candidates']), count)
                self.assertEqual(len({canonical(c['plan']) for c in task['candidates']}), count)
                h.recursive_input_check(task)
                self.assertNotIn('expected_best', task)
                self.assertNotIn('role_ids', task)
            self.assertEqual(len(set(residuals)), 1)
        partial = [c for c in self.g[0]['input']['candidates'] if len(c['plan']) == 2]
        self.assertEqual(len(partial), 8)
        self.assertTrue(all(a['op'] == 'set' for c in partial for a in c['plan']))

    def test_terminal_only_selector_is_wrong_on_both_maximum_variants(self):
        for row in self.x:
            oracle, labels = h._dual(row['input'])
            guessed = h._terminal_only_best(row['input'], oracle)
            self.assertEqual(self.roles(row, guessed), frozenset(('P',)))
            self.assertEqual(guessed == labels['best_candidate_ids'], row['variant'] in (0, 2))


class IndependentHCompositionCustody(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='h-preregistration-custody-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve() / 'goal_conditioned_decision_data'
        (self.root / 'configs').mkdir(parents=True)
        self.spec = json.loads((h.ROOT / h.SPEC_PATH).read_text())
        (self.root / h.SPEC_PATH).write_text(canonical(self.spec) + '\n')
        self.code = {'independent_metadata_fixture.py': 'a' * 64}
        development_inputs = h.make_bundles(113, 17)
        root_patch = patch.object(h, 'ROOT', self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)
        source_patch = patch.object(h, 'source_hashes', return_value=self.code)
        source_patch.start(); self.addCleanup(source_patch.stop)
        constructor = patch.object(h, 'make_bundles', side_effect=lambda seed, index: deepcopy(development_inputs))
        constructor.start(); self.addCleanup(constructor.stop)
        for name in ('_dual', 'generate_group'):
            hook = patch.object(h, name, side_effect=AssertionError('custody replay must not construct/execute selected tasks'))
            hook.start(); self.addCleanup(hook.stop)
        frozen = {
            'version': 'h-composition-experiment-freeze-1', 'specification': self.spec,
            'specification_sha256': file_hash(self.root / h.SPEC_PATH),
            'code_hashes': self.code, 'source_digest': digest(self.code),
            'input_commitments': [{'seed': seed, 'goal_inputs': ['1' * 64] * 3,
                                   'constraint_inputs': ['2' * 64] * 4}
                                  for seed in range(430000, 430032)],
            'publication': False, 'training_launched': False,
        }
        self.write(h.FREEZE_PATH, frozen)
        self.freeze_hash = file_hash(self.root / h.FREEZE_PATH)
        self.started = {'version': 'h-composition-experiment-started-1',
                        'freeze_sha256': self.freeze_hash, 'source_digest': digest(self.code),
                        'started_at_ns': 123}
        self.write(h.STARTED_PATH, self.started)
        self.result = {
            'version': 'h-composition-controlled-result-1', 'prototype_version': h.VERSION,
            'verdict': 'PASS', 'source_digest': digest(self.code),
            'freeze_sha256': self.freeze_hash, 'started_sha256': file_hash(self.root / h.STARTED_PATH),
            'counts': deepcopy(h.RESULT_COUNTS), 'expected_counts': deepcopy(h.RESULT_COUNTS),
            'goal_blind_best_membership': {key: 0 for key in h._blind_choices(development_inputs[0][0]['input'])},
            'supplied_operations': sorted(h.OPERATIONS), 'oracle_or_label_mismatches': 0,
            'reproduction_mismatches': 0, 'canonical_records_published': 0, 'training_launched': False,
            # These arrays satisfy superficial counts but contain no evidence.
            'parent_receipts': [{'seed': seed, 'goal': [{}, {}, {}], 'constraints': [{}, {}, {}, {}]}
                                for seed in range(430000, 430032)],
        }

    def write(self, relative, value):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical(value) + '\n')

    def write_result(self):
        self.result['integrity_sha256'] = digest({key: value for key, value in self.result.items()
                                                  if key != 'integrity_sha256'})
        self.write(h.RESULT_PATH, self.result)

    def test_fabricated_aggregate_pass_with_empty_parent_evidence_is_refused(self):
        self.write_result()
        with self.assertRaisesRegex(ValueError, 'custody|evidence|receipt|commitment'):
            h.measure()

    def test_rehashed_wrong_started_body_cannot_authenticate_a_completed_attempt(self):
        self.started['source_digest'] = 'f' * 64
        self.write(h.STARTED_PATH, self.started)
        self.result['started_sha256'] = file_hash(self.root / h.STARTED_PATH)
        self.write_result()
        with self.assertRaisesRegex(ValueError, 'start|custody'):
            h.measure()

    def test_completed_result_reader_enforces_declared_byte_bound(self):
        self.result['oversized_untrusted_field'] = 'x' * self.spec['max_result_bytes']
        self.write_result()
        self.assertGreater((self.root / h.RESULT_PATH).stat().st_size, self.spec['max_result_bytes'])
        with self.assertRaisesRegex(ValueError, 'bound|byte|MiB|size'):
            h.measure()


if __name__ == '__main__':
    unittest.main()
