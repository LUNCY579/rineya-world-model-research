"""Executable v2 construction and namespace regressions on fixture seeds only."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.generators import build as v1, version2 as v2
from goal_conditioned_decision_data.auditors.integrity import (audit_record, recursive_input_check,
                                                            semantic_fingerprint, validate_auxiliary)
from goal_conditioned_decision_data.environments.core import Blocksworld
from goal_conditioned_decision_data.oracles.primary import evaluate as primary
from goal_conditioned_decision_data.oracles.independent import evaluate as independent
from goal_conditioned_decision_data.pipelines.common import canonical, digest, file_hash


class Version2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = v2.curriculum()
        cls.group = v2.generate_group(703, split='dev')

    def test_declared_pools_are_disjoint_same_grammar_and_fixed_profile(self):
        self.assertEqual(digest(self.cfg), v2.CONFIG_HASH)
        self.assertEqual(file_hash(v2.ROOT / 'configs/curriculum-v2.json'), '90005d13ea7e0178bc0b96ee43974cd777c5a69b65af50e55bf7fd33e48117cd')
        tokens = [word for split in self.cfg['lexicons'].values() for pool in split.values() for word in pool]
        self.assertEqual(len(tokens), 384)
        self.assertEqual(len(set(tokens)), 384)
        self.assertTrue(all(re.fullmatch('[a-z]{8}', word) for word in tokens))
        for split in self.cfg['lexicons'].values():
            self.assertEqual(set(split), set(v2.NAMESPACES))
            self.assertTrue(all(len(pool) == 16 for pool in split.values()))
        bad = deepcopy(self.cfg); bad['c_bundle']['max_proposals'] = 25
        with self.assertRaisesRegex(ValueError, 'profile'):
            v2.generate_goal_triplet(704, config=bad)

    def test_fixed_six_candidate_triplet_changes_only_complete_noninitial_goal(self):
        records, diagnostics = v2.generate_goal_triplet(704)
        self.assertEqual(diagnostics['status'], 'VALID')
        self.assertLessEqual(diagnostics['proposals_attempted'], 24)
        self.assertEqual(len(records), 3)
        self.assertEqual((records, diagnostics), v2.generate_goal_triplet(704))
        self.assertEqual(len({row['id'] for row in records}), 3)
        self.assertEqual(len({row['metadata']['goal_bundle_id'] for row in records}), 1)
        baseline = deepcopy(records[0]['input']); baseline.pop('goal')
        for record in records:
            task = record['input']
            other = deepcopy(task); other.pop('goal')
            self.assertEqual(other, baseline)
            self.assertEqual(len(task['candidates']), 6)
            self.assertEqual(len(task['goal']['on']), 4)
            self.assertNotEqual(task['goal'], task['initial_state'])
            self.assertEqual(v2.bundle_identity(task), record['metadata']['goal_bundle_id'])
            self.assertEqual(record['provenance']['generator_version'], 'generator-2.0')
            self.assertEqual(record['provenance']['config_version'], 'curriculum-2.0')
            self.assertEqual(v1.record_id(record), record['id'])
            for candidate in task['candidates']:
                self.assertRegex(candidate['id'], r'^c_[0-9a-f]{24}$')
                self.assertLessEqual(len(candidate['plan']), 4)
            self.assertEqual(primary(task), independent(task))
            audit_record(record)
            recursive_input_check(task)
        best = [set(record['targets']['decision']['best_candidate_ids']) for record in records]
        for left, right in combinations(best, 2):
            self.assertTrue(left); self.assertTrue(right); self.assertFalse(left & right)
        self.assertEqual(len({canonical(record['input']['goal']) for record in records}), 3)

    def test_exact_prefix_reverse_and_empty_candidates_and_equal_path_costs(self):
        records, _ = v2.generate_goal_triplet(705)
        record = records[0]
        roles = record['metadata']['candidate_roles']
        by_role = {roles[candidate['id']]: candidate['plan'] for candidate in record['input']['candidates']}
        self.assertEqual(by_role['prefix_0'], by_role['path_0'][:-1])
        self.assertEqual(by_role['reverse_0'], list(reversed(by_role['path_0'])))
        self.assertEqual(by_role['empty'], [])
        self.assertEqual([len(by_role[role]) for role in ('path_0', 'path_1', 'path_2')], [4, 4, 4])
        self.assertTrue(all(action['op'] == 'move' for role in ('path_0', 'path_1', 'path_2') for action in by_role[role]))
        cost = record['input']['rules']['action_costs']['move']
        outcomes = record['targets']['oracle']['candidates']
        self.assertTrue(all(candidate['outcomes'][0]['cost'] == 4 * cost for candidate in outcomes if roles[candidate['id']].startswith('path_')))

    def test_caps_and_inadequate_parent_are_explicit_not_replacement_sampling(self):
        initial = next(row for row in v1.generate_group(706, config=self.cfg, split='dev') if row['family'] == 'C')['input']['initial_state']
        with patch.object(v2, '_path_proposal', return_value=([], deepcopy(initial))) as proposal:
            records, diagnostics = v2.generate_goal_triplet(706)
            self.assertEqual(records, [])
            self.assertEqual(proposal.call_count, 24)
            self.assertEqual(diagnostics['status'], 'EXCLUDED')
            self.assertEqual(diagnostics['reason'], 'insufficient_distinct_noninitial_endpoints')
            self.assertEqual(diagnostics['proposals_attempted'], 24)
        with patch.object(v2, '_path_proposal', return_value=([], deepcopy(initial))):
            with self.assertRaises(v2.GenerationError) as caught:
                v2.generate_group(706, split='dev')
            self.assertEqual(caught.exception.diagnostics['reason'], 'insufficient_distinct_noninitial_endpoints')

    def test_cheaper_prefix_and_complete_tie_sets_are_retained(self):
        source = next(row for row in v1.generate_group(707, config=self.cfg, split='dev') if row['family'] == 'C')
        names = list(source['input']['initial_state']['on'])
        a, b, c, d = names
        source['input']['initial_state'] = {'on': {name: 'table' for name in names}}
        def move(block, destination):
            return {'op': 'move', 'block': block, 'destination': destination}
        paths = [[move(a,b), move(a,c), move(a,b), move(c,d)],
                 [move(a,c), move(a,d), move(a,c), move(a,b)],
                 [move(b,c), move(b,d), move(b,c), move(b,a)]]
        proposals = []
        for path in paths:
            state = deepcopy(source['input']['initial_state'])
            for action in path:
                state, error = Blocksworld().transition(state, action, source['input']['rules'])
                self.assertIsNone(error)
            proposals.append((path, state))
        with patch.object(v2, '_path_proposal', side_effect=proposals):
            records, _ = v2._goal_triplet(source, self.cfg, 707, 'dev')
        roles = records[0]['metadata']['candidate_roles']
        best_roles = [{roles[name] for name in row['targets']['decision']['best_candidate_ids']} for row in records]
        self.assertEqual(best_roles[0], {'path_0', 'reverse_0'})
        self.assertEqual(best_roles[1], {'prefix_0'})
        self.assertEqual(best_roles[2], {'path_2'})
        for record in records:
            audit_record(record)

    def test_oracle_disagreement_is_a_failure_with_evidence(self):
        with patch.object(v2, 'independent', return_value={'deliberate_fixture_disagreement': True}):
            with self.assertRaises(v2.GenerationError) as caught:
                v2.generate_goal_triplet(708)
            self.assertEqual(caught.exception.diagnostics['status'], 'FAIL')
            self.assertEqual(caught.exception.diagnostics['reason'], 'oracle_disagreement')
            self.assertIn('primary', caught.exception.diagnostics)
            self.assertIn('independent', caught.exception.diagnostics)

    def test_all_core_families_have_reserved_maps_and_rechecked_targets(self):
        self.assertEqual(set(row['family'] for row in self.group), set('ABCDEFGH'))
        self.assertEqual(sum(row['family'] == 'C' for row in self.group), 3)
        for row in self.group:
            self.assertEqual(row['metadata']['vocabulary']['stratum'], 'reserved_dev')
            restored = v2.restore_source_names(row)
            self.assertEqual(semantic_fingerprint(row), v1.semantic_fingerprint({'input': restored}))
            self.assertEqual(row['targets']['oracle'], primary(row['input']))
            self.assertEqual(primary(row['input']), independent(row['input']))
            validate_auxiliary(row)
            audit_record(row)

    def test_a_goal_maps_and_c_maps_remain_shared_but_d_surfaces_distinct(self):
        a = [row for row in self.group if row['family'] == 'A']
        c = [row for row in self.group if row['family'] == 'C']
        for family in (a, c):
            self.assertEqual(len({canonical(row['metadata']['vocabulary']['maps']) for row in family}), 1)
            self.assertEqual(len({canonical(row['input']['candidates']) for row in family}), 1)
        for kind in ('entity_rename', 'status_vocabulary'):
            surface = next(row for row in self.group if row['metadata']['perturbation'].get('kind') == kind)
            self.assertNotEqual(surface['input'], a[0]['input'])
            self.assertEqual(semantic_fingerprint(surface), semantic_fingerprint(a[0]))
            self.assertEqual({item['id']: item for item in surface['targets']['decision']['candidates']},
                             {item['id']: item for item in a[0]['targets']['decision']['candidates']})

    def test_vocabulary_pair_is_total_unknown_refs_preserved_and_controls_separate(self):
        records, diagnostic = v2.generate_vocabulary_pair(709)
        self.assertEqual(diagnostic['strata'], ['familiar_control', 'reserved_dev'])
        self.assertEqual((records, diagnostic), v2.generate_vocabulary_pair(709))
        self.assertEqual(v2.restore_source_names(records[0]), v2.restore_source_names(records[1]))
        self.assertEqual(semantic_fingerprint(records[0]), semantic_fingerprint(records[1]))
        self.assertEqual(records[0]['targets']['decision'], records[1]['targets']['decision'])
        for row, pool_split in zip(records, ('train', 'dev')):
            maps = row['metadata']['vocabulary']['maps']
            for namespace, mapping in maps.items():
                self.assertTrue(set(mapping.values()) <= set(self.cfg['lexicons'][pool_split][namespace]))
            invalid = maps['entries']['absent']
            self.assertNotIn(invalid, row['input']['initial_state']['entries'])
            invalid_outcomes = [candidate['outcomes'][0] for candidate in row['targets']['oracle']['candidates'] if candidate['outcomes'][0]['failure_step'] is not None]
            self.assertTrue(any(outcome['steps'][0]['error'] == 'unknown_entry' for outcome in invalid_outcomes))
            audit_record(row)
        with self.assertRaisesRegex(ValueError, 'DEV only'):
            v2.generate_vocabulary_pair(709, split='train')

    def test_mapping_tamper_partial_unknown_map_pool_and_stratum_fail_closed(self):
        row = v2.generate_vocabulary_pair(710)[0][1]
        for edit in ('missing_unknown', 'out_of_pool', 'wrong_stratum', 'noninjective', 'source_hash'):
            bad = deepcopy(row); vocabulary = bad['metadata']['vocabulary']
            if edit == 'missing_unknown':
                vocabulary['maps']['entries'].pop('absent')
            elif edit == 'out_of_pool':
                vocabulary['maps']['entries']['absent'] = 'outsidepool'
            elif edit == 'wrong_stratum':
                vocabulary['stratum'] = 'reserved_train'
            elif edit == 'source_hash':
                vocabulary['source_namespace_task_sha256'] = '0' * 64
            else:
                vocabulary['maps']['entries']['absent'] = vocabulary['maps']['entries']['e']
            bad['id'] = v1.record_id(bad)
            with self.assertRaises(ValueError):
                v2.restore_source_names(bad)

    def test_observation_paths_status_roles_registers_and_table_are_explicit(self):
        missing = next(row for row in self.group if 'observations' in row['input'])
        entry = missing['metadata']['vocabulary']['maps']['entries']['e']
        self.assertEqual(missing['input']['observations']['missing'][0], 'entries.' + entry + '.amount')
        register = next(row for row in self.group if row['input']['environment'] == 'register')
        mapping = register['metadata']['vocabulary']['maps']['registers']
        self.assertIn('absent', mapping)
        self.assertNotIn(mapping['absent'], register['input']['initial_state']['registers'])
        blocks = next(row for row in self.group if row['family'] == 'C')
        maps = blocks['metadata']['vocabulary']['maps']
        self.assertIn(blocks['input']['rules']['table'], self.cfg['lexicons']['dev']['table'])
        self.assertNotIn(blocks['input']['rules']['table'], set(maps['blocks'].values()))
        states = [v2.generate_vocabulary_pair(seed)[0][1]['input']['rules']['statuses'] for seed in (711,712)]
        self.assertNotEqual(states[0], states[1])
        self.assertTrue(all(set(state) == {'pending', 'settled', 'void'} for state in states))

    def test_explicit_block_table_register_copy_swap_and_unknown_references(self):
        blocks = v2.restore_source_names(next(row for row in self.group if row['family'] == 'C'))
        names = list(blocks['initial_state']['on']); table = blocks['rules']['table']
        blocks['constraints'] = {'forbidden_on': [[names[0], table]], 'protected_blocks': [names[1]]}
        blocks['candidates'].append({'id': 'invalid_fixture', 'plan': [
            {'op': 'move', 'block': table, 'destination': 'missing_block'}]})
        maps = v2._maps_for(blocks, 'fixture_parent', 'dev', 'base', self.cfg)
        mapped = v2.rename_task(blocks, maps)
        self.assertEqual(mapped['rules']['table'], maps['table'][table])
        self.assertEqual(mapped['constraints']['forbidden_on'], [[maps['blocks'][names[0]], maps['table'][table]]])
        self.assertEqual(mapped['constraints']['protected_blocks'], [maps['blocks'][names[1]]])
        invalid = mapped['candidates'][-1]['plan'][0]
        self.assertEqual(invalid['block'], maps['blocks'][table])
        self.assertNotEqual(invalid['block'], mapped['rules']['table'])
        self.assertEqual(invalid['destination'], maps['blocks']['missing_block'])
        self.assertEqual(invalid['op'], 'move')
        inverse = {namespace: {new: old for old, new in mapping.items()} for namespace, mapping in maps.items()}
        self.assertEqual(v2.rename_task(mapped, inverse), blocks)
        register = v2.restore_source_names(next(row for row in self.group if row['input']['environment'] == 'register'))
        names = list(register['initial_state']['registers'])
        register['constraints'] = {'minimum_values': {names[0]: -10000}, 'maximum_values': {names[1]: 10000}, 'protected_registers': [names[0]]}
        register['candidates'] = [{'id': 'register_fixture', 'plan': [
            {'op': 'copy', 'source': names[0], 'destination': names[1]},
            {'op': 'swap', 'left': names[0], 'right': names[1]},
            {'op': 'add', 'register': 'missing_register', 'value': 1}]}]
        maps = v2._maps_for(register, 'fixture_parent', 'dev', 'base', self.cfg)
        mapped = v2.rename_task(register, maps)
        self.assertEqual(mapped['candidates'][0]['plan'][0], {'op': 'copy', 'source': maps['registers'][names[0]], 'destination': maps['registers'][names[1]]})
        self.assertEqual(mapped['candidates'][0]['plan'][1], {'op': 'swap', 'left': maps['registers'][names[0]], 'right': maps['registers'][names[1]]})
        self.assertNotIn(mapped['candidates'][0]['plan'][2]['register'], mapped['initial_state']['registers'])
        self.assertEqual(primary(mapped), independent(mapped))
        self.assertEqual(primary(mapped)['candidates'][0]['outcomes'][0]['steps'][-1]['error'], 'unknown_register')

    def test_v1_sources_and_generation_remain_unchanged(self):
        self.assertEqual(file_hash(v2.ROOT / 'generators/build.py'), '78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575')
        self.assertEqual(file_hash(v2.ROOT / 'configs/curriculum.json'), 'c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780')
        before = v1.generate_group(713, split='dev')
        v2.generate_group(713, split='dev')
        self.assertEqual(before, v1.generate_group(713, split='dev'))
        self.assertTrue(all(row['provenance']['generator_version'] == 'generator-1.0' for row in before))


if __name__ == '__main__':
    unittest.main()
