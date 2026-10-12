"""Independent adversarial generator-v2/profile review.

Expectations below were specified independently of the v2 implementation.
Production access is limited to nonsealed accepted rows and their receipts.
No test publishes real data, reads sealed labels, or treats executable checks
as release approval. This reviewer authors only this file and its report.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors.integrity import audit_record, validate_schema
from goal_conditioned_decision_data.generators import build
from goal_conditioned_decision_data.oracles.primary import evaluate as primary
from goal_conditioned_decision_data.oracles.independent import evaluate as independent
from goal_conditioned_decision_data.pipelines.common import canonical, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = Path('/Users/chengli/.codex/gcdl001-host/worktree/goal_conditioned_decision_data')
V1_CONFIG_CANONICAL = 'd98651723bc9469105627dc55d78f0217eea67b98428020a5b1552452b4ca2f2'
V1_CONFIG_BYTES = 'c16450547db433d008001e4fd182978697c5b7e8029a2956e9f21145054c3780'
V1_GENERATOR_BYTES = '78387f73de0bd8cdda4cbef446d9e735edfea175badb1575bb567fb087e4a575'
PRODUCTION_ACCEPTED_COLLECTION = '45cb9526e674ec8d2a59c9fa40e219890f40dc3961b48198fd3c9c9f8d294d02'

# These commitments were captured from the already approved read-only v1
# implementation before this reviewer imported any v2 module.
V1_OUTPUTS = {
    (77, 'train'): (30, '75c798dd6eb085ab7b934349ba8fd969f26dd1d70b058e1c33c74367a9b754b5'),
    (78, 'dev'): (27, 'dfae05a3e4d3c198e2e7d0e210b58adc6d53b8a9d894ca410047e01486c1a941'),
    (79, 'ood'): (29, '123a74496173fe7bc8b412b1b3700dc34e01d06800e2aa97c3b1fbf340e5611e'),
    (1000000, 'train'): (30, '5802d57dfb63cf0b752315c0fa1aae12905eed4acf3c7fdbd34deaec0c4ed054'),
    (1000003, 'dev'): (27, '374a042967fd8faab87d7f97c66d60a71a4d87c59a83a9a7c11bfe6f4ff6005b'),
    (1000004, 'ood'): (29, 'a1dcac1f25044cfdb2ae64d3545002a268c058d286e74f99671c5fd1a61d4109'),
}
V1_RENDERED = {
    (77, 'train', 'A', 'pointwise'): 'c116f22d9bd8f3468d7490542ee4ab51278511085b1583500dd268671c223b65',
    (77, 'train', 'A', 'pairwise'): 'fa58a369b5cc942353f1b138ce46b95ff7929b980e4b06fbf8d40bd2646ec36f',
    (77, 'train', 'A', 'listwise'): 'a018346bc7ad5131250895c455c1d4c29d1ba12fbbd61913f0ba401feba96b50',
    (1000003, 'dev', 'C', 'pointwise'): '3b71fb74a3428f6b11394beafdb336a75470c52221fed0481e659433f2847bc5',
    (1000003, 'dev', 'C', 'pairwise'): 'cb0c0e22b5ae7108e6b4bf8e9364e893c64610618bb8e8605e718effaeed6abc',
    (1000003, 'dev', 'C', 'listwise'): 'a8724ece36f10b3fd1c75ffe74abebb7b5780ae6984c32345210e887bb795be7',
    (1000004, 'ood', 'C', 'pointwise'): '5518bd8ad536f063396371eb795bda523abab1eaccc4c3bdb5b06e13953fe8c5',
    (1000004, 'ood', 'C', 'pairwise'): '18542d73299e2f6cdc1c187fa7baaf743f10df246467c56c4353f8d8342236cf',
    (1000004, 'ood', 'C', 'listwise'): 'c1422d28dd967acf2a1b02d49ee4c5e4be5b0d3b52e3e33053edcf85e3102cd4',
}


def exact(value):
    return Fraction(value['numerator'], value['denominator']) if isinstance(value, dict) else Fraction(value)


def manual_ledger():
    # A +3 pending credit at balance 10: settling gives 13 at cost 1;
    # settling then voiding restores 10 at total cost 3. Empty costs zero.
    return {
        'environment': 'ledger',
        'initial_state': {'accounts': {'oak': 10, 'pine': 4},
                          'entries': {'bill': {'account': 'oak', 'amount': 3, 'status': 'x_wait'}}},
        'rules': {'statuses': {'pending': 'x_wait', 'settled': 'x_done', 'void': 'x_void'},
                  'void_effect': 'reverse', 'allow_overdraft': False,
                  'balance_bounds': [-100, 100],
                  'action_costs': {'settle': 1, 'void': 2, 'transfer': 1, 'noop': 0}},
        'goal': {'balances': {'oak': 13}, 'statuses': {'bill': 'x_done'}},
        'constraints': {'minimum_balances': {'oak': 0}, 'forbidden_statuses': ['x_void']},
        'candidates': [
            {'id': 'c_000000000000000000000001', 'plan': [{'op': 'settle', 'entry': 'bill'}]},
            {'id': 'c_000000000000000000000002', 'plan': [{'op': 'settle', 'entry': 'bill'}, {'op': 'void', 'entry': 'bill'}]},
            {'id': 'c_000000000000000000000003', 'plan': []}],
        'utility_definition': {
            'comparison': 'lexicographic',
            'components': ['expected_feasibility', 'expected_goal_atoms_satisfied', 'negative_expected_successful_action_cost'],
            'feasibility': 'legal execution and no constraint violations at initial or intermediate states',
            'all_infeasible': 'abstain; no best candidate'},
    }


def without_goal(task):
    task = deepcopy(task)
    task.pop('goal')
    return task


def independent_forest(plan, start, table, max_height):
    """Small hand-defined clear-block executor; never calls production moves."""
    on = deepcopy(start)
    for action in plan:
        if action['op'] == 'noop':
            continue
        if action['op'] != 'move':
            return None
        source, destination = action['block'], action['destination']
        if (source not in on or destination not in {*on, table}
                or source == destination or on[source] == destination
                or source in on.values()
                or (destination != table and destination in on.values())):
            return None
        height, cursor = 1, destination
        while cursor != table:
            height += 1
            cursor = on[cursor]
        if height > max_height:
            return None
        on[source] = destination
    return on


def independent_plan(task, plan):
    """Manually calculate legal prefix, move/noop cost, and terminal forest."""
    current = deepcopy(task['initial_state']['on'])
    spent = 0
    for action in plan:
        following = independent_forest([action], current, task['rules']['table'], task['rules']['max_height'])
        if following is None:
            return current, False, spent
        current = following
        spent += task['rules']['action_costs'][action['op']]
    return current, True, spent


def candidate_relationship_selector(task):
    """Fixed goal-blind prefix/reverse selector, with opaque-ID tie break."""
    candidates = task['candidates']
    qualifiers = [c['id'] for c in candidates if len(c['plan']) >= 2
                  and any(x['plan'] == c['plan'][:-1] for x in candidates)
                  and any(x['plan'] == list(reversed(c['plan'])) for x in candidates)]
    return min(qualifiers) if qualifiers else min(c['id'] for c in candidates)


class IndependentV1Compatibility(unittest.TestCase):
    def test_v1_generator_and_default_curriculum_remain_exact_bytes(self):
        self.assertEqual(file_hash(ROOT / 'generators/build.py'), V1_GENERATOR_BYTES)
        self.assertEqual(file_hash(ROOT / 'configs/curriculum.json'), V1_CONFIG_BYTES)
        self.assertEqual(digest(json.loads((ROOT / 'configs/curriculum.json').read_text())), V1_CONFIG_CANONICAL)

    def test_frozen_v1_outputs_train_dev_ood_and_production_strata(self):
        for (seed, split), (count, commitment) in V1_OUTPUTS.items():
            with self.subTest(seed=seed, split=split):
                rows = build.generate_group(seed, split=split)
                self.assertEqual(len(rows), count)
                self.assertEqual(digest(rows), commitment)
                self.assertEqual(rows, build.generate_group(seed, split=split))

    def test_v1_rendered_version_and_complete_outputs_remain_exact(self):
        from goal_conditioned_decision_data.decision_labels.rendering import render_training_views
        for (seed, split, family, view), wanted in V1_RENDERED.items():
            with self.subTest(seed=seed, split=split, view=view):
                row = next(r for r in build.generate_group(seed, split=split) if r['family'] == family)
                row['quality_gate'] = {'status': 'accepted', 'rejection_reasons': []}
                row['verification'] = audit_record(row)
                rendered = list(render_training_views(row, view, 3))
                self.assertTrue(all(r['surface']['rendering_version'] == 'gcdl-rendering-1.0' for r in rendered))
                self.assertEqual(digest(rendered), wanted)

    def test_three_predeclared_production_batches_match_ids_candidates_labels_fingerprints(self):
        for seed, split in ((1000000, 'train'), (1000003, 'dev'), (1000004, 'ood')):
            path = PRODUCTION / 'data/accepted' / ('%s-%d.jsonl' % (split, seed))
            batch = json.loads((PRODUCTION / 'manifests' / ('batch-' + path.stem + '.json')).read_text())
            persisted = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
            frozen_fingerprints = {r[0]: r[1] for r in batch['protected_index']}
            regenerated = build.generate_group(seed, split=split)
            self.assertEqual({r['id'] for r in regenerated}, set(persisted))
            for row in regenerated:
                with self.subTest(seed=seed, record=row['id']):
                    original = persisted[row['id']]
                    self.assertEqual(canonical(row['input']['candidates']), canonical(original['input']['candidates']))
                    self.assertEqual(row['targets'], original['targets'])
                    self.assertEqual(row['provenance'], original['provenance'])
                    self.assertEqual(build.semantic_fingerprint(row), frozen_fingerprints[row['id']])

    def test_all_1936_nonsealed_archived_rows_validate_without_rewriting_receipts(self):
        paths = sorted((PRODUCTION / 'data/accepted').glob('*.jsonl'))
        before = {str(path.relative_to(PRODUCTION)): file_hash(path) for path in paths}
        self.assertEqual(len(paths), 66)
        self.assertEqual(digest(before), PRODUCTION_ACCEPTED_COLLECTION)
        count = 0
        receipt_bytes = {}
        for path in paths:
            manifest_path = PRODUCTION / 'manifests' / ('batch-' + path.stem + '.json')
            receipt_bytes[str(manifest_path)] = file_hash(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['sha256'], before[str(path.relative_to(PRODUCTION))])
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.assertIn(row['split'], ('train', 'dev', 'ood'))
                original = canonical(row)
                source_map = deepcopy(row['verification']['code_hashes'])
                self.assertEqual(source_map, manifest['code_hashes'])
                self.assertEqual(row['provenance']['config_hash'], V1_CONFIG_CANONICAL)
                self.assertEqual(audit_record(row)['differential_oracle'], 'PASS')
                self.assertEqual(canonical(row), original)
                self.assertEqual(row['verification']['code_hashes'], source_map)
                count += 1
        self.assertEqual(count, 1936)
        self.assertEqual(before, {str(path.relative_to(PRODUCTION)): file_hash(path) for path in paths})
        self.assertEqual(receipt_bytes, {name: file_hash(name) for name in receipt_bytes})


class IndependentNamespaceSemantics(unittest.TestCase):
    def test_manual_status_goals_constraints_cost_and_reverse_semantics(self):
        task = manual_ledger()
        a, b = primary(deepcopy(task)), independent(deepcopy(task))
        self.assertEqual(a, b)
        projections = [(tuple(exact(v) for v in c['utility']), exact(c['success_probability']),
                        exact(c['feasible_probability'])) for c in a['candidates']]
        self.assertEqual(projections, [((1, 2, -1), 1, 1), ((0, 0, -3), 0, 0), ((1, 0, 0), 0, 1)])
        self.assertEqual(a['candidates'][0]['outcomes'][0]['terminal_state']['accounts']['oak'], 13)
        self.assertEqual(a['candidates'][1]['outcomes'][0]['terminal_state']['accounts']['oak'], 10)

    def test_status_values_rotate_consistently_but_role_keys_are_semantic(self):
        original = manual_ledger()
        rotated = deepcopy(original)
        mapping = {'x_wait': 'x_done', 'x_done': 'x_void', 'x_void': 'x_wait'}
        rotated['rules']['statuses'] = {k: mapping[v] for k, v in original['rules']['statuses'].items()}
        rotated['initial_state']['entries']['bill']['status'] = mapping['x_wait']
        rotated['goal']['statuses']['bill'] = mapping['x_done']
        rotated['constraints']['forbidden_statuses'] = [mapping['x_void']]
        self.assertEqual(build.semantic_fingerprint({'input': original}), build.semantic_fingerprint({'input': rotated}))
        self.assertEqual(build.decision_labels(primary(original)), build.decision_labels(primary(rotated)))
        # Merely changing semantic-role declarations without changing their
        # references converts the initial pending credit into a void credit.
        incorrect = deepcopy(original)
        incorrect['rules']['statuses'] = deepcopy(rotated['rules']['statuses'])
        self.assertNotEqual(build.semantic_fingerprint({'input': original}), build.semantic_fingerprint({'input': incorrect}))
        for evaluator in (primary, independent):
            result = evaluator(incorrect)
            self.assertEqual(exact(result['candidates'][0]['legal_probability']), 0)
            self.assertEqual(result['candidates'][0]['outcomes'][0]['terminal_state']['accounts']['oak'], 10)

    def test_unknown_action_refs_do_not_normalize_to_valid_entities(self):
        task = manual_ledger()
        wrong = deepcopy(task)
        wrong['candidates'][0]['plan'][0]['entry'] = 'undeclared_entry'
        self.assertNotEqual(build.semantic_fingerprint({'input': task}), build.semantic_fingerprint({'input': wrong}))
        for evaluator in (primary, independent):
            result = evaluator(wrong)
            self.assertEqual(exact(result['candidates'][0]['legal_probability']), 0)
            self.assertEqual(exact(result['candidates'][0]['success_probability']), 0)

    def test_manual_three_goals_reversal_illegal_and_empty_noninitial(self):
        move = lambda a, b: {'op': 'move', 'block': a, 'destination': b}
        p0 = [move('a', 'b'), move('a', 'c'), move('a', 'd'), move('b', 'c')]
        p1 = [move('c', 'd'), move('c', 'a'), move('c', 'b'), move('d', 'a')]
        p2 = [move('d', 'a'), move('d', 'b'), move('d', 'c'), move('a', 'b')]
        task = {'environment': 'blocksworld',
                'initial_state': {'on': dict.fromkeys('abcd', 'table')},
                'rules': {'table': 'table', 'max_height': 4, 'action_costs': {'move': 2, 'noop': 0}},
                'constraints': {}, 'candidates': [
                    {'id': 'p0', 'plan': p0}, {'id': 'p1', 'plan': p1}, {'id': 'p2', 'plan': p2},
                    {'id': 'prefix', 'plan': p0[:-1]}, {'id': 'reverse', 'plan': list(reversed(p0))},
                    {'id': 'empty', 'plan': []}]}
        goals = [dict(a='d', b='c', c='table', d='table'),
                 dict(a='table', b='table', c='b', d='a'),
                 dict(a='b', b='table', c='table', d='c')]
        scores = [[4, 0, 1, 3, 4, 2], [0, 4, 1, 1, 0, 2], [1, 1, 4, 2, 1, 2]]
        legal = [1, 1, 1, 1, 0, 1]
        costs = [8, 8, 8, 6, 4, 0]
        for index, goal in enumerate(goals):
            task['goal'] = {'on': goal}
            with self.subTest(goal=index):
                a, b = primary(deepcopy(task)), independent(deepcopy(task))
                self.assertEqual(a, b)
                wanted = list(zip(legal, scores[index], [-v for v in costs]))
                self.assertEqual([tuple(exact(v) for v in c['utility']) for c in a['candidates']], wanted)
                self.assertEqual(build.decision_labels(a)['best_candidate_ids'], ['p%d' % index])
                self.assertEqual(a['candidates'][4]['outcomes'][0]['failure_step'], 3)
                self.assertEqual(exact(a['candidates'][5]['success_probability']), 0)


class IndependentProfileCustody(unittest.TestCase):
    def test_exact_v1_identity_archive_and_default_resolve(self):
        from goal_conditioned_decision_data.generators import profiles
        profile = profiles.resolve_profile()
        self.assertEqual(profile['id'], 'v1')
        self.assertEqual(profile['generator_version'], 'generator-1.0')
        self.assertEqual(profile['environment_version'], 'environments-1.0')
        self.assertEqual(profile['config_version'], 'curriculum-1.0')
        self.assertEqual(profile['config_hash'], V1_CONFIG_CANONICAL)
        self.assertEqual(profile['config_file_sha256'], V1_CONFIG_BYTES)
        self.assertNotEqual(profile['config_hash'], profile['config_file_sha256'])
        self.assertEqual(Path(profile['config_path']).name, 'curriculum-v1.json')
        self.assertEqual(file_hash(profile['config_path']), V1_CONFIG_BYTES)
        provenance = build.generate_group(700)[0]['provenance']
        self.assertEqual(profiles.resolve_record_profile(provenance)['id'], 'v1')
        profile['config']['split_templates']['dev']['blocks'] = 99
        self.assertEqual(profiles.resolve_profile()['config']['split_templates']['dev']['blocks'], 4)

    def test_unknown_profile_ids_and_mixed_record_identity_fail_closed(self):
        from goal_conditioned_decision_data.generators import profiles
        for value in ('v3', 'generator-2.0', '../v1', 'generators.build', '', None, 1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profiles.resolve_profile(value)
        valid = build.generate_group(701)[0]
        for field, value in (
                ('config_hash', '0' * 64), ('config_hash', V1_CONFIG_BYTES),
                ('config_version', 'curriculum-2.0'), ('generator_version', 'generator-2.0'),
                ('environment_version', 'environments-2.0')):
            with self.subTest(field=field, value=value):
                changed = deepcopy(valid)
                changed['provenance'][field] = value
                changed['id'] = build.record_id(changed)
                with self.assertRaises(ValueError):
                    profiles.resolve_record_profile(changed['provenance'])
                with self.assertRaises(ValueError):
                    validate_schema(changed)
        for missing in profiles.IDENTITY_FIELDS:
            provenance = deepcopy(valid['provenance']); provenance.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                profiles.resolve_record_profile(provenance)

    def test_raw_archive_tamper_fails_even_if_json_value_is_unchanged(self):
        from goal_conditioned_decision_data.generators import profiles
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'configs', root / 'configs')
            archived = root / 'configs/curriculum-v1.json'
            original = archived.read_bytes()
            before = profiles.resolve_profile('v1', root=root)
            archived.write_bytes(original + b'\n')
            self.assertEqual(digest(json.loads(archived.read_text())), before['config_hash'])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                profiles.resolve_profile('v1', root=root)
            archived.write_bytes(original)
            altered = json.loads(archived.read_text()); altered['split_templates']['dev']['blocks'] = 3
            archived.write_text(canonical(altered))
            with self.assertRaises(ValueError):
                profiles.resolve_profile('v1', root=root)

    def test_registry_bytes_path_and_archive_symlinks_fail_closed(self):
        from goal_conditioned_decision_data.generators import profiles
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / 'configs', root / 'configs')
            registry = root / 'configs/generation_profiles.json'
            original = registry.read_bytes()
            registry.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                profiles.resolve_profile('v1', root=root)
            registry.write_bytes(original)
            archived = root / 'configs/curriculum-v1.json'
            archived.unlink(); archived.symlink_to(ROOT / 'configs/curriculum-v1.json')
            with self.assertRaisesRegex(ValueError, 'nonsymlink'):
                profiles.resolve_profile('v1', root=root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').symlink_to(ROOT / 'configs', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                profiles.resolve_profile('v1', root=root)

    def test_record_provenance_paths_cannot_select_executable_or_archive(self):
        from goal_conditioned_decision_data.generators import profiles
        row = build.generate_group(702)[0]
        row['provenance']['config_path'] = '/dev/null'
        row['provenance']['module'] = 'os.system'
        row['provenance']['profile_id'] = 'v2'
        # Such extra, nonmodel fields do not choose the trusted adapter.
        self.assertEqual(profiles.resolve_record_profile(row['provenance'])['id'], 'v1')
        source = profiles.resolve_profile('v1')['config']
        self.assertEqual(source['version'], 'curriculum-1.0')

    def test_all_archives_and_registry_are_in_source_hash_inventory(self):
        from goal_conditioned_decision_data.pipelines.common import source_hashes
        hashes = source_hashes()
        for name in ('configs/curriculum-v1.json', 'configs/curriculum-v2.json',
                     'configs/generation_profiles.json'):
            self.assertIn(name, hashes)
            self.assertEqual(hashes[name], file_hash(ROOT / name))


class IndependentGoalBundle(unittest.TestCase):
    def check_triplet(self, rows, diagnostics, split='dev'):
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r['id'] for r in rows}), 3)
        self.assertEqual(len({r['provenance']['parent_world_id'] for r in rows}), 1)
        first = rows[0]['input']
        self.assertEqual(first['environment'], 'blocksworld')
        self.assertEqual(first['constraints'], {})
        self.assertEqual(len(first['candidates']), 6)
        ids = [c['id'] for c in first['candidates']]
        self.assertEqual(len(set(ids)), 6)
        for name in ids:
            self.assertRegex(name, r'^c_[a-f0-9]{24}$')
        blocks = first['initial_state']['on']
        table = first['rules']['table']
        self.assertNotIn(table, blocks)
        goals = [r['input']['goal']['on'] for r in rows]
        self.assertEqual(len({canonical(g) for g in goals}), 3)
        for goal in goals:
            self.assertEqual(set(goal), set(blocks))
            self.assertNotEqual(goal, blocks)
        best_sets = []
        chosen = []
        for row in rows:
            task = row['input']
            self.assertEqual(without_goal(task), without_goal(first))
            self.assertEqual(row['split'], split)
            self.assertEqual(row['provenance']['generator_version'], 'generator-2.0')
            self.assertEqual(row['provenance']['config_version'], 'curriculum-2.0')
            self.assertEqual([c['id'] for c in task['candidates']], ids)
            self.assertNotIn('goal_index', task)
            self.assertNotIn('candidate_roles', canonical(task))
            self.assertNotIn('best_candidate_ids', canonical(task))
            a, b = primary(deepcopy(task)), independent(deepcopy(task))
            self.assertEqual(a, b)
            self.assertEqual(row['targets']['oracle'], b)
            expected = {}
            goal_reached = False
            for candidate in task['candidates']:
                terminal, legal, cost = independent_plan(task, candidate['plan'])
                goal_score = sum(terminal[k] == v for k, v in task['goal']['on'].items())
                expected[candidate['id']] = (int(legal), goal_score, -cost)
                goal_reached |= legal and terminal == task['goal']['on']
            self.assertTrue(goal_reached)
            best_score = max(expected.values())
            expected_best = sorted(name for name, utility in expected.items() if utility == best_score)
            self.assertEqual(row['targets']['decision']['best_candidate_ids'], expected_best)
            self.assertTrue(expected_best)
            for projected in row['targets']['decision']['candidates']:
                self.assertEqual(tuple(exact(v) for v in projected['utility']), expected[projected['id']])
            self.assertEqual(audit_record(row)['differential_oracle'], 'PASS')
            best_sets.append(set(expected_best))
            chosen.append(candidate_relationship_selector(task))
        for i in range(3):
            for j in range(i):
                self.assertFalse(best_sets[i] & best_sets[j])
        self.assertEqual(len(set(chosen)), 1)
        self.assertLessEqual(sum(chosen[i] in best_sets[i] for i in range(3)), 1)
        lengths = sorted(len(c['plan']) for c in first['candidates'])
        horizon = max(lengths)
        self.assertEqual(lengths, [0, horizon - 1, horizon, horizon, horizon, horizon])
        qualifying = [c for c in first['candidates'] if len(c['plan']) == horizon
                      and any(x['plan'] == c['plan'][:-1] for x in first['candidates'])
                      and any(x['plan'] == list(reversed(c['plan'])) for x in first['candidates'])]
        self.assertTrue(qualifying)
        return chosen[0]

    def test_goal_only_triplets_and_exact_targets_on_independent_fixture_seeds(self):
        from goal_conditioned_decision_data.generators import version2
        valid = 0
        for seed in range(700, 712):
            with self.subTest(seed=seed):
                rows, diagnostics = version2.generate_goal_triplet(seed, split='dev')
                self.assertEqual((rows, diagnostics), version2.generate_goal_triplet(seed, split='dev'))
                if rows:
                    self.check_triplet(rows, diagnostics)
                    valid += 1
                else:
                    self.assertEqual(diagnostics['reason'], 'insufficient_distinct_noninitial_endpoints')
                    self.assertEqual(diagnostics['proposals_attempted'], 24)
        self.assertGreaterEqual(valid, 9)

    def test_failed_proposals_stop_at_declared_24_without_replacement(self):
        from goal_conditioned_decision_data.generators import version2
        original = next(r for r in build.generate_group(799, split='dev') if r['family'] == 'C')
        with patch.object(version2, '_path_proposal', return_value=([], original['input']['initial_state'])) as propose:
            rows, diagnostic = version2.generate_goal_triplet(799, split='dev')
        self.assertEqual(rows, [])
        self.assertEqual(propose.call_count, 24)
        self.assertEqual(diagnostic['status'], 'EXCLUDED')
        self.assertEqual(diagnostic['reason'], 'insufficient_distinct_noninitial_endpoints')
        self.assertEqual(diagnostic['proposals_attempted'], 24)
        self.assertEqual(diagnostic['seed'], 799)

    def test_cheaper_prefix_winner_is_retained_in_full_best_set(self):
        from goal_conditioned_decision_data.generators import version2
        move = lambda a, b: {'op': 'move', 'block': a, 'destination': b}
        p0 = [move('a', 'b'), move('a', 'c'), move('a', 'd'), move('b', 'c')]
        p1 = [move('a', 'b'), move('a', 'c'), move('a', 'table'), move('a', 'd')]
        p2 = [move('d', 'a'), move('d', 'b'), move('d', 'c'), move('a', 'b')]
        source = next(r for r in build.generate_group(798, split='dev') if r['family'] == 'C')
        source['input'] = {'environment': 'blocksworld',
            'initial_state': {'on': dict.fromkeys('abcd', 'table')},
            'rules': {'table': 'table', 'max_height': 4, 'action_costs': {'move': 2, 'noop': 0}},
            'goal': {'on': dict(a='d', b='c', c='table', d='table')}, 'constraints': {},
            'candidates': [{'id': 'dummy', 'plan': []}],
            'utility_definition': deepcopy(manual_ledger()['utility_definition'])}
        terminals = [dict(a='d', b='c', c='table', d='table'),
                     dict(a='d', b='table', c='table', d='table'),
                     dict(a='b', b='table', c='table', d='c')]
        proposed = [(p, {'on': g}) for p, g in zip((p0, p1, p2), terminals)]
        with patch.object(version2.v1, 'generate_group', return_value=[source]), \
                patch.object(version2, '_path_proposal', side_effect=proposed):
            rows, diagnostic = version2.generate_goal_triplet(798, split='dev')
        self.check_triplet(rows, diagnostic)
        middle = rows[1]
        best = middle['targets']['decision']['best_candidate_ids']
        self.assertEqual(len(best), 1)
        candidate = next(c for c in middle['input']['candidates'] if c['id'] == best[0])
        self.assertEqual(len(candidate['plan']), 3)
        projected = next(c for c in middle['targets']['decision']['candidates'] if c['id'] == best[0])
        self.assertEqual(tuple(exact(v) for v in projected['utility']), (1, 4, -6))

    def test_c_candidate_ids_order_and_parent_identity_do_not_depend_on_gold(self):
        from goal_conditioned_decision_data.generators import version2
        normal, _ = version2.generate_goal_triplet(797, split='dev')
        real_labels = version2.v1.decision_labels
        def fabricated(oracle):
            labels = real_labels(oracle)
            ids = [c['id'] for c in oracle['candidates']]
            remap = dict(zip(ids, ids[1:] + ids[:1]))
            labels['best_candidate_ids'] = sorted(remap[name] for name in labels['best_candidate_ids'])
            labels['uncertainty']['probability_definition'] = 'PRIVATE_GOLD_CANARY'
            return labels
        with patch.object(version2.v1, 'decision_labels', side_effect=fabricated):
            altered, _ = version2.generate_goal_triplet(797, split='dev')
        for original, changed in zip(normal, altered):
            self.assertEqual(original['input'], changed['input'])
            self.assertEqual(original['provenance']['parent_world_id'], changed['provenance']['parent_world_id'])
            self.assertNotEqual(original['id'], changed['id'])
            self.assertNotIn('PRIVATE_GOLD_CANARY', canonical(changed['input']))

    def test_legal_reverse_tie_is_retained_without_intended_winner_filtering(self):
        from goal_conditioned_decision_data.generators import version2
        move = lambda a, b: {'op': 'move', 'block': a, 'destination': b}
        p0 = [move('a', 'b'), move('a', 'table'), move('d', 'c'), move('a', 'b')]
        p1 = [move('a', 'b'), move('a', 'table'), move('d', 'b'), move('a', 'c')]
        p2 = [move('a', 'table'), move('a', 'd'), move('a', 'b'), move('c', 'd')]
        source = next(r for r in build.generate_group(796, split='dev') if r['family'] == 'C')
        source['input'] = {'environment': 'blocksworld',
            'initial_state': {'on': dict(a='c', b='table', c='table', d='table')},
            'rules': {'table': 'table', 'max_height': 4, 'action_costs': {'move': 2, 'noop': 0}},
            'goal': {'on': dict(a='b', b='table', c='table', d='c')}, 'constraints': {},
            'candidates': [{'id': 'dummy', 'plan': []}],
            'utility_definition': deepcopy(manual_ledger()['utility_definition'])}
        terminals = [dict(a='b', b='table', c='table', d='c'),
                     dict(a='c', b='table', c='table', d='b'),
                     dict(a='b', b='table', c='d', d='table')]
        proposed = [(p, {'on': g}) for p, g in zip((p0, p1, p2), terminals)]
        with patch.object(version2.v1, 'generate_group', return_value=[source]), \
                patch.object(version2, '_path_proposal', side_effect=proposed):
            rows, diagnostic = version2.generate_goal_triplet(796, split='dev')
        self.check_triplet(rows, diagnostic)
        first = rows[0]
        best = first['targets']['decision']['best_candidate_ids']
        self.assertEqual(len(best), 2)
        self.assertEqual(first['targets']['decision']['uncertainty']['label'], 'tied_best')
        for candidate in first['targets']['decision']['candidates']:
            if candidate['id'] in best:
                self.assertEqual(tuple(exact(v) for v in candidate['utility']), (1, 4, -8))

    def test_actual_production_stride_six_and_rendered_goal_only_inputs(self):
        from goal_conditioned_decision_data.generators import version2
        from goal_conditioned_decision_data.decision_labels.rendering import render_training_views, parse_input_text
        for split, start in (('dev', 1000003), ('ood', 1000004)):
            for offset in range(32):
                seed = start + 6 * offset
                with self.subTest(seed=seed, split=split):
                    rows, diagnostics = version2.generate_goal_triplet(seed, split=split)
                    if not rows:
                        self.assertEqual(diagnostics['reason'], 'insufficient_distinct_noninitial_endpoints')
                        continue
                    self.check_triplet(rows, diagnostics, split)
                    rendered = []
                    for row in rows:
                        row['quality_gate'] = {'status': 'accepted', 'rejection_reasons': []}
                        row['verification'] = audit_record(row)
                        original = canonical(row)
                        rendered.append(list(render_training_views(row, 'listwise', 3)))
                        self.assertEqual(canonical(row), original)
                    self.assertEqual([len(r) for r in rendered], [3, 3, 3])
                    for variant in range(3):
                        tasks = [parse_input_text(rendered[i][variant]['input_text']) for i in range(3)]
                        self.assertEqual(without_goal(tasks[0]), without_goal(tasks[1]))
                        self.assertEqual(without_goal(tasks[0]), without_goal(tasks[2]))
                        # Equality of Python mappings ignores key order. The
                        # declared goal-only presentation also keeps seeded
                        # nested ordering constant over the shared parent.
                        skeletons = [json.dumps(without_goal(t), ensure_ascii=False) for t in tasks]
                        self.assertEqual(len(set(skeletons)), 1)
                        versions = {rendered[i][variant]['surface']['rendering_version'] for i in range(3)}
                        self.assertEqual(len(versions), 1)
                        self.assertNotIn('gcdl-rendering-1.0', versions)


class IndependentV2Namespaces(unittest.TestCase):
    def test_predeclared_lexicons_have_uniform_shape_and_no_global_or_opcode_collisions(self):
        from goal_conditioned_decision_data.generators import version2
        cfg = version2.curriculum()
        words = []
        reserved = {'environment', 'initial_state', 'rules', 'goal', 'constraints', 'candidates', 'plan', 'op',
                    'ledger', 'blocksworld', 'register', 'accounts', 'entries', 'blocks', 'registers', 'statuses',
                    'pending', 'settled', 'void', 'table', 'on', 'values', 'source', 'destination', 'left', 'right',
                    'settle', 'transfer', 'noop', 'move', 'add', 'multiply', 'set', 'copy', 'swap'}
        self.assertEqual(set(cfg['lexicons']), {'train', 'dev', 'ood', 'sealed'})
        for split, namespaces in cfg['lexicons'].items():
            self.assertEqual(set(namespaces), set(version2.NAMESPACES))
            for namespace, pool in namespaces.items():
                with self.subTest(split=split, namespace=namespace):
                    self.assertEqual(len(pool), 16)
                    self.assertEqual(len(set(pool)), 16)
                    self.assertFalse(set(pool) & reserved)
                    for word in pool:
                        self.assertRegex(word, r'^[a-z]{8}$')
                    words.extend(pool)
        self.assertEqual(len(words), 384)
        self.assertEqual(len(set(words)), 384)

    def test_paired_control_and_reserved_dev_have_exact_labels_and_separate_pools(self):
        from goal_conditioned_decision_data.generators import version2
        from goal_conditioned_decision_data.auditors.integrity import semantic_fingerprint
        cfg = version2.curriculum()
        for namespace in version2.NAMESPACES:
            self.assertFalse(set(cfg['lexicons']['train'][namespace]) & set(cfg['lexicons']['dev'][namespace]))
            self.assertFalse(set(cfg['lexicons']['dev'][namespace]) & set(cfg['lexicons']['ood'][namespace]))
        for seed in range(720, 728):
            with self.subTest(seed=seed):
                pair, diagnostic = version2.generate_vocabulary_pair(seed)
                self.assertEqual((pair, diagnostic), version2.generate_vocabulary_pair(seed))
                self.assertEqual(len(pair), 2)
                control, holdout = pair
                self.assertEqual(control['metadata']['vocabulary']['stratum'], 'familiar_control')
                self.assertEqual(holdout['metadata']['vocabulary']['stratum'], 'reserved_dev')
                self.assertEqual(control['metadata']['vocabulary']['lexicon_split'], 'train')
                self.assertEqual(holdout['metadata']['vocabulary']['lexicon_split'], 'dev')
                self.assertEqual(control['split'], 'dev'); self.assertEqual(holdout['split'], 'dev')
                self.assertEqual(control['provenance']['parent_world_id'], holdout['provenance']['parent_world_id'])
                self.assertEqual(version2.restore_source_names(control), version2.restore_source_names(holdout))
                self.assertEqual(control['targets']['decision'], holdout['targets']['decision'])
                self.assertEqual(semantic_fingerprint(control), semantic_fingerprint(holdout))
                for row, lexicon in ((control, 'train'), (holdout, 'dev')):
                    self.assertEqual(audit_record(row)['differential_oracle'], 'PASS')
                    vocabulary = row['metadata']['vocabulary']
                    maps = vocabulary['maps']
                    self.assertEqual(set(maps), set(version2.NAMESPACES))
                    for namespace, mapping in maps.items():
                        self.assertEqual(len(mapping), len(set(mapping.values())))
                        self.assertLessEqual(set(mapping.values()), set(cfg['lexicons'][lexicon][namespace]))
                        for value in mapping.values():
                            self.assertRegex(value, r'^[a-z]{8}$')
                        if lexicon == 'dev':
                            self.assertFalse(set(mapping.values()) & set(cfg['lexicons']['train'][namespace]))
                    self.assertEqual(len(row['input']['initial_state']['accounts']), 3)
                    self.assertEqual(len(row['input']['initial_state']['entries']), 2)
                    self.assertEqual(set(row['input']['rules']['statuses']), {'pending', 'settled', 'void'})

    def test_forged_missing_colliding_cross_split_and_source_maps_are_rejected(self):
        from goal_conditioned_decision_data.generators import version2
        original = version2.generate_vocabulary_pair(729)[0][1]
        cfg = version2.curriculum()
        mutations = []
        absent = deepcopy(original); absent['metadata'].pop('vocabulary'); mutations.append(absent)
        missing = deepcopy(original)
        missing['metadata']['vocabulary']['maps']['accounts'].pop(next(iter(missing['metadata']['vocabulary']['maps']['accounts'])))
        mutations.append(missing)
        collided = deepcopy(original)
        accounts = collided['metadata']['vocabulary']['maps']['accounts']; names = list(accounts)
        accounts[names[1]] = accounts[names[0]]; mutations.append(collided)
        wrong_pool = deepcopy(original)
        accounts = wrong_pool['metadata']['vocabulary']['maps']['accounts']
        accounts[next(iter(accounts))] = cfg['lexicons']['train']['accounts'][0]; mutations.append(wrong_pool)
        stratum = deepcopy(original); stratum['metadata']['vocabulary']['stratum'] = 'familiar_control'; mutations.append(stratum)
        source_hash = deepcopy(original); source_hash['metadata']['vocabulary']['source_namespace_task_sha256'] = '0' * 64; mutations.append(source_hash)
        parent = deepcopy(original); parent['provenance']['parent_world_id'] += '_forged'; mutations.append(parent)
        for index, changed in enumerate(mutations):
            changed['id'] = build.record_id(changed)
            with self.subTest(mutation=index):
                with self.assertRaises(ValueError):
                    version2.restore_source_names(changed)
                with self.assertRaises(ValueError):
                    audit_record(changed)

    def test_familiar_controls_are_dev_only_and_v2_modified_config_cannot_replay(self):
        from goal_conditioned_decision_data.generators import version2
        for split in ('train', 'ood'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                version2.generate_vocabulary_pair(730, split=split)
        changed = version2.curriculum(); changed['c_bundle']['max_proposals'] = 25
        with self.assertRaises(ValueError):
            version2.generate_goal_triplet(730, config=changed)
        changed = version2.curriculum(); changed['lexicons']['dev']['accounts'][0] = 'abcdefgh'
        with self.assertRaises(ValueError):
            version2.generate_vocabulary_pair(730, config=changed)

    def test_explicit_ledger_namespace_mapping_preserves_all_refs_and_status_roles(self):
        from goal_conditioned_decision_data.generators import version2
        task = manual_ledger()
        task['possible_worlds'] = [deepcopy(task['initial_state']), deepcopy(task['initial_state'])]
        task['possible_worlds'][1]['entries']['bill']['amount'] = 4
        task['initial_state']['entries']['bill']['amount'] = None
        task['information'] = 'missing'
        task['probability_assumption'] = 'uniform_over_possible_worlds'
        maps = {'accounts': {'oak': 'acct_a', 'pine': 'acct_b'}, 'entries': {'bill': 'entry_a'},
                'statuses': {'x_wait': 'phase_c', 'x_done': 'phase_a', 'x_void': 'phase_b'},
                'blocks': {}, 'registers': {}, 'table': {}}
        renamed = version2.rename_task(task, maps)
        self.assertEqual(task['initial_state']['entries']['bill']['status'], 'x_wait')
        self.assertEqual(renamed['rules']['statuses'], {'pending': 'phase_c', 'settled': 'phase_a', 'void': 'phase_b'})
        self.assertEqual(renamed['goal'], {'balances': {'acct_a': 13}, 'statuses': {'entry_a': 'phase_a'}})
        self.assertEqual(renamed['constraints'], {'minimum_balances': {'acct_a': 0}, 'forbidden_statuses': ['phase_b']})
        self.assertEqual(renamed['candidates'][0]['plan'], [{'op': 'settle', 'entry': 'entry_a'}])
        for world in [renamed['initial_state']] + renamed['possible_worlds']:
            self.assertEqual(set(world['accounts']), {'acct_a', 'acct_b'})
            self.assertEqual(set(world['entries']), {'entry_a'})
            self.assertEqual(world['entries']['entry_a']['account'], 'acct_a')
            self.assertEqual(world['entries']['entry_a']['status'], 'phase_c')
        self.assertEqual(build.semantic_fingerprint({'input': task}), build.semantic_fingerprint({'input': renamed}))
        original_labels = build.decision_labels(primary(task))
        self.assertEqual(original_labels, build.decision_labels(primary(renamed)))
        self.assertEqual(original_labels, build.decision_labels(independent(renamed)))
        self.assertEqual(exact(original_labels['candidates'][0]['success_probability']), Fraction(1, 2))

    def test_unknown_entry_and_account_references_stay_invalid_after_renaming(self):
        from goal_conditioned_decision_data.generators import version2
        task = manual_ledger()
        task['candidates'] = [
            {'id': 'wrong_entry', 'plan': [{'op': 'settle', 'entry': 'absent_entry'}]},
            {'id': 'wrong_account', 'plan': [{'op': 'transfer', 'source': 'absent_account', 'destination': 'oak', 'amount': 1}]},
            {'id': 'valid_credit', 'plan': [{'op': 'settle', 'entry': 'bill'}]}]
        maps = {'accounts': {'oak': 'acct_a', 'pine': 'acct_b', 'absent_account': 'acct_c'},
                'entries': {'bill': 'entry_a', 'absent_entry': 'entry_b'},
                'statuses': {'x_wait': 'phase_c', 'x_done': 'phase_a', 'x_void': 'phase_b'},
                'blocks': {}, 'registers': {}, 'table': {}}
        renamed = version2.rename_task(task, maps)
        self.assertEqual(renamed['candidates'][0]['plan'][0]['entry'], 'entry_b')
        self.assertNotIn('entry_b', renamed['initial_state']['entries'])
        self.assertEqual(renamed['candidates'][1]['plan'][0]['source'], 'acct_c')
        self.assertNotIn('acct_c', renamed['initial_state']['accounts'])
        wanted = build.decision_labels(primary(task))
        for evaluator in (primary, independent):
            result = evaluator(renamed)
            self.assertEqual(build.decision_labels(result), wanted)
            self.assertEqual([exact(c['legal_probability']) for c in result['candidates']], [0, 0, 1])

    def test_block_table_constraints_and_unknown_destination_are_namespace_aware(self):
        from goal_conditioned_decision_data.generators import version2
        task = {'environment': 'blocksworld', 'initial_state': {'on': {'a': 'desk', 'b': 'a', 'c': 'desk'}},
                'rules': {'table': 'desk', 'max_height': 3, 'action_costs': {'move': 2, 'noop': 0}},
                'goal': {'on': {'a': 'c'}}, 'constraints': {'forbidden_on': [['b', 'c']], 'protected_blocks': ['c']},
                'candidates': [
                    {'id': 'valid', 'plan': [{'op': 'move', 'block': 'b', 'destination': 'desk'},
                                             {'op': 'move', 'block': 'a', 'destination': 'c'}]},
                    {'id': 'unknown', 'plan': [{'op': 'move', 'block': 'a', 'destination': 'absent'}]}]}
        maps = {'accounts': {}, 'entries': {}, 'statuses': {}, 'registers': {},
                'blocks': {'a': 'aa', 'b': 'bb', 'c': 'cc', 'absent': 'dd'}, 'table': {'desk': 'surface'}}
        renamed = version2.rename_task(task, maps)
        self.assertEqual(renamed['initial_state']['on'], {'aa': 'surface', 'bb': 'aa', 'cc': 'surface'})
        self.assertEqual(renamed['rules']['table'], 'surface')
        self.assertEqual(renamed['goal'], {'on': {'aa': 'cc'}})
        self.assertEqual(renamed['constraints'], {'forbidden_on': [['bb', 'cc']], 'protected_blocks': ['cc']})
        self.assertEqual(renamed['candidates'][0]['plan'][0], {'op': 'move', 'block': 'bb', 'destination': 'surface'})
        self.assertEqual(renamed['candidates'][1]['plan'][0]['destination'], 'dd')
        self.assertEqual(build.decision_labels(primary(task)), build.decision_labels(independent(renamed)))
        self.assertEqual([exact(c['legal_probability']) for c in primary(renamed)['candidates']], [1, 0])

    def test_register_args_constraints_and_opcodes_keep_declared_roles(self):
        from goal_conditioned_decision_data.generators import version2
        task = {'environment': 'register', 'initial_state': {'registers': {'r1': 2, 'r2': 5}},
                'rules': {'value_bounds': [-20, 20], 'overflow': 'reject',
                          'operations': ['copy', 'swap', 'add', 'multiply', 'set', 'noop'],
                          'action_costs': dict.fromkeys(['copy', 'swap', 'add', 'multiply', 'set', 'noop'], 1)},
                'goal': {'values': {'r1': 5, 'r2': 5}},
                'constraints': {'minimum_values': {'r1': 0}, 'maximum_values': {'r2': 10}, 'protected_registers': ['r2']},
                'candidates': [
                    {'id': 'copy', 'plan': [{'op': 'copy', 'source': 'r2', 'destination': 'r1'}]},
                    {'id': 'swap', 'plan': [{'op': 'swap', 'left': 'r1', 'right': 'r2'}]},
                    {'id': 'add', 'plan': [{'op': 'add', 'register': 'r1', 'value': 3}]},
                    {'id': 'multiply', 'plan': [{'op': 'multiply', 'register': 'r1', 'value': 2}]},
                    {'id': 'set', 'plan': [{'op': 'set', 'register': 'r1', 'value': 5}]},
                    {'id': 'unknown', 'plan': [{'op': 'set', 'register': 'absent', 'value': 5}]}]}
        maps = {'accounts': {}, 'entries': {}, 'statuses': {}, 'blocks': {}, 'table': {},
                'registers': {'r1': 'aa', 'r2': 'bb', 'absent': 'cc'}}
        renamed = version2.rename_task(task, maps)
        self.assertEqual(renamed['initial_state']['registers'], {'aa': 2, 'bb': 5})
        self.assertEqual(renamed['rules'], task['rules'])
        self.assertEqual(renamed['goal'], {'values': {'aa': 5, 'bb': 5}})
        self.assertEqual(renamed['constraints'], {'minimum_values': {'aa': 0}, 'maximum_values': {'bb': 10}, 'protected_registers': ['bb']})
        self.assertEqual(renamed['candidates'][0]['plan'][0], {'op': 'copy', 'source': 'bb', 'destination': 'aa'})
        self.assertEqual(renamed['candidates'][1]['plan'][0], {'op': 'swap', 'left': 'aa', 'right': 'bb'})
        wanted = [(1, 2, -1), (0, 1, -1), (1, 2, -1), (1, 1, -1), (1, 2, -1), (0, 1, 0)]
        for evaluator in (primary, independent):
            result = evaluator(renamed)
            self.assertEqual([tuple(exact(v) for v in c['utility']) for c in result['candidates']], wanted)
            self.assertEqual(build.decision_labels(result)['best_candidate_ids'], ['add', 'copy', 'set'])


class IndependentWorkerProfileCustody(unittest.TestCase):
    def setUp(self):
        from goal_conditioned_decision_data.pipelines import worker
        self.worker = worker
        self.config = json.loads((ROOT / 'configs/production.json').read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        root_patch = patch.object(worker, 'ROOT', self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)
        disk_patch = patch.object(worker.shutil, 'disk_usage',
                                  return_value=type('Disk', (), {'free': 100 * 1024 ** 3})())
        disk_patch.start(); self.addCleanup(disk_patch.stop)
        self.db = worker.connection()
        self.addCleanup(self.db.close)

    def enqueue(self, seed, profile):
        self.db.execute('INSERT INTO jobs (seed,split,status,profile) VALUES (?,"train","pending",?)', (seed, profile))
        self.db.commit()

    def test_legacy_jobs_migrate_to_v1_and_config_cannot_reinterpret_them(self):
        # Re-create the actual pre-profile schema in an empty temporary DB.
        self.db.execute('DROP TABLE jobs')
        self.db.execute('CREATE TABLE jobs (seed INTEGER, split TEXT, status TEXT, accepted INTEGER DEFAULT 0, quarantined INTEGER DEFAULT 0, report TEXT, PRIMARY KEY(seed,split))')
        self.db.execute('INSERT INTO jobs (seed,split,status) VALUES (735,"train","pending")')
        self.db.commit()
        self.config['active_generation_profile'] = 'v2'
        real = build.generate_group
        with patch.object(build, 'generate_group', wraps=real) as generate:
            manifest = self.worker.process(735, 'train', self.config, self.db)
        self.assertEqual(manifest['generation_profile'], 'v1')
        self.assertEqual(self.db.execute('SELECT profile FROM jobs').fetchone()[0], 'v1')
        self.assertEqual(generate.call_count, 2)
        for call in generate.call_args_list:
            self.assertEqual(call.kwargs['config']['version'], 'curriculum-1.0')
        self.assertEqual(manifest['generation_profile_evidence']['config_hash'], V1_CONFIG_CANONICAL)
        self.assertEqual(manifest['generation_profile_evidence']['config_file_sha256'], V1_CONFIG_BYTES)

    def test_explicit_v2_generation_and_rerun_share_identical_trusted_profile(self):
        from goal_conditioned_decision_data.generators import version2
        self.enqueue(736, 'v2')
        self.config['active_generation_profile'] = 'v1'
        with patch.object(version2, 'generate_group', wraps=version2.generate_group) as generate:
            manifest = self.worker.process(736, 'train', self.config, self.db)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args_list[0], generate.call_args_list[1])
        self.assertEqual(generate.call_args_list[0].kwargs['config']['version'], 'curriculum-2.0')
        self.assertEqual(manifest['generation_profile'], 'v2')
        rows = list(map(json.loads, (self.root / manifest['shard']).read_text().splitlines()))
        self.assertTrue(rows)
        self.assertEqual({r['provenance']['generator_version'] for r in rows}, {'generator-2.0'})
        self.assertTrue(all(r['verification']['code_hashes'] == manifest['code_hashes'] for r in rows))

    def test_replay_cannot_change_existing_batch_profile_before_generation(self):
        from goal_conditioned_decision_data.generators import version2
        self.enqueue(737, 'v1')
        first = self.worker.process(737, 'train', self.config, self.db)
        before = file_hash(self.root / first['shard'])
        self.db.execute('UPDATE jobs SET profile="v2",status="pending"'); self.db.commit()
        with patch.object(version2, 'generate_group', wraps=version2.generate_group) as generate:
            with self.assertRaises(ValueError):
                self.worker.process(737, 'train', self.config, self.db)
        self.assertEqual(generate.call_count, 0)
        self.assertEqual(file_hash(self.root / first['shard']), before)
        self.assertFalse((self.root / 'data/candidates/train-737.jsonl').exists())

    def test_unknown_queued_profile_fails_before_any_generation_or_publication(self):
        self.enqueue(738, 'untrusted_adapter')
        with patch.object(build, 'generate_group', wraps=build.generate_group) as generate:
            with self.assertRaises(ValueError):
                self.worker.process(738, 'train', self.config, self.db)
        self.assertEqual(generate.call_count, 0)
        self.assertFalse((self.root / 'data').exists())

    def test_actual_v2_manifest_cannot_be_downgraded_to_legacy_by_removing_profile_fields(self):
        self.enqueue(741, 'v2')
        manifest = self.worker.process(741, 'train', self.config, self.db)
        self.assertGreater(manifest['accepted'], 0)
        manifest.pop('generation_profile'); manifest.pop('generation_profile_evidence')
        path = self.root / 'manifests/batch-train-741.json'
        self.db.execute('DELETE FROM jobs'); self.db.execute('DELETE FROM seen'); self.db.commit()
        for version in {manifest['version'], 'gcdl-manifest-1'}:
            manifest['version'] = version
            path.write_text(canonical(manifest))
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.worker.recover_inventory(self.db)
            self.assertEqual(self.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 0)

    def test_one_bad_v2_goal_quarantines_the_whole_supplied_batch(self):
        from goal_conditioned_decision_data.generators import version2
        self.enqueue(739, 'v2')
        rows = version2.generate_group(739, split='train')
        broken = next(r for r in rows if r['family'] == 'C')
        broken['targets']['decision']['best_candidate_ids'] = []
        broken['id'] = build.record_id(broken)
        with patch.object(version2, 'generate_group', side_effect=lambda *args, **kwargs: deepcopy(rows)):
            manifest = self.worker.process(739, 'train', self.config, self.db)
        self.assertEqual(manifest['accepted'], 0)
        self.assertEqual(manifest['quarantined'], len(rows))
        self.assertEqual((self.root / manifest['shard']).read_bytes(), b'')
        quarantined = list(map(json.loads, (self.root / 'data/quarantined/train-739.jsonl').read_text().splitlines()))
        self.assertEqual(len(quarantined), len(rows))
        self.assertTrue(all(r['quality_gate']['status'] == 'quarantined' for r in quarantined))

    def test_legacy_recovery_defaults_v1_but_v2_missing_or_false_commitment_fails(self):
        self.enqueue(740, 'v1')
        manifest = self.worker.process(740, 'train', self.config, self.db)
        path = self.root / 'manifests/batch-train-740.json'
        manifest['version'] = 'gcdl-manifest-1'
        manifest.pop('generation_profile'); manifest.pop('generation_profile_evidence')
        path.write_text(canonical(manifest))
        self.db.execute('DELETE FROM jobs'); self.db.execute('DELETE FROM seen'); self.db.commit()
        self.worker.recover_inventory(self.db)
        self.assertEqual(self.db.execute('SELECT profile FROM jobs').fetchone()[0], 'v1')
        manifest['generation_profile'] = 'v2'
        path.write_text(canonical(manifest))
        self.db.execute('DELETE FROM jobs'); self.db.commit()
        with self.assertRaises(ValueError):
            self.worker.recover_inventory(self.db)
        from goal_conditioned_decision_data.generators.profiles import resolve_profile
        profile = resolve_profile('v2')
        manifest['generation_profile_evidence'] = {k: profile[k] for k in ('id', 'generator_version', 'environment_version', 'config_version', 'config_hash', 'config_file_sha256')}
        manifest['generation_profile_evidence']['config_hash'] = V1_CONFIG_CANONICAL
        path.write_text(canonical(manifest))
        with self.assertRaisesRegex(ValueError, 'commitment mismatch'):
            self.worker.recover_inventory(self.db)


class IndependentV2CustodyAndBounds(unittest.TestCase):
    def test_c_schema_recomputes_bundle_and_rejects_invalid_goals_indices_or_roles(self):
        from goal_conditioned_decision_data.generators import version2
        original = version2.generate_goal_triplet(731)[0][0]
        mutations = []
        identity = deepcopy(original); identity['metadata']['goal_bundle_id'] = '0' * 64; mutations.append(identity)
        partial = deepcopy(original); partial['input']['goal']['on'].pop(next(iter(partial['input']['goal']['on']))); mutations.append(partial)
        initial = deepcopy(original); initial['input']['goal'] = deepcopy(initial['input']['initial_state']); mutations.append(initial)
        index_bool = deepcopy(original); index_bool['metadata']['goal_index'] = True; mutations.append(index_bool)
        index_high = deepcopy(original); index_high['metadata']['goal_index'] = 3; mutations.append(index_high)
        candidates = deepcopy(original); candidates['input']['candidates'].pop(); mutations.append(candidates)
        family = deepcopy(original); family['family'] = 'D'; mutations.append(family)
        leak = deepcopy(original); leak['input']['goal_index'] = 0; mutations.append(leak)
        for number, row in enumerate(mutations):
            row['id'] = build.record_id(row)
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                validate_schema(row)

    def test_goal_triplets_and_vocabulary_pairs_must_be_complete_and_cannot_mix_profiles(self):
        from goal_conditioned_decision_data.generators import version2
        from goal_conditioned_decision_data.auditors.integrity import validate_v2_group
        triplet = version2.generate_goal_triplet(732)[0]
        pair = version2.generate_vocabulary_pair(733)[0]
        for row in triplet + pair:
            audit_record(row)
        validate_v2_group(triplet)
        validate_v2_group(pair)
        for incomplete in (triplet[:1], triplet[:2], triplet + [triplet[0]], pair[:1], pair[1:]):
            with self.subTest(size=len(incomplete)), self.assertRaises(ValueError):
                validate_v2_group(incomplete)
        with self.assertRaises(ValueError):
            validate_v2_group(triplet + [build.generate_group(734)[0]])
        with self.assertRaises(ValueError):
            validate_v2_group(triplet, require_full_curriculum=True)
        changed = deepcopy(triplet)
        changed[1]['input']['instruction'] += ' altered sibling'
        changed[1]['id'] = build.record_id(changed[1])
        with self.assertRaises(ValueError):
            validate_v2_group(changed)

    def test_profile_bounds_precede_factorial_normalization(self):
        from goal_conditioned_decision_data.auditors.integrity import semantic_fingerprint
        from goal_conditioned_decision_data.generators import version2
        for group in (build.generate_group(734, split='dev'), version2.generate_group(734, split='dev')):
            for environment, field, count in (('blocksworld', 'on', 7), ('blocksworld', 'on', 3),
                                               ('ledger', 'accounts', 5), ('register', 'registers', 5)):
                row = next(r for r in group if r['input']['environment'] == environment)
                oversized = deepcopy(row)
                value = row['input']['rules'].get('table', 0)
                oversized['input']['initial_state'][field] = {'fake_%d' % i: value for i in range(count)}
                # Spy avoids ever trying the attack's factorial work; the
                # gate must reject before invoking the v1 normalizer.
                with self.subTest(profile=row['provenance']['generator_version'], environment=environment, count=count):
                    with patch.object(build, 'semantic_fingerprint', return_value='NORMALIZER_CALLED') as normalize:
                        with self.assertRaises(ValueError):
                            semantic_fingerprint(oversized)
                    self.assertEqual(normalize.call_count, 0)




