"""Independent rendered-surface review, with hand-calculated utility fixtures.

This reviewer did not author the renderer/exporter/generator/oracles. Tests read
only train/dev/ood and code/config; sealed gold is neither read nor generated.
Temporary exports are deliberately confined to temporary fixture roots.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.decision_labels import rendering as surface
from goal_conditioned_decision_data.auditors.integrity import audit_record, semantic_fingerprint
from goal_conditioned_decision_data.generators.build import (
    generate_group, record_id, decision_labels)
from goal_conditioned_decision_data.oracles.primary import evaluate
from goal_conditioned_decision_data.pipelines.common import canonical, digest, file_hash, source_hashes
from goal_conditioned_decision_data.pipelines import rendered as pipeline

ROOT = Path(__file__).resolve().parents[1]


def rat(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def score(candidate):
    return tuple(Fraction(v['numerator'], v['denominator'])
                 for v in candidate['utility'])


def manual_task(kind='ordinary'):
    # Oak begins at 10. Settling +3 achieves 13 for cost 1; settling and
    # voiding reverses the credit for total cost 3; empty costs 0.
    task = {
        'environment': 'ledger',
        'initial_state': {'accounts': {'oak': 10, 'pine': 4},
                          'entries': {'bill': {'account': 'oak', 'amount': 3, 'status': 'waiting'}}},
        'rules': {'statuses': {'pending': 'waiting', 'settled': 'recorded', 'void': 'void'},
                  'void_effect': 'reverse', 'allow_overdraft': False,
                  'balance_bounds': [-100, 100],
                  'action_costs': {'settle': 1, 'void': 2, 'transfer': 1, 'noop': 0}},
        'goal': {'balances': {'oak': 13}}, 'constraints': {},
        'utility_definition': {'comparison': 'lexicographic',
            'components': ['expected_feasibility', 'expected_goal_atoms_satisfied', 'negative_expected_successful_action_cost'],
            'feasibility': 'legal execution and no constraint violations at initial or intermediate states',
            'all_infeasible': 'abstain; no best candidate'},
        'candidates': [
            {'id': 'c_000000000000000000000001', 'plan': [{'op': 'settle', 'entry': 'bill'}]},
            {'id': 'c_000000000000000000000002', 'plan': [{'op': 'settle', 'entry': 'bill'}, {'op': 'void', 'entry': 'bill'}]},
            {'id': 'c_000000000000000000000003', 'plan': []}],
    }
    if kind == 'tie':
        task['goal'] = {'balances': {'oak': 10}}
        task['candidates'] = [
            {'id': 'c_000000000000000000000001', 'plan': [{'op': 'noop'}]},
            {'id': 'c_000000000000000000000003', 'plan': []}]
    elif kind == 'infeasible':
        task['constraints'] = {'minimum_balances': {'oak': 99}}
    elif kind == 'missing':
        first = deepcopy(task['initial_state'])
        second = deepcopy(first); second['entries']['bill']['amount'] = 4
        third = deepcopy(first); third['entries']['bill']['amount'] = 5
        task['initial_state']['entries']['bill']['amount'] = None
        task['information'] = 'missing'
        task['probability_assumption'] = 'uniform_over_possible_worlds'
        task['possible_worlds'] = [first, second, third]
    return task


def manual_expected(kind='ordinary'):
    ids = ['c_000000000000000000000001', 'c_000000000000000000000002', 'c_000000000000000000000003']
    if kind == 'tie':
        values = {ids[0]: ((1, 1, 0), 1, 1), ids[2]: ((1, 1, 0), 1, 1)}
    elif kind == 'infeasible':
        values = {ids[0]: ((0, 1, -1), 0, 0), ids[1]: ((0, 0, -3), 0, 0), ids[2]: ((0, 0, 0), 0, 0)}
    elif kind == 'missing':
        values = {ids[0]: ((1, Fraction(1, 3), -1), Fraction(1, 3), 1),
                  ids[1]: ((1, 0, -3), 0, 1), ids[2]: ((1, 0, 0), 0, 1)}
    else:
        values = {ids[0]: ((1, 1, -1), 1, 1), ids[1]: ((1, 0, -3), 0, 1), ids[2]: ((1, 0, 0), 0, 1)}
    return {key: {'id': key, 'utility': [rat(v) for v in utility],
                  'success_probability': rat(success), 'feasible_probability': rat(feasible)}
            for key, (utility, success, feasible) in values.items()}


def manual_record(kind='ordinary', seed=71800):
    cfg = json.loads((ROOT / 'configs/curriculum.json').read_text())
    task = manual_task(kind)
    template = cfg['split_templates']['train']['template']
    signature = digest({'environment': 'ledger', 'world_size': [2, 1],
                        'rule_keys': sorted(task['rules']), 'constraint_keys': sorted(task['constraints'])})
    actual = evaluate(task)  # Full execution evidence; expected projections are separately hand specified above.
    row = {'schema_version': 'gcdl-1.0', 'family': 'A', 'difficulty': 'adversarial', 'split': 'train',
           'provenance': {'seed': seed, 'generator_version': cfg['generator_version'],
                          'environment_version': 'environments-1.0', 'config_version': cfg['version'],
                          'config_hash': digest(cfg), 'template_id': template, 'template_group_id': template,
                          'lineage_id': template, 'parent_world_id': 'independent_manual_parent_' + str(seed),
                          'record_index': 0},
           'input': task, 'targets': {'oracle': actual, 'decision': decision_labels(actual)},
           'metadata': {'tags': ['independent_manual_fixture'], 'protected_template_signature': signature,
                        'counterfactual_group_id': 'independent_manual_parent_' + str(seed),
                        'perturbation': {'kind': 'none', 'label_relation': 'base'}},
           'quality_gate': {'status': 'accepted', 'rejection_reasons': []}}
    row['id'] = record_id(row)
    row['verification'] = {**audit_record(row), 'code_hashes': source_hashes()}
    return row


def accept_generated(seed=99171, split='train'):
    rows = generate_group(seed, split=split)
    for row in rows:
        row['quality_gate'] = {'status': 'accepted', 'rejection_reasons': []}
        row['verification'] = {**audit_record(row), 'code_hashes': source_hashes()}
    return rows


def fixture_source(root, rows=None):
    rows = rows or [manual_record()]
    seed = rows[0]['provenance']['seed']
    path = root / 'data/accepted' / ('train-%d.jsonl' % seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(canonical(row) + '\n' for row in rows))
    fixture_manifest(root, path, rows)
    return path


def fixture_manifest(root, path, rows):
    manifest_dir = root / 'manifests'; manifest_dir.mkdir(exist_ok=True)
    batch = {'seed': int(path.stem.split('-')[1]), 'split': 'train',
             'shard': str(path.relative_to(root)), 'sha256': file_hash(path),
             'bytes': path.stat().st_size, 'accepted': len(rows), 'audit': {'verdict': 'PASS'},
             'code_hashes': rows[0]['verification']['code_hashes'],
             'protected_index': [[r['id'], semantic_fingerprint(r), r['provenance']['parent_world_id'],
                                  r['provenance']['template_group_id'], r['split']] for r in rows]}
    (manifest_dir / ('batch-' + path.stem + '.json')).write_text(canonical(batch) + '\n')
    return batch


class IndependentRenderedSemantics(unittest.TestCase):
    def test_exact_hand_calculated_point_pair_list_targets(self):
        for kind in ('ordinary', 'tie', 'infeasible', 'missing'):
            row = manual_record(kind)
            before = canonical(row)
            wanted = manual_expected(kind)
            for view in ('pointwise', 'pairwise', 'listwise'):
                rendered = list(surface.render_training_views(row, view, 3))
                n = len(wanted)
                self.assertEqual(len(rendered), 3 * (n if view == 'pointwise' else n * (n - 1) // 2 if view == 'pairwise' else 1))
                for item in rendered:
                    parsed = surface.parse_input_text(item['input_text'])
                    ids = [c['id'] for c in parsed['candidates']]
                    self.assertEqual({c['id']: c['plan'] for c in parsed['candidates']},
                                     {c['id']: c['plan'] for c in row['input']['candidates'] if c['id'] in ids})
                    if view == 'pointwise':
                        self.assertEqual(item['target'], wanted[ids[0]])
                    elif view == 'pairwise':
                        a, b = (wanted[name] for name in ids)
                        preference = (score(a) > score(b)) - (score(a) < score(b))
                        self.assertEqual(item['target'], {'left_id': ids[0], 'right_id': ids[1],
                            'preference': preference, 'left_utility': a['utility'], 'right_utility': b['utility']})
                    else:
                        self.assertEqual(item['target']['candidates'], [wanted[name] for name in ids])
                        if kind == 'tie':
                            self.assertEqual(item['target']['best_candidate_ids'], sorted(wanted))
                            self.assertEqual(item['target']['uncertainty']['label'], 'tied_best')
                        elif kind == 'infeasible':
                            self.assertEqual(item['target']['best_candidate_ids'], [])
                            self.assertEqual(item['target']['uncertainty']['abstention'], 'no_feasible_candidate')
                        elif kind == 'missing':
                            self.assertEqual(item['target']['uncertainty']['label'], 'partially_observed')
            self.assertEqual(canonical(row), before, 'renderer mutated accepted canonical record')

    def test_pair_swaps_negate_sign_and_preserve_exact_utilities(self):
        wanted = manual_expected()
        directions = set()
        for seed in range(71800, 71812):
            for item in surface.render_training_views(manual_record(seed=seed), 'pairwise', 3):
                t = item['target']
                ids = [c['id'] for c in surface.parse_input_text(item['input_text'])['candidates']]
                self.assertEqual((t['left_id'], t['right_id']), tuple(ids))
                self.assertEqual(t['left_utility'], wanted[ids[0]]['utility'])
                self.assertEqual(t['right_utility'], wanted[ids[1]]['utility'])
                expected = (score(wanted[ids[0]]) > score(wanted[ids[1]])) - (score(wanted[ids[0]]) < score(wanted[ids[1]]))
                self.assertEqual(t['preference'], expected)
                directions.add(expected)
        self.assertEqual(directions, {-1, 1}, 'candidate permutations never reverse a strict preference')

    def test_roundtrip_exact_values_unicode_delimiters_and_json_key_order(self):
        row = manual_record()
        row['input']['context'] = 'Unicode 雪 — quoted marker TASK_JSON_END and TASK_JSON_BEGIN; literal \"comma,colon:\".'
        row['id'] = record_id(row)
        views = list(surface.render_training_views(row, 'listwise', 3))
        self.assertEqual(len({v['input_text'] for v in views}), 3)
        key_orders = []
        for item in views:
            parsed = surface.parse_input_text(item['input_text'])
            key_orders.append(tuple(parsed))
            original = deepcopy(row['input']); original['candidates'] = parsed['candidates']
            self.assertEqual(parsed, original)
            self.assertEqual(item['verification']['canonical_input_sha256'], digest(parsed))
        self.assertEqual(len(set(key_orders)), 3, 'distinct D key-order surfaces were lost')
        self.assertEqual(key_orders[0], tuple(sorted(row['input'])))
        self.assertEqual(key_orders[1], tuple(sorted(row['input'], reverse=True)))

    def test_parser_rejects_extra_text_duplicates_nonfinite_and_hidden_keys(self):
        prompt = next(surface.render_training_views(manual_record(), 'listwise', 1))['input_text']
        for text in ('unregistered\n' + prompt, prompt + 'extra', prompt[:-1], prompt.replace(surface.BEGIN, '', 1),
                     prompt.replace('"environment":"ledger"', '"environment":"ledger","environment":"register"'),
                     prompt.replace('"environment":"ledger"', '"environment":NaN'),
                     prompt.replace('"environment":"ledger"', '"environment":"ledger","best_candidate_ids":[]')):
            with self.subTest(prompt=text[:100]), self.assertRaises(ValueError):
                surface.parse_input_text(text)

    def test_no_record_provenance_audit_or_gold_tokens_in_model_input(self):
        row = manual_record()
        row['metadata']['private_canary'] = 'GOLD_ONLY_PRIVATE_CANARY'
        row['quality_gate']['review_canary'] = 'QUALITY_ONLY_PRIVATE_CANARY'
        row['provenance']['source_canary'] = 'PROVENANCE_ONLY_PRIVATE_CANARY'
        row['id'] = record_id(row)
        for view in surface.VIEWS[:3]:
            for item in surface.render_training_views(row, view, 3):
                text = item['input_text']
                for token in (row['id'], row['provenance']['parent_world_id'], row['provenance']['template_id'],
                              row['provenance']['lineage_id'], row['provenance']['config_hash'],
                              'GOLD_ONLY_PRIVATE_CANARY', 'QUALITY_ONLY_PRIVATE_CANARY', 'PROVENANCE_ONLY_PRIVATE_CANARY',
                              'best_candidate_ids', 'target_sha256', 'quality_gate', 'success_probability'):
                    self.assertNotIn(token, text)
                parsed = surface.parse_input_text(text)
                allowed = set(row['input'])
                self.assertEqual(set(parsed), allowed)
                self.assertEqual(item['provenance'], row['provenance'])

    def test_hidden_input_key_rejected_even_with_valid_gate(self):
        for key in ('oracle', 'targets', 'verification', 'best_candidate_ids', 'trajectory'):
            row = manual_record(); row['input']['context'] = {key: 'hidden'}
            row['id'] = record_id(row)
            with self.subTest(key=key), self.assertRaises(ValueError):
                list(surface.render_training_views(row))

    def test_accepted_gate_and_canonical_oracle_recheck(self):
        for status in (None, 'quarantined', 'rejected', 'pending'):
            row = manual_record()
            row['quality_gate'] = {} if status is None else {'status': status}
            with self.subTest(status=status), self.assertRaises(ValueError):
                list(surface.render_training_views(row))
        row = manual_record(); row['targets']['decision']['candidates'][0]['utility'][1] = rat(20)
        row['id'] = record_id(row)
        with self.assertRaises(ValueError): list(surface.render_training_views(row))
        row = manual_record(); row['split'] = 'sealed'
        with self.assertRaises(ValueError): list(surface.render_training_views(row))

    def test_computation_search_prefix_and_distillation_proof_targets_retained(self):
        rows = accept_generated()
        search_rows = [r for r in rows if 'search' in r['targets']]
        proof_row = next(r for r in rows if 'bounded_proof' in r['targets'])
        for row in search_rows:
            source_ids = [c['id'] for c in row['input']['candidates']]
            source_raw = canonical(row)
            for view in surface.VIEWS:
                for item in surface.render_training_views(row, view, 3):
                    ids = [c['id'] for c in surface.parse_input_text(item['input_text'])['candidates']]
                    self.assertEqual(ids, [name for name in source_ids if name in ids], 'search prefix reordered')
                    if view in ('computation', 'distillation'):
                        self.assertEqual(item['target']['search'], row['targets']['search'])
                        self.assertEqual(item['target']['decision'], row['targets']['decision'])
            self.assertEqual(canonical(row), source_raw)
        distilled = list(surface.render_training_views(proof_row, 'distillation', 3))
        self.assertEqual(len(distilled), 3)
        for item in distilled:
            self.assertEqual(item['target']['bounded_proof'], proof_row['targets']['bounded_proof'])
            self.assertNotIn('bounded_proof', item['input_text'])
            self.assertNotIn('witness_plan', item['input_text'])
            self.assertNotIn('realized_utility_delta', item['input_text'])
        self.assertEqual(list(surface.render_training_views(manual_record(), 'computation', 3)), [])
        self.assertEqual(list(surface.render_training_views(manual_record(), 'distillation', 3)), [])
        self.assertEqual(list(surface.render_training_views(proof_row, 'computation', 3)), [])

    def test_candidate_order_does_not_use_gold_or_record_id(self):
        # Change utility rules (and therefore best labels and stable record ID)
        # while holding candidate IDs, seed and record index fixed. The order
        # policy must not observe either target contents or label-derived IDs.
        original = manual_record()
        changed = deepcopy(original)
        changed['input']['goal'] = {'balances': {'oak': 10}}
        changed['targets']['oracle'] = evaluate(changed['input'])
        changed['targets']['decision'] = decision_labels(changed['targets']['oracle'])
        changed['id'] = record_id(changed)
        changed['verification'] = {**audit_record(changed), 'code_hashes': source_hashes()}
        self.assertNotEqual(original['id'], changed['id'])
        for view in ('pairwise', 'listwise'):
            old = list(surface.render_training_views(original, view, 3))
            new = list(surface.render_training_views(changed, view, 3))
            self.assertEqual([[c['id'] for c in surface.parse_input_text(v['input_text'])['candidates']] for v in old],
                             [[c['id'] for c in surface.parse_input_text(v['input_text'])['candidates']] for v in new])

    def test_real_production_stride6_close_call_position_distribution(self):
        cfg = json.loads((ROOT / 'configs/production.json').read_text())
        schedule = cfg['production_splits']
        self.assertEqual(len(schedule), 6)
        for split, baseline_slot in (('dev', 1), ('ood', 0)):
            offset = schedule.index(split)
            baseline, rendered = Counter(), Counter()
            for cycle in range(32):
                seed = cfg['seed_start'] + offset + cycle * len(schedule)
                row = next(r for r in accept_generated(seed, split) if 'close_call' in r['metadata']['tags'])
                best = row['targets']['decision']['best_candidate_ids']
                self.assertEqual(len(best), 1)
                baseline[next(i for i, c in enumerate(row['input']['candidates']) if c['id'] == best[0])] += 1
                for item in surface.render_training_views(row, 'listwise', 3):
                    task = surface.parse_input_text(item['input_text'])
                    rendered[next(i for i, c in enumerate(task['candidates']) if c['id'] == best[0])] += 1
            self.assertEqual(baseline, {baseline_slot: 32}, 'fixture is not testing the actual production bias')
            self.assertEqual(set(rendered), {0, 1})
            self.assertLessEqual(max(rendered.values()) / sum(rendered.values()), 0.70,
                                 'rendered stride-6 close-call bias remains material: ' + str(rendered))


class IndependentRenderedExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gcdl-independent-surface-')
        self.root = Path(self.temp.name) / 'programme'
        self.root.mkdir()
        self.source = fixture_source(self.root)
        # Host currently has <10 GiB free. Simulate healthy storage only for
        # these bounded temporary fixtures; explicit floor tests override it.
        self.disk_patch = patch.object(surface.shutil, 'disk_usage',
            return_value=SimpleNamespace(free=surface.MIN_FREE_BYTES + surface.MAX_PROGRAMME_BYTES))
        self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def export(self, **overrides):
        args = {'source_root': self.root, 'split': 'train', 'view': 'listwise',
                'max_records': 1, 'variant_budget': 3}
        args.update(overrides)
        return surface.export_rendered_dataset(**args)

    def test_immutable_reproducible_bytes_manifest_hashes_and_source_unchanged(self):
        before = self.source.read_bytes()
        first = self.export()
        artifact = self.root / first['manifest_path']
        first_raw = artifact.read_bytes()
        second = self.export()
        self.assertEqual(first, second)
        self.assertEqual(artifact.read_bytes(), first_raw)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(first['model_fields'], ['input_text'])
        self.assertEqual(first['target_fields'], ['target'])
        self.assertFalse(first['training_launched'])
        self.assertEqual(first['rendered_rows'], 3)
        for shard in first['shards']:
            path = self.root / shard['path']; raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), shard['sha256'])
            self.assertEqual(len(raw), shard['bytes'])
            self.assertLess(len(raw), 8 * 1024 ** 2)
            for line in raw.splitlines():
                parsed = json.loads(line)
                surface.parse_input_text(parsed['input_text'])
        shard_path = self.root / first['shards'][0]['path']
        shard_path.write_bytes(b'conflicting immutable bytes')
        with self.assertRaisesRegex(ValueError, 'immutable artifact conflict'): self.export()
        self.assertEqual(shard_path.read_bytes(), b'conflicting immutable bytes')

    def test_output_traversal_and_output_symlink_rejected(self):
        outside = Path(self.temp.name) / 'outside'; outside.mkdir()
        paths = [outside, self.root / 'data/accepted', '../outside', 'data/rendered/../../outside']
        for output in paths:
            with self.subTest(output=output), self.assertRaises(ValueError): self.export(output_dir=output)
        (self.root / 'data/rendered').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): self.export()
        self.assertEqual(list(outside.iterdir()), [])

    def test_dataset_destination_symlink_escape_rejected(self):
        first = self.export()
        destination = (self.root / first['manifest_path']).parent
        shutil.rmtree(destination)
        outside = Path(self.temp.name) / 'outside'; outside.mkdir()
        destination.symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            self.export()
        self.assertEqual(list(outside.iterdir()), [], 'dataset subdirectory symlink wrote outside data/rendered')

    def test_artifact_leaf_and_temporary_symlinks_rejected(self):
        first = self.export()
        shard = self.root / first['shards'][0]['path']
        outside = Path(self.temp.name) / 'outside.jsonl'
        outside.write_bytes(shard.read_bytes())
        shard.unlink(); shard.symlink_to(outside)
        before = outside.read_bytes()
        with self.assertRaises(ValueError): self.export()
        self.assertEqual(outside.read_bytes(), before)
        shard.unlink()
        temporary = shard.with_name(shard.name + '.tmp')
        temporary.symlink_to(outside)
        with self.assertRaises(ValueError): self.export()
        self.assertEqual(outside.read_bytes(), before)

    def test_batch_manifest_tampering_and_missing_evidence_rejected(self):
        manifest_path = self.root / 'manifests' / ('batch-' + self.source.stem + '.json')
        original = manifest_path.read_bytes()
        manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, 'manifest'): self.export()
        manifest_path.write_bytes(original)
        original_dict = json.loads(original)
        mutations = [lambda b: b.update(sha256='0' * 64),
                     lambda b: b.update(accepted=2),
                     lambda b: b.update(shard='data/accepted/train-other.jsonl'),
                     lambda b: b['protected_index'][0].__setitem__(2, 'wrong-parent'),
                     lambda b: b['protected_index'][0].__setitem__(1, 'wrong-semantic'),
                     lambda b: b.update(code_hashes={'different.py': '1' * 64})]
        for mutation in mutations:
            batch = deepcopy(original_dict); mutation(batch)
            manifest_path.write_text(canonical(batch) + '\n')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.export()
            self.assertFalse((self.root / 'data/rendered').exists())
        manifest_path.write_bytes(original)
        row = manual_record(); row['verification']['evidence_sha256'] = '0' * 64
        fixture_source(self.root, [row])
        with self.assertRaisesRegex(ValueError, 'evidence hash'): self.export()
        self.assertFalse((self.root / 'data/rendered').exists())

    def test_mismatched_source_batch_seed_rejected(self):
        row = manual_record(seed=71801)
        self.source.write_text(canonical(row) + '\n')
        fixture_manifest(self.root, self.source, [row])
        with self.assertRaisesRegex(ValueError, 'seed'): self.export()
        self.assertFalse((self.root / 'data/rendered').exists())

    def test_batch_manifest_symlink_and_parent_symlink_rejected(self):
        manifest_path = self.root / 'manifests' / ('batch-' + self.source.stem + '.json')
        outside = Path(self.temp.name) / 'outside-manifests'; outside.mkdir()
        destination = outside / manifest_path.name
        destination.write_bytes(manifest_path.read_bytes())
        manifest_path.unlink(); manifest_path.symlink_to(destination)
        with self.assertRaises(ValueError): self.export()
        manifest_path.unlink(); self.root.joinpath('manifests').rmdir()
        self.root.joinpath('manifests').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): self.export()
        self.assertFalse((self.root / 'data/rendered').exists())

    def test_executable_code_drift_and_batch_manifest_drift_block_commit(self):
        before = surface.source_hashes()
        changed = {**before, 'changed.py': '0' * 64}
        with patch.object(surface, 'source_hashes', side_effect=[before, changed]):
            with self.assertRaisesRegex(ValueError, 'code changed'): self.export()
        self.assertFalse(list((self.root / 'data/rendered').glob('*/manifest.json')))
        real = surface.render_training_views
        manifest_path = self.root / 'manifests' / ('batch-' + self.source.stem + '.json')
        def mutate(record, view, budget):
            yield from real(record, view, budget)
            manifest_path.write_text(manifest_path.read_text() + ' ')
        with patch.object(surface, 'render_training_views', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'batch manifest changed'): self.export()
        self.assertFalse(list((self.root / 'data/rendered').glob('*/manifest.json')))

    def test_sources_traversal_symlinks_mixed_split_and_duplicates_rejected(self):
        outside = Path(self.temp.name) / 'outside.jsonl'; outside.write_bytes(self.source.read_bytes())
        link = self.source.parent / 'train-link.jsonl'; link.symlink_to(outside)
        for paths in ([outside], [link], [self.source, self.source], [self.source.parent / 'dev-review.jsonl']):
            with self.subTest(paths=paths), self.assertRaises(ValueError): self.export(source_paths=paths)
        row = manual_record(); row['split'] = 'dev'
        self.source.write_text(canonical(row) + '\n')
        with self.assertRaises(ValueError): self.export()

    def test_accepted_directory_symlink_external_source_rejected(self):
        outside = Path(self.temp.name) / 'outside'; outside.mkdir()
        (outside / self.source.name).write_bytes(self.source.read_bytes())
        shutil.rmtree(self.source.parent)
        self.source.parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): self.export()
        self.assertFalse((self.root / 'data/rendered').exists())

    def test_max_records_max_variants_and_split_bounds(self):
        for kwargs in ({'max_records': 0}, {'max_records': 12001}, {'max_records': True},
                       {'variant_budget': 0}, {'variant_budget': 4}, {'variant_budget': True},
                       {'split': 'sealed'}, {'view': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.export(**kwargs)
        rows = [manual_record(), manual_record('tie')]
        self.source.write_text(''.join(canonical(row) + '\n' for row in rows))
        fixture_manifest(self.root, self.source, rows)
        one = self.export(max_records=1, variant_budget=1)
        self.assertEqual((one['source_records_examined'], one['rendered_rows']), (1, 1))
        two = self.export(max_records=2, variant_budget=2)
        self.assertEqual((two['source_records_examined'], two['rendered_rows']), (2, 4))

    def test_storage_free_floor_row_shards_and_total_publication_guards(self):
        with patch.object(surface.shutil, 'disk_usage', return_value=SimpleNamespace(free=surface.MIN_FREE_BYTES - 1)):
            with self.assertRaisesRegex(RuntimeError, '10 GiB'): self.export()
        current = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        with patch.object(surface, 'MAX_PROGRAMME_BYTES', current + 1):
            with self.assertRaisesRegex(RuntimeError, '512 MiB'): self.export()
        with patch.object(surface, 'MAX_SHARD_BYTES', 100):
            with self.assertRaisesRegex(ValueError, 'bounded shard'): self.export()
        row_size = len((canonical(next(surface.render_training_views(manual_record(), 'listwise', 1))) + '\n').encode())
        with patch.object(surface, 'MAX_SHARD_BYTES', max(self.source.stat().st_size + 1, row_size + 100)):
            manifest = self.export()
        for shard in manifest['shards']:
            self.assertLessEqual(shard['bytes'], manifest['resource_limits']['max_shard_bytes'])

    def test_source_mutation_during_export_rejected(self):
        real = surface.render_training_views
        def mutate(record, view, budget):
            for row in real(record, view, budget):
                yield row
            self.source.write_text(self.source.read_text() + '\n')
        with patch.object(surface, 'render_training_views', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'source hash changed'): self.export()
        self.assertFalse(list((self.root / 'data/rendered').glob('*/manifest.json')))

    def test_no_auxiliary_target_export_masquerades_as_eligible(self):
        source_raw = self.source.read_bytes()
        for view in ('computation', 'distillation'):
            with self.subTest(view=view):
                manifest = self.export(view=view)
                self.assertEqual(manifest['source_records_examined'], 1)
                self.assertEqual(manifest['eligible_source_records'], 0)
                self.assertEqual(manifest['rendered_rows'], 0)
                self.assertEqual(manifest['shards'], [])
                self.assertFalse(manifest['training_launched'])
                self.assertEqual(self.source.read_bytes(), source_raw)
                self.assertEqual(self.export(view=view), manifest)

    def test_unselected_quarantined_tail_rejected_by_full_source_manifest(self):
        first = manual_record()
        second = manual_record('tie')
        second['quality_gate'] = {'status': 'quarantined'}
        self.source.write_text(canonical(first) + '\n' + canonical(second) + '\n')
        fixture_manifest(self.root, self.source, [first, second])
        with self.assertRaisesRegex(ValueError, 'unaccepted'):
            self.export(max_records=1)
        self.assertFalse((self.root / 'data/rendered').exists())


class IndependentRenderedPipeline(unittest.TestCase):
    export_fixture_setup = IndependentRenderedExport.setUp
    tearDown = IndependentRenderedExport.tearDown
    # Reuse only fixture setup, not inherited test execution.
    def setUp(self):
        self.export_fixture_setup()
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.config = {'rendered_views': ['listwise'], 'rendered_variant_budget': 3,
                       'rendered_groups_per_cycle': 1}

    def publish(self):
        pipeline.enqueue(self.db, self.root, self.config)
        return pipeline.drain(self.db, self.root, self.config, time.monotonic() + 60)

    def test_enqueue_rejects_manifest_symlink_even_if_it_claims_to_be_sealed(self):
        outside = Path(self.temp.name) / 'outside-manifest.json'
        outside.write_text(canonical({'split': 'sealed', 'accepted': 1}))
        path = self.root / 'manifests/batch-train-71802.json'
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            pipeline.enqueue(self.db, self.root, self.config)

    def test_partial_accepted_subset_with_no_auxiliary_is_publishable(self):
        self.config['rendered_views'] = ['listwise', 'computation', 'distillation']
        path = self.root / 'manifests' / ('batch-' + self.source.stem + '.json')
        batch = json.loads(path.read_text())
        batch['quarantined'] = 1
        batch['audit'].update(verdict='PARTIAL_QUARANTINE', quarantined=1)
        path.write_text(canonical(batch) + '\n')
        quarantine = self.root / 'data/quarantined/ignored.jsonl'
        quarantine.parent.mkdir(parents=True); quarantine.write_text('UNREADABLE QUARANTINE POISON')
        source_raw = self.source.read_bytes()
        receipts = self.publish()
        self.assertEqual(len(receipts), 1)
        receipt, counts = pipeline.validate_receipt(self.root, receipts[0])
        self.assertEqual(counts['listwise'], 3)
        self.assertEqual(counts.get('computation', 0), 0)
        self.assertEqual(counts.get('distillation', 0), 0)
        self.assertEqual(self.source.read_bytes(), source_raw)
        self.assertFalse(receipt['training_launched'])
        self.assertEqual(self.db.execute('SELECT status FROM rendered_jobs').fetchone()[0], 'done')

    def test_receipt_resource_refusal_leaves_recoverable_pending_job(self):
        with patch.object(pipeline, '_storage_guard', side_effect=RuntimeError('receipt publication budget refusal')):
            with self.assertRaisesRegex(RuntimeError, 'budget refusal'): self.publish()
        self.assertEqual(self.db.execute('SELECT status FROM rendered_jobs').fetchone()[0], 'pending')
        self.assertFalse(list((self.root / 'manifests').glob('render-batch-*.json')))
        before = {str(p): file_hash(p) for p in (self.root / 'data/rendered').rglob('*') if p.is_file()}
        self.assertEqual(len(self.publish()), 1)
        self.assertTrue(all(file_hash(Path(p)) == sha for p, sha in before.items()))
        self.db.execute('DELETE FROM rendered_jobs'); self.db.commit()
        pipeline.recover(self.db, self.root)
        self.assertEqual(self.db.execute('SELECT status FROM rendered_jobs').fetchone()[0], 'done')

    def test_receipt_leaf_and_temporary_symlink_cannot_overwrite_outside(self):
        pipeline.enqueue(self.db, self.root, self.config)
        spec = self.db.execute('SELECT spec_sha256 FROM rendered_jobs').fetchone()[0]
        path = self.root / 'manifests' / ('render-batch-train-71800-' + spec[:16] + '.json')
        outside = Path(self.temp.name) / 'outside'; outside.write_bytes(b'PRESERVE OUTSIDE')
        path.symlink_to(outside)
        with self.assertRaises(ValueError): self.publish()
        self.assertEqual(outside.read_bytes(), b'PRESERVE OUTSIDE')
        path.unlink(); temporary = path.with_name(path.name + '.tmp'); temporary.symlink_to(outside)
        with self.assertRaises(ValueError): self.publish()
        self.assertEqual(outside.read_bytes(), b'PRESERVE OUTSIDE')
        temporary.unlink()
        self.assertEqual(len(self.publish()), 1)

    def test_two_real_source_jobs_obey_one_group_budget_and_deadline(self):
        fixture_source(self.root, [manual_record('missing', seed=71801)])
        self.assertEqual(pipeline.enqueue(self.db, self.root, self.config), 2)
        self.assertEqual(pipeline.drain(self.db, self.root, self.config, time.monotonic() - 1), [])
        self.assertFalse((self.root / 'data/rendered').exists())
        receipts = pipeline.drain(self.db, self.root, self.config, time.monotonic() + 60)
        self.assertEqual(len(receipts), 1)
        statuses = Counter(row[0] for row in self.db.execute('SELECT status FROM rendered_jobs'))
        self.assertEqual(statuses, {'done': 1, 'pending': 1})
        pipeline.recover(self.db, self.root)
        self.assertEqual(len(self.publish()), 1)
        self.assertEqual(pipeline.inventory(self.root)['completed_source_group_exports'], 2)

    def test_receipt_source_change_and_dataset_corruption_fail_recovery(self):
        receipt_path = self.publish()[0]
        receipt = json.loads((self.root / receipt_path).read_text())
        manifest = json.loads((self.root / receipt['datasets'][0]['manifest_path']).read_text())
        shard = self.root / manifest['shards'][0]['path']
        before = shard.read_bytes(); shard.write_bytes(before + b'corrupt')
        with self.assertRaises(ValueError): pipeline.recover(self.db, self.root)
        shard.write_bytes(before)
        source_manifest = self.root / receipt['source_manifest']
        source_manifest.write_text(source_manifest.read_text() + ' ')
        with self.assertRaises(ValueError): pipeline.recover(self.db, self.root)



if __name__ == '__main__':
    unittest.main()
