"""Surface exports retain executable canonical gold and resource boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors.integrity import audit_record, semantic_fingerprint
from goal_conditioned_decision_data.decision_labels import rendering as render
from goal_conditioned_decision_data.generators.build import generate_group, record_id
from goal_conditioned_decision_data.oracles.primary import evaluate as primary
from goal_conditioned_decision_data.oracles.independent import evaluate as independent
from goal_conditioned_decision_data.pipelines.common import canonical, digest, file_hash


def fixtures(split='dev', seed=742):
    records = generate_group(seed, split=split)
    for record in records:
        record['quality_gate'] = {'status': 'accepted'}
        record['verification'] = {**audit_record(record), 'code_hashes': {'fixture-code.py': 'f' * 64}}
    return records


def candidate_map(oracle):
    return {candidate['id']: candidate for candidate in oracle['candidates']}


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = fixtures()
        cls.record = cls.records[0]

    def test_three_actual_nested_orders_roundtrip_oracles_and_source_immutability(self):
        source = deepcopy(self.record)
        rows = list(render.render_training_views(source))
        self.assertEqual(rows, list(render.render_training_views(source)))
        self.assertEqual(source, self.record)
        self.assertEqual(len({row['view_id'] for row in rows}), 3)
        tasks = [render.parse_input_text(row['input_text']) for row in rows]
        orders = [list(task) for task in tasks]
        self.assertEqual(orders[0], sorted(source['input']))
        self.assertEqual(orders[1], list(reversed(orders[0])))
        self.assertNotIn(orders[2], orders[:2])
        self.assertEqual(list(tasks[0]['rules']), sorted(source['input']['rules']))
        self.assertEqual(list(tasks[1]['rules']), list(reversed(sorted(source['input']['rules']))))
        for row, task in zip(rows, tasks):
            self.assertTrue(render._source_equivalent(task, source['input']))
            self.assertEqual(primary(task), independent(task))
            self.assertEqual(candidate_map(primary(task)), candidate_map(source['targets']['oracle']))
            self.assertEqual(row['record_id'], source['id'])
            self.assertEqual(row['split'], source['split'])
            self.assertEqual(row['provenance'], source['provenance'])
            self.assertEqual(row['target']['ranking'], source['targets']['decision']['ranking'])
            self.assertEqual(row['target']['best_candidate_ids'], source['targets']['decision']['best_candidate_ids'])

    def test_gold_independent_candidate_permutation_and_pair_target_orientation(self):
        rows = list(render.render_training_views(self.record, 'pairwise'))
        self.assertEqual(len(rows), 3 * 6)
        by_id = {candidate['id']: candidate for candidate in self.record['targets']['decision']['candidates']}
        for row in rows:
            task = render.parse_input_text(row['input_text'])
            ids = [candidate['id'] for candidate in task['candidates']]
            expected = sorted(ids, key=lambda name: digest([
                self.record['provenance']['seed'], self.record['provenance']['record_index'],
                row['surface']['name'], 'candidate_order', name]))
            self.assertEqual(ids, expected)
            target = row['target']
            self.assertEqual([target['left_id'], target['right_id']], ids)
            self.assertEqual(target['left_utility'], by_id[ids[0]]['utility'])
            self.assertEqual(target['right_utility'], by_id[ids[1]]['utility'])
            a, b = (render._score(by_id[name]) for name in ids)
            self.assertEqual(target['preference'], (a > b) - (a < b))
        # Even the nonmodel record ID and gold can change without affecting
        # the candidate-order recipe. Only target projection changes.
        altered = deepcopy(self.record)
        altered['id'] = 'not_used_for_order'
        altered['targets']['decision']['best_candidate_ids'] = []
        left = render._variant_task(self.record, self.record['input'], self.record['targets']['decision'], 'listwise', render.VARIANTS[0])[0]
        right = render._variant_task(altered, altered['input'], altered['targets']['decision'], 'listwise', render.VARIANTS[0])[0]
        self.assertEqual(left, right)

    def test_search_prefix_order_preserved_and_computation_gold_complete(self):
        for source in self.records:
            if 'search' not in source['targets']:
                continue
            for view in ('listwise', 'computation', 'distillation'):
                for row in render.render_training_views(source, view):
                    task = render.parse_input_text(row['input_text'])
                    self.assertEqual(task, source['input'])
                    self.assertEqual(row['surface']['candidate_order'], 'source_order_preserved')
                    if view != 'listwise':
                        self.assertEqual(row['target']['search'], source['targets']['search'])
                        self.assertEqual(row['target']['decision'], source['targets']['decision'])
                    self.assertNotIn('realized_utility_delta', row['input_text'])
                    self.assertNotIn('best_candidate_ids', row['input_text'])

    def test_distillation_retains_all_f_and_g_auxiliary_targets(self):
        selected = [record for record in self.records if record['family'] in ('F', 'G')]
        self.assertTrue(selected)
        for source in selected:
            rows = list(render.render_training_views(source, 'distillation'))
            self.assertEqual(len(rows), 3)
            for row in rows:
                for key in ('search', 'bounded_proof'):
                    if key in source['targets']:
                        self.assertEqual(row['target'][key], source['targets'][key])
                if 'bounded_proof' in source['targets']:
                    proof = row['target']['bounded_proof']
                    self.assertEqual(proof['scope'], 'finite_declared_action_alphabet_and_horizon')
                    self.assertFalse(proof['global_unbounded_optimality_claim'])
                    self.assertNotIn('witness_plan', row['input_text'])
                    self.assertNotIn('solution_exists_in_bound', row['input_text'])
        self.assertEqual(list(render.render_training_views(self.record, 'computation')), [])
        self.assertEqual(list(render.render_training_views(self.record, 'distillation')), [])

    def test_every_view_requires_accepted_and_disallows_sealed(self):
        for view in render.VIEWS:
            for status in (None, 'quarantined', 'rejected', 'pending'):
                source = deepcopy(self.record)
                source['quality_gate'] = {} if status is None else {'status': status}
                with self.assertRaisesRegex(ValueError, 'accepted'):
                    list(render.render_training_views(source, view))
            source = fixtures('sealed')[0]
            with self.assertRaisesRegex(ValueError, 'sealed'):
                list(render.render_training_views(source, view))
        for budget in (0, 4, True, 1.2):
            with self.assertRaises(ValueError):
                list(render.render_training_views(self.record, variant_budget=budget))

    def test_model_text_is_only_fixed_framing_and_inference_input(self):
        source = deepcopy(self.record)
        source['metadata']['secret_audit_note'] = 'THIS_AUDIT_SECRET_MUST_NOT_APPEAR'
        source['verification']['internal'] = 'THIS_VERIFICATION_SECRET_MUST_NOT_APPEAR'
        source['id'] = record_id(source)
        for view in ('pointwise', 'pairwise', 'listwise'):
            for row in render.render_training_views(source, view):
                self.assertNotIn('THIS_AUDIT_SECRET', row['input_text'])
                self.assertNotIn('THIS_VERIFICATION_SECRET', row['input_text'])
                task = render.parse_input_text(row['input_text'])
                self.assertFalse({'metadata', 'targets', 'oracle', 'quality_gate', 'verification'} & set(task))
        bad = deepcopy(self.record)
        bad['input']['context'] = {'terminal_state': {'secret': 1}}
        bad['id'] = record_id(bad)
        with self.assertRaisesRegex(ValueError, 'oracle-only'):
            list(render.render_training_views(bad))

    def test_parser_rejects_duplicate_nonfinite_extra_text_and_handles_delimiter_strings(self):
        row = next(render.render_training_views(self.record))
        with self.assertRaises(ValueError):
            render.parse_input_text(row['input_text'] + 'unregistered suffix')
        prefix, suffix = render._FRAMING[0] + render.BEGIN, render.END + render._REQUEST['listwise']
        for task_json in ('{"goal":1,"goal":2}', '{"x":NaN}', '[]'):
            with self.assertRaises(ValueError):
                render.parse_input_text(prefix + task_json + suffix)
        source = deepcopy(self.record)
        source['input']['context'] = '\nTASK_JSON_END\nTASK_JSON_BEGIN\n'
        source['id'] = record_id(source)
        for row in render.render_training_views(source):
            self.assertEqual(render.parse_input_text(row['input_text'])['context'], source['input']['context'])


class ExportTests(unittest.TestCase):
    def setUp(self):
        # Fixture correctness must not depend on unrelated host disk usage.
        # The explicit low-space test below overrides this patch.
        disk = type('Disk', (), {'free': 100 * 1024 ** 3})()
        self.disk_patch = patch.object(render.shutil, 'disk_usage', return_value=disk)
        self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)

    def create_source(self, directory, records=None):
        root = Path(directory)/'goal_conditioned_decision_data'
        root.mkdir()
        accepted = root / 'data' / 'accepted'
        accepted.mkdir(parents=True)
        records = records or fixtures()[:3]
        path = accepted / 'dev-742.jsonl'
        path.write_text(''.join(canonical(record) + '\n' for record in records), encoding='utf-8')
        manifests = root / 'manifests'
        manifests.mkdir(exist_ok=True)
        manifest = {'seed': 742, 'split': 'dev', 'shard': 'data/accepted/dev-742.jsonl',
                    'sha256': file_hash(path), 'bytes': path.stat().st_size, 'accepted': len(records),
                    'audit': {'verdict': 'PASS'}, 'code_hashes': records[0]['verification']['code_hashes'],
                    'protected_index': [[record['id'], semantic_fingerprint(record), record['provenance']['parent_world_id'],
                                         record['provenance']['template_group_id'], 'dev'] for record in records]}
        (manifests / 'batch-dev-742.json').write_text(canonical(manifest) + '\n')
        return root, path

    def test_bounded_immutable_export_manifest_hashes_replay_and_gold_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory)
            original = source.read_bytes()
            manifest = render.export_rendered_dataset(root, 'dev', 'listwise', 2, 2)
            self.assertEqual(manifest['source_records_examined'], 2)
            self.assertEqual(manifest['rendered_rows'], 4)
            self.assertFalse(manifest['training_launched'])
            self.assertTrue(manifest['code_hashes'])
            self.assertEqual(manifest['model_fields'], ['input_text'])
            self.assertEqual(source.read_bytes(), original)
            for shard in manifest['shards']:
                path = root / shard['path']
                self.assertEqual(file_hash(path), shard['sha256'])
                self.assertEqual(path.stat().st_size, shard['bytes'])
                self.assertLess(shard['bytes'], 8 * 1024 ** 2)
            before = {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}
            self.assertEqual(manifest, render.export_rendered_dataset(root, 'dev', 'listwise', 2, 2))
            self.assertEqual(before, {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()})
            shard_path = root / manifest['shards'][0]['path']
            shard_path.write_bytes(b'tampered\n')
            with self.assertRaisesRegex(ValueError, 'immutable artifact conflict'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 2, 2)

    def test_no_sealed_nonaccepted_source_or_destination_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory)
            with self.assertRaisesRegex(ValueError, 'sealed'):
                render.export_rendered_dataset(root, 'sealed', 'listwise', 2, 2)
            with self.assertRaisesRegex(ValueError, 'output_dir'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 2, 2, root / 'data' / 'accepted')
            sealed = root / 'data' / 'sealed' / 'dev-secret.jsonl'
            sealed.parent.mkdir()
            sealed.write_text('this sealed content must never be decoded')
            with self.assertRaisesRegex(ValueError, 'data/accepted'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 2, 2, source_paths=[sealed])
            records = fixtures()[:1]
            records[0]['quality_gate']['status'] = 'quarantined'
            source.write_text(canonical(records[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'unaccepted'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            self.assertFalse((root / 'data' / 'rendered').exists())

    def test_publication_reserves_both_storage_and_free_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory)
            with patch.object(render, 'MAX_PROGRAMME_BYTES', source.stat().st_size + 1):
                with self.assertRaisesRegex(RuntimeError, 'storage'):
                    render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            disk = type('Disk', (), {'free': render.MIN_FREE_BYTES})()
            with patch.object(render.shutil, 'disk_usage', return_value=disk):
                with self.assertRaisesRegex(RuntimeError, 'free'):
                    render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            for max_records in (0, 12001, True):
                with self.assertRaises(ValueError):
                    render.export_rendered_dataset(root, 'dev', 'listwise', max_records, 1)

    def test_selected_shard_and_filtered_auxiliary_count(self):
        records = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory, records)
            manifest = render.export_rendered_dataset(root, 'dev', 'computation', len(records), 1, source_paths=[source])
            expected = sum('search' in record['targets'] for record in records)
            self.assertEqual(manifest['source_records_examined'], len(records))
            self.assertEqual(manifest['eligible_source_records'], expected)
            self.assertEqual(manifest['rendered_rows'], expected)
            self.assertEqual(manifest['sources'][0]['sha256'], file_hash(source))

    def test_unmanaged_mutated_receipt_and_false_verification_sources_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory)
            receipt = root / 'manifests' / 'batch-dev-742.json'
            raw = receipt.read_bytes()
            receipt.unlink()
            with self.assertRaisesRegex(ValueError, 'manifest'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            receipt.write_bytes(raw)
            batch = json.loads(raw)
            batch['accepted'] += 1
            receipt.write_text(canonical(batch))
            with self.assertRaisesRegex(ValueError, 'row-count'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            receipt.write_bytes(raw)
            source.write_bytes(source.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
        source = deepcopy(fixtures()[0])
        source['verification']['evidence_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'evidence hash'):
            list(render.render_training_views(source))

    def test_mixed_batch_exports_only_accepted_shard_with_explicit_consistent_counts(self):
        records = fixtures()
        accepted, quarantined = records[:3], deepcopy(records[3])
        quarantined['quality_gate'] = {'status': 'quarantined', 'rejection_reasons': ['review fixture']}
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory, accepted)
            quarantine_path = root / 'data' / 'quarantined' / 'dev-742.jsonl'
            quarantine_path.parent.mkdir()
            quarantine_path.write_text(canonical(quarantined) + '\n')
            receipt = root / 'manifests' / 'batch-dev-742.json'
            batch = json.loads(receipt.read_text())
            batch.update(quarantined=1, audit={'verdict': 'PARTIAL_QUARANTINE', 'accepted': 3, 'quarantined': 1})
            receipt.write_text(canonical(batch) + '\n')
            manifest = render.export_rendered_dataset(root, 'dev', 'listwise', 3, 3)
            self.assertEqual(manifest['source_records_examined'], 3)
            self.assertEqual(manifest['rendered_rows'], 9)
            rows = [json.loads(line) for shard in manifest['shards'] for line in (root / shard['path']).read_text().splitlines()]
            self.assertEqual({row['record_id'] for row in rows}, {record['id'] for record in accepted})
            self.assertNotIn(quarantined['id'], {row['record_id'] for row in rows})
            for change in ({'quarantined': 0}, {'quarantined': True},
                           {'audit': {'verdict': 'PARTIAL_QUARANTINE', 'accepted': 4, 'quarantined': 1}},
                           {'audit': {'verdict': 'PARTIAL_QUARANTINE', 'accepted': 3, 'quarantined': 2}},
                           {'audit': {'verdict': 'FAIL', 'accepted': 3, 'quarantined': 1}}):
                invalid = {**batch, **change}
                receipt.write_text(canonical(invalid) + '\n')
                with self.assertRaisesRegex(ValueError, 'audit.*count mismatch'):
                    render.export_rendered_dataset(root, 'dev', 'listwise', 3, 1)
            invalid = deepcopy(batch)
            invalid.pop('quarantined')
            receipt.write_text(canonical(invalid) + '\n')
            with self.assertRaisesRegex(ValueError, 'audit.*count mismatch'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 3, 1)
            # Even a correctly hashed, consistent partial-batch receipt cannot
            # make a quarantined row inside the accepted shard exportable.
            source.write_text(source.read_text() + canonical(quarantined) + '\n')
            poisoned = deepcopy(batch)
            poisoned.update(sha256=file_hash(source), bytes=source.stat().st_size, accepted=4)
            poisoned['audit']['accepted'] = 4
            poisoned['protected_index'].append([quarantined['id'], semantic_fingerprint(quarantined),
                                                quarantined['provenance']['parent_world_id'],
                                                quarantined['provenance']['template_group_id'], 'dev'])
            receipt.write_text(canonical(poisoned) + '\n')
            # Full shard flag validation prevents an unselected quarantined
            # tail from being treated as an accepted source either.
            for limit in (1, 4):
                with self.assertRaisesRegex(ValueError, 'unaccepted'):
                    render.export_rendered_dataset(root, 'dev', 'listwise', limit, 1)

    def test_auxiliary_ineligible_accepted_source_has_deterministic_empty_manifest(self):
        accepted = fixtures()[:3]
        self.assertTrue(all(not {'search', 'bounded_proof'}.intersection(row['targets']) for row in accepted))
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.create_source(directory, accepted)
            original = source.read_bytes()
            for view in ('computation', 'distillation'):
                manifest = render.export_rendered_dataset(root, 'dev', view, 3, 3)
                self.assertEqual(manifest['source_records_examined'], 3)
                self.assertEqual(manifest['eligible_source_records'], 0)
                self.assertEqual(manifest['rendered_rows'], 0)
                self.assertEqual(manifest['shards'], [])
                self.assertEqual({item['record_id'] for item in manifest['selected_records']}, {row['id'] for row in accepted})
                self.assertEqual(manifest['sources'][0]['sha256'], file_hash(source))
                self.assertEqual(manifest, render.export_rendered_dataset(root, 'dev', view, 3, 3))
                self.assertFalse(manifest['training_launched'])
                self.assertEqual(json.loads((root / manifest['manifest_path']).read_text()), manifest)
            self.assertEqual(source.read_bytes(), original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'goal_conditioned_decision_data'
            root.mkdir()
            (root / 'data' / 'accepted').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'no selected accepted records'):
                render.export_rendered_dataset(root, 'dev', 'computation', 1, 1)
            self.assertFalse((root / 'data' / 'rendered').exists())

    def test_dataset_shard_manifest_and_temporary_symlink_escapes_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root, source = self.create_source(directory)
            manifest = render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            destination = (root / manifest['manifest_path']).parent
            render.shutil.rmtree(destination)
            destination.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            self.assertEqual(list(Path(outside).iterdir()), [])
            destination.unlink()
            manifest = render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            shard = root / manifest['shards'][0]['path']
            contents = shard.read_bytes()
            outside_shard = Path(outside) / 'shard.jsonl'
            outside_shard.write_bytes(contents)
            shard.unlink()
            shard.symlink_to(outside_shard)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            shard.unlink()
            temporary = shard.with_name(shard.name + '.tmp')
            temporary.symlink_to(outside_shard)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                render.export_rendered_dataset(root, 'dev', 'listwise', 1, 1)
            self.assertEqual(outside_shard.read_bytes(), contents)

    def test_cli_requires_explicit_bounds_and_excludes_sealed_choice(self):
        script = Path(render.__file__).resolve().parents[1] / 'scripts' / 'export_training.py'
        result = subprocess.run([sys.executable, str(script), '--split', 'sealed', '--max-records', '1', '--variant-budget', '1'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)
        result = subprocess.run([sys.executable, str(script), '--split', 'dev'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--max-records', result.stderr)
        self.assertIn('--variant-budget', result.stderr)


if __name__ == '__main__':
    unittest.main()
