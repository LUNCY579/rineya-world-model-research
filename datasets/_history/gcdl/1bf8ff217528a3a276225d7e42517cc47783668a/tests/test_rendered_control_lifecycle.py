"""Independent control lifecycle checks using hand-specified temporary metadata.

Rendered/source payloads are opaque fixture bytes, never parsed prompts/gold.
No H/v2 generation, official experiment, production mutation or real transport.
The separately owned metadata reviewer challenges qualified projection custody.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors import release
from goal_conditioned_decision_data.decision_labels import rendering
from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.pipelines import render_controls, rendered, worker
from goal_conditioned_decision_data.pipelines.common import canonical, digest
from goal_conditioned_decision_data.scripts import archive_cycle as cycle


class LabelFreeInventory:
    """Small manually authenticated metadata facade; no raw row decoding."""
    def __init__(self, root):
        self.root = root
        self.cold = {}
        self.reads = []
        self.offline = False
        catalog = 'a' * 64
        snapshot = 'b' * 40
        self.index = {'files': {}, 'receipts': {}, 'catalogs': {catalog: {
            'catalog_commit': snapshot, 'archive_ref': archive.archive_ref(snapshot),
            'catalog_path': archive.catalog_path(catalog), 'catalog_sha256': 'c' * 64,
            'catalog_git_blob_sha1': 'd' * 40,
        }}}

    def write(self, relative, raw):
        if not isinstance(raw, bytes):
            raw = (canonical(raw) + '\n').encode()
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def assert_root(self, root):
        if Path(root).resolve() != self.root:
            raise AssertionError('fixture escaped its programme')

    def raw(self, relative):
        path = self.root / relative
        if path.is_file():
            return path.read_bytes()
        if relative in self.cold:
            if self.offline:
                raise OSError('fixture cold header unavailable')
            return self.cold[relative]
        raise FileNotFoundError(relative)

    def read_bytes(self, root, relative, *, max_bytes=archive.MAX_READ_BYTES, **kwargs):
        self.assert_root(root); self.reads.append(relative)
        raw = self.raw(relative)
        if len(raw) > max_bytes:
            raise archive.ArchiveError('fixture byte bound exceeded')
        return raw

    def read_text(self, *args, **kwargs):
        return self.read_bytes(*args, **kwargs).decode()

    def payload_info(self, root, relative):
        self.assert_root(root)
        path = self.root / relative
        raw = path.read_bytes() if path.is_file() else self.cold.get(relative)
        if raw is None:
            raise FileNotFoundError(relative)
        return {'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob_sha1': archive.git_blob_sha1(raw),
                'bytes': len(raw), 'archived': relative in self.index['files']}

    def iter_paths(self, root, pattern):
        self.assert_root(root)
        paths = {str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file()}
        paths.update(p for p in self.cold if fnmatch.fnmatchcase(p, pattern))
        return sorted(paths)

    def index_paths(self, paths, *, remove=False, catalog_id=None):
        for relative in paths:
            raw = self.raw(relative); info = self.payload_info(self.root, relative)
            catalog = catalog_id or self.index['files'].get(relative, {}).get('catalog_id', 'a' * 64)
            self.index['files'][relative] = {key: info[key] for key in ('sha256', 'git_blob_sha1', 'bytes')}
            self.index['files'][relative].update(commit='e' * 40, catalog_id=catalog, receipt_id='f' * 64)
            self.cold[relative] = raw
            if remove:
                (self.root / relative).unlink()

    def cohort(self, seed, *, empty_auxiliary=False, split='dev', accepted=1, rows_by_view=None):
        code = {'fixture.py': '1' * 64}
        protected = [[f'fixture-record-{seed}-{i}', f'fixture-semantic-{seed}-{i}',
                      f'fixture-parent-{seed}', 'fixture-template', split] for i in range(accepted)]
        source = f'data/accepted/{split}-{seed}.jsonl'
        raw_source = self.write(source, b'opaque source fixture, not record JSON\n')
        batch_path = f'manifests/batch-{split}-{seed}.json'
        batch = {'seed': seed, 'split': split, 'shard': source,
                 'sha256': hashlib.sha256(raw_source).hexdigest(), 'bytes': len(raw_source),
                 'accepted': accepted, 'quarantined': 0, 'audit': {'verdict': 'PASS'},
                 'code_hashes': code, 'protected_index': protected}
        raw_batch = self.write(batch_path, batch)
        specification = {'export_version': rendering.EXPORT_VERSION,
                         'rendering_version': rendering.RENDERING_POLICY_SET_VERSION,
                         'supported_rendering_versions': list(rendering.SUPPORTED_RENDERING_VERSIONS),
                         'views': ['listwise', 'computation', 'distillation'], 'variant_budget': 3,
                         'code_sha256': digest(code)}
        datasets = []; payloads = []
        for view in specification['views']:
            identifier = 'gcdl_rendered_' + digest([seed, view])[:32]
            header = f'data/rendered/{identifier}/manifest.json'
            count = 0 if empty_auxiliary and view != 'listwise' else 3
            if rows_by_view is not None:
                count = rows_by_view[view]
            eligible = (count // 3) if rows_by_view is not None else int(bool(count))
            shards = []
            if count:
                shard = f'data/rendered/{identifier}/shard-00000.jsonl'
                raw = self.write(shard, ('opaque rendered ' + view + '\n').encode())
                shards.append({'path': shard, 'sha256': hashlib.sha256(raw).hexdigest(),
                               'bytes': len(raw), 'rows': count})
                payloads.append(shard)
            manifest = {'manifest_path': header, 'dataset_id': identifier, 'view': view, 'split': split,
                        'export_version': rendering.EXPORT_VERSION, 'variant_budget': 3,
                        'rendering_version': rendering.SUPPORTED_RENDERING_VERSIONS[0],
                        'rendering_versions': [rendering.SUPPORTED_RENDERING_VERSIONS[0]],
                        'rendering_policy_set_version': rendering.RENDERING_POLICY_SET_VERSION,
                        'source_records_examined': accepted, 'eligible_source_records': eligible,
                        'rendered_rows': count, 'code_hashes': code, 'shards': shards,
                        'sources': [{'path': source, 'sha256': batch['sha256'],
                                     'source_batch_manifest': batch_path,
                                     'source_batch_manifest_sha256': hashlib.sha256(raw_batch).hexdigest(),
                                     'source_code_hashes': code, 'protected_index': protected}],
                        'training_launched': False}
            raw = self.write(header, manifest); payloads.append(header)
            datasets.append({'view': view, 'dataset_id': identifier, 'manifest_path': header,
                             'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'rendered_rows': count,
                             'eligible_source_records': eligible})
        receipt = {'version': 'gcdl-rendered-batch-1', 'seed': seed, 'split': split,
                   'source_manifest': batch_path, 'source_sha256': hashlib.sha256(raw_batch).hexdigest(),
                   'source_shard_sha256': batch['sha256'], 'specification': specification,
                   'spec_sha256': digest(specification), 'code_hashes': code, 'datasets': datasets,
                   'training_launched': False}
        receipt_path = f'manifests/render-batch-{split}-{seed}-{receipt["spec_sha256"][:16]}.json'
        self.write(receipt_path, receipt)
        return receipt_path, receipt, sorted(payloads)


class MockControlSnapshot:
    """Only label-free control bundle objects, retained at fixed fake commits."""
    def __init__(self):
        self.deadline = time.monotonic() + 600
        self.refs = {archive.archive_ref('b' * 40): 'b' * 40}
        self.trees = {}; self.blobs = {}
        self.calls = []

    def add(self, root, relative, commit):
        raw = (root / relative).read_bytes(); oid = archive.git_blob_sha1(raw)
        self.refs[archive.archive_ref(commit)] = commit
        self.trees[commit] = {archive.PROGRAMME + '/' + relative:
                              {'type': 'blob', 'mode': '100644', 'sha': oid, 'size': len(raw)}}
        self.blobs[oid] = raw

    def fetch_ref(self, reference):
        self.calls.append(('ref', reference)); return self.refs.get(reference)

    def fetch_tree(self, commit):
        self.calls.append(('tree', commit)); return deepcopy(self.trees[commit])

    def fetch_blob(self, oid, max_bytes):
        self.calls.append(('blob', oid)); raw = self.blobs[oid]
        if len(raw) > max_bytes:
            raise archive.ArchiveError('mock control snapshot byte bound')
        return raw


class IndependentRenderedControlLifecycle(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gcdl-render-control-lifecycle-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / archive.PROGRAMME
        (self.root / 'state').mkdir(parents=True)
        (self.root / 'state/worker.lock').touch()
        self.inventory = LabelFreeInventory(self.root)
        for name in ('read_bytes', 'read_text', 'payload_info', 'iter_paths'):
            hook = patch.object(archive, name, side_effect=getattr(self.inventory, name))
            hook.start(); self.addCleanup(hook.stop)
        hook = patch.object(archive, 'load_index', side_effect=lambda root: deepcopy(self.inventory.index))
        hook.start(); self.addCleanup(hook.stop)
        hook = patch.object(archive, '_run', side_effect=AssertionError('no real authentication/network in lifecycle tests'))
        hook.start(); self.addCleanup(hook.stop)

    def raw_files(self, paths):
        return {relative: (self.root / relative).read_bytes() for relative in paths}

    def test_cohort_selection_respects_twelve_file_limit_including_runtime_priority(self):
        _, _, first = self.inventory.cohort(201)
        _, _, second = self.inventory.cohort(202)
        priority = 'data/backups/runtime-fixture.tar.gz'
        self.inventory.write(priority, b'opaque runtime')
        self.assertEqual(cycle.MAX_FILES, 12)
        self.assertEqual(cycle.MAX_BYTES, 128 * 1024 ** 2)
        selected = cycle._select_payloads(self.root, self.inventory.index, [priority])
        self.assertEqual(selected[0], priority)
        self.assertLessEqual(len(selected), 12)
        for paths in (first, second):
            overlap = set(paths) & set(selected)
            self.assertIn(len(overlap), (0, len(paths)))
        self.assertEqual(sum(bool(set(paths) & set(selected)) for paths in (first, second)), 1)

    def test_partial_cold_cohort_precedes_new_cohort_and_selects_only_fresh_members(self):
        _, _, first = self.inventory.cohort(210)
        _, _, second = self.inventory.cohort(211)
        _, _, third = self.inventory.cohort(212)
        self.inventory.index_paths(first[:1], remove=True)
        priority = 'data/backups/runtime-fixture.tar.gz'; self.inventory.write(priority, b'opaque runtime')
        selected = cycle._select_payloads(self.root, self.inventory.index, [priority])
        self.assertEqual(selected[1:6], sorted(set(first) - set(first[:1])))
        self.assertNotIn(first[0], selected)
        self.assertTrue(set(first[1:]) <= set(selected))
        self.assertEqual(sum(bool(set(paths) & set(selected)) for paths in (second, third)), 1)
        self.assertLessEqual(len(selected), 12)

    def test_byte_budget_never_splits_a_completed_cohort(self):
        _, _, first = self.inventory.cohort(220)
        _, _, second = self.inventory.cohort(221)
        first_size = sum((self.root / p).stat().st_size for p in first)
        second_size = sum((self.root / p).stat().st_size for p in second)
        before = self.raw_files(first + second)
        with patch.object(cycle, 'MAX_BYTES', first_size + second_size - 1):
            selected = cycle._select_payloads(self.root, self.inventory.index, [])
        self.assertTrue(set(first) <= set(selected))
        self.assertFalse(set(second) & set(selected))
        self.assertEqual(self.raw_files(first + second), before)
        with patch.object(cycle, 'MAX_BYTES', first_size - 1):
            with self.assertRaisesRegex(RuntimeError, 'cohort exceeds'):
                cycle._select_payloads(self.root, self.inventory.index, [])

    def test_zero_row_auxiliary_headers_are_in_the_same_cohort(self):
        _, _, paths = self.inventory.cohort(230, empty_auxiliary=True)
        self.assertEqual(len(paths), 4)
        selected = cycle._select_payloads(self.root, self.inventory.index, [])
        self.assertTrue(set(paths) <= set(selected))

    def test_prepared_capsules_without_authority_do_not_repair_database_while_raw_is_unavailable(self):
        receipt_path, receipt, paths = self.inventory.cohort(240)
        self.inventory.index_paths(paths)
        staged = render_controls.prepare(self.root, [receipt_path])
        self.assertEqual(len(staged), 1)
        self.assertTrue((self.root / staged[0]).is_file())
        self.assertFalse((self.root / render_controls.POINTER).exists())
        headers = [item['manifest_path'] for item in receipt['datasets']]
        self.inventory.index_paths(headers, remove=True); self.inventory.offline = True
        db = sqlite3.connect(':memory:'); self.addCleanup(db.close)
        rendered.ensure_queue(db)
        db.execute('INSERT INTO rendered_jobs VALUES (?,?,?,?,?,?,?)',
                   (receipt['source_manifest'], receipt['source_sha256'], receipt['spec_sha256'],
                    receipt['seed'], receipt['split'], 'pending', None)); db.commit()
        with self.assertRaisesRegex(OSError, 'cold header unavailable'):
            rendered.recover(db, self.root)
        self.assertEqual(db.execute('SELECT status,receipt FROM rendered_jobs').fetchall(), [('pending', None)])
        self.assertFalse(any((self.root / path).exists() for path in headers))

    def test_prepared_unqualified_capsules_use_full_raw_fallback_when_available(self):
        relative, receipt, paths = self.inventory.cohort(241)
        self.inventory.index_paths(paths)
        render_controls.prepare(self.root, [relative]); self.inventory.reads.clear()
        _, rows = rendered.validate_receipt(self.root, relative)
        self.assertEqual(rows, {'listwise': 3, 'computation': 3, 'distillation': 3})
        self.assertTrue({d['manifest_path'] for d in receipt['datasets']} <= set(self.inventory.reads))

    def test_publication_hook_backs_up_then_pins_then_activates_then_backs_up_authority(self):
        calls = []; provider = object(); deadline = time.monotonic() + 30
        paths = ['manifests/render-batch-dev-fixture.json']
        with patch.object(render_controls, 'pending', return_value=paths), \
             patch.object(cycle, 'current_independent_review', return_value=True), \
             patch.object(render_controls, 'prepare', side_effect=lambda *a, **k: calls.append('prepare') or ['capsule']), \
             patch.object(render_controls, 'plan_bundle', side_effect=lambda *a, **k: calls.append('plan') or {'bundle_path': 'bundle'}), \
             patch.object(cycle, '_backup', side_effect=lambda *a, **k: calls.append('backup') or 'b' * 40), \
             patch.object(cycle, '_publish_snapshot', side_effect=lambda *a, **k: calls.append('snapshot')), \
             patch.object(render_controls, 'activate', side_effect=lambda *a, **k: calls.append('activate') or {'qualified': True}):
            result = cycle._publish_render_controls(self.root, deadline, provider, None)
        self.assertEqual(calls, ['prepare', 'plan', 'backup', 'snapshot', 'activate', 'backup'])
        self.assertEqual(result, {'qualified': True})

    def test_operator_pause_during_plan_preserves_raw_and_refuses_activation(self):
        relative, _, paths = self.inventory.cohort(242)
        before = self.raw_files(paths)
        operator_pause = (canonical({'reason': 'operator requested pause'}) + '\n').encode()
        def pause_during_plan(*args, **kwargs):
            self.inventory.write('state/PAUSED.json', operator_pause)
            return {'bundle_path': 'bundle'}
        with patch.object(render_controls, 'pending', return_value=[relative]), \
             patch.object(cycle, 'current_independent_review', return_value=True), \
             patch.object(render_controls, 'prepare', return_value=['capsule']), \
             patch.object(render_controls, 'plan_bundle', side_effect=pause_during_plan), \
             patch.object(render_controls, 'activate') as activate, \
             patch.object(cycle, '_backup') as backup, \
             patch.object(archive, 'evict_verified') as evict:
            with self.assertRaisesRegex(RuntimeError, 'pause changed'):
                cycle._publish_render_controls(self.root, time.monotonic() + 30, object(), None)
        activate.assert_not_called(); backup.assert_not_called(); evict.assert_not_called()
        self.assertEqual((self.root / 'state/PAUSED.json').read_bytes(), operator_pause)
        self.assertEqual(self.raw_files(paths), before)
        self.assertFalse((self.root / render_controls.POINTER).exists())

    def test_changed_source_approval_during_plan_is_not_self_approved(self):
        relative, _, paths = self.inventory.cohort(243)
        before = self.raw_files(paths); version = {'current': 'approved'}
        def change_source(*args, **kwargs):
            version['current'] = 'unapproved-change'
            return {'bundle_path': 'bundle'}
        with patch.object(render_controls, 'pending', return_value=[relative]), \
             patch.object(cycle, 'current_independent_review', side_effect=lambda **kwargs: version['current'] == 'approved'), \
             patch.object(render_controls, 'prepare', return_value=['capsule']), \
             patch.object(render_controls, 'plan_bundle', side_effect=change_source), \
             patch.object(render_controls, 'activate') as activate, \
             patch.object(cycle, '_backup') as backup, \
             patch.object(archive, 'evict_verified') as evict:
            with self.assertRaisesRegex(RuntimeError, 'independent review changed'):
                cycle._publish_render_controls(self.root, time.monotonic() + 30, object(), None)
        activate.assert_not_called(); backup.assert_not_called(); evict.assert_not_called()
        self.assertEqual(version['current'], 'unapproved-change')
        self.assertEqual(self.raw_files(paths), before)
        self.assertFalse((self.root / render_controls.POINTER).exists())

    def test_stop_after_backup_prevents_the_next_snapshot_or_activation(self):
        relative, _, paths = self.inventory.cohort(244)
        before = self.raw_files(paths)
        def stopped_backup(*args, **kwargs):
            self.inventory.write('state/STOP', b'operator STOP during backup\n')
            return 'b' * 40
        with patch.object(render_controls, 'pending', return_value=[relative]), \
             patch.object(cycle, 'current_independent_review', return_value=True), \
             patch.object(render_controls, 'prepare', return_value=['capsule']), \
             patch.object(render_controls, 'plan_bundle', return_value={'bundle_path': 'bundle'}), \
             patch.object(cycle, '_backup', side_effect=stopped_backup), \
             patch.object(cycle, '_publish_snapshot') as snapshot, \
             patch.object(render_controls, 'activate') as activate, \
             patch.object(archive, 'evict_verified') as evict:
            with self.assertRaisesRegex(archive.ArchiveError, 'STOP'):
                cycle._publish_render_controls(self.root, time.monotonic() + 30, object(), None)
        snapshot.assert_not_called(); activate.assert_not_called(); evict.assert_not_called()
        self.assertEqual(self.raw_files(paths), before)
        self.assertTrue((self.root / 'state/STOP').exists())

    def archive_run_patches(self, paths):
        self.inventory.index_paths(paths)
        receipt_id = 'f' * 64
        descriptor = 'manifests/remote_custody/custody-' + receipt_id + '.json'
        self.inventory.write(descriptor, {'receipt_id': receipt_id, 'files': {p: self.inventory.index['files'][p] for p in paths}})
        self.inventory.index['receipts'][receipt_id] = {'path': descriptor, 'sha256': '1' * 64}
        self.inventory.write('state/PAUSED.json', {'reason': 'resource gate: fewer than 10 GiB free'})
        self.config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                       'archive_trigger_free_bytes': 10 * 1024 ** 3, 'min_free_bytes': 10 * 1024 ** 3}
        return (
            patch.object(cycle, 'current_independent_review', return_value=True),
            patch.object(cycle.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 * 1024 ** 3)),
            patch.object(cycle, '_command', return_value=(cycle.BRANCH + '\n').encode()),
            patch.object(cycle, '_backup', return_value='b' * 40),
            patch.object(cycle, '_drop_from_active_tree', return_value=[]),
            patch.object(cycle, '_remove_development_original', return_value=0),
        )

    def test_publication_failure_prevents_eviction_preserves_raw_and_original_pause(self):
        _, _, paths = self.inventory.cohort(250)
        hooks = self.archive_run_patches(paths)
        for hook in hooks:
            hook.start(); self.addCleanup(hook.stop)
        before = self.raw_files(paths); pause = (self.root / 'state/PAUSED.json').read_bytes()
        with patch.object(cycle, '_publish_render_controls', side_effect=RuntimeError('fixture control snapshot push failed')), \
             patch.object(archive, 'evict_verified') as evict:
            with self.assertRaisesRegex(RuntimeError, 'control snapshot push failed'):
                cycle.run(self.root, self.config, transport=object(), compact=False)
        evict.assert_not_called()
        self.assertEqual(self.raw_files(paths), before)
        self.assertEqual((self.root / 'state/PAUSED.json').read_bytes(), pause)

    def test_partial_indexed_completed_cohort_cannot_evade_control_gate(self):
        _, receipt, paths = self.inventory.cohort(251)
        first_header = receipt['datasets'][0]['manifest_path']
        hooks = self.archive_run_patches([first_header])
        for hook in hooks:
            hook.start(); self.addCleanup(hook.stop)
        before = self.raw_files(paths)
        with patch.object(archive, 'evict_verified') as evict:
            with self.assertRaisesRegex(RuntimeError, 'cohort requires qualified controls'):
                cycle.run(self.root, self.config, transport=object(), compact=False)
        evict.assert_not_called(); self.assertEqual(self.raw_files(paths), before)

    def test_stop_and_foreign_pause_never_publish_controls(self):
        config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                  'archive_trigger_free_bytes': 10 * 1024 ** 3, 'min_free_bytes': 10 * 1024 ** 3}
        for reason in ('STOP', 'oracle disagreement'):
            if reason == 'STOP':
                self.inventory.write('state/STOP', b'operator STOP\n')
            else:
                (self.root / 'state/STOP').unlink()
                self.inventory.write('state/PAUSED.json', {'reason': reason})
            with patch.object(cycle, '_publish_render_controls') as publish, \
                 patch.object(cycle, '_backup') as backup:
                self.assertIsNone(cycle.run(self.root, config, transport=object(), compact=False))
            publish.assert_not_called(); backup.assert_not_called()

    def test_recovery_deadline_yields_before_creating_canonical_jobs_or_running_generator(self):
        config = json.loads((worker.ROOT / 'configs/production.json').read_text())
        config.update(archive_enabled=False)
        self.inventory.write('configs/production.json', config)
        self.inventory.write('manifests/VALIDATED_CODE.json', {})
        self.inventory.write('manifests/SEALED_FREEZE.json', {})
        args = SimpleNamespace(no_backup=False, seal=False, render_only=False, groups=1)
        with patch.object(worker, 'ROOT', self.root), \
             patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40)), \
             patch.object(release, 'current_independent_review', return_value=True), \
             patch.object(worker, 'recover_inventory'), \
             patch.object(rendered, 'recover', side_effect=archive.ReadDeadlineExceeded('fixture metadata deadline')), \
             patch.object(worker, 'process') as process, \
             patch.object(worker, 'backup_if_due') as backup, redirect_stdout(io.StringIO()):
            self.assertEqual(worker.run(args), 0)
        process.assert_not_called(); backup.assert_not_called()
        db = sqlite3.connect(self.root / 'state/queue.sqlite'); self.addCleanup(db.close)
        self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 0)
        self.assertEqual(json.loads((self.root / 'state/heartbeat.json').read_text())['status'], 'yielded')
        self.assertFalse((self.root / 'state/PAUSED.json').exists())

    def test_control_storage_bound_refuses_growth_before_any_write(self):
        occupied = render_controls.PREFIX + 'already-staged.bin'
        self.inventory.write(occupied, b'12345678')
        new_path = render_controls.PREFIX + 'new-staged.bin'
        with patch.object(render_controls, 'MAX_CONTROL_STORAGE_BYTES', 10):
            with self.assertRaisesRegex(archive.ArchiveError, 'storage bound'):
                render_controls._write(self.root, new_path, b'abc')
        self.assertEqual((self.root / occupied).read_bytes(), b'12345678')
        self.assertFalse((self.root / new_path).exists())

    def test_seventy_two_cold_manifests_recover_once_from_latest_qualified_aggregate(self):
        # Fixed miniature inventories reproduce the current 2,976-row split/view
        # counts; these are manual metadata, not production records or gold.
        cohorts = []
        second_catalog = '2' * 64; second_snapshot = '3' * 40
        self.inventory.index['catalogs'][second_catalog] = {
            'catalog_commit': second_snapshot, 'archive_ref': archive.archive_ref(second_snapshot),
            'catalog_path': archive.catalog_path(second_catalog), 'catalog_sha256': '4' * 64,
            'catalog_git_blob_sha1': '5' * 40,
        }
        for version in (1, 2):
            for split, number, source_count in (('train', 8, 30), ('dev', 2, 27), ('ood', 2, 29)):
                accepted = source_count + (2 if version == 2 else 0)
                for _ in range(number):
                    seed = 300 + len(cohorts)
                    counts = {'listwise': 3 * accepted, 'computation': 15, 'distillation': 18}
                    cohort = self.inventory.cohort(seed, split=split, accepted=accepted, rows_by_view=counts)
                    self.inventory.index_paths(cohort[2], catalog_id='a' * 64 if version == 1 else second_catalog)
                    cohorts.append(cohort)
        self.assertEqual(len(cohorts), 24)
        self.assertEqual(sum(len(receipt['datasets']) for _, receipt, _ in cohorts), 72)
        provider = MockControlSnapshot(); latest = None
        provider.refs[archive.archive_ref(second_snapshot)] = second_snapshot
        deadline = time.monotonic() + 60
        for start in range(0, 24, 4):
            paths = [relative for relative, _, _ in cohorts[start:start + 4]]
            capsules = render_controls.prepare(self.root, paths, transport=provider, deadline=deadline)
            plan = render_controls.plan_bundle(self.root, capsules, transport=provider)
            commit = format(9000 + start, '040x')
            provider.add(self.root, plan['bundle_path'], commit)
            latest = render_controls.activate(self.root, plan['bundle_path'], commit,
                                              transport=provider, deadline=deadline)
        self.assertIsNotNone(latest)
        self.assertEqual({p.name for p in (self.root / render_controls.PREFIX).glob('bundle-slot-*.json')},
                         {'bundle-slot-0.json', 'bundle-slot-1.json'})
        # The original raw metadata and opaque shards are all cold. The facade
        # refuses reads of any missing raw body after this point.
        for _, _, paths in cohorts:
            self.inventory.index_paths(paths, remove=True)
        self.inventory.offline = True; self.inventory.reads.clear(); provider.calls.clear()
        db = sqlite3.connect(':memory:'); self.addCleanup(db.close)
        expected = {'train/listwise': 1488, 'train/computation': 240, 'train/distillation': 288,
                    'dev/listwise': 336, 'dev/computation': 60, 'dev/distillation': 72,
                    'ood/listwise': 360, 'ood/computation': 60, 'ood/distillation': 72}
        with patch.object(archive, 'GitHubTransport', return_value=provider):
            with archive.read_session(self.root, deadline=deadline):
                rendered.recover(db, self.root)
                rendered.recover(db, self.root)  # Database repair remains idempotent.
                inventory = rendered.inventory(self.root)
        self.assertEqual(db.execute('SELECT COUNT(*) FROM rendered_jobs WHERE status="done"').fetchone()[0], 24)
        self.assertEqual(inventory['completed_source_group_exports'], 24)
        self.assertEqual(inventory['rendered_rows_by_split_view'], expected)
        self.assertEqual(sum(inventory['rendered_rows_by_split_view'].values()), 2976)
        self.assertFalse(any(relative.startswith('data/rendered/') for relative in self.inventory.reads),
                         'qualified recovery fetched a cold header or rendered shard')
        self.assertEqual([kind for kind, _ in provider.calls].count('blob'), 1)
        self.assertEqual([kind for kind, _ in provider.calls].count('tree'), 1)
        references = [reference for kind, reference in provider.calls if kind == 'ref']
        expected_references = sorted((latest['archive_ref'], archive.archive_ref('b' * 40),
                                      archive.archive_ref(second_snapshot)))
        self.assertEqual(references, expected_references + expected_references)
        self.assertFalse(any((self.root / path).exists() for _, _, paths in cohorts for path in paths))


if __name__ == '__main__':
    unittest.main()
