"""Independent destructive archival/cold-custody review.

The reviewer did not implement the archival boundary. Payloads are tiny,
manually specified opaque bytes, fake remote storage lives in RAM, and only
small temporary files may be removed by these fixtures. No production
eviction, actual GitHub operation, sealed-label parse, or DEV pilot rerun.
"""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


class RemoteBytes:
    """Manually built fake server: paths, commit/ref identities, and raw bytes."""
    source_commit = 'a' * 40
    snapshot_commit = 'b' * 40

    def __init__(self):
        self.trees, self.blobs, self.refs, self.calls = {}, {}, {}, []
        self.offline = False
        self.on_blob = None

    def add(self, commit, relative, raw):
        oid = git_blob(raw)
        self.blobs[oid] = raw
        self.trees.setdefault(commit, {})['goal_conditioned_decision_data/' + relative] = {
            'type': 'blob', 'mode': '100644', 'sha': oid, 'size': len(raw)}

    def fetch_ref(self, reference):
        self.calls.append(('ref', reference))
        if self.offline:
            raise archive.ArchiveError('fixture offline')
        return self.refs.get(reference)

    def fetch_tree(self, commit):
        self.calls.append(('tree', commit))
        if self.offline:
            raise archive.ArchiveError('fixture offline')
        return deepcopy(self.trees.get(commit, {}))

    def fetch_blob(self, oid, max_bytes):
        self.calls.append(('blob', oid, max_bytes))
        if self.offline:
            raise archive.ArchiveError('fixture offline')
        if self.on_blob:
            self.on_blob(oid)
        raw = self.blobs[oid]
        if len(raw) > max_bytes:
            raise archive.ArchiveError('fixture overlong raw body')
        return raw


class IndependentArchiveCustody(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gcdl-archive-custody-')
        self.addCleanup(self.directory.cleanup)
        self.root = (Path(self.directory.name) / 'goal_conditioned_decision_data').resolve()
        self.root.mkdir()
        self.paths = {'data/accepted/train-7.jsonl': b'MANUAL ACCEPTED OPAQUE BYTES\n',
                      'data/sealed/sealed-8.jsonl': b'OPAQUE SEAL: NO LABEL PARSE\n'}
        for relative, raw in self.paths.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.remote = RemoteBytes()
        files = {}
        for relative, raw in self.paths.items():
            files[relative] = {'commit': self.remote.source_commit, 'git_blob_sha1': git_blob(raw),
                               'sha256': sha(raw), 'bytes': len(raw)}
            self.remote.add(self.remote.source_commit, relative, raw)
            self.remote.add(self.remote.snapshot_commit, relative, raw)
        body = {'version': 'gcdl-archive-catalog-1', 'repository': 'LUNCY579/rineya-world-model-research',
                'programme': 'goal_conditioned_decision_data', 'files': files}
        self.catalog = {**body, 'catalog_id': sha(canonical(body).encode())}
        self.catalog_relative = 'manifests/remote_catalogs/catalog-' + self.catalog['catalog_id'] + '.json'
        self.remote.add(self.remote.snapshot_commit, self.catalog_relative,
                        (canonical(self.catalog) + '\n').encode())
        self.reference = 'refs/heads/archive/gcdl-001/' + self.remote.snapshot_commit
        self.remote.refs[self.reference] = self.remote.snapshot_commit
        self.git_calls = []
        # Git commit/status acceptance is isolated here; filesystem and every
        # remote tree/raw-byte/custody/index check remains the real implementation.
        hook = patch.object(archive, '_git_file', side_effect=lambda root, path, meta:
                            self.git_calls.append((str(root), path, deepcopy(meta))))
        hook.start(); self.addCleanup(hook.stop)
        archive._CACHE.clear(); archive._CACHE_SIZE = 0
        self.addCleanup(lambda: (archive._CACHE.clear(), setattr(archive, '_CACHE_SIZE', 0)))

    def verify(self):
        return archive.verify_archive(self.root, self.catalog, self.remote.snapshot_commit,
                                      transport=self.remote)

    def assert_original_present(self):
        for relative, raw in self.paths.items():
            self.assertEqual((self.root / relative).read_bytes(), raw)

    def test_verified_eviction_then_authenticated_cold_read_has_no_payload_restore(self):
        receipt = self.verify()
        result = archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual(result['deleted'], list(self.catalog['files']))
        self.assertEqual(result['deleted_bytes'], sum(map(len, self.paths.values())))
        self.assertFalse(result['Git_objects_removed'])
        self.assertEqual(archive.iter_paths(self.root, 'data/*/*.jsonl'), sorted(self.paths))
        relative = 'data/accepted/train-7.jsonl'
        self.assertEqual(archive.read_bytes(self.root, relative, transport=self.remote), self.paths[relative])
        self.assertFalse((self.root / relative).exists())
        custody = archive.deletion_custody(self.root, relative)
        self.assertEqual(custody['archive_ref'], self.reference)
        self.assertEqual(custody['sha256'], sha(self.paths[relative]))

    def test_catalog_identity_tamper_refuses_before_remote_or_custody_writes(self):
        self.catalog['files']['data/accepted/train-7.jsonl']['sha256'] = '0' * 64
        with self.assertRaisesRegex(archive.ArchiveError, 'catalog identity'):
            self.verify()
        self.assertEqual(self.remote.calls, [])
        self.assertFalse((self.root / archive.INDEX_PATH).exists())
        self.assert_original_present()

    def test_missing_or_moved_snapshot_ref_preserves_all_local_bytes(self):
        for target in (None, 'c' * 40):
            with self.subTest(target=target):
                self.remote.refs[self.reference] = target
                with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
                    self.verify()
                self.assertFalse((self.root / archive.INDEX_PATH).exists())
                self.assert_original_present()

    def test_snapshot_ref_moving_during_raw_proof_cannot_create_custody(self):
        self.remote.on_blob = lambda oid: self.remote.refs.update({self.reference: 'c' * 40})
        with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
            self.verify()
        self.assertFalse((self.root / archive.INDEX_PATH).exists())
        self.assert_original_present()

    def test_snapshot_ref_moving_during_cold_read_cannot_populate_ram_cache(self):
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        self.remote.on_blob = lambda oid: self.remote.refs.update({self.reference: 'c' * 40})
        with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
            archive.read_bytes(self.root, 'data/accepted/train-7.jsonl', transport=self.remote)
        self.assertEqual(list(archive._CACHE), [])
        self.assertFalse((self.root / 'data/accepted/train-7.jsonl').exists())

    def test_catalog_only_snapshot_cannot_custody_unrelated_source_commit(self):
        # A catalogue can mention an orphan commit that exists today but is not
        # retained by the durable snapshot branch. Its catalogue alone is unsafe.
        for relative in self.paths:
            del self.remote.trees[self.remote.snapshot_commit]['goal_conditioned_decision_data/' + relative]
        with self.assertRaises(archive.ArchiveError):
            self.verify()
        self.assertFalse((self.root / archive.INDEX_PATH).exists())
        self.assert_original_present()

    def test_raw_byte_tamper_same_length_is_rejected_despite_correct_tree_oid(self):
        oid = git_blob(self.paths['data/accepted/train-7.jsonl'])
        self.remote.blobs[oid] = b'X' * len(self.remote.blobs[oid])
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            self.verify()
        self.assertFalse((self.root / archive.INDEX_PATH).exists())
        self.assert_original_present()

    def test_wrong_tree_path_mode_oid_or_size_refuses(self):
        key = 'goal_conditioned_decision_data/data/accepted/train-7.jsonl'
        good = deepcopy(self.remote.trees[self.remote.source_commit][key])
        for field, value in [('mode', '120000'), ('sha', '0' * 40), ('size', 999), ('type', 'tree')]:
            with self.subTest(field=field):
                self.remote.trees[self.remote.source_commit][key] = {**good, field: value}
                with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID/size'):
                    self.verify()
                self.assert_original_present()
        self.remote.trees[self.remote.source_commit][key] = good

    def test_offline_verification_and_offline_evict_cannot_remove_payloads(self):
        self.remote.offline = True
        with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
            self.verify()
        self.assert_original_present()
        self.remote.offline = False
        receipt = self.verify()
        self.remote.offline = True
        with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
            archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assert_original_present()

    def test_remote_corruption_after_verification_preserves_whole_local_batch(self):
        receipt = self.verify()
        oid = git_blob(self.paths['data/sealed/sealed-8.jsonl'])
        self.remote.blobs[oid] = b'Q' * len(self.remote.blobs[oid])
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assert_original_present()

    def test_inflight_local_write_prevents_custody_and_preserves_other_file(self):
        relative = 'data/accepted/train-7.jsonl'
        replacement = b'NEW LOCAL CONTENT MUST SURVIVE\n'
        self.remote.on_blob = lambda oid: (self.root / relative).write_bytes(replacement)
        with self.assertRaisesRegex(archive.ArchiveError, 'changed after remote'):
            self.verify()
        self.assertEqual((self.root / relative).read_bytes(), replacement)
        self.assertEqual((self.root / 'data/sealed/sealed-8.jsonl').read_bytes(), self.paths['data/sealed/sealed-8.jsonl'])
        self.assertFalse((self.root / archive.INDEX_PATH).exists())

    def test_same_byte_inode_replacement_after_verify_is_not_evicted(self):
        receipt = self.verify()
        relative = 'data/accepted/train-7.jsonl'
        path = self.root / relative
        before = path.stat().st_ino
        temporary = path.with_name('replacement.tmp')
        temporary.write_bytes(self.paths[relative]); os.replace(temporary, path)
        self.assertNotEqual(path.stat().st_ino, before)
        with self.assertRaisesRegex(archive.ArchiveError, 'hash/inode'):
            archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assert_original_present()

    def test_immediate_callback_replacement_keeps_new_file(self):
        receipt = self.verify()
        relative = 'data/accepted/train-7.jsonl'
        replacement = b'NEW FILE AFTER WHOLE BATCH CHECK\n'
        def replace(root, path):
            if path == relative:
                temporary = self.root / 'replacement.tmp'
                temporary.write_bytes(replacement)
                os.replace(temporary, self.root / relative)
        with patch.object(archive, '_before_unlink', side_effect=replace):
            with self.assertRaises(archive.ArchiveError):
                archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual((self.root / relative).read_bytes(), replacement)
        self.assertTrue((self.root / 'data/sealed/sealed-8.jsonl').exists())

    def test_replacement_in_final_check_to_move_gap_is_restored_and_kept(self):
        receipt = self.verify()
        relative = 'data/accepted/train-7.jsonl'
        replacement = b'NEW INODE INSERTED AFTER FINAL HASH\n'
        def replace(root, path):
            if path == relative:
                temporary = self.root / 'late-replacement.tmp'
                temporary.write_bytes(replacement)
                os.replace(temporary, self.root / relative)
        with patch.object(archive, '_before_move', side_effect=replace):
            with self.assertRaisesRegex(archive.ArchiveError, 'moved file mismatch'):
                archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual((self.root / relative).read_bytes(), replacement)
        self.assertEqual((self.root / 'data/sealed/sealed-8.jsonl').read_bytes(), self.paths['data/sealed/sealed-8.jsonl'])

    def test_replaced_parent_directory_prevents_unlinking_new_ancestry(self):
        receipt = self.verify()
        relative = 'data/accepted/train-7.jsonl'
        replacement = b'NEW DIRECTORY CONTENT\n'
        def replace_parent(root, path):
            if path == relative:
                old = self.root / 'data/accepted'
                old.rename(self.root / 'retained-original-directory')
                old.mkdir(); (old / 'train-7.jsonl').write_bytes(replacement)
        with patch.object(archive, '_before_move', side_effect=replace_parent):
            with self.assertRaisesRegex(archive.ArchiveError, 'ancestry changed'):
                archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual((self.root / relative).read_bytes(), replacement)
        self.assertEqual((self.root / 'retained-original-directory/train-7.jsonl').read_bytes(), self.paths[relative])

    def test_crash_after_atomic_move_recovers_pending_inode_then_revalidates_remote(self):
        receipt = self.verify()
        relative = 'data/accepted/train-7.jsonl'
        # Independently specify durable pending marker and actual interrupted
        # filesystem state, without invoking the implementation's move helper.
        holding = ('data/accepted/.gcdl-evacuate-' + receipt['receipt_id'] + '-' +
                   sha(relative.encode())[:16] + '/payload')
        path = self.root / holding; path.parent.mkdir(mode=0o700)
        os.rename(self.root / relative, path)
        journal = self.root / ('state/archive_evacuations/' + receipt['receipt_id'] + '.json')
        journal.parent.mkdir(parents=True)
        journal.write_text(canonical({'receipt_id': receipt['receipt_id'], 'verified_remote': True,
            'deleted': [], 'status': 'evacuating', 'pending': {'relative': relative,
            'holding': holding, 'identity': receipt['local_identities'][relative],
            'parents': receipt['local_ancestry'][relative]}}) + '\n')
        self.remote.offline = True
        with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
            archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual((self.root / relative).read_bytes(), self.paths[relative])
        self.assertFalse(path.exists())
        self.remote.offline = False
        result = archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual(result['deleted'], list(self.paths))

    def test_partial_eviction_is_retryable_without_losing_missing_file_custody(self):
        receipt = self.verify()
        second = 'data/sealed/sealed-8.jsonl'
        def stop_after_first(root, path):
            if path == second:
                raise archive.ArchiveError('fixture interruption')
        with patch.object(archive, '_before_unlink', side_effect=stop_after_first):
            with self.assertRaisesRegex(archive.ArchiveError, 'fixture interruption'):
                archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertFalse((self.root / 'data/accepted/train-7.jsonl').exists())
        self.assertEqual((self.root / second).read_bytes(), self.paths[second])
        resumed = archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual(resumed['deleted'], [second])
        self.assertEqual(archive.read_bytes(self.root, 'data/accepted/train-7.jsonl', transport=self.remote),
                         self.paths['data/accepted/train-7.jsonl'])

    def test_missing_without_prior_verified_custody_refuses(self):
        (self.root / 'data/accepted/train-7.jsonl').unlink()
        with self.assertRaisesRegex(archive.ArchiveError, 'orphan or missing unverified'):
            self.verify()
        self.assertFalse((self.root / archive.INDEX_PATH).exists())

    def test_default_sealed_cold_read_refused_before_any_remote_call(self):
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        self.remote.calls.clear()
        relative = 'data/sealed/sealed-8.jsonl'
        self.assertEqual(archive.payload_info(self.root, relative)['sha256'], sha(self.paths[relative]))
        self.assertEqual(self.remote.calls, [])
        with self.assertRaisesRegex(archive.ArchiveError, 'sealed raw read'):
            archive.read_bytes(self.root, relative, transport=self.remote)
        self.assertEqual(self.remote.calls, [])

    def test_real_cold_index_worker_recovery_keeps_exact_counts_without_blob_read(self):
        from goal_conditioned_decision_data.generators.profiles import resolve_profile
        from goal_conditioned_decision_data.pipelines import worker
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        config = resolve_profile('v1')['config']
        for relative, split, seed in [('data/accepted/train-7.jsonl', 'train', 7),
                                      ('data/sealed/sealed-8.jsonl', 'sealed', 8)]:
            record_index = [split + '-opaque-id', split + '-opaque-semantic', split + '-opaque-parent',
                            config['split_templates'][split]['template'], split]
            batch = {'seed': seed, 'split': split, 'protected_index': [record_index],
                     'accepted': 1, 'quarantined': 0, 'audit': {'verdict': 'PASS'},
                     'shard': relative, 'sha256': sha(self.paths[relative]), 'bytes': len(self.paths[relative])}
            (self.root / ('manifests/batch-' + split + '-' + str(seed) + '.json')).write_text(canonical(batch))
        (self.root / 'manifests/SEALED_FREEZE.json').write_text(canonical({
            'shards': {'data/sealed/sealed-8.jsonl': sha(self.paths['data/sealed/sealed-8.jsonl'])}}))
        database = sqlite3.connect(':memory:')
        self.addCleanup(database.close)
        worker.ensure_job_profiles(database)
        database.execute('CREATE TABLE seen(id TEXT PRIMARY KEY, semantic TEXT, parent TEXT, template TEXT, split TEXT)')
        self.remote.calls.clear(); self.remote.offline = True
        with patch.object(worker, 'ROOT', self.root), \
             patch.object(archive, 'GitHubTransport', side_effect=AssertionError('no remote content read permitted')):
            worker.recover_inventory(database); worker.recover_inventory(database)
        self.assertEqual(database.execute('SELECT seed,split,status,accepted,profile FROM jobs ORDER BY seed').fetchall(),
                         [(7, 'train', 'done', 1, 'v1'), (8, 'sealed', 'done', 1, 'v1')])
        self.assertEqual(database.execute('SELECT COUNT(*) FROM seen').fetchone()[0], 2)
        self.assertEqual(self.remote.calls, [])
        self.assertFalse(any((self.root / relative).exists() for relative in self.paths))

    def test_bound_refuses_before_remote_fetch_and_cold_read_is_ram_only(self):
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        self.remote.calls.clear()
        relative = 'data/accepted/train-7.jsonl'
        with self.assertRaisesRegex(archive.ArchiveError, 'caller read bound'):
            archive.read_bytes(self.root, relative, max_bytes=len(self.paths[relative]) - 1, transport=self.remote)
        self.assertEqual(self.remote.calls, [])
        self.assertEqual(archive.read_bytes(self.root, relative, max_bytes=len(self.paths[relative]),
                                           transport=self.remote), self.paths[relative])
        self.assertFalse((self.root / relative).exists())
        self.assertEqual(sorted(path.name for path in (self.root / 'state').iterdir()),
                         ['archive_evacuations', 'worker.lock'])

    def test_index_snapshot_or_receipt_tamper_refuses_before_remote_read(self):
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        relative = 'manifests/remote_custody/custody-' + receipt['receipt_id'] + '.json'
        path = self.root / relative
        path.write_bytes(path.read_bytes() + b' ')
        self.remote.calls.clear()
        with self.assertRaisesRegex(archive.ArchiveError, 'receipt hash'):
            archive.read_bytes(self.root, 'data/accepted/train-7.jsonl', transport=self.remote)
        self.assertEqual(self.remote.calls, [])

    def test_symlink_and_hardlink_payloads_refuse_without_foreign_file_changes(self):
        relative = 'data/accepted/train-7.jsonl'; path = self.root / relative
        outside = self.root.parent / 'foreign'; outside.write_bytes(b'FOREIGN SENTINEL')
        path.unlink(); path.symlink_to(outside)
        with self.assertRaisesRegex(archive.ArchiveError, 'symlink'):
            self.verify()
        self.assertEqual(outside.read_bytes(), b'FOREIGN SENTINEL')
        path.unlink(); os.link(outside, path)
        with self.assertRaisesRegex(archive.ArchiveError, 'hardlink'):
            self.verify()
        self.assertEqual(outside.read_bytes(), b'FOREIGN SENTINEL')

    def test_protected_live_controls_and_noncanonical_paths_have_no_deletion_custody(self):
        protected = ['state/queue.sqlite', 'state/queue.sqlite-wal', 'state/PAUSED.json',
                     'state/STOP', 'state/worker.lock', 'configs/production.json',
                     'manifests/REMOTE_ARCHIVE.json', 'manifests/SEALED_FREEZE.json',
                     'reports/INDEPENDENT_REVIEW.md', 'reports/independent_review.MD',
                     'reports/history/REMOTE_ARCHIVE.json', 'reports/history/script.py',
                     'reports/.private/history.json', 'data/backups/worker.lock', 'pipelines/worker.py']
        for relative in protected:
            with self.subTest(relative=relative):
                self.assertFalse(archive.is_payload_path(relative))
                with self.assertRaisesRegex(archive.ArchiveError, 'protected'):
                    archive.deletion_custody(self.root, relative)
        for relative in ['../data/accepted/x', '/data/accepted/x', 'data//accepted/x',
                         'data/accepted/./x', 'data/accepted/../x', 'data\\accepted\\x']:
            with self.subTest(relative=relative):
                with self.assertRaises(archive.ArchiveError):
                    archive.read_bytes(self.root, relative, transport=self.remote)
        self.assertEqual(self.remote.calls, [])

    def test_auto_evict_requires_exact_authorization_and_preserves_pause_marker(self):
        receipt = self.verify()
        marker = self.root / 'state/PAUSED.json'
        marker.write_bytes(b'{"reason":"resource gate: fewer than 10 GiB free","original":3}\n')
        before = marker.read_bytes()
        config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                  'archive_trigger_free_bytes': 10 * 1024 ** 3}
        with patch.object(archive.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            for options in ({'archive_enabled': False}, {'archive_authorization': 'unqualified'}):
                self.assertIsNone(archive.maybe_archive(self.root, {**config, **options}, transport=self.remote))
                self.assert_original_present()
            with self.assertRaisesRegex(archive.ArchiveError, '10 GiB'):
                archive.maybe_archive(self.root, {**config, 'archive_trigger_free_bytes': 1}, transport=self.remote)
            result = archive.maybe_archive(self.root, config, transport=self.remote)
        self.assertEqual(result['receipt_id'], receipt['receipt_id'])
        self.assertEqual(result['status'], 'evacuated')
        self.assertEqual(marker.read_bytes(), before)

    def test_floor_met_is_noop_and_low_disk_without_custody_requires_backup(self):
        config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                  'archive_trigger_free_bytes': 10 * 1024 ** 3}
        with patch.object(archive.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024 ** 3)):
            self.assertIsNone(archive.maybe_archive(self.root, config, transport=self.remote))
        with patch.object(archive.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            result = archive.maybe_archive(self.root, config, transport=self.remote)
        self.assertEqual(result['status'], 'backup_required')
        self.assertEqual(result['deleted'], [])
        self.assertEqual(self.remote.calls, [])
        self.assert_original_present()

    def test_stop_created_while_remote_proof_runs_retains_whole_batch(self):
        receipt = self.verify()
        self.remote.on_blob = lambda oid: (self.root / 'state/STOP').write_text('operator stop during proof')
        with self.assertRaisesRegex(archive.ArchiveError, '[Ss][Tt][Oo][Pp]|operator stop'):
            archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assert_original_present()

    def test_existing_stop_refuses_evict_before_remote_and_preserves_marker(self):
        receipt = self.verify()
        marker = self.root / 'state/STOP'; marker.write_bytes(b'operator stop\n')
        self.remote.calls.clear()
        with self.assertRaisesRegex(archive.ArchiveError, '[Ss][Tt][Oo][Pp]|operator stop'):
            archive.evict_verified(self.root, receipt, transport=self.remote)
        self.assertEqual(self.remote.calls, [])
        self.assertEqual(marker.read_bytes(), b'operator stop\n')
        self.assert_original_present()

    def backup_fixture(self, *, ref_target=None, wrong_tree=False, offline=False, late_stage=False):
        receipt = self.verify()
        archive.evict_verified(self.root, receipt, transport=self.remote)
        (self.root / 'configs').mkdir(exist_ok=True)
        (self.root / 'configs/production.json').write_text('{"backup_transport_attempts":1}')
        namespace = runpy.run_path(str(ROOT / 'scripts/backup.py'))
        main = namespace['main']; globals_ = main.__globals__
        branch = 'research/goal-conditioned-decision-data-v1'
        code_commit = 'd' * 40
        calls, writes = [], []
        full_paths = ['goal_conditioned_decision_data/' + path for path in self.paths]
        target = self.remote.snapshot_commit if ref_target is None else ref_target
        def execute(command, **kwargs):
            calls.append(list(command))
            def result(stdout='', returncode=0):
                return SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)
            if command[:3] == ['git', 'branch', '--show-current']:
                return result(branch + '\n')
            if command[:3] == ['git', 'status', '--porcelain']:
                return result(''.join(' D ' + path + '\0' for path in full_paths))
            if command[:3] == ['git', 'remote', 'get-url']:
                return result('https://github.com/LUNCY579/rineya-world-model-research.git\n')
            if command[:3] == ['gh', 'auth', 'token']:
                return result('FIXTURE_TOKEN')
            if 'ls-remote' in command:
                requested = command[-1]
                if requested.endswith(self.remote.snapshot_commit):
                    if offline:
                        raise RuntimeError('fixture offline transport')
                    return result(target + '\t' + self.reference + '\n')
                return result(code_commit + '\trefs/heads/' + branch + '\n')
            if command[:3] == ['git', 'rev-parse', 'HEAD']:
                return result(code_commit + '\n')
            if command[:2] == ['git', 'rev-parse'] and command[2].startswith('HEAD:'):
                relative = command[2].split(':', 1)[1].removeprefix('goal_conditioned_decision_data/')
                return result(git_blob(self.paths[relative]) + '\n')
            if command[:2] == ['gh', 'api']:
                if '/git/commits/' in command[2]:
                    return result(json.dumps({'sha': self.remote.snapshot_commit,
                                              'tree': {'sha': 'e' * 40}}))
                tree = []
                for path, raw in self.paths.items():
                    tree.append({'path': 'goal_conditioned_decision_data/' + path,
                                 'type': 'blob', 'sha': '0' * 40 if wrong_tree else git_blob(raw),
                                 'mode': '100644', 'size': len(raw)})
                return result(json.dumps({'truncated': False, 'sha': 'e' * 40, 'tree': tree}))
            if command[:2] == ['git', 'add']:
                self.assertIn('--sparse', command)
                return result()
            if command[:3] == ['git', 'diff', '--cached']:
                if '--name-only' in command:
                    return result('\0'.join(full_paths + (['goal_conditioned_decision_data/configs/production.json'] if late_stage else [])) + '\0')
                return result(returncode=1)
            if command[:2] == ['git', 'commit'] or 'pull' in command or 'push' in command:
                return result()
            raise AssertionError('unexpected backup fixture command: ' + repr(command))
        overrides = {'ROOT': self.root, 'REPO': self.root.parent, 'execute': execute,
                     'atomic_json': lambda path, value: writes.append(deepcopy(value))}
        with patch.dict(globals_, overrides), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main()
        return result, calls, writes

    def test_valid_cold_custody_allows_sparse_code_branch_deletion_and_keeps_snapshot(self):
        result, calls, writes = self.backup_fixture()
        self.assertEqual(result, 0, writes)
        self.assertEqual(writes[-1]['status'], 'confirmed')
        self.assertTrue(any(command[:3] == ['git', 'add', '--sparse'] for command in calls))
        self.assertTrue(any(command[:2] == ['git', 'commit'] for command in calls))
        self.assertTrue(any('ls-remote' in command and command[-1] == self.reference for command in calls))
        self.assertFalse(any('push' in command and command[-1].endswith(self.reference) for command in calls))

    def test_moved_snapshot_or_remote_tree_mismatch_refuses_before_staging(self):
        for options in ({'ref_target': 'c' * 40}, {'wrong_tree': True}, {'offline': True}):
            with self.subTest(options=options):
                # Fresh catalogue/index identities, same tiny bytes per attempt.
                for relative, raw in self.paths.items():
                    (self.root / relative).write_bytes(raw)
                result, calls, writes = self.backup_fixture(**options)
                self.assertEqual(result, 1)
                self.assertEqual(writes[-1]['status'], 'failed')
                self.assertFalse(any(command[:2] in (['git', 'add'], ['git', 'commit']) for command in calls))

    def test_new_staged_control_deletion_refuses_commit_even_after_valid_custody(self):
        result, calls, writes = self.backup_fixture(late_stage=True)
        self.assertEqual(result, 1)
        self.assertIn('deletion inventory changed', writes[-1]['reason'])
        self.assertFalse(any(command[:2] == ['git', 'commit'] for command in calls))

    def cycle_existing_fixture(self, *, backup_failure=False, change_pause_in_proof=False,
                               no_initial_pause=False):
        from goal_conditioned_decision_data.scripts import archive_cycle as cycle
        self.verify()
        pause = self.root / 'state/PAUSED.json'
        original = b'{"reason":"resource gate: fewer than 10 GiB free","origin":11}\n'
        if not no_initial_pause:
            pause.write_bytes(original)
        changed = b'{"reason":"operator investigation","origin":12}\n'
        if change_pause_in_proof:
            self.remote.on_blob = lambda oid: pause.write_bytes(changed)
        config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                  'archive_trigger_free_bytes': 10 * 1024 ** 3, 'min_free_bytes': 10 * 1024 ** 3}
        commands, stages = [], []
        def command(root, args, deadline, environment=None):
            commands.append(list(args))
            if args == ['git', 'branch', '--show-current']:
                return (cycle.BRANCH + '\n').encode()
            if args[:2] == ['git', 'ls-files']:
                return ''.join('goal_conditioned_decision_data/' + path + '\0' for path in self.paths).encode()
            if args[:2] == ['git', 'status']:
                return b''
            if args[:2] == ['git', 'rm']:
                self.assertIn('--cached', args); self.assertIn('--sparse', args)
                return b''
            raise AssertionError('unexpected/no real network cycle command: ' + repr(args))
        def backup(root, deadline):
            stages.append('confirmed-backup' if not backup_failure else 'failed-backup')
            if backup_failure:
                raise RuntimeError('fixture backup failed before durable custody')
            return self.remote.snapshot_commit
        original_evict = archive.evict_verified
        def evict(*args, **kwargs):
            stages.append('evict')
            self.assertIsNotNone(kwargs.get('lock_fd'), 'outer acquired worker FD must reach nested archiver')
            return original_evict(*args, **kwargs)
        overrides = [patch.object(cycle, '_command', side_effect=command),
                     patch.object(cycle, '_backup', side_effect=backup),
                     patch.object(cycle, 'current_independent_review', return_value=True),
                     patch.object(cycle.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)),
                     patch.object(archive, 'evict_verified', side_effect=evict)]
        for override in overrides:
            override.start(); self.addCleanup(override.stop)
        return cycle, config, pause, original, changed, stages, commands

    def test_cycle_default_acquired_lock_forwards_fd_and_preserves_original_pause(self):
        cycle, config, pause, original, _, stages, commands = self.cycle_existing_fixture()
        result = cycle.run(self.root, config, transport=self.remote, compact=False)
        self.assertEqual(result['deleted_files'], 2)
        self.assertEqual(result['remaining_local_payloads'], 0)
        self.assertEqual(pause.read_bytes(), original)
        self.assertIn('evict', stages)
        self.assertTrue(any(command[:2] == ['git', 'rm'] for command in commands))

    def test_cycle_unbacked_local_custody_cannot_evict_before_confirmed_metadata_backup(self):
        cycle, config, pause, original, _, stages, _ = self.cycle_existing_fixture(backup_failure=True)
        with self.assertRaisesRegex(RuntimeError, 'fixture backup failed'):
            cycle.run(self.root, config, transport=self.remote, compact=False)
        self.assertNotIn('evict', stages)
        self.assertEqual(pause.read_bytes(), original)
        self.assert_original_present()

    def test_cycle_pause_changed_during_proof_retains_bytes_and_operator_cause(self):
        cycle, config, pause, _, changed, _, _ = self.cycle_existing_fixture(change_pause_in_proof=True)
        with self.assertRaisesRegex(RuntimeError, 'pause changed'):
            cycle.run(self.root, config, transport=self.remote, compact=False)
        self.assertEqual(pause.read_bytes(), changed)
        self.assert_original_present()

    def test_cycle_new_maintenance_pause_clears_only_after_success(self):
        cycle, config, pause, _, _, _, _ = self.cycle_existing_fixture(no_initial_pause=True)
        result = cycle.run(self.root, config, transport=self.remote, compact=False)
        self.assertEqual(result['deleted_files'], 2)
        self.assertFalse(pause.exists())

    def test_cycle_stop_or_unrelated_pause_is_noop_before_review_lock_and_network(self):
        from goal_conditioned_decision_data.scripts import archive_cycle as cycle
        (self.root / 'state').mkdir(exist_ok=True)
        config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                  'archive_trigger_free_bytes': 10 * 1024 ** 3, 'min_free_bytes': 10 * 1024 ** 3}
        for relative, raw in [('state/STOP', b'operator stop'),
                              ('state/PAUSED.json', b'{"reason":"quality investigation"}')]:
            path = self.root / relative; path.write_bytes(raw)
            with patch.object(cycle, 'current_independent_review') as review, \
                 patch.object(cycle, '_command') as command, patch.object(archive, '_lock') as lock:
                self.assertIsNone(cycle.run(self.root, config, transport=self.remote))
                review.assert_not_called(); command.assert_not_called(); lock.assert_not_called()
            self.assertEqual(path.read_bytes(), raw)
            path.unlink()

    def development_fixture(self):
        raw = b'MANUAL OPAQUE DEVELOPMENT ARCHIVE\n'
        relative = 'data/backups/development_archive.tar.gz'
        destination = self.root / relative; destination.parent.mkdir(parents=True)
        destination.write_bytes(raw)
        source = self.root / 'state/development_archive.tar.gz'; source.parent.mkdir(exist_ok=True)
        source.write_bytes(raw)
        manifest = self.root / 'manifests/DEVELOPMENT_ARCHIVE.json'; manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(canonical({'bytes': len(raw), 'sha256': sha(raw),
                                      'archive': 'state/development_archive.tar.gz'}))
        body = {'version': 'gcdl-archive-catalog-1', 'repository': 'LUNCY579/rineya-world-model-research',
                'programme': 'goal_conditioned_decision_data', 'files': {relative: {
                    'commit': self.remote.source_commit, 'git_blob_sha1': git_blob(raw),
                    'sha256': sha(raw), 'bytes': len(raw)}}}
        self.catalog = {**body, 'catalog_id': sha(canonical(body).encode())}
        self.remote.add(self.remote.source_commit, relative, raw)
        self.remote.add(self.remote.snapshot_commit, relative, raw)
        self.remote.add(self.remote.snapshot_commit,
            'manifests/remote_catalogs/catalog-' + self.catalog['catalog_id'] + '.json',
            (canonical(self.catalog) + '\n').encode())
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        return source, relative, raw

    def test_development_duplicate_cached_bytes_do_not_replace_fresh_remote_custody(self):
        from goal_conditioned_decision_data.scripts import archive_cycle as cycle
        source, relative, raw = self.development_fixture()
        self.assertEqual(archive.read_bytes(self.root, relative, transport=self.remote), raw)
        self.remote.offline = True
        with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
            cycle._remove_development_original(self.root, transport=self.remote)
        self.assertEqual(source.read_bytes(), raw)

    def test_development_duplicate_failed_holding_restore_never_overwrites_new_source(self):
        from goal_conditioned_decision_data.scripts import archive_cycle as cycle
        source, _, raw = self.development_fixture()
        replacement = b'NEW SOURCE MUST SURVIVE RESTORATION CONFLICT\n'
        original_held = archive._held_meta
        def create_conflict(directory_fd, name):
            original_held(directory_fd, name)
            source.write_bytes(replacement)
            raise archive.ArchiveError('fixture failure after original moved')
        with patch.object(archive, '_held_meta', side_effect=create_conflict):
            with self.assertRaisesRegex(archive.ArchiveError, 'fixture failure'):
                cycle._remove_development_original(self.root, transport=self.remote)
        self.assertEqual(source.read_bytes(), replacement)
        retained = [path for path in (self.root / 'state').rglob('*') if path.is_file()]
        self.assertTrue(any(path.read_bytes() == raw for path in retained), 'old bytes require an owned holding pointer')
        self.assertNotIn('local_payload_evicted', json.loads((self.root / 'manifests/DEVELOPMENT_ARCHIVE.json').read_text()))

    def test_cycle_pending_git_cleanup_retains_owned_pause_and_retries_above_floor(self):
        cycle, config, pause, _, _, _, _ = self.cycle_existing_fixture(no_initial_pause=True)
        calls = []
        def compact(root, **kwargs):
            calls.append(kwargs)
            result = {'status': 'compacted_cleanup_pending' if len(calls) == 1 else 'compacted',
                      'repository': archive.REPOSITORY, 'programme': archive.PROGRAMME,
                      'commit': self.remote.snapshot_commit, 'transaction': '.git-compaction-' + '0' * 32,
                      'archived_blob_objects_remaining': 1 if len(calls) == 1 else 0}
            (self.root / 'state/git_compaction.json').write_text(canonical(result))
            return result
        with patch.object(cycle.compact_git, 'compact', side_effect=compact):
            first = cycle.run(self.root, config, transport=self.remote, compact=True)
            self.assertEqual(first['status'], 'archive_cleanup_pending')
            self.assertTrue(pause.exists(), 'pending local Git copies cannot report a finished maintenance cycle')
            with patch.object(cycle.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024 ** 3)):
                second = cycle.run(self.root, config, transport=self.remote, compact=True)
        self.assertEqual(len(calls), 2, 'pending cleanup must be retried even after free space rises above the floor')
        self.assertEqual(second['git_cleanup']['status'], 'compacted')
        self.assertTrue(all(call.get('lock_fd') is not None for call in calls))

    def test_cleanup_deferral_cli_requires_explicit_manual_operation(self):
        namespace = runpy.run_path(str(ROOT / 'scripts/archive_cycle.py'))
        main = namespace['main']
        calls = []
        with patch.dict(main.__globals__, {'run': lambda **kwargs: calls.append(kwargs) or {'status': 'fixture'}}), \
             patch.object(sys, 'argv', ['archive_cycle.py', '--defer-git-cleanup']), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as refused:
                main()
        self.assertEqual(refused.exception.code, 2)
        self.assertEqual(calls, [])
        with patch.dict(main.__globals__, {'run': lambda **kwargs: calls.append(kwargs) or {'status': 'fixture'}}), \
             patch.object(sys, 'argv', ['archive_cycle.py', '--manual', '--defer-git-cleanup']), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 0)
        self.assertEqual(calls, [{'automatic': False, 'compact': False}])

    def test_review_expired_deadline_refuses_before_any_source_or_payload_read(self):
        from goal_conditioned_decision_data.auditors import release
        with patch.object(release.time, 'monotonic', return_value=11), \
             patch.object(archive, 'read_bytes') as read, patch.object(release, 'source_hashes') as source:
            self.assertFalse(release.current_independent_review(deadline=10))
            read.assert_not_called(); source.assert_not_called()

    def test_review_future_deadline_matches_original_and_expiry_during_hashing_refuses(self):
        from goal_conditioned_decision_data.auditors import release
        code = {'fixture.py': '1' * 64}
        source_digest = sha(canonical(code).encode())
        evidence = {}
        for relative, raw in [('manifests/V2_EXPERIMENT_FREEZE.json', b'{"fixture":"freeze"}\n'),
                              ('reports/V2_CONTROLLED_EXPERIMENT.json', b'{"fixture":"result"}\n')]:
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw); evidence[relative] = sha(raw)
        review = ('FINAL_VERDICT: PASS\n' + source_digest + '\n' + canonical(evidence)).encode()
        (self.root / 'reports/INDEPENDENT_REVIEW.md').write_bytes(review)
        (self.root / 'manifests/VALIDATED_CODE.json').write_text(canonical({
            'code_hashes': code, 'source_digest': source_digest, 'required_evidence': evidence,
            'independent_review_sha256': sha(review)}))
        (self.root / 'configs').mkdir()
        (self.root / 'configs/production.json').write_text('{"active_generation_profile":"v2"}')
        clock_value = [1]
        def hashes():
            return code
        with patch.object(release, 'ROOT', self.root), patch.object(release, 'source_hashes', side_effect=hashes), \
             patch.object(release.time, 'monotonic', side_effect=lambda: clock_value[0]):
            self.assertTrue(release.current_independent_review())
            self.assertTrue(release.current_independent_review(deadline=10))
            def expire_during_hashes():
                clock_value[0] = 11
                return code
            with patch.object(release, 'source_hashes', side_effect=expire_during_hashes):
                self.assertFalse(release.current_independent_review(deadline=10))

    def test_cycle_last_file_hidden_holding_restores_before_custody_backup_then_evicts(self):
        relative = 'data/accepted/train-7.jsonl'
        # Exactly one payload, so no other visible sibling can accidentally
        # trigger the right receipt recovery and hide the last-file defect.
        (self.root / 'data/sealed/sealed-8.jsonl').unlink()
        self.paths = {relative: self.paths[relative]}
        body = {key: deepcopy(value) for key, value in self.catalog.items() if key != 'catalog_id'}
        body['files'] = {relative: body['files'][relative]}
        self.catalog = {**body, 'catalog_id': sha(canonical(body).encode())}
        self.remote.add(self.remote.snapshot_commit,
            'manifests/remote_catalogs/catalog-' + self.catalog['catalog_id'] + '.json',
            (canonical(self.catalog) + '\n').encode())
        cycle, config, pause, original, _, stages, _ = self.cycle_existing_fixture()
        index = archive.load_index(self.root)
        identifier = index['files'][relative]['receipt_id']
        receipt = json.loads((self.root / index['receipts'][identifier]['path']).read_text())
        holding = ('data/accepted/.gcdl-evacuate-' + identifier + '-' + sha(relative.encode())[:16] + '/payload')
        target = self.root / holding; target.parent.mkdir(mode=0o700)
        os.rename(self.root / relative, target)
        journal = self.root / ('state/archive_evacuations/' + identifier + '.json')
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(canonical({'receipt_id': identifier, 'verified_remote': True,
            'deleted': [], 'status': 'evacuating', 'pending': {'relative': relative, 'holding': holding,
            'identity': receipt['local_identities'][relative], 'parents': receipt['local_ancestry'][relative]}}))
        backup_count = [0]
        def backup(root, deadline):
            backup_count[0] += 1
            if backup_count[0] == 1:
                self.assertEqual((self.root / relative).read_bytes(), self.paths[relative],
                                 'restore must happen before Git can stage an accidental hidden copy')
                self.assertFalse(target.exists())
            stages.append('confirmed-backup')
            return self.remote.snapshot_commit
        with patch.object(cycle, '_backup', side_effect=backup), \
             patch.object(cycle.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024 ** 3)):
            result = cycle.run(self.root, config, transport=self.remote, compact=False)
        self.assertEqual(result['deleted_files'], 1)
        self.assertEqual(result['remaining_local_payloads'], 0)
        self.assertLess(stages.index('confirmed-backup'), stages.index('evict'))
        self.assertFalse(target.exists()); self.assertFalse((self.root / relative).exists())
        self.assertEqual(pause.read_bytes(), original)

    def test_cycle_malformed_or_orphan_pending_pointer_refuses_before_backup_or_remote_proof(self):
        for orphan in (False, True):
            with self.subTest(orphan=orphan):
                cycle, config, pause, original, _, stages, _ = self.cycle_existing_fixture()
                index = archive.load_index(self.root)
                relative = 'data/accepted/train-7.jsonl'
                identifier = '0' * 64 if orphan else index['files'][relative]['receipt_id']
                directory = self.root / 'state/archive_evacuations'; directory.mkdir(exist_ok=True)
                path = directory / (identifier + '.json')
                path.write_text(canonical({'receipt_id': identifier, 'status': 'evacuating',
                    'pending': {'relative': 'data/accepted/UNRESERVED.jsonl', 'holding': '../foreign',
                                'identity': [], 'parents': []}}))
                sentinel = self.root.parent / 'foreign'; sentinel.write_bytes(b'FOREIGN SENTINEL')
                self.remote.calls.clear()
                with self.assertRaises((RuntimeError, ValueError)):
                    cycle.run(self.root, config, transport=self.remote, compact=False)
                self.assertEqual(stages, [], 'unbound pending custody must fail before Git auth/backup')
                self.assertEqual(self.remote.calls, [])
                self.assertEqual(sentinel.read_bytes(), b'FOREIGN SENTINEL')
                self.assertEqual(pause.read_bytes(), original)
                self.assert_original_present()
                path.unlink()

    def test_cycle_compaction_pending_cannot_overwrite_new_operator_pause(self):
        cycle, config, pause, _, _, _, _ = self.cycle_existing_fixture(no_initial_pause=True)
        operator = b'{"reason":"operator investigation during Git cleanup","origin":27}\n'
        def compact(root, **kwargs):
            pause.write_bytes(operator)
            return {'status': 'compacted_cleanup_pending', 'archived_blob_objects_remaining': 1}
        with patch.object(cycle.compact_git, 'compact', side_effect=compact):
            with self.assertRaisesRegex(RuntimeError, 'pause changed'):
                cycle.run(self.root, config, transport=self.remote, compact=True)
        self.assertEqual(pause.read_bytes(), operator)

    def test_completed_git_cleanup_is_bound_to_the_exact_current_archive_index(self):
        from goal_conditioned_decision_data.scripts import archive_cycle as cycle
        receipt = self.verify(); archive.evict_verified(self.root, receipt, transport=self.remote)
        index = archive.load_index(self.root)
        # Crash after the last unlink has no pending holding pointer. The
        # uncompleted Git store still needs work independently of free space.
        self.assertTrue(cycle.cleanup_pending(self.root))
        record = self.root / 'state/git_compaction.json'
        record.write_text(canonical({'status': 'compacted', 'archive_index_digest': '0' * 64}))
        self.assertTrue(cycle.cleanup_pending(self.root))
        record.write_text(canonical({'status': 'compacted', 'archive_index_digest': index['digest']}))
        self.assertFalse(cycle.cleanup_pending(self.root))
        record.write_text(canonical({'status': 'compacted_cleanup_pending', 'archive_index_digest': index['digest']}))
        self.assertTrue(cycle.cleanup_pending(self.root))


class IndependentArchiveBackupSafety(unittest.TestCase):
    def reject_deletion(self, status):
        namespace = runpy.run_path(str(ROOT / 'scripts/backup.py'))
        main = namespace['main']
        globals_ = main.__globals__
        calls, writes = [], []
        def execute(command, **kwargs):
            calls.append(command)
            if command[:3] == ['git', 'branch', '--show-current']:
                return SimpleNamespace(stdout='research/goal-conditioned-decision-data-v1\n', stderr='', returncode=0)
            if command[:3] == ['git', 'status', '--porcelain']:
                return SimpleNamespace(stdout=status, stderr='', returncode=0)
            if command[:2] == ['git', 'status']:
                return SimpleNamespace(stdout=status, stderr='', returncode=0)
            if command[:3] == ['git', 'diff', '--cached']:
                return SimpleNamespace(stdout='D\tgoal_conditioned_decision_data/data/accepted/train-1.jsonl\n', stderr='', returncode=1)
            raise AssertionError('operation beyond the destructive guard: ' + repr(command))
        with tempfile.TemporaryDirectory(prefix='gcdl-archive-review-') as temp:
            root = Path(temp) / 'goal_conditioned_decision_data'
            (root / 'configs').mkdir(parents=True)
            # Control configuration only: no dataset or programme copy.
            (root / 'configs/production.json').write_bytes((ROOT / 'configs/production.json').read_bytes())
            overrides = {'ROOT': root, 'REPO': root.parent, 'execute': execute,
                         'atomic_json': lambda path, value: writes.append(deepcopy(value))}
            with patch.dict(globals_, overrides), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = main()
        self.assertEqual(result, 1)
        self.assertTrue(writes)
        self.assertIn('delet', writes[-1]['reason'].lower())
        self.assertFalse(any(command[:2] in (['git', 'add'], ['git', 'commit'], ['gh', 'auth']) for command in calls))

    def test_missing_tracked_payload_cannot_be_committed_as_remote_deletion(self):
        self.reject_deletion(' D goal_conditioned_decision_data/data/accepted/train-1.jsonl\0')

    def test_already_staged_payload_deletion_is_refused_before_authentication(self):
        self.reject_deletion('D  goal_conditioned_decision_data/data/accepted/train-1.jsonl\0')


class IndependentArchiveGitBoundary(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='gcdl-archive-git-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / 'goal_conditioned_decision_data'
        self.relative = 'data/accepted/train-9.jsonl'
        path = self.root / self.relative; path.parent.mkdir(parents=True)
        self.raw = b'SMALL TRACKED BYTES\n'; path.write_bytes(self.raw)
        self.commit = 'f' * 40; self.oid = git_blob(self.raw)
        self.full_path = 'goal_conditioned_decision_data/' + self.relative
        self.commands = []
        self.stage = '100644 ' + self.oid + ' 0\t' + self.full_path + '\0'
        self.tree = '100644 blob ' + self.oid + '\t' + self.full_path + '\0'
        self.diff = b''
        self.remote = 'https://github.com/LUNCY579/rineya-world-model-research.git'
        def run(root, command, *args, **kwargs):
            self.assertEqual(Path(root), self.root.resolve())
            self.commands.append(command)
            if command == ['git', 'rev-parse', '--show-toplevel']:
                return (str(self.root.parent) + '\n').encode()
            if command == ['git', 'config', '--get', 'remote.origin.url']:
                return (self.remote + '\n').encode()
            if command[:2] == ['git', 'ls-tree']:
                self.assertEqual(command[3], self.commit)
                return self.tree.encode()
            if command[:2] == ['git', 'ls-files']:
                return self.stage.encode()
            if command[:2] == ['git', 'diff']:
                return self.diff
            raise AssertionError('unsafe/unexpected metadata command: ' + repr(command))
        hook = patch.object(archive, '_run', side_effect=run)
        hook.start(); self.addCleanup(hook.stop)

    def plan(self):
        return archive.plan_archive(self.root, [self.relative], self.commit)

    def test_plan_matches_independent_byte_hash_and_never_reads_git_blob(self):
        catalog = self.plan()
        self.assertEqual(catalog['files'], {self.relative: {
            'commit': self.commit, 'git_blob_sha1': self.oid, 'sha256': sha(self.raw), 'bytes': len(self.raw)}})
        self.assertEqual(catalog['catalog_id'], sha(canonical({key: value for key, value in catalog.items()
                                                             if key != 'catalog_id'}).encode()))
        self.assertFalse(any('cat-file' in command or 'fetch' in command or 'add' in command
                             for command in self.commands))

    def test_untracked_staged_changed_and_conflicted_files_refuse_plan(self):
        good = self.stage
        for staged in ['', good.replace(self.oid, '0' * 40), good.replace(' 0\t', ' 2\t')]:
            with self.subTest(stage=staged):
                self.stage = staged
                with self.assertRaisesRegex(archive.ArchiveError, 'untracked, staged-changed or conflicted'):
                    self.plan()
                self.assertFalse((self.root / 'manifests/remote_catalogs').exists())
        self.assertEqual((self.root / self.relative).read_bytes(), self.raw)

    def test_changed_worktree_or_wrong_commit_blob_refuses(self):
        self.diff = (self.full_path + '\n').encode()
        with self.assertRaisesRegex(archive.ArchiveError, 'changed committed'):
            self.plan()
        self.diff = b''; self.tree = self.tree.replace(self.oid, '0' * 40)
        with self.assertRaisesRegex(archive.ArchiveError, 'committed payload blob'):
            self.plan()
        self.assertFalse((self.root / 'manifests/remote_catalogs').exists())
        self.assertEqual((self.root / self.relative).read_bytes(), self.raw)

    def test_wrong_owner_remote_protected_path_and_explicit_batch_bounds_refuse(self):
        self.remote = 'https://github.com/other/project.git'
        with self.assertRaisesRegex(archive.ArchiveError, 'outside the owner'):
            self.plan()
        self.remote = 'https://github.com/LUNCY579/rineya-world-model-research.git'
        with self.assertRaisesRegex(archive.ArchiveError, 'protected code/control'):
            archive.plan_archive(self.root, ['state/queue.sqlite'], self.commit)
        with self.assertRaisesRegex(archive.ArchiveError, 'exceeds bounded batch'):
            archive.plan_archive(self.root, [self.relative], self.commit, max_bytes=1)
        self.assertFalse((self.root / 'manifests/remote_catalogs').exists())

    def test_held_worker_lock_prevents_planning_before_any_git_call(self):
        import fcntl
        lock = self.root / 'state/worker.lock'; lock.parent.mkdir()
        with lock.open('w') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(archive.ArchiveError, 'worker/archive lock is held'):
                self.plan()
        self.assertEqual(self.commands, [])


class IndependentArchiveTransportBounds(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='gcdl-archive-transport-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / 'goal_conditioned_decision_data'
        self.root.mkdir()

    def test_streaming_command_cap_refuses_small_overlong_body(self):
        # Real pipe/selector limit, tiny harmless process; no network or data.
        with self.assertRaisesRegex(archive.ArchiveError, 'bound'):
            archive._run(self.root, [sys.executable, '-c', 'print("x" * 256)'], max_output=64)

    def test_deadline_refuses_before_spawning_transport(self):
        import time
        with patch.object(archive.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(archive.ArchiveError, 'deadline'):
                archive._run(self.root, ['NEVER EXECUTED'], deadline=time.monotonic() - 1)
            spawn.assert_not_called()

    def test_owner_auth_discards_other_tokens_hosts_debug_and_keeps_token_process_local(self):
        calls = []
        def run(root, command, deadline=None, env=None, **kwargs):
            calls.append((command, dict(env), kwargs))
            self.assertEqual(command, ['gh', 'auth', 'token', '--user', 'LUNCY579', '--hostname', 'github.com'])
            self.assertTrue(all(key not in env for key in
                ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_ENTERPRISE_TOKEN', 'GITHUB_ENTERPRISE_TOKEN', 'GH_DEBUG', 'GH_HOST')))
            return b'FIXTURE_OWNER_TOKEN\n'
        polluted = {'GH_TOKEN': 'UNRELATED', 'GITHUB_TOKEN': 'UNRELATED', 'GH_HOST': 'other.example',
                    'GH_DEBUG': 'api', 'GH_ENTERPRISE_TOKEN': 'UNRELATED', 'GITHUB_ENTERPRISE_TOKEN': 'UNRELATED'}
        with patch.dict(os.environ, polluted), patch.object(archive, '_run', side_effect=run):
            transport = archive.GitHubTransport(self.root)
            environment = transport._environment()
            self.assertEqual(environment['GH_TOKEN'], 'FIXTURE_OWNER_TOKEN')
            self.assertEqual(os.environ['GH_TOKEN'], 'UNRELATED')
            self.assertIs(transport._environment(), environment)
            self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_truncated_or_unbound_github_tree_is_never_accepted(self):
        commit = 'a' * 40; tree = 'b' * 40
        for malformed in ({'sha': tree, 'truncated': True, 'tree': []},
                          {'sha': tree, 'tree': []},
                          {'sha': 'c' * 40, 'truncated': False, 'tree': []}):
            with self.subTest(tree=malformed):
                transport = archive.GitHubTransport(self.root)
                with patch.object(transport, '_api', side_effect=[{'sha': commit, 'tree': {'sha': tree}}, malformed]):
                    with self.assertRaisesRegex(archive.ArchiveError, 'tree identity/truncation'):
                        transport.fetch_tree(commit)

    def test_blob_raw_media_and_caller_bound_are_forwarded_exactly(self):
        transport = archive.GitHubTransport(self.root)
        with patch.object(transport, '_api', return_value=b'opaque\x00\xff\n') as api:
            self.assertEqual(transport.fetch_blob('a' * 40, max_bytes=9), b'opaque\x00\xff\n')
            api.assert_called_once_with('git/blobs/' + 'a' * 40, 9, raw=True)


class IndependentArchiveSnapshotCreation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='gcdl-archive-create-ref-')
        self.addCleanup(directory.cleanup)
        self.root = (Path(directory.name) / 'goal_conditioned_decision_data').resolve()
        self.root.mkdir()
        self.commit = 'b' * 40
        self.reference = 'refs/heads/archive/gcdl-001/' + self.commit

    def publish(self, *, initial=None, race=False, wrong_reply=None):
        from goal_conditioned_decision_data.scripts import archive_cycle as cycle
        import time
        state = {'ref': initial}; calls = []
        transport = SimpleNamespace(_environment=lambda: {'GH_TOKEN': 'FIXTURE_OWNER_TOKEN'})
        def command(root, args, deadline, environment=None):
            self.assertEqual(Path(root), self.root)
            self.assertEqual(environment['GIT_ALLOW_PROTOCOL'], 'https')
            calls.append(list(args))
            if 'ls-remote' in args:
                return b'' if state['ref'] is None else (state['ref'] + '\t' + self.reference + '\n').encode()
            if race:
                # A concurrent actor created an ancestor after our absent read.
                state['ref'] = 'a' * 40
            if args[:2] == ['gh', 'api']:
                self.assertEqual(args, ['gh', 'api', '--hostname', 'github.com', '--method', 'POST',
                    'repos/LUNCY579/rineya-world-model-research/git/refs', '-f', 'ref=' + self.reference,
                    '-f', 'sha=' + self.commit])
                if state['ref'] is not None:
                    raise RuntimeError('fixture422 reference already exists')
                state['ref'] = self.commit
                reply = {'ref': self.reference, 'object': {'type': 'commit', 'sha': self.commit}}
                if wrong_reply:
                    if wrong_reply == 'ref': reply['ref'] = 'refs/heads/other'
                    elif wrong_reply == 'type': reply['object']['type'] = 'tree'
                    else: reply['object']['sha'] = 'c' * 40
                return canonical(reply).encode()
            if 'push' in args:
                # Model the unsafe historical fast-forward accurately: no force
                # flag is needed to advance a newly created ancestor ref.
                state['ref'] = self.commit
                return b''
            raise AssertionError('unexpected snapshot command: ' + repr(args))
        with patch.object(cycle, '_command', side_effect=command):
            try:
                result = cycle._publish_snapshot(self.root, self.commit, time.monotonic() + 10, transport)
            except RuntimeError as error:
                return error, state, calls
        return result, state, calls

    def test_new_snapshot_uses_exact_create_reference_post_and_never_git_push(self):
        result, state, calls = self.publish()
        self.assertEqual(result, self.reference)
        self.assertEqual(state['ref'], self.commit)
        self.assertEqual(sum(args[:2] == ['gh', 'api'] for args in calls), 1)
        self.assertFalse(any('push' in args or 'PATCH' in args for args in calls))

    def test_existing_snapshot_same_commit_is_idempotent_other_commit_refused_without_write(self):
        result, state, calls = self.publish(initial=self.commit)
        self.assertEqual(result, self.reference)
        self.assertTrue(all('ls-remote' in args for args in calls))
        result, state, calls = self.publish(initial='a' * 40)
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(state['ref'], 'a' * 40)
        self.assertTrue(all('ls-remote' in args for args in calls))

    def test_concurrent_ancestor_reference_creation_is_never_fast_forwarded(self):
        result, state, calls = self.publish(race=True)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn('already exists', str(result))
        self.assertEqual(state['ref'], 'a' * 40)
        self.assertFalse(any('push' in args or 'PATCH' in args for args in calls))

    def test_creation_reply_wrong_reference_object_type_or_commit_identity_refuses(self):
        for field in ('ref', 'type', 'sha'):
            with self.subTest(field=field):
                result, _, calls = self.publish(wrong_reply=field)
                self.assertIsInstance(result, RuntimeError)
                self.assertFalse(any('push' in args or 'PATCH' in args for args in calls))

