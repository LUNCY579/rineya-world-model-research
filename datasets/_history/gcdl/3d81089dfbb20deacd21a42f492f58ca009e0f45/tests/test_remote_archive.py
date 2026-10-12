"""Small real-Git custody fixtures; remote storage is bounded, opaque RAM bytes."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive


class MemoryRemote:
    def __init__(self, catalog, snapshot, raws):
        self.catalog = catalog; self.snapshot = snapshot; self.calls = []
        self.trees = {}; self.blobs = {}; self.offline = False
        for relative, meta in catalog['files'].items():
            for commit in (meta['commit'], snapshot):
                self.add(commit, relative, raws[relative])
        self.add(snapshot, archive.catalog_path(catalog['catalog_id']), (archive._canonical(catalog) + '\n').encode())

    def add(self, commit, relative, raw):
        oid = archive.git_blob_sha1(raw); self.blobs[oid] = raw
        self.trees.setdefault(commit, {})[archive.PROGRAMME + '/' + relative] = {
            'type': 'blob', 'mode': '100644', 'sha': oid, 'size': len(raw)}

    def check(self):
        if self.offline:
            raise archive.ArchiveError('fixture offline')

    def fetch_ref(self, reference):
        self.check(); self.calls.append(('ref', reference))
        return self.snapshot if reference == archive.archive_ref(self.snapshot) else None

    def fetch_tree(self, commit):
        self.check(); self.calls.append(('tree', commit))
        return deepcopy(self.trees.get(commit, {}))

    def fetch_blob(self, oid, max_bytes):
        self.check(); self.calls.append(('blob', oid, max_bytes))
        raw = self.blobs.get(oid)
        if raw is None or len(raw) > max_bytes:
            raise archive.ArchiveError('missing/overlong fixture blob')
        return raw


class RemoteArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gcdl-archive-git-')
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name).resolve()
        self.root = self.repo / archive.PROGRAMME; self.root.mkdir()
        self.raws = {'data/accepted/train-17.jsonl': b'fixture canonical bytes\n',
                     'data/sealed/sealed-18.jsonl': b'opaque sealed bytes; no JSON parser\n'}
        for relative, raw in self.raws.items():
            target = self.root / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        (self.root / 'code.py').write_text('keep code\n')
        self.git('init', '-q'); self.git('remote', 'add', 'origin', 'https://github.com/' + archive.REPOSITORY + '.git')
        self.git('add', archive.PROGRAMME)
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'payload fixture')
        self.source = self.git('rev-parse', 'HEAD').strip()
        archive._CACHE.clear(); archive._CACHE_SIZE = 0
        self.addCleanup(archive._CACHE.clear)

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.repo, check=True, capture_output=True, text=True).stdout

    def prepare(self):
        catalog = archive.plan_archive(self.root, list(self.raws))
        self.git('add', archive.PROGRAMME + '/' + archive.catalog_path(catalog['catalog_id']))
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'catalog fixture')
        snapshot = self.git('rev-parse', 'HEAD').strip()
        remote = MemoryRemote(catalog, snapshot, self.raws)
        receipt = archive.verify_archive(self.root, catalog, snapshot, transport=remote)
        return catalog, remote, receipt

    def assert_present(self):
        for relative, raw in self.raws.items():
            self.assertEqual((self.root / relative).read_bytes(), raw)

    def test_real_git_committed_catalog_verified_remote_and_ram_cold_read(self):
        catalog, remote, receipt = self.prepare()
        self.assert_present()
        self.assertNotEqual(receipt['catalog_commit'], self.source)
        self.assertEqual({meta['commit'] for meta in catalog['files'].values()}, {self.source})
        result = archive.evict_verified(self.root, receipt, transport=remote)
        self.assertEqual(set(result['deleted']), set(self.raws)); self.assertFalse(result['Git_objects_removed'])
        self.assertEqual((self.root / 'code.py').read_text(), 'keep code\n')
        self.assertEqual(archive.iter_paths(self.root, 'data/*/*.jsonl'), sorted(self.raws))
        relative = 'data/accepted/train-17.jsonl'
        self.assertEqual(archive.read_bytes(self.root, relative, transport=remote), self.raws[relative])
        self.assertFalse((self.root / relative).exists())
        custody = archive.deletion_custody(self.root, relative)
        self.assertEqual(custody['catalog_commit'], remote.snapshot)
        self.assertEqual(custody['commit'], self.source)
        self.assertEqual(custody['archive_ref'], archive.archive_ref(remote.snapshot))
        self.assertLessEqual(archive._CACHE_SIZE, archive.RAM_CACHE_BYTES)
        calls = len(remote.calls)
        self.assertEqual(archive.payload_info(self.root, 'data/sealed/sealed-18.jsonl')['sha256'], hashlib.sha256(self.raws['data/sealed/sealed-18.jsonl']).hexdigest())
        self.assertEqual(len(remote.calls), calls)
        with self.assertRaisesRegex(archive.ArchiveError, 'sealed raw'):
            archive.read_bytes(self.root, 'data/sealed/sealed-18.jsonl', transport=remote)
        self.assertEqual(archive.read_bytes(self.root, 'data/sealed/sealed-18.jsonl', allow_sealed=True, transport=remote), self.raws['data/sealed/sealed-18.jsonl'])

    def test_untracked_or_changed_payload_refuses_catalog(self):
        new = self.root / 'data/accepted/untracked.jsonl'; new.write_bytes(b'new')
        with self.assertRaisesRegex(archive.ArchiveError, 'not committed'):
            archive.plan_archive(self.root, ['data/accepted/untracked.jsonl'])
        relative = 'data/accepted/train-17.jsonl'; (self.root / relative).write_bytes(b'changed')
        with self.assertRaises(archive.ArchiveError):
            archive.plan_archive(self.root, [relative])
        self.assertEqual((self.root / relative).read_bytes(), b'changed')

    def test_staged_changed_blob_and_wrong_remote_refuse(self):
        relative = 'data/accepted/train-17.jsonl'; (self.root / relative).write_bytes(b'changed')
        self.git('add', archive.PROGRAMME + '/' + relative)
        with self.assertRaises(archive.ArchiveError):
            archive.plan_archive(self.root, [relative], self.source)
        self.git('remote', 'set-url', 'origin', 'https://github.com/other/repo.git')
        with self.assertRaisesRegex(archive.ArchiveError, 'owner programme scope'):
            archive.plan_archive(self.root, [relative])

    def test_symlink_hardlink_traversal_and_protected_paths(self):
        relative = 'data/accepted/train-17.jsonl'; path = self.root / relative
        copy = self.repo / 'outside'; copy.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(copy)
        with self.assertRaisesRegex(archive.ArchiveError, 'symlink'):
            archive.plan_archive(self.root, [relative])
        path.unlink(); os.link(copy, path)
        with self.assertRaisesRegex(archive.ArchiveError, 'hardlink'):
            archive.plan_archive(self.root, [relative])
        for protected in ('state/worker.lock', 'state/progress.sqlite3', 'code.py', 'configs/production.json',
                          'manifests/SEALED_FREEZE.json', 'reports/INDEPENDENT_REVIEW.md',
                          'reports/QUALITY_REPORT.json', 'reports/PRODUCTION_STATUS.md', 'reports/history/source.py',
                          'reports/.hidden/proof.json', 'reports/STOP', 'reports/VALIDATED_CODE.json'):
            self.assertFalse(archive.is_payload_path(protected))
            with self.assertRaises(archive.ArchiveError):
                archive.plan_archive(self.root, [protected])
        for historical in ('reports/OPERATION_COMPOSITION.json', 'reports/V2_CONTROLLED_EXPERIMENT.md',
                           'reports/history/old-review.md', 'reports/review_history/evidence.json'):
            self.assertTrue(archive.is_payload_path(historical))
        for invalid in ('../data/x', '/tmp/x', 'data//x', 'data/./x', 'data\\x'):
            with self.assertRaises(archive.ArchiveError):
                archive.is_payload_path(invalid)

    def test_offline_and_corrupt_remote_retain_entire_batch(self):
        _, remote, receipt = self.prepare(); remote.offline = True
        with self.assertRaises(archive.ArchiveError):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present(); remote.offline = False
        oid = receipt['files']['data/sealed/sealed-18.jsonl']['git_blob_sha1']
        remote.blobs[oid] = b'x' * len(remote.blobs[oid])
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present()

    def test_index_corruption_and_missing_custody_snapshot_refuse_no_deletion(self):
        _, remote, receipt = self.prepare()
        index_path = self.root / archive.INDEX_PATH; original = index_path.read_bytes()
        index_path.write_bytes(original.replace(b'gcdl-remote-archive-1', b'gcdl-remote-archive-X'))
        with self.assertRaisesRegex(archive.ArchiveError, 'corrupt'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present(); index_path.write_bytes(original)
        index = json.loads(original); snapshot = self.root / ('manifests/remote_indexes/index-' + index['digest'] + '.json')
        snapshot.unlink()
        with self.assertRaises((archive.ArchiveError, FileNotFoundError)):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present()

    def test_snapshot_must_keep_payload_not_only_catalog(self):
        catalog, remote, receipt = self.prepare()
        relative = next(iter(catalog['files']))
        del remote.trees[remote.snapshot][archive.PROGRAMME + '/' + relative]
        with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present()

    def test_changed_inode_after_verification_refuses_all(self):
        _, remote, receipt = self.prepare(); relative = next(iter(self.raws)); path = self.root / relative
        replacement = path.with_suffix('.replacement'); replacement.write_bytes(path.read_bytes()); os.replace(replacement, path)
        with self.assertRaisesRegex(archive.ArchiveError, 'hash/inode'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present()

    def test_replacement_in_check_to_move_gap_is_restored_never_deleted(self):
        _, remote, receipt = self.prepare(); relative = next(iter(self.raws)); path = self.root / relative
        def replace_before_move(root, candidate):
            if candidate == relative:
                replacement = path.with_suffix('.replacement'); replacement.write_bytes(b'new user bytes')
                os.replace(replacement, path)
        with patch.object(archive, '_before_move', side_effect=replace_before_move):
            with self.assertRaisesRegex(archive.ArchiveError, 'moved file mismatch'):
                archive.evict_verified(self.root, receipt, transport=remote)
        self.assertEqual(path.read_bytes(), b'new user bytes')
        self.assertEqual((self.root / 'data/sealed/sealed-18.jsonl').read_bytes(), self.raws['data/sealed/sealed-18.jsonl'])

    def test_interrupted_holding_move_restores_then_reverifies(self):
        _, remote, receipt = self.prepare(); relative = next(iter(self.raws))
        holding = archive._holding_path(relative, receipt['receipt_id']); target = self.root / holding
        target.parent.mkdir(mode=0o700); os.rename(self.root / relative, target)
        journal = 'state/archive_evacuations/' + receipt['receipt_id'] + '.json'
        archive._journal(self.root, journal, {'receipt_id': receipt['receipt_id'], 'deleted': [], 'pending': {
            'relative': relative, 'holding': holding, 'identity': receipt['local_identities'][relative],
            'parents': receipt['local_ancestry'][relative]}})
        remote.offline = True
        with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present(); self.assertFalse(target.exists())
        # Durable pending custody survives the offline retry and explains only
        # the rename-induced ctime change. No fresh receipt is needed.
        remote.offline = False
        self.assertEqual(set(archive.evict_verified(self.root, receipt, transport=remote)['deleted']), set(self.raws))

    def prepare_catalog(self, receipt):
        return json.loads((self.root / receipt['catalog_path']).read_bytes())

    def test_partial_eviction_resume_retains_remaining_and_fetches_proof_again(self):
        _, remote, receipt = self.prepare(); second = list(receipt['files'])[1]
        def stop(root, relative):
            if relative == second:
                raise archive.ArchiveError('fixture interruption')
        with patch.object(archive, '_before_unlink', side_effect=stop):
            with self.assertRaisesRegex(archive.ArchiveError, 'interruption'):
                archive.evict_verified(self.root, receipt, transport=remote)
        self.assertFalse((self.root / list(receipt['files'])[0]).exists())
        self.assertEqual((self.root / second).read_bytes(), self.raws[second])
        remote.offline = True
        with self.assertRaises(archive.ArchiveError):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assertTrue((self.root / second).exists()); remote.offline = False
        result = archive.evict_verified(self.root, receipt, transport=remote)
        self.assertEqual(result['deleted'], [second])

    def test_missing_unverified_or_orphan_source_refuses(self):
        catalog = archive.plan_archive(self.root, list(self.raws)); snapshot = 'b' * 40
        remote = MemoryRemote(catalog, snapshot, self.raws)
        (self.root / next(iter(self.raws))).unlink()
        with self.assertRaisesRegex(archive.ArchiveError, 'orphan or missing'):
            archive.verify_archive(self.root, catalog, snapshot, transport=remote)
        self.assertEqual(remote.calls, [])

    def test_plan_bounds_and_shared_lock_enforced(self):
        with self.assertRaisesRegex(archive.ArchiveError, 'bounded batch'):
            archive.plan_archive(self.root, list(self.raws), max_files=1)
        self.assertEqual(len(archive.plan_archive(self.root, max_files=1)['files']), 1)
        with archive._lock(self.root):
            with self.assertRaisesRegex(archive.ArchiveError, 'lock is held'):
                archive.plan_archive(self.root, list(self.raws))

    def test_actual_subprocess_output_limit_and_deadline(self):
        with self.assertRaisesRegex(archive.ArchiveError, 'bounded bytes'):
            archive._run(self.root, [sys.executable, '-c', 'import sys;sys.stdout.buffer.write(b"x"*65536)'], max_output=128)
        import time
        with self.assertRaisesRegex(archive.ArchiveError, 'deadline'):
            archive._run(self.root, [sys.executable, '-c', 'import time;time.sleep(1)'], deadline=time.monotonic() + 0.05)
        # A blocked fetch is tested against a local fixture path, without any
        # possible network or authentication. Host Git 2.39 recognizes this
        # hard transport guard even if it ignores GIT_NO_LAZY_FETCH.
        with self.assertRaises(archive.ArchiveError):
            archive._run(self.root, ['git', 'ls-remote', str(self.repo)])

    def test_owner_token_stays_in_perprocess_environment(self):
        calls = []
        def run(root, command, deadline=None, env=None, max_output=None):
            calls.append((command, dict(env), max_output))
            if command[1:3] == ['auth', 'token']:
                return b'ghp_fixtureSecret123\n'
            return b'{"ref":"refs/heads/archive/gcdl-001/' + b'a' * 40 + b'","object":{"type":"commit","sha":"' + b'a' * 40 + b'"}}'
        with patch.dict(os.environ, {'GH_TOKEN': 'wrong', 'GITHUB_TOKEN': 'wrong2', 'GH_DEBUG': 'api'}), patch.object(archive, '_run', side_effect=run):
            transport = archive.GitHubTransport(self.root)
            self.assertEqual(transport.fetch_ref(archive.archive_ref('a' * 40)), 'a' * 40)
            self.assertEqual(os.environ['GH_TOKEN'], 'wrong')
        self.assertEqual(calls[0][0], ['gh', 'auth', 'token', '--user', 'LUNCY579', '--hostname', 'github.com'])
        self.assertNotIn('GH_TOKEN', calls[0][1]); self.assertNotIn('GITHUB_TOKEN', calls[0][1])
        self.assertEqual(calls[1][1]['GH_TOKEN'], 'ghp_fixtureSecret123'); self.assertNotIn('GH_DEBUG', calls[1][1])
        self.assertTrue(all('switch' not in command for command, _, _ in calls))

    def test_stop_before_and_during_remote_proof_or_atomic_move(self):
        _, remote, receipt = self.prepare(); stop = self.root / 'state/STOP'
        stop.write_text('operator stop')
        calls = len(remote.calls)
        with self.assertRaisesRegex(archive.ArchiveError, 'STOP'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assertEqual(len(remote.calls), calls); self.assert_present(); stop.unlink()
        fetch = remote.fetch_blob
        def stop_in_proof(oid, max_bytes):
            stop.write_text('new stop during proof')
            return fetch(oid, max_bytes)
        with patch.object(remote, 'fetch_blob', side_effect=stop_in_proof):
            with self.assertRaisesRegex(archive.ArchiveError, 'STOP'):
                archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present(); stop.unlink()
        with patch.object(archive, '_before_move', side_effect=lambda root, relative: stop.write_text('new stop before move')):
            with self.assertRaisesRegex(archive.ArchiveError, 'STOP'):
                archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present()

    def test_cli_plan_outputs_control_pointer_and_evict_requires_review(self):
        from goal_conditioned_decision_data.scripts import archive as command
        output = io.StringIO()
        with patch.object(command, 'ROOT', self.root), redirect_stdout(output):
            self.assertEqual(command.main(['plan', '--path', 'data/accepted/train-17.jsonl', '--max-files', '1']), 0)
        result = json.loads(output.getvalue()); self.assertEqual(result['deleted'], [])
        self.assertTrue((self.root / result['catalog_path']).exists())
        (self.root / 'configs').mkdir()
        (self.root / 'configs/production.json').write_text(json.dumps({'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION}))
        with patch.object(command, 'ROOT', self.root), patch('goal_conditioned_decision_data.auditors.release.current_independent_review', return_value=False), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                command.main(['evict', '--receipt', 'manifests/remote_custody/missing.json'])
            self.assertEqual(failure.exception.code, 1)
        self.assert_present()

    def test_low_disk_policy_is_explicit_and_one_verified_batch_only(self):
        _, remote, receipt = self.prepare()
        config = {'archive_enabled': True, 'archive_authorization': archive.AUTHORIZATION,
                  'archive_trigger_free_bytes': 10 * 1024 ** 3}
        low = type('Disk', (), {'free': 1})()
        with patch.object(archive.shutil, 'disk_usage', return_value=low):
            self.assertIsNone(archive.maybe_archive(self.root, {}))
            self.assertIsNone(archive.maybe_archive(self.root, {**config, 'archive_authorization': 'other'}))
            result = archive.maybe_archive(self.root, config, transport=remote)
        self.assertEqual(result['receipt_id'], receipt['receipt_id'])
        self.assertEqual(set(result['deleted']), set(self.raws))

    def test_parent_directory_replacement_after_verified_receipt_refuses(self):
        _, remote, receipt = self.prepare()
        parent = self.root / 'data'; saved = self.root / 'previous-data'
        parent.rename(saved); parent.mkdir()
        # Move entire child directories, preserving every file's full stat
        # identity. Only the common ancestor inode changes.
        os.rename(saved / 'accepted', parent / 'accepted')
        os.rename(saved / 'sealed', parent / 'sealed')
        for relative in self.raws:
            self.assertEqual(archive._local(self.root, relative, want_bytes=False)[2], receipt['local_identities'][relative])
        with self.assertRaisesRegex(archive.ArchiveError, 'hash/inode'):
            archive.evict_verified(self.root, receipt, transport=remote)
        self.assert_present()


if __name__ == '__main__':
    unittest.main()
