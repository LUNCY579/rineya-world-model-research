"""Actual isolated Git graph/store swaps; transport/auth/review/index stay fake.

The fixture remote is a local bare Git repository outside the owned fixture
host. Its immutable archive branch keeps opaque payload blobs while the active
branch drops them. No production path, credential or network is used.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.scripts import compact_git as compact


class ObjectRemote:
    def __init__(self, catalog, snapshot, raw):
        self.catalog = catalog; self.snapshot = snapshot; self.calls = []; self.offline = False
        self.trees = {}; self.blobs = {}
        for relative, meta in catalog['files'].items():
            for commit in (meta['commit'], snapshot):
                self.add(commit, relative, raw)
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
        self.check(); self.calls.append(('blob', oid))
        raw = self.blobs[oid]
        if len(raw) > max_bytes:
            raise archive.ArchiveError('fixture bound')
        return raw


class ActualGitCompaction(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='gcdl-compact-real-git-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve(); self.host = self.base / 'gcdl001-host'; self.host.mkdir()
        self.repository = self.host / 'repository'; self.repository.mkdir()
        self.git(self.repository, 'init', '-q', '-b', 'main')
        self.original_root = self.repository / archive.PROGRAMME; self.original_root.mkdir()
        (self.repository / '.gitignore').write_text(archive.PROGRAMME + '/state/\n')
        (self.original_root / 'code.py').write_text('fixture code kept exactly\n')
        self.relative = 'data/accepted/train-17.jsonl'; self.raw = b'opaque archived fixture payload\n'
        payload = self.original_root / self.relative; payload.parent.mkdir(parents=True); payload.write_bytes(self.raw)
        self.git(self.repository, 'add', '.'); self.commit(self.repository, 'opaque payload')
        self.source = self.git(self.repository, 'rev-parse', 'HEAD').strip()
        meta = {'commit': self.source, 'git_blob_sha1': archive.git_blob_sha1(self.raw),
                'sha256': hashlib.sha256(self.raw).hexdigest(), 'bytes': len(self.raw)}
        body = {'version': 'gcdl-archive-catalog-1', 'repository': archive.REPOSITORY,
                'programme': archive.PROGRAMME, 'files': {self.relative: meta}}
        self.catalog = {**body, 'catalog_id': archive._digest(body)}
        self.catalog_relative = archive.catalog_path(self.catalog['catalog_id'])
        catalog_file = self.original_root / self.catalog_relative; catalog_file.parent.mkdir(parents=True)
        catalog_file.write_text(archive._canonical(self.catalog) + '\n')
        self.git(self.repository, 'add', '.'); self.commit(self.repository, 'immutable catalog')
        self.snapshot = self.git(self.repository, 'rev-parse', 'HEAD').strip()
        self.git(self.repository, 'branch', compact.BRANCH)
        self.worktree = self.host / 'worktree'
        self.git(self.repository, 'worktree', 'add', '-q', str(self.worktree), compact.BRANCH)
        self.root = self.worktree / archive.PROGRAMME
        self.git(self.worktree, 'rm', '-q', archive.PROGRAMME + '/' + self.relative)
        self.commit(self.worktree, 'active code drops cold payload')
        self.head = self.git(self.worktree, 'rev-parse', 'HEAD').strip()
        self.bare = self.base / 'remote.git'; self.git(self.base, 'init', '--bare', '-q', str(self.bare))
        self.git(self.bare, 'config', 'uploadpack.allowFilter', 'true')
        self.git(self.bare, 'config', 'uploadpack.allowAnySHA1InWant', 'true')
        self.url = self.bare.as_uri()
        self.git(self.repository, 'remote', 'add', 'origin', compact.REMOTE)
        self.git(self.repository, 'push', '-q', self.url, 'main', compact.BRANCH,
                 self.snapshot + ':' + archive.archive_ref(self.snapshot))
        self.git(self.repository, 'update-ref', 'refs/remotes/origin/main', self.snapshot)
        self.git(self.repository, 'update-ref', 'refs/remotes/origin/' + compact.BRANCH, self.head)
        self.state = self.root / 'state'; self.state.mkdir()
        (self.state / 'PAUSED.json').write_text('{"reason":"disk fixture"}\n')
        (self.state / 'backup.json').write_text(json.dumps({'status': 'confirmed', 'commit': self.head, 'remote_commit': self.head}))
        (self.state / 'worker.lock').touch()
        (self.state / 'queue.db').write_bytes(b'fixture runtime bytes retained\n')
        self.remote = ObjectRemote(self.catalog, self.snapshot, self.raw)
        cat_raw = (archive._canonical(self.catalog) + '\n').encode()
        descriptor = {'catalog_commit': self.snapshot, 'archive_ref': archive.archive_ref(self.snapshot),
                      'catalog_path': self.catalog_relative, 'catalog_sha256': hashlib.sha256(cat_raw).hexdigest(),
                      'catalog_git_blob_sha1': archive.git_blob_sha1(cat_raw)}
        self.index = {'files': {self.relative: {**meta, 'catalog_id': self.catalog['catalog_id'], 'receipt_id': '1' * 64}},
                      'catalogs': {self.catalog['catalog_id']: descriptor}}
        self.index['digest'] = archive._digest(self.index)
        self.git_calls = []; self.original_git = compact._git
        def fixture_git(directory, args, environment=None, timeout=45):
            args = list(args); self.git_calls.append((Path(directory), args.copy()))
            if args[0] == 'clone':
                args = [self.url if item == compact.REMOTE else item for item in args]
            elif args[:3] == ['ls-remote', '--heads', 'origin']:
                args[2] = self.url
            return self.original_git(directory, args, environment, timeout)
        for hook in (patch.object(compact, '_git', side_effect=fixture_git),
                     patch.object(compact, '_owner_environment', return_value={**os.environ, 'GIT_ALLOW_PROTOCOL': 'file'}),
                     patch.object(archive, 'load_index', side_effect=lambda root: deepcopy(self.index)),
                     patch.object(archive, 'deletion_custody', side_effect=lambda root, relative: self.index['files'][relative]),
                     patch('goal_conditioned_decision_data.auditors.release.current_independent_review', return_value=True)):
            hook.start(); self.addCleanup(hook.stop)

    def git(self, directory, *args):
        return subprocess.run(['git', *args], cwd=directory, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True).stdout

    def commit(self, directory, message):
        self.git(directory, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', message)

    def runtime_identity(self):
        return {name: (archive._identity((self.state / name).stat()), (self.state / name).read_bytes())
                for name in ('PAUSED.json', 'backup.json', 'worker.lock', 'queue.db')}

    def assert_live(self, expected_runtime, directory_inodes):
        self.assertEqual(self.runtime_identity(), expected_runtime)
        self.assertEqual([self.worktree.stat().st_ino, self.root.stat().st_ino, self.state.stat().st_ino], directory_inodes)
        self.assertEqual(self.git(self.worktree, 'rev-parse', 'HEAD').strip(), self.head)
        self.assertEqual((self.root / 'code.py').read_text(), 'fixture code kept exactly\n')

    def test_real_shallow_filtered_store_removes_old_objects_preserves_snapshot_and_inodes(self):
        old_objects = compact._object_ids(self.worktree); oid = archive.git_blob_sha1(self.raw)
        self.assertIn(oid, old_objects)
        runtime = self.runtime_identity(); inodes = [self.worktree.stat().st_ino, self.root.stat().st_ino, self.state.stat().st_ino]
        result = compact.compact(self.root, transport=self.remote)
        self.assertEqual(result['status'], 'compacted'); self.assertEqual(result['archived_blob_objects_remaining'], 0)
        self.assertEqual(result['archive_index_digest'], self.index['digest'])
        self.assertEqual(json.loads((self.state / 'git_compaction.json').read_text())['archive_index_digest'], self.index['digest'])
        self.assertNotIn(oid, compact._object_ids(self.worktree))
        self.assertEqual(self.git(self.repository, 'rev-parse', '--is-shallow-repository').strip(), 'true')
        self.assertEqual(self.git(self.repository, 'config', 'remote.origin.partialclonefilter').strip(), 'blob:none')
        self.assertEqual(self.git(self.repository, 'config', '--get', 'remote.origin.url').strip(), compact.REMOTE)
        self.assert_live(runtime, inodes)
        self.assertEqual(self.git(self.bare, 'show', archive.archive_ref(self.snapshot) + ':' + archive.PROGRAMME + '/' + self.relative).encode(), self.raw)
        self.assertEqual(self.git(self.bare, 'ls-tree', compact.BRANCH, '--', archive.PROGRAMME + '/' + self.relative), '')
        self.assertFalse(list(self.host.glob('.git-compaction-*')))

    def test_second_compaction_accepts_exact_metadata_only_primary_and_is_idempotent(self):
        self.assertEqual(compact.compact(self.root, transport=self.remote)['status'], 'compacted')
        self.assertEqual({path.name for path in self.repository.iterdir()}, {'.git'})
        self.assertFalse((self.repository / '.git/index').exists())
        inode = self.repository.stat().st_ino
        result = compact.compact(self.root, transport=self.remote)
        self.assertEqual(result['status'], 'already_compact')
        self.assertEqual(result['archive_index_digest'], self.index['digest'])
        self.assertEqual(json.loads((self.state / 'git_compaction.json').read_text())['status'], 'already_compact')
        self.assertEqual(self.repository.stat().st_ino, inode)

    def test_clean_detached_primary_ancestor_of_confirmed_remote_tip_is_preserved(self):
        self.git(self.repository, 'checkout', '--detach', '-q', self.source)
        self.assertEqual(self.git(self.repository, 'status', '--porcelain'), '')
        self.assertNotEqual(self.source, self.snapshot)
        result = compact.compact(self.root, transport=self.remote)
        self.assertEqual(result['status'], 'compacted')
        self.assertNotIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))
        self.assertTrue(any(args[:2] == ['merge-base', self.source] for _, args in self.git_calls))

    def test_postinstall_validation_failure_rolls_back_before_cleanup(self):
        pointer = (self.worktree / '.git').read_bytes(); store_inode = self.repository.stat().st_ino
        runtime = self.runtime_identity(); inodes = [self.worktree.stat().st_ino, self.root.stat().st_ino, self.state.stat().st_ino]
        with patch('goal_conditioned_decision_data.auditors.release.current_independent_review', side_effect=[True, False]):
            with self.assertRaisesRegex(RuntimeError, 'review guard failed'):
                compact.compact(self.root, transport=self.remote)
        self.assertEqual((self.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(self.repository.stat().st_ino, store_inode)
        self.assertIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))
        self.assert_live(runtime, inodes)
        self.assertFalse(list(self.host.glob('.git-compaction-*')))

    def test_partial_retired_cleanup_keeps_healthy_replacement_and_retries(self):
        old_inode = self.repository.stat().st_ino
        remover = compact._remove_tree; cut = []
        def partial(path):
            if path.name == 'retired' and not cut:
                cut.append(True)
                # Deliberately corrupt the old store. Rolling back to it would
                # make HEAD unreadable and lose the verified healthy cache.
                (path / '.git/HEAD').unlink()
                raise OSError('fixture partial cleanup')
            return remover(path)
        with patch.object(compact, '_remove_tree', side_effect=partial):
            result = compact.compact(self.root, transport=self.remote)
        self.assertEqual(result['status'], 'compacted_cleanup_pending')
        self.assertEqual(result['archive_index_digest'], self.index['digest'])
        self.assertNotEqual(self.repository.stat().st_ino, old_inode)
        self.assertEqual(self.git(self.worktree, 'rev-parse', 'HEAD').strip(), self.head)
        self.assertNotIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))
        self.assertTrue((self.host / result['transaction'] / 'retired').exists())
        done = compact.compact(self.root, transport=self.remote)
        self.assertEqual(done['status'], 'compacted'); self.assertFalse(list(self.host.glob('.git-compaction-*')))

    def test_dirty_and_untracked_primary_refuse_before_clone_or_deletion(self):
        for kind in ('tracked', 'untracked', 'ignored'):
            with self.subTest(kind=kind):
                path = self.original_root / ('code.py' if kind == 'tracked' else ('state/lost.txt' if kind == 'ignored' else 'untracked.txt'))
                original = path.read_bytes() if path.exists() else None; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'user unpublished bytes')
                with self.assertRaisesRegex(RuntimeError, 'primary Git checkout is dirty/untracked'):
                    compact.compact(self.root, transport=self.remote)
                self.assertEqual(path.read_bytes(), b'user unpublished bytes')
                self.assertFalse(any(args[0] == 'clone' for _, args in self.git_calls))
                if original is None:path.unlink()
                else:path.write_bytes(original)
                if kind == 'ignored':path.parent.rmdir()

    def test_clone_failure_keeps_original_and_cleans_only_owned_preparation(self):
        inode = self.repository.stat().st_ino; actual = compact._git
        def fail_clone(directory, args, environment=None, timeout=45):
            if args[0] == 'clone':raise RuntimeError('fixture clone failure')
            return actual(directory, args, environment, timeout)
        with patch.object(compact, '_git', side_effect=fail_clone):
            with self.assertRaisesRegex(RuntimeError, 'clone failure'):
                compact.compact(self.root, transport=self.remote)
        self.assertEqual(self.repository.stat().st_ino, inode)
        self.assertEqual(self.git(self.worktree, 'rev-parse', 'HEAD').strip(), self.head)
        self.assertFalse(list(self.host.glob('.git-compaction-*')))

    def test_transaction_collision_is_not_cleaned_or_overwritten(self):
        suffix = 'a' * 32; occupied = self.host / ('.git-compaction-' + suffix); occupied.mkdir()
        (occupied / 'foreign.txt').write_bytes(b'foreign sibling bytes')
        with patch.object(compact.uuid, 'uuid4', return_value=SimpleNamespace(hex=suffix)):
            with self.assertRaises(FileExistsError):compact.compact(self.root, transport=self.remote)
        self.assertEqual((occupied / 'foreign.txt').read_bytes(), b'foreign sibling bytes')
        self.assertIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))

    def test_pointer_install_failure_rolls_back_without_untracked_root_temp(self):
        original = (self.worktree / '.git').read_bytes(); inode = self.repository.stat().st_ino
        replace = os.replace
        def refuse_pointer(source, destination):
            if Path(destination) == self.worktree / '.git':raise OSError('fixture pointer install failure')
            return replace(source, destination)
        with patch.object(compact.os, 'replace', side_effect=refuse_pointer):
            with self.assertRaisesRegex(OSError, 'pointer install failure'):
                compact.compact(self.root, transport=self.remote)
        self.assertEqual((self.worktree / '.git').read_bytes(), original)
        self.assertEqual(self.repository.stat().st_ino, inode)
        self.assertEqual(self.git(self.worktree, 'status', '--porcelain'), '')
        self.assertFalse(list(self.host.glob('.git-compaction-*')))
        self.assertFalse((self.worktree / '.git.compact-tmp').exists())
        self.assertIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))

    def test_process_output_and_total_compaction_deadlines_are_enforced(self):
        with self.assertRaisesRegex(RuntimeError, 'bounded bytes'):
            compact._command(self.host, [sys.executable, '-c', 'import sys;sys.stdout.buffer.write(b"x"*65536)'], os.environ.copy(), max_output=64)
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            compact.compact(self.root, transport=self.remote, seconds=0.001)
        self.assertLess(time.monotonic() - started, 1)
        self.assertIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))
        for seconds in (0, 181, float('nan'), float('inf'), True):
            with self.assertRaisesRegex(RuntimeError, 'bounded compaction deadline'):
                compact.compact(self.root, transport=self.remote, seconds=seconds)

    def test_operator_pause_change_during_remote_proof_preserves_original_store(self):
        inode = self.repository.stat().st_ino; pointer = (self.worktree / '.git').read_bytes()
        lock_inode = (self.state / 'worker.lock').stat().st_ino; fetch = self.remote.fetch_blob
        operator = b'{"reason":"operator requested investigation"}\n'
        def change_pause(oid, max_bytes):
            (self.state / 'PAUSED.json').write_bytes(operator)
            return fetch(oid, max_bytes)
        with patch.object(self.remote, 'fetch_blob', side_effect=change_pause):
            with self.assertRaisesRegex(RuntimeError, 'operator pause changed'):
                compact.compact(self.root, transport=self.remote)
        self.assertEqual((self.state / 'PAUSED.json').read_bytes(), operator)
        self.assertEqual((self.state / 'worker.lock').stat().st_ino, lock_inode)
        self.assertEqual(self.repository.stat().st_ino, inode)
        self.assertEqual((self.worktree / '.git').read_bytes(), pointer)
        self.assertIn(archive.git_blob_sha1(self.raw), compact._object_ids(self.worktree))
        self.assertFalse(any(args[0] == 'clone' for _, args in self.git_calls))


if __name__ == '__main__':
    unittest.main()
