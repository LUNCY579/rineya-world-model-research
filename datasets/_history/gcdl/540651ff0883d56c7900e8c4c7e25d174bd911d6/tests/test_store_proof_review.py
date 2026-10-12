"""Independent adversarial review of bounded Git-store custody proof.

The raw bodies are hand specified opaque bytes. Remote transport is bounded
RAM, while compaction uses the existing disposable, actual-Git fixture and its
local bare server. No production files, credentials, scientific experiment,
sealed target decoding or real GitHub access is involved.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.scripts import compact_git as compact
from goal_conditioned_decision_data.tests import test_compact_git as git_fixture


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def blob_id(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class RawServer:
    def __init__(self):
        self.refs, self.trees, self.blobs, self.calls = {}, {}, {}, []
        self.offline = False
        self.after_blob = None

    def add(self, commit, relative, raw):
        oid = blob_id(raw)
        self.blobs[oid] = raw
        self.trees.setdefault(commit, {})[archive.PROGRAMME + '/' + relative] = {
            'type': 'blob', 'mode': '100644', 'sha': oid, 'size': len(raw)}

    def check(self):
        if self.offline:
            raise archive.ArchiveError('independent transport failure')

    def fetch_ref(self, reference):
        self.calls.append(('ref', reference)); self.check()
        return self.refs.get(reference)

    def fetch_tree(self, commit):
        self.calls.append(('tree', commit)); self.check()
        return deepcopy(self.trees.get(commit, {}))

    def fetch_blob(self, oid, max_bytes):
        self.calls.append(('blob', oid, max_bytes)); self.check()
        if oid not in self.blobs:
            raise archive.ArchiveError('independent missing raw bytes')
        raw = self.blobs[oid]
        if len(raw) > max_bytes:
            raise archive.ArchiveError('independent overlong raw bytes')
        if self.after_blob is not None:
            self.after_blob(oid)
        return raw


def make_catalog(paths, source):
    files = {relative: {'commit': source, 'git_blob_sha1': blob_id(raw),
                        'sha256': sha(raw), 'bytes': len(raw)}
             for relative, raw in paths.items()}
    body = {'version': 'gcdl-archive-catalog-1', 'repository': archive.REPOSITORY,
            'programme': archive.PROGRAMME, 'files': files}
    return {**body, 'catalog_id': sha(canonical(body).encode())}


def publish(server, catalog, snapshot, paths):
    for relative, meta in catalog['files'].items():
        server.add(meta['commit'], relative, paths[relative])
        server.add(snapshot, relative, paths[relative])
    server.add(snapshot, archive.catalog_path(catalog['catalog_id']),
               (canonical(catalog) + '\n').encode())
    server.refs[archive.archive_ref(snapshot)] = snapshot


class IndependentSelectiveRawProof(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gcdl-selective-proof-review-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / archive.PROGRAMME
        self.root.mkdir()
        self.paths = {'data/accepted/train-old.jsonl': b'old opaque historical payload\n',
                      'data/accepted/train-new.jsonl': b'new opaque newly retained payload\n'}
        self.old, self.new = self.paths
        self.source, self.snapshot = 'a' * 40, 'b' * 40
        self.catalog = make_catalog(self.paths, self.source)
        self.server = RawServer(); publish(self.server, self.catalog, self.snapshot, self.paths)
        self.reference = archive.archive_ref(self.snapshot)
        self.new_oid, self.old_oid = blob_id(self.paths[self.new]), blob_id(self.paths[self.old])
        for relative, raw in self.paths.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        archive._CACHE.clear(); archive._CACHE_SIZE = 0
        self.addCleanup(lambda: (archive._CACHE.clear(), setattr(archive, '_CACHE_SIZE', 0)))

    def proof(self, selected=None):
        if selected is None:
            return archive._remote_proof(self.root, self.catalog, self.snapshot, self.server)
        return archive._remote_proof(self.root, self.catalog, self.snapshot, self.server,
                                     required_blob_ids=selected)

    def blob_calls(self):
        return [call[1] for call in self.server.calls if call[0] == 'blob']

    def assert_originals(self):
        for relative, raw in self.paths.items():
            self.assertEqual((self.root / relative).read_bytes(), raw)

    def test_default_proof_reads_both_payloads_and_rejects_either_corruption(self):
        self.proof()
        self.assertIn(self.old_oid, self.blob_calls()); self.assertIn(self.new_oid, self.blob_calls())
        for oid in (self.old_oid, self.new_oid):
            with self.subTest(oid=oid):
                good = self.server.blobs[oid]
                self.server.blobs[oid] = b'X' * len(good)
                with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
                    self.proof()
                self.server.blobs[oid] = good
                self.assert_originals()

    def test_public_verification_keeps_full_raw_default_and_cannot_issue_corrupt_custody(self):
        for oid in (self.old_oid, self.new_oid):
            with self.subTest(oid=oid):
                good = self.server.blobs[oid]; self.server.blobs[oid] = b'X' * len(good)
                original = archive._remote_proof
                with patch.object(archive, '_git_file'), patch.object(archive, '_remote_proof', wraps=original) as proof:
                    with self.assertRaises(archive.ArchiveError):
                        archive.verify_archive(self.root, self.catalog, self.snapshot, transport=self.server)
                self.assertTrue(proof.called)
                self.assertNotIn('required_blob_ids', proof.call_args.kwargs)
                self.assertFalse((self.root / archive.INDEX_PATH).exists())
                self.assert_originals(); self.server.blobs[oid] = good

    def test_public_eviction_keeps_full_raw_default_after_prior_success(self):
        with patch.object(archive, '_git_file'):
            receipt = archive.verify_archive(self.root, self.catalog, self.snapshot, transport=self.server)
        for oid in (self.old_oid, self.new_oid):
            with self.subTest(oid=oid):
                good = self.server.blobs[oid]; self.server.blobs[oid] = b'X' * len(good)
                original = archive._remote_proof
                with patch.object(archive, '_remote_proof', wraps=original) as proof:
                    with self.assertRaises(archive.ArchiveError):
                        archive.evict_verified(self.root, receipt, transport=self.server)
                self.assertTrue(proof.called)
                self.assertNotIn('required_blob_ids', proof.call_args.kwargs)
                self.assert_originals(); self.server.blobs[oid] = good

    def test_selected_new_payload_skips_old_raw_but_proves_all_metadata_and_refs(self):
        descriptor = self.proof({self.new_oid})
        calls = self.blob_calls()
        self.assertIn(self.new_oid, calls); self.assertNotIn(self.old_oid, calls)
        catalog_raw = (canonical(self.catalog) + '\n').encode()
        self.assertIn(blob_id(catalog_raw), calls)
        self.assertIn(('tree', self.source), self.server.calls)
        self.assertIn(('tree', self.snapshot), self.server.calls)
        self.assertEqual([call for call in self.server.calls if call[0] == 'ref'],
                         [('ref', self.reference), ('ref', self.reference)])
        self.assertEqual(descriptor['catalog_sha256'], sha(catalog_raw))
        self.assertEqual(descriptor['archive_ref'], self.reference)

    def test_selected_payload_same_length_corruption_is_rejected(self):
        self.server.blobs[self.new_oid] = b'X' * len(self.paths[self.new])
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            self.proof({self.new_oid})
        self.assertNotIn(self.old_oid, self.blob_calls()); self.assert_originals()

    def test_selected_payload_short_and_overlong_bodies_are_rejected(self):
        for raw in (self.paths[self.new][:-1], self.paths[self.new] + b'extra'):
            with self.subTest(length=len(raw)):
                self.server.blobs[self.new_oid] = raw
                with self.assertRaises(archive.ArchiveError):
                    self.proof({self.new_oid})
                self.assert_originals()

    def test_selected_catalog_sha256_mismatch_is_rejected_with_authentic_catalog_bytes(self):
        body = deepcopy(self.catalog); body.pop('catalog_id')
        body['files'][self.new]['sha256'] = '0' * 64
        self.catalog = {**body, 'catalog_id': sha(canonical(body).encode())}
        publish(self.server, self.catalog, self.snapshot, self.paths)
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            self.proof({self.new_oid})
        self.assert_originals()

    def test_selected_source_and_snapshot_wrong_oid_are_rejected(self):
        key = archive.PROGRAMME + '/' + self.new
        for commit in (self.source, self.snapshot):
            with self.subTest(commit=commit):
                good = deepcopy(self.server.trees[commit][key])
                self.server.trees[commit][key]['sha'] = 'c' * 40
                with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID/size'):
                    self.proof({self.new_oid})
                self.server.trees[commit][key] = good
                self.assert_originals()

    def test_empty_set_authenticates_catalog_and_all_tree_metadata_without_payload_raw(self):
        self.proof(frozenset())
        self.assertNotIn(self.old_oid, self.blob_calls()); self.assertNotIn(self.new_oid, self.blob_calls())
        self.assertEqual(self.blob_calls(), [blob_id((canonical(self.catalog) + '\n').encode())])
        self.assertEqual([call for call in self.server.calls if call[0] == 'ref'],
                         [('ref', self.reference), ('ref', self.reference)])

    def test_malformed_or_unknown_selectors_fail_before_transport(self):
        invalid = [[], [self.new_oid], (), (self.new_oid,), self.new_oid, {}, 0, True,
                   {None}, {'not-an-oid'}, {self.new_oid.upper()}, {'c' * 40}]
        for value in invalid:
            with self.subTest(value=repr(value)):
                self.server.calls.clear()
                with self.assertRaisesRegex(archive.ArchiveError, 'selector'):
                    self.proof(value)
                self.assertEqual(self.server.calls, [])
                self.assert_originals()

    def test_unselected_old_payload_missing_or_changed_snapshot_ref_still_refuses(self):
        for value in (None, 'c' * 40):
            with self.subTest(value=value):
                self.server.refs[self.reference] = value
                with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
                    self.proof({self.new_oid})
                self.assert_originals()

    def test_unselected_old_payload_catalog_raw_corruption_still_refuses(self):
        oid = blob_id((canonical(self.catalog) + '\n').encode())
        self.server.blobs[oid] = b'X' * len(self.server.blobs[oid])
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            self.proof({self.new_oid})
        self.assertNotIn(self.old_oid, self.blob_calls()); self.assert_originals()

    def test_unselected_old_source_tree_missing_mode_type_size_or_oid_still_refuses(self):
        key = archive.PROGRAMME + '/' + self.old
        good = deepcopy(self.server.trees[self.source][key])
        for field, value in [('missing', None), ('mode', '120000'), ('type', 'tree'),
                             ('size', len(self.paths[self.old]) + 1), ('sha', 'c' * 40)]:
            with self.subTest(field=field):
                self.server.trees[self.source][key] = deepcopy(good)
                if field == 'missing':
                    del self.server.trees[self.source][key]
                else:
                    self.server.trees[self.source][key][field] = value
                with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID/size'):
                    self.proof({self.new_oid})
                self.assert_originals()
        self.assertNotIn(self.old_oid, self.blob_calls())

    def test_unselected_old_snapshot_tree_missing_mode_type_size_or_oid_still_refuses(self):
        key = archive.PROGRAMME + '/' + self.old
        good = deepcopy(self.server.trees[self.snapshot][key])
        for field, value in [('missing', None), ('mode', '120000'), ('type', 'tree'),
                             ('size', len(self.paths[self.old]) + 1), ('sha', 'c' * 40)]:
            with self.subTest(field=field):
                self.server.trees[self.snapshot][key] = deepcopy(good)
                if field == 'missing':
                    del self.server.trees[self.snapshot][key]
                else:
                    self.server.trees[self.snapshot][key][field] = value
                with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID/size'):
                    self.proof({self.new_oid})
                self.assert_originals()
        self.assertNotIn(self.old_oid, self.blob_calls())

    def test_empty_selection_ref_moving_after_catalog_bytes_still_refuses(self):
        self.server.after_blob = lambda oid: self.server.refs.update({self.reference: 'c' * 40})
        with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
            self.proof(set())
        self.assertNotIn(self.old_oid, self.blob_calls()); self.assertNotIn(self.new_oid, self.blob_calls())
        self.assert_originals()


class IndependentStoreProofSelection(unittest.TestCase):
    def setUp(self):
        self.fixture = git_fixture.ActualGitCompaction(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)

    def pending_cleanup(self, remove_head=False):
        f = self.fixture
        remover = compact._remove_tree
        def retain_retired(path):
            if path.name == 'retired':
                if remove_head:
                    (path / '.git/HEAD').unlink()
                raise OSError('independent interruption before retired cleanup')
            return remover(path)
        with patch.object(compact, '_remove_tree', side_effect=retain_retired):
            result = compact.compact(f.root, transport=f.remote)
        self.assertEqual(result['status'], 'compacted_cleanup_pending')
        self.assertNotIn(blob_id(f.raw), compact._object_ids(f.worktree))
        retired = f.host / result['transaction'] / 'retired'
        self.assertTrue(retired.is_dir())
        return result, retired

    @staticmethod
    def file_snapshot(directory):
        return {str(path.relative_to(directory)): (path.stat().st_ino, path.read_bytes())
                for path in directory.rglob('*') if path.is_file()}

    def add_second_catalog_with_present_object(self, *, reuse_path=False, retain_old=False):
        f = self.fixture
        self.assertEqual(compact.compact(f.root, transport=f.remote)['status'], 'compacted')
        old_oid = blob_id(f.raw)
        self.assertNotIn(old_oid, compact._object_ids(f.worktree))
        catalogue_paths = []
        if reuse_path:
            # Historical reports may reuse an eligible path; accepted dataset
            # path identities remain immutable. The RAM remote constructs the
            # two report snapshots independently of the fixture bare server.
            f.relative = 'reports/history/reused-review.json'
            f.catalog = make_catalog({f.relative: f.raw}, f.source)
            f.catalog_relative = archive.catalog_path(f.catalog['catalog_id'])
            original_raw = (canonical(f.catalog) + '\n').encode()
            original_path = f.root / f.catalog_relative
            original_path.parent.mkdir(parents=True, exist_ok=True); original_path.write_bytes(original_raw)
            catalogue_paths.append(f.catalog_relative)
            f.index['files'] = {f.relative: {**f.catalog['files'][f.relative],
                                             'catalog_id': f.catalog['catalog_id'], 'receipt_id': '1' * 64}}
            f.index['catalogs'] = {f.catalog['catalog_id']: {
                'catalog_commit': f.snapshot, 'archive_ref': archive.archive_ref(f.snapshot),
                'catalog_path': f.catalog_relative, 'catalog_sha256': sha(original_raw),
                'catalog_git_blob_sha1': blob_id(original_raw)}}
        raw = b'hand specified newly present common-store object\n'
        relative = f.relative if reuse_path else 'data/accepted/train-independent-new.jsonl'
        source, snapshot = 'd' * 40, 'e' * 40
        catalog = make_catalog({relative: raw}, source)
        path = f.root / archive.catalog_path(catalog['catalog_id'])
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(canonical(catalog) + '\n')
        catalogue_paths.append(archive.catalog_path(catalog['catalog_id']))
        f.git(f.worktree, 'add', *(archive.PROGRAMME + '/' + relative for relative in catalogue_paths))
        f.commit(f.worktree, 'new independent catalogue control')
        f.head = f.git(f.worktree, 'rev-parse', 'HEAD').strip()
        f.git(f.worktree, 'push', '-q', f.url, compact.BRANCH)
        f.git(f.worktree, 'update-ref', 'refs/remotes/origin/' + compact.BRANCH, f.head)
        (f.state / 'backup.json').write_text(json.dumps({'status': 'confirmed', 'commit': f.head, 'remote_commit': f.head}))
        written = subprocess.run(['git', 'hash-object', '-w', '--stdin'], cwd=f.worktree,
                                 input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode().strip()
        self.assertEqual(written, blob_id(raw))
        if retain_old:
            old_written = subprocess.run(['git', 'hash-object', '-w', '--stdin'], cwd=f.worktree,
                                         input=f.raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode().strip()
            self.assertEqual(old_written, old_oid)
        cat_raw = (canonical(catalog) + '\n').encode()
        f.index['catalogs'][catalog['catalog_id']] = {
            'catalog_commit': snapshot, 'archive_ref': archive.archive_ref(snapshot),
            'catalog_path': archive.catalog_path(catalog['catalog_id']),
            'catalog_sha256': sha(cat_raw), 'catalog_git_blob_sha1': blob_id(cat_raw)}
        f.index['files'][relative] = {**catalog['files'][relative], 'catalog_id': catalog['catalog_id'],
                                      'receipt_id': '3' * 64}
        f.index.pop('digest'); f.index['digest'] = archive._digest(f.index)
        remote = RawServer()
        publish(remote, f.catalog, f.snapshot, {f.relative: f.raw})
        publish(remote, catalog, snapshot, {relative: raw})
        return remote, catalog, raw

    def test_real_common_store_intersection_drives_exact_per_catalog_proof(self):
        f = self.fixture
        remote, new_catalog, new_raw = self.add_second_catalog_with_present_object()
        old_oid, new_oid = blob_id(f.raw), blob_id(new_raw)
        before = compact._object_ids(f.worktree)
        self.assertNotIn(old_oid, before); self.assertIn(new_oid, before)
        captured = []; original = archive._remote_proof
        def observe(root, catalog, commit, server, **kwargs):
            captured.append((catalog['catalog_id'], kwargs.get('required_blob_ids')))
            return original(root, catalog, commit, server, **kwargs)
        with patch.object(archive, '_remote_proof', side_effect=observe):
            result = compact.compact(f.root, transport=remote)
        self.assertEqual(dict(captured), {f.catalog['catalog_id']: set(), new_catalog['catalog_id']: {new_oid}})
        self.assertEqual(result['archived_blob_objects_removed'], 1)
        self.assertEqual(result['archived_blob_objects_remaining'], 0)
        self.assertNotIn(new_oid, compact._object_ids(f.worktree))
        blobs = [call[1] for call in remote.calls if call[0] == 'blob']
        self.assertIn(new_oid, blobs); self.assertNotIn(old_oid, blobs)
        for snapshot in (f.snapshot, 'e' * 40):
            self.assertEqual([call for call in remote.calls if call == ('ref', archive.archive_ref(snapshot))],
                             [('ref', archive.archive_ref(snapshot))] * 2)

    def test_real_common_store_selected_corrupt_bytes_preserve_original_store(self):
        f = self.fixture
        remote, _, raw = self.add_second_catalog_with_present_object()
        oid = blob_id(raw); remote.blobs[oid] = b'X' * len(raw)
        pointer, inode, objects = (f.worktree / '.git').read_bytes(), f.repository.stat().st_ino, compact._object_ids(f.worktree)
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            compact.compact(f.root, transport=remote)
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(f.repository.stat().st_ino, inode)
        self.assertEqual(compact._object_ids(f.worktree), objects)
        self.assertFalse(list(f.host.glob('.git-compaction-*')))

    def test_historical_versions_of_reused_path_are_both_raw_proved_and_removed(self):
        f = self.fixture
        remote, new_catalog, new_raw = self.add_second_catalog_with_present_object(reuse_path=True, retain_old=True)
        old_oid, new_oid = blob_id(f.raw), blob_id(new_raw)
        self.assertEqual(len(f.index['files']), 1)
        self.assertEqual(f.index['files'][f.relative]['git_blob_sha1'], new_oid)
        self.assertTrue({old_oid, new_oid}.issubset(compact._object_ids(f.worktree)))
        captured = []; original = archive._remote_proof
        def observe(root, catalog, commit, server, **kwargs):
            captured.append((catalog['catalog_id'], kwargs.get('required_blob_ids')))
            return original(root, catalog, commit, server, **kwargs)
        with patch.object(archive, '_remote_proof', side_effect=observe):
            result = compact.compact(f.root, transport=remote)
        self.assertEqual(dict(captured), {f.catalog['catalog_id']: {old_oid}, new_catalog['catalog_id']: {new_oid}})
        self.assertEqual(result['archived_blob_objects_removed'], 2)
        self.assertEqual(result['archived_blob_objects_remaining'], 0)
        self.assertFalse({old_oid, new_oid} & compact._object_ids(f.worktree))
        blobs = [call[1] for call in remote.calls if call[0] == 'blob']
        self.assertIn(old_oid, blobs); self.assertIn(new_oid, blobs)

    def test_corrupt_historical_path_version_preserves_both_present_store_objects(self):
        f = self.fixture
        remote, _, new_raw = self.add_second_catalog_with_present_object(reuse_path=True, retain_old=True)
        old_oid, new_oid = blob_id(f.raw), blob_id(new_raw)
        remote.blobs[old_oid] = b'X' * len(f.raw)
        pointer, inode = (f.worktree / '.git').read_bytes(), f.repository.stat().st_ino
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            compact.compact(f.root, transport=remote)
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(f.repository.stat().st_ino, inode)
        self.assertTrue({old_oid, new_oid}.issubset(compact._object_ids(f.worktree)))
        self.assertFalse(list(f.host.glob('.git-compaction-*')))

    def test_historical_path_version_head_alias_refuses_before_remote_proof(self):
        f = self.fixture
        remote, _, new_raw = self.add_second_catalog_with_present_object(reuse_path=True, retain_old=True)
        old_oid, new_oid = blob_id(f.raw), blob_id(new_raw)
        self.assertEqual(f.index['files'][f.relative]['git_blob_sha1'], new_oid)
        (f.root / 'code.py').write_bytes(f.raw)
        f.git(f.worktree, 'add', archive.PROGRAMME + '/code.py')
        f.commit(f.worktree, 'historical old payload required by active HEAD')
        f.head = f.git(f.worktree, 'rev-parse', 'HEAD').strip()
        f.git(f.worktree, 'update-ref', 'refs/remotes/origin/' + compact.BRANCH, f.head)
        (f.state / 'backup.json').write_text(json.dumps({'status': 'confirmed', 'commit': f.head, 'remote_commit': f.head}))
        pointer, inode = (f.worktree / '.git').read_bytes(), f.repository.stat().st_ino
        with patch.object(archive, '_remote_proof', side_effect=AssertionError('historical alias reached remote')) as proof:
            with self.assertRaisesRegex(RuntimeError, 'byte alias'):
                compact.compact(f.root, transport=remote)
        proof.assert_not_called()
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(f.repository.stat().st_ino, inode)
        self.assertEqual((f.root / 'code.py').read_bytes(), f.raw)
        self.assertTrue({old_oid, new_oid}.issubset(compact._object_ids(f.worktree)))

    def test_new_cold_object_appearing_during_proof_cannot_be_deleted_unverified(self):
        f = self.fixture
        remote, _, new_raw = self.add_second_catalog_with_present_object()
        old_oid, new_oid = blob_id(f.raw), blob_id(new_raw)
        self.assertNotIn(old_oid, compact._object_ids(f.worktree))
        pointer, inode = (f.worktree / '.git').read_bytes(), f.repository.stat().st_ino
        injected = []
        def add_old_object(oid):
            if not injected:
                injected.append(True)
                result = subprocess.run(['git', 'hash-object', '-w', '--stdin'], cwd=f.worktree,
                                        input=f.raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self.assertEqual(result.stdout.decode().strip(), old_oid)
        remote.after_blob = add_old_object
        with self.assertRaisesRegex(RuntimeError, 'Git objects changed during proof'):
            compact.compact(f.root, transport=remote)
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(f.repository.stat().st_ino, inode)
        self.assertTrue({old_oid, new_oid}.issubset(compact._object_ids(f.worktree)))
        self.assertNotIn(old_oid, [call[1] for call in remote.calls if call[0] == 'blob'])
        self.assertFalse(list(f.host.glob('.git-compaction-*')))

    def test_every_head_byte_alias_refuses_before_remote_proof_or_store_deletion(self):
        f = self.fixture
        for relative in ('code.py', 'configs/alias.json', 'manifests/protected-alias.json'):
            with self.subTest(relative=relative):
                path = f.root / relative
                original = path.read_bytes() if path.exists() else None
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f.raw)
                f.git(f.worktree, 'add', archive.PROGRAMME + '/' + relative)
                f.commit(f.worktree, 'independent active HEAD byte alias')
                f.head = f.git(f.worktree, 'rev-parse', 'HEAD').strip()
                (f.state / 'backup.json').write_text(json.dumps({'status': 'confirmed', 'commit': f.head, 'remote_commit': f.head}))
                pointer, inode = (f.worktree / '.git').read_bytes(), f.repository.stat().st_ino
                with patch.object(archive, '_remote_proof', side_effect=AssertionError('alias proof reached remote')) as proof:
                    with self.assertRaisesRegex(RuntimeError, 'byte alias'):
                        compact.compact(f.root, transport=f.remote)
                proof.assert_not_called()
                self.assertEqual(path.read_bytes(), f.raw)
                self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
                self.assertEqual(f.repository.stat().st_ino, inode)
                self.assertIn(blob_id(f.raw), compact._object_ids(f.worktree))
                if original is None:
                    f.git(f.worktree, 'rm', '-q', archive.PROGRAMME + '/' + relative)
                else:
                    path.write_bytes(original); f.git(f.worktree, 'add', archive.PROGRAMME + '/' + relative)
                f.commit(f.worktree, 'restore independent retained source')

    def test_transport_failure_cannot_replace_or_delete_original_store(self):
        f = self.fixture
        pointer, inode, objects = (f.worktree / '.git').read_bytes(), f.repository.stat().st_ino, compact._object_ids(f.worktree)
        f.remote.offline = True
        with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
            compact.compact(f.root, transport=f.remote)
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(f.repository.stat().st_ino, inode)
        self.assertEqual(compact._object_ids(f.worktree), objects)
        self.assertFalse(list(f.host.glob('.git-compaction-*')))

    def test_pending_retired_only_blobs_require_fresh_full_raw_proof(self):
        f = self.fixture
        result, retired = self.pending_cleanup()
        before = self.file_snapshot(retired)
        pointer = (f.worktree / '.git').read_bytes()
        oid = blob_id(f.raw); f.remote.blobs[oid] = b'X' * len(f.raw)
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            compact.compact(f.root, transport=f.remote)
        self.assertEqual(self.file_snapshot(retired), before)
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(json.loads((f.state / 'git_compaction.json').read_text())['status'], result['status'])
        self.assertNotIn(oid, compact._object_ids(f.worktree))

    def test_partial_retired_metadata_loss_cannot_skip_corrupt_raw_proof(self):
        f = self.fixture
        _, retired = self.pending_cleanup(remove_head=True)
        self.assertFalse((retired / '.git/HEAD').exists())
        before = self.file_snapshot(retired)
        oid = blob_id(f.raw); f.remote.blobs[oid] = b'X' * len(f.raw)
        with self.assertRaisesRegex(archive.ArchiveError, 'mismatch'):
            compact.compact(f.root, transport=f.remote)
        self.assertEqual(self.file_snapshot(retired), before)
        self.assertNotIn(oid, compact._object_ids(f.worktree))

    def test_invalid_pending_journal_and_replaced_transaction_refuse_before_proof(self):
        f = self.fixture
        result, retired = self.pending_cleanup()
        journal = f.state / 'git_compaction.json'; valid = journal.read_bytes()
        before = self.file_snapshot(retired)
        for field, value in [('archive_index_digest', '0' * 64),
                             ('transaction_identity', [0, 0]), ('commit', 'c' * 40)]:
            with self.subTest(field=field):
                record = deepcopy(result); record[field] = value
                journal.write_text(json.dumps(record))
                with patch.object(archive, '_remote_proof', side_effect=AssertionError('invalid journal reached proof')) as proof:
                    with self.assertRaisesRegex(RuntimeError, 'receipt|replaced'):
                        compact.compact(f.root, transport=f.remote)
                proof.assert_not_called()
                self.assertEqual(self.file_snapshot(retired), before)
                self.assertNotIn(blob_id(f.raw), compact._object_ids(f.worktree))
                journal.write_bytes(valid)

    def test_pending_journal_appearing_after_selective_proof_cannot_authorize_cleanup(self):
        f = self.fixture
        _, retired = self.pending_cleanup()
        journal = f.state / 'git_compaction.json'; valid = journal.read_bytes(); journal.unlink()
        before = self.file_snapshot(retired)
        pointer = (f.worktree / '.git').read_bytes()
        f.remote.calls.clear()
        original = archive._remote_proof
        captured = []
        def install_journal_after_proof(root, catalog, commit, server, **kwargs):
            captured.append(kwargs.get('required_blob_ids'))
            descriptor = original(root, catalog, commit, server, **kwargs)
            journal.write_bytes(valid)
            return descriptor
        with patch.object(archive, '_remote_proof', side_effect=install_journal_after_proof):
            with self.assertRaisesRegex(RuntimeError, 'cleanup receipt changed during proof'):
                compact.compact(f.root, transport=f.remote)
        self.assertEqual(captured, [set()])
        self.assertNotIn(blob_id(f.raw), [call[1] for call in f.remote.calls if call[0] == 'blob'])
        self.assertEqual(self.file_snapshot(retired), before)
        self.assertEqual((f.worktree / '.git').read_bytes(), pointer)
        self.assertEqual(journal.read_bytes(), valid)


if __name__ == '__main__':
    unittest.main()
