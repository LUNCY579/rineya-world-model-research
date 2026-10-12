"""Cold-read sessions use tiny opaque RAM fixtures and never contact GitHub."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def blob_oid(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class ColdCustody:
    """Construct valid control commitments without exercising evacuation."""
    source = 'a' * 40
    snapshot = 'b' * 40

    def __init__(self, root, count=12, metadata_override=None, *, start=0, source=None, snapshot=None):
        self.root = root
        self.source = source or self.source
        self.snapshot = snapshot or self.snapshot
        self.raws = {'data/accepted/manifests/manifest-%03d.json' % number:
                     ('{"fixture":%d}\n' % number).encode() for number in range(start, start + count)}
        files = {}
        for relative, raw in self.raws.items():
            files[relative] = {'commit': self.source, 'git_blob_sha1': blob_oid(raw),
                               'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        if metadata_override:
            metadata_override(files)
        catalog = {'version': 'gcdl-archive-catalog-1', 'repository': archive.REPOSITORY,
                   'programme': archive.PROGRAMME, 'files': files}
        catalog['catalog_id'] = digest(catalog)
        self.catalog = catalog
        self.catalog_relative = 'manifests/remote_catalogs/catalog-' + catalog['catalog_id'] + '.json'
        self.catalog_raw = canonical(catalog) + b'\n'
        descriptor = {'catalog_commit': self.snapshot,
                      'archive_ref': 'refs/heads/archive/gcdl-001/' + self.snapshot,
                      'catalog_path': self.catalog_relative,
                      'catalog_sha256': hashlib.sha256(self.catalog_raw).hexdigest(),
                      'catalog_git_blob_sha1': blob_oid(self.catalog_raw)}
        receipt = {'version': 'gcdl-archive-custody-1', 'status': 'remote_verified',
                   'repository': archive.REPOSITORY, 'programme': archive.PROGRAMME,
                   'catalog_id': catalog['catalog_id'], **descriptor, 'files': files,
                   'local_identities': {}, 'local_ancestry': {}}
        receipt['receipt_id'] = digest(receipt)
        receipt_relative = 'manifests/remote_custody/custody-' + receipt['receipt_id'] + '.json'
        receipt_raw = canonical(receipt) + b'\n'
        index = {'version': 'gcdl-remote-archive-1', 'repository': archive.REPOSITORY,
                 'programme': archive.PROGRAMME,
                 'files': {relative: {**meta, 'catalog_id': catalog['catalog_id'],
                                      'receipt_id': receipt['receipt_id']}
                           for relative, meta in files.items()},
                 'catalogs': {catalog['catalog_id']: descriptor},
                 'receipts': {receipt['receipt_id']: {
                     'path': receipt_relative, 'sha256': hashlib.sha256(receipt_raw).hexdigest()}}}
        index['digest'] = digest(index)
        self.index = index
        index_raw = canonical(index) + b'\n'
        self.write(self.catalog_relative, self.catalog_raw)
        self.write(receipt_relative, receipt_raw)
        self.write('manifests/remote_indexes/index-' + index['digest'] + '.json', index_raw)
        self.write(archive.INDEX_PATH, index_raw)
        self.write('code.py', b'fixture live code remains local\n')
        self.write('reports/PRODUCTION_STATUS.md', b'fixture live status remains local\n')
        self.write('state/KEEP', b'fixture state remains local\n')

    def write(self, relative, raw):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


class FakeGitHubApi:
    """Bounded gh API responses; authentication is a captured fake process call."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.calls = []
        self.counts = Counter()
        self.blobs = {}
        self.catalog_oids = set()
        self.references = {}
        self.ref_sha = fixture.snapshot
        self.move_after_payload = False
        self.offline = False
        self.root_trees = {}
        self.sub_trees = {}
        self.entries = {}
        self.add_fixture(fixture)

    def add_fixture(self, fixture):
        self.blobs.update({meta['git_blob_sha1']: fixture.raws[relative]
                           for relative, meta in fixture.catalog['files'].items()})
        self.blobs[blob_oid(fixture.catalog_raw)] = fixture.catalog_raw
        self.catalog_oids.add(blob_oid(fixture.catalog_raw))
        self.references['refs/heads/archive/gcdl-001/' + fixture.snapshot] = fixture.snapshot
        for commit in (fixture.source, fixture.snapshot):
            self.root_trees[commit] = hashlib.sha1(('fixture root ' + commit).encode()).hexdigest()
            self.sub_trees[commit] = hashlib.sha1(('fixture subtree ' + commit).encode()).hexdigest()
            self.entries[commit] = [
                {'path': relative, 'mode': '100644', 'type': 'blob',
                 'sha': meta['git_blob_sha1'], 'size': meta['bytes']}
                for relative, meta in fixture.catalog['files'].items()]
        self.entries[fixture.snapshot].append({
            'path': fixture.catalog_relative, 'mode': '100644', 'type': 'blob',
            'sha': blob_oid(fixture.catalog_raw), 'size': len(fixture.catalog_raw)})

    def __call__(self, root, command, deadline=None, env=None, max_output=archive.MAX_READ_BYTES):
        if Path(root).resolve() != self.fixture.root.resolve():
            raise AssertionError('fixture API root mismatch')
        if deadline is not None and deadline <= time.monotonic():
            raise archive.ArchiveError('fixture API deadline exceeded')
        if self.offline:
            raise archive.ArchiveError('fixture API offline')
        self.calls.append((list(command), deadline, deepcopy(env), max_output))
        if command == ['gh', 'auth', 'token', '--user', archive.OWNER, '--hostname', 'github.com']:
            self.counts['auth'] += 1
            return b'ghp_fixtureReadSessionToken123\n'
        if command[:6] != ['gh', 'api', '--hostname', 'github.com', '--method', 'GET']:
            raise AssertionError('unexpected fixture command: ' + repr(command))
        prefix = 'repos/' + archive.REPOSITORY + '/'
        if not command[6].startswith(prefix):
            raise AssertionError('unexpected fixture repository')
        endpoint = command[6][len(prefix):]
        if endpoint.startswith('git/ref/'):
            self.counts['ref'] += 1
            reference = 'refs/' + endpoint[len('git/ref/'):]
            sha = self.ref_sha if reference.endswith(self.fixture.snapshot) else self.references[reference]
            value = {'ref': reference, 'object': {'type': 'commit', 'sha': sha}}
        elif endpoint.startswith('git/commits/'):
            self.counts['commit'] += 1
            commit = endpoint.rsplit('/', 1)[-1]
            value = {'sha': commit, 'tree': {'sha': self.root_trees[commit]}}
        elif endpoint.startswith('git/trees/'):
            self.counts['tree'] += 1
            tree = endpoint.split('/', 2)[-1].split('?', 1)[0]
            if tree in self.root_trees.values():
                commit = next(commit for commit, oid in self.root_trees.items() if oid == tree)
                value = {'sha': tree, 'truncated': False, 'tree': [{
                    'path': archive.PROGRAMME, 'mode': '040000', 'type': 'tree',
                    'sha': self.sub_trees[commit]}]}
            else:
                commit = next(commit for commit, oid in self.sub_trees.items() if oid == tree)
                value = {'sha': tree, 'truncated': False, 'tree': deepcopy(self.entries[commit])}
        elif endpoint.startswith('git/blobs/'):
            oid = endpoint.rsplit('/', 1)[-1]
            self.counts['catalog_blob' if oid in self.catalog_oids else 'payload_blob'] += 1
            raw = self.blobs[oid]
            if len(raw) > max_output:
                raise archive.ArchiveError('fixture API response exceeded bounded bytes')
            if self.move_after_payload and oid not in self.catalog_oids:
                self.ref_sha = '9' * 40
            return raw
        else:
            raise AssertionError('unexpected fixture API endpoint: ' + endpoint)
        raw = canonical(value)
        if len(raw) > max_output:
            raise archive.ArchiveError('fixture API response exceeded bounded bytes')
        return raw


class ArchiveReadSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gcdl-read-session-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve() / archive.PROGRAMME
        self.root.mkdir()
        self.fixture = ColdCustody(self.root)
        self.api = FakeGitHubApi(self.fixture)
        self.clear_raw_cache()
        self.addCleanup(self.clear_raw_cache)

    def clear_raw_cache(self):
        archive._CACHE.clear()
        archive._CACHE_SIZE = 0

    def local_files(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}

    def assert_cold(self):
        self.assertTrue(all(not (self.root / relative).exists() for relative in self.fixture.raws))

    def read_all(self):
        for relative, expected in self.fixture.raws.items():
            self.assertEqual(archive.read_bytes(self.root, relative), expected)

    def merge_indexes(self, *fixtures):
        index = {'version': 'gcdl-remote-archive-1', 'repository': archive.REPOSITORY,
                 'programme': archive.PROGRAMME, 'files': {}, 'catalogs': {}, 'receipts': {}}
        for fixture in fixtures:
            for field in ('files', 'catalogs', 'receipts'):
                index[field].update(fixture.index[field])
        index['digest'] = digest(index)
        raw = canonical(index) + b'\n'
        self.fixture.write('manifests/remote_indexes/index-' + index['digest'] + '.json', raw)
        self.fixture.write(archive.INDEX_PATH, raw)

    def test_same_catalog_cold_reads_share_auth_transport_and_metadata(self):
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                self.read_all()
        self.assertEqual(self.api.counts['auth'], 1)
        self.assertEqual(self.api.counts['catalog_blob'], 1)
        self.assertEqual(self.api.counts['payload_blob'], len(self.fixture.raws))
        self.assertEqual(self.api.counts['commit'], 2)
        self.assertEqual(self.api.counts['tree'], 4)
        self.assertEqual(self.api.counts['ref'], 2 * len(self.fixture.raws))
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_session_reduces_api_calls_against_standalone_default_reads(self):
        with patch.object(archive, '_run', side_effect=self.api):
            self.read_all()
        standalone = len(self.api.calls)
        standalone_auth = self.api.counts['auth']
        self.clear_raw_cache()
        session_api = FakeGitHubApi(self.fixture)
        with patch.object(archive, '_run', side_effect=session_api):
            with archive.read_session(self.root):
                self.read_all()
        self.assertEqual(standalone_auth, len(self.fixture.raws))
        self.assertEqual(session_api.counts['auth'], 1)
        self.assertLess(len(session_api.calls), standalone // 2)
        self.assert_cold()

    def test_same_root_nested_session_reuses_outer_transport(self):
        first, second = list(self.fixture.raws)[:2]
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                self.assertEqual(archive.read_bytes(self.root, first), self.fixture.raws[first])
                with archive.read_session(self.root):
                    self.assertEqual(archive.read_bytes(self.root, second), self.fixture.raws[second])
        self.assertEqual(self.api.counts['auth'], 1)
        self.assertEqual(self.api.counts['catalog_blob'], 1)

    def test_local_reads_do_not_create_transport_or_authenticate(self):
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                self.assertEqual(archive.read_bytes(self.root, 'code.py'), before['code.py'])
                self.assertEqual(archive.read_text(self.root, 'reports/PRODUCTION_STATUS.md'),
                                 before['reports/PRODUCTION_STATUS.md'].decode())
                self.assertIsNone(session._transport)
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.local_files(), before)

    def test_explicit_transport_is_honored_inside_session(self):
        first = next(iter(self.fixture.raws))
        explicit = archive.GitHubTransport(self.root)
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                with patch.object(explicit, 'fetch_blob', wraps=explicit.fetch_blob) as fetched:
                    self.assertEqual(archive.read_bytes(self.root, first, transport=explicit), self.fixture.raws[first])
                self.assertEqual(fetched.call_count, 2)
        self.assertEqual(self.api.counts['auth'], 1)
        self.assert_cold()

    def test_different_root_nested_session_and_default_read_fail_closed(self):
        other = Path(self.directory.name).resolve() / 'other' / archive.PROGRAMME
        other.mkdir(parents=True)
        other_fixture = ColdCustody(other, count=1)
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                with self.assertRaises(archive.ArchiveError):
                    with archive.read_session(other):
                        pass
                with self.assertRaises(archive.ArchiveError):
                    archive.read_bytes(other, next(iter(other_fixture.raws)))
        self.assertEqual(self.api.counts['auth'], 0)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_session_scope_resets_after_exception(self):
        first, second = list(self.fixture.raws)[:2]
        with patch.object(archive, '_run', side_effect=self.api):
            with self.assertRaisesRegex(RuntimeError, 'fixture body failure'):
                with archive.read_session(self.root):
                    archive.read_bytes(self.root, first)
                    raise RuntimeError('fixture body failure')
            archive.read_bytes(self.root, second)
        self.assertEqual(self.api.counts['auth'], 2)
        self.assertEqual(self.api.counts['catalog_blob'], 2)

    def test_corrupted_local_index_is_refused_even_after_metadata_is_warm(self):
        first, second = list(self.fixture.raws)[:2]
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                archive.read_bytes(self.root, first)
                counts = self.api.counts.copy()
                self.fixture.write(archive.INDEX_PATH, b'{"corrupt":true}\n')
                with self.assertRaises(archive.ArchiveError):
                    archive.read_bytes(self.root, second)
        self.assertEqual(self.api.counts, counts)
        self.assert_cold()

    def test_corrupted_local_catalog_is_refused_even_after_remote_catalog_is_warm(self):
        first, second = list(self.fixture.raws)[:2]
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                archive.read_bytes(self.root, first)
                counts = self.api.counts.copy()
                self.fixture.write(self.fixture.catalog_relative, b'{"corrupt":true}\n')
                with self.assertRaises(archive.ArchiveError):
                    archive.read_bytes(self.root, second)
        self.assertEqual(self.api.counts, counts)
        self.assert_cold()

    def test_snapshot_ref_checked_before_each_cold_read_with_warm_metadata(self):
        first, second = list(self.fixture.raws)[:2]
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                archive.read_bytes(self.root, first)
                self.api.ref_sha = '9' * 40
                payloads = self.api.counts['payload_blob']
                with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
                    archive.read_bytes(self.root, second)
        self.assertEqual(self.api.counts['payload_blob'], payloads)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_snapshot_ref_movement_after_payload_read_refuses_return(self):
        first = next(iter(self.fixture.raws))
        before = self.local_files()
        self.api.move_after_payload = True
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                with self.assertRaisesRegex(archive.ArchiveError, 'snapshot ref'):
                    archive.read_bytes(self.root, first)
        self.assertEqual(self.api.counts['ref'], 2)
        self.assertEqual(archive._CACHE_SIZE, 0)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_cold_raw_length_sha256_and_git_oid_are_verified(self):
        for corruption in ('length', 'sha256', 'git_oid'):
            with self.subTest(corruption=corruption):
                self.clear_raw_cache()
                metadata_override = None
                if corruption == 'git_oid':
                    metadata_override = lambda files: files[next(iter(files))].update(git_blob_sha1='8' * 40)
                fixture = ColdCustody(self.root, count=1, metadata_override=metadata_override)
                api = FakeGitHubApi(fixture)
                first = next(iter(fixture.raws))
                oid = fixture.catalog['files'][first]['git_blob_sha1']
                if corruption == 'length':
                    api.blobs[oid] += b'extra'
                elif corruption == 'sha256':
                    api.blobs[oid] = b'x' * len(api.blobs[oid])
                before = self.local_files()
                with patch.object(archive, '_run', side_effect=api):
                    with archive.read_session(self.root):
                        with self.assertRaises(archive.ArchiveError):
                            archive.read_bytes(self.root, first)
                self.assertEqual(archive._CACHE_SIZE, 0)
                self.assertEqual(self.local_files(), before)
                self.assertFalse((self.root / first).exists())

    def test_cold_snapshot_tree_path_type_mode_oid_and_size_are_verified(self):
        first = next(iter(self.fixture.raws))
        before = self.local_files()
        for corruption in ('missing_path', 'type', 'mode', 'oid', 'size'):
            with self.subTest(corruption=corruption):
                self.clear_raw_cache()
                api = FakeGitHubApi(self.fixture)
                entry = next(entry for entry in api.entries[self.fixture.snapshot] if entry['path'] == first)
                if corruption == 'missing_path':
                    api.entries[self.fixture.snapshot].remove(entry)
                else:
                    entry.update({'type': {'type': 'tree'}, 'mode': {'mode': '120000'},
                                  'oid': {'sha': '9' * 40}, 'size': {'size': True}}[corruption])
                with patch.object(archive, '_run', side_effect=api):
                    with archive.read_session(self.root):
                        with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID/size'):
                            archive.read_bytes(self.root, first)
                self.assertEqual(api.counts['payload_blob'], 0)
                self.assertEqual(archive._CACHE_SIZE, 0)
                self.assertEqual(self.local_files(), before)
                self.assert_cold()

    def test_truncated_github_tree_is_refused_before_any_payload_read(self):
        first = next(iter(self.fixture.raws))
        before = self.local_files()
        def truncated(root, command, deadline=None, env=None, max_output=archive.MAX_READ_BYTES):
            raw = self.api(root, command, deadline, env, max_output)
            if command[1] == 'api' and command[6].endswith('?recursive=1'):
                body = json.loads(raw)
                body['truncated'] = True
                return canonical(body)
            return raw
        with patch.object(archive, '_run', side_effect=truncated):
            with archive.read_session(self.root):
                with self.assertRaisesRegex(archive.ArchiveError, 'truncation'):
                    archive.read_bytes(self.root, first)
        self.assertEqual(self.api.counts['payload_blob'], 0)
        self.assertEqual(self.api.counts['catalog_blob'], 0)
        self.assertEqual(archive._CACHE_SIZE, 0)
        self.assertEqual(self.local_files(), before)

    def test_shared_deadline_bounds_auth_and_all_api_calls(self):
        deadline = time.monotonic() + 10
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root, deadline=deadline):
                self.read_all()
        self.assertTrue(self.api.calls)
        self.assertTrue(all(call[1] is not None and call[1] <= deadline for call in self.api.calls))
        self.assertEqual(len({call[1] for call in self.api.calls}), 1)

    def test_expired_session_refuses_cold_and_cached_reads(self):
        first, second = list(self.fixture.raws)[:2]
        start = time.monotonic()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root, deadline=start + 10):
                archive.read_bytes(self.root, first)
                counts = self.api.counts.copy()
                with patch.object(archive.time, 'monotonic', return_value=start + 11):
                    for relative in (second, first):
                        with self.subTest(relative=relative):
                            with self.assertRaisesRegex(archive.ArchiveError, 'deadline'):
                                archive.read_bytes(self.root, relative)
        self.assertEqual(self.api.counts, counts)
        self.assert_cold()

    def test_expired_deadline_refuses_without_auth_or_local_changes(self):
        before = self.local_files()
        first = next(iter(self.fixture.raws))
        with patch.object(archive, '_run', side_effect=self.api):
            with self.assertRaisesRegex(archive.ArchiveError, 'deadline'):
                with archive.read_session(self.root, deadline=time.monotonic() - 1):
                    archive.read_bytes(self.root, first)
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.local_files(), before)

    def test_offline_session_retains_all_control_and_live_local_files(self):
        before = self.local_files()
        self.api.offline = True
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                with self.assertRaisesRegex(archive.ArchiveError, 'offline'):
                    archive.read_bytes(self.root, next(iter(self.fixture.raws)))
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_owner_auth_is_once_process_local_and_does_not_switch_account(self):
        with patch.dict(os.environ, {'GH_TOKEN': 'wrong', 'GITHUB_TOKEN': 'wrong2', 'GH_DEBUG': 'api'}):
            with patch.object(archive, '_run', side_effect=self.api):
                with archive.read_session(self.root):
                    self.read_all()
            self.assertEqual(os.environ['GH_TOKEN'], 'wrong')
            self.assertEqual(os.environ['GITHUB_TOKEN'], 'wrong2')
        auth_calls = [call for call in self.api.calls if call[0][1:3] == ['auth', 'token']]
        self.assertEqual(len(auth_calls), 1)
        self.assertNotIn('GH_TOKEN', auth_calls[0][2])
        self.assertNotIn('GITHUB_TOKEN', auth_calls[0][2])
        self.assertTrue(all('switch' not in call[0] for call in self.api.calls))
        api_calls = [call for call in self.api.calls if call[0][1] == 'api']
        self.assertTrue(all(call[2]['GH_TOKEN'] == 'ghp_fixtureReadSessionToken123' for call in api_calls))
        self.assertTrue(all('GH_DEBUG' not in call[2] for call in api_calls))

    def test_tree_metadata_lru_is_bounded_and_keyed_by_commit_hash(self):
        mapping_sizes = [len(canonical({archive.PROGRAMME + '/' + entry['path']: entry
                                       for entry in entries})) for entries in self.api.entries.values()]
        budget = max(mapping_sizes)
        first, second = list(self.fixture.raws)[:2]
        before = self.local_files()
        with patch.object(archive, 'TREE_CACHE_BYTES', budget), patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                for relative in (first, second):
                    archive.read_bytes(self.root, relative)
                    transport = session.transport
                    self.assertLessEqual(transport._tree_bytes, budget)
                    self.assertEqual(transport._tree_bytes,
                                     sum(len(canonical(mapping)) for mapping in transport._trees.values()))
                    self.assertEqual(len(transport._trees), 1)
                    self.assertTrue(set(transport._trees) <= {self.fixture.source, self.fixture.snapshot})
                self.assertEqual(self.api.counts['commit'], 4)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_oversized_metadata_and_raw_payloads_are_not_cached(self):
        first, second = list(self.fixture.raws)[:2]
        before = self.local_files()
        with patch.object(archive, 'TREE_CACHE_BYTES', 1), patch.object(archive, 'CATALOG_CACHE_BYTES', 1), \
                patch.object(archive, 'RAM_CACHE_BYTES', 1), patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                for relative in (first, second):
                    self.assertEqual(archive.read_bytes(self.root, relative), self.fixture.raws[relative])
                    self.assertEqual(session.catalogs, {})
                    self.assertEqual(session.catalog_bytes, 0)
                    self.assertEqual(session.transport._trees, {})
                    self.assertEqual(session.transport._tree_bytes, 0)
                    self.assertEqual(archive._CACHE, {})
                    self.assertEqual(archive._CACHE_SIZE, 0)
        self.assertEqual(self.api.counts['catalog_blob'], 2)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_catalog_lru_budget_evicts_old_catalog_and_rechecks_remote_bytes(self):
        other = ColdCustody(self.root, count=2, start=100, source='1' * 40, snapshot='2' * 40)
        self.merge_indexes(self.fixture, other)
        self.api.add_fixture(other)
        budget = max(len(self.fixture.catalog_raw), len(other.catalog_raw))
        first, second = list(self.fixture.raws)[:2]
        other_first = next(iter(other.raws))
        before = self.local_files()
        with patch.object(archive, 'CATALOG_CACHE_BYTES', budget), patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                for relative in (first, other_first, second):
                    archive.read_bytes(self.root, relative)
                    self.assertLessEqual(session.catalog_bytes, budget)
                    self.assertEqual(session.catalog_bytes, sum(len(raw) for raw in session.catalogs.values()))
                    self.assertEqual(len(session.catalogs), 1)
                self.assertEqual(self.api.counts['catalog_blob'], 3)
                key = next(iter(session.catalogs))
                self.assertIn(self.fixture.snapshot, key)
                self.assertIn(blob_oid(self.fixture.catalog_raw), key)
                self.assertIn(hashlib.sha256(self.fixture.catalog_raw).hexdigest(), key)
        self.assertEqual(session.catalog_bytes, 0)
        self.assertEqual(session.catalogs, {})
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_legitimate_new_index_cannot_reuse_old_path_catalog_commitment(self):
        first = next(iter(self.fixture.raws))
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                archive.read_bytes(self.root, first)
                other = ColdCustody(self.root, count=1, source='1' * 40, snapshot='2' * 40)
                self.api.add_fixture(other)
                self.assertEqual(archive.read_bytes(self.root, first), other.raws[first])
                self.assertEqual(self.api.counts['auth'], 1)
                self.assertEqual(self.api.counts['catalog_blob'], 2)
                self.assertEqual(self.api.counts['payload_blob'], 2)
                self.assertEqual(len(session.catalogs), 2)
        self.assert_cold()

    def test_explicit_provider_cannot_reuse_default_catalog_cache(self):
        first, second = list(self.fixture.raws)[:2]
        explicit = archive.GitHubTransport(self.root)
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                archive.read_bytes(self.root, first)
                default = session.transport
                archive.read_bytes(self.root, second, transport=explicit)
                self.assertEqual(self.api.counts['auth'], 2)
                self.assertEqual(self.api.counts['catalog_blob'], 2)
                self.assertEqual(len(session.catalogs), 1)
                self.assertEqual({key[0] for key in session.catalogs}, {default})
        self.assert_cold()

    def test_corrupt_ram_catalog_is_revalidated_before_cold_payload(self):
        first, second = list(self.fixture.raws)[:2]
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                archive.read_bytes(self.root, first)
                key = next(iter(session.catalogs))
                session.catalogs[key] = b'x' * len(session.catalogs[key])
                payloads = self.api.counts['payload_blob']
                with self.assertRaisesRegex(archive.ArchiveError, 'length/SHA256/Git object'):
                    archive.read_bytes(self.root, second)
        self.assertEqual(self.api.counts['payload_blob'], payloads)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_corrupt_ram_tree_entry_is_refused_before_cold_payload(self):
        first, second = list(self.fixture.raws)[:2]
        before = self.local_files()
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                archive.read_bytes(self.root, first)
                session.transport._trees[self.fixture.snapshot][archive.PROGRAMME + '/' + second]['sha'] = '9' * 40
                payloads = self.api.counts['payload_blob']
                with self.assertRaisesRegex(archive.ArchiveError, 'tree path/OID/size'):
                    archive.read_bytes(self.root, second)
        self.assertEqual(self.api.counts['payload_blob'], payloads)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_raw_cache_budget_evicts_bytes_without_payload_restoration(self):
        first, second = list(self.fixture.raws)[:2]
        budget = max(len(self.fixture.raws[first]), len(self.fixture.raws[second]))
        before = self.local_files()
        with patch.object(archive, 'RAM_CACHE_BYTES', budget), patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root):
                for relative in (first, second, first):
                    archive.read_bytes(self.root, relative)
                    self.assertLessEqual(archive._CACHE_SIZE, budget)
                    self.assertEqual(archive._CACHE_SIZE, sum(len(raw) for raw in archive._CACHE.values()))
                self.assertEqual(self.api.counts['auth'], 1)
                self.assertEqual(self.api.counts['catalog_blob'], 1)
                self.assertEqual(self.api.counts['payload_blob'], 3)
        self.assertEqual(self.local_files(), before)
        self.assert_cold()

    def test_nested_deadline_tightens_existing_provider_and_restores_outer_bound(self):
        first, second, third = list(self.fixture.raws)[:3]
        start = time.monotonic()
        outer_deadline = start + 20
        inner_deadline = start + 5
        with patch.object(archive.time, 'monotonic', return_value=start), \
                patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root, deadline=outer_deadline) as outer:
                archive.read_bytes(self.root, first)
                provider = outer.transport
                with archive.read_session(self.root, deadline=inner_deadline) as inner:
                    self.assertIs(inner, outer)
                    archive.read_bytes(self.root, second)
                    self.assertIs(inner.transport, provider)
                    self.assertEqual(inner.deadline, inner_deadline)
                    self.assertEqual(inner.transport.deadline, inner_deadline)
                self.assertEqual(outer.deadline, outer_deadline)
                with patch.object(archive.time, 'monotonic', return_value=start + 6):
                    archive.read_bytes(self.root, third)
                    self.assertIsNot(outer.transport, provider)
                    self.assertLessEqual(outer.transport.deadline, outer_deadline)
                    self.assertEqual({key[0] for key in outer.catalogs}, {outer.transport})
        self.assertEqual(self.api.counts['auth'], 2)
        self.assert_cold()

    def test_nested_longer_deadline_cannot_extend_outer_operation(self):
        first = next(iter(self.fixture.raws))
        start = time.monotonic()
        with patch.object(archive.time, 'monotonic', return_value=start), \
                patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root, deadline=start + 5) as outer:
                with archive.read_session(self.root, deadline=start + 50) as inner:
                    self.assertIs(inner, outer)
                    self.assertEqual(inner.deadline, start + 5)
                    archive.read_bytes(self.root, first)
                    self.assertLessEqual(inner.transport.deadline, start + 5)
                    with patch.object(archive.time, 'monotonic', return_value=start + 6):
                        with self.assertRaises(archive.ReadDeadlineExceeded):
                            archive.read_bytes(self.root, first)
        self.assertEqual(self.api.counts['auth'], 1)
        self.assert_cold()

    def test_invalid_deadline_is_refused_without_authentication(self):
        for invalid in (float('nan'), float('inf'), 'later', True):
            with self.subTest(deadline=invalid):
                with patch.object(archive, '_run', side_effect=self.api):
                    with self.assertRaises(archive.ArchiveError):
                        with archive.read_session(self.root, deadline=invalid):
                            pass
        self.assertEqual(self.api.calls, [])

    def test_falsey_explicit_transport_is_still_honored(self):
        class FalseyTransport(archive.GitHubTransport):
            def __bool__(self):
                return False
        explicit = FalseyTransport(self.root)
        first = next(iter(self.fixture.raws))
        with patch.object(archive, '_run', side_effect=self.api):
            with archive.read_session(self.root) as session:
                with patch.object(explicit, 'fetch_blob', wraps=explicit.fetch_blob) as fetched:
                    archive.read_bytes(self.root, first, transport=explicit)
                self.assertEqual(fetched.call_count, 2)
                self.assertIsNone(session._transport)
        self.assert_cold()


if __name__ == '__main__':
    unittest.main()
