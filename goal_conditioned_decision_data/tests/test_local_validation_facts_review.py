"""Independent boolean-fact memo and original transaction-order review.
No live writes, payload labels, real network, or experiment runs.
"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import rendered, render_controls, remote_archive, worker


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(raw_json(value)[:-1])


def old_recover(db, root):
    """Exact pre-optimization loop, before/after source commitments unchanged."""
    root = Path(root).resolve()
    rendered.ensure_queue(db)
    for (relative,) in db.execute('SELECT receipt FROM rendered_jobs WHERE status="done"').fetchall():
        rendered.validate_receipt(root, relative)
    for relative in rendered._archive_call('iter_paths', root, 'manifests/render-batch-*.json'):
        receipt, _ = rendered.validate_receipt(root, relative)
        db.execute('INSERT OR REPLACE INTO rendered_jobs VALUES (?,?,?,?,?,?,?)',
                   (receipt['source_manifest'], receipt['source_sha256'], receipt['spec_sha256'],
                    receipt['seed'], receipt['split'], 'done', relative))
    db.commit()


class TraceConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upserts = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, sql, parameters=(), /):
        if sql.startswith('INSERT OR REPLACE INTO rendered_jobs'):
            self.upserts.append(tuple(parameters))
        return super().execute(sql, parameters)

    def commit(self):
        self.commits += 1
        return super().commit()

    def rollback(self):
        self.rollbacks += 1
        return super().rollback()

    def reset_trace(self):
        self.upserts.clear()
        self.commits = self.rollbacks = 0


class RecoveryFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gcdl-recovery-once-review-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'goal_conditioned_decision_data'
        self.root.mkdir()
        self.projections = {}
        self.payload_metadata = {}
        self.discovery = []
        self.after_discovery = None
        self.discoveries = 0
        self.validation_calls = []
        self.database_number = 0
        self.a = self.cohort(113)
        self.b = self.cohort(114)
        self.db, self.observer = self.database()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(rendered, '_archive_call', side_effect=self.archive_call))
        stack.enter_context(patch.object(render_controls, 'try_projection', side_effect=self.project))
        original_validate = rendered.validate_receipt

        def checked_validate(root, relative, **kwargs):
            self.validation_calls.append(relative)
            return original_validate(root, relative, **kwargs)

        stack.enter_context(patch.object(rendered, 'validate_receipt', side_effect=checked_validate))
        forbidden = AssertionError('metadata-only recovery fixture forbids raw/network access')
        for owner, name in ((remote_archive, '_run'), (remote_archive, 'GitHubTransport'),
                            (remote_archive, 'read_bytes'), (subprocess, 'run'),
                            (socket, 'create_connection')):
            stack.enter_context(patch.object(owner, name, side_effect=forbidden))

    def database(self):
        self.database_number += 1
        path = self.root / ('queue-fixture-%d.sqlite' % self.database_number)
        db = sqlite3.connect(path, factory=TraceConnection)
        observer = sqlite3.connect(path)
        self.addCleanup(db.close)
        self.addCleanup(observer.close)
        rendered.ensure_queue(db)
        db.execute('CREATE TABLE events (at REAL,kind TEXT,detail TEXT)')
        db.commit()
        db.reset_trace()
        return db, observer

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else raw_json(value))

    def cohort(self, seed):
        code = {'manual-metadata-fixture.py': '1' * 64}
        protected = [['manual-id-%d' % seed, 'manual-fingerprint-%d' % seed,
                      'manual-parent-%d' % seed, 'manual-template', 'dev']]
        source = 'data/accepted/dev-%d.jsonl' % seed
        source_sha = identity(['opaque-source-commitment', seed])
        self.payload_metadata[source] = {'sha256': source_sha, 'bytes': 99}
        batch_path = 'manifests/batch-dev-%d.json' % seed
        batch = {'seed': seed, 'split': 'dev', 'shard': source, 'sha256': source_sha,
                 'bytes': 99, 'accepted': 1, 'audit': {'verdict': 'PASS'},
                 'quarantined': 0, 'code_hashes': code, 'protected_index': protected}
        self.write(batch_path, batch)
        batch_sha = sha(raw_json(batch))
        specification = {'export_version': 'manual-export-v1', 'rendering_version': 'manual-render-v1',
                         'views': ['listwise'], 'variant_budget': 3, 'code_sha256': identity(code)}
        spec_sha = identity(specification)
        dataset_id = 'gcdl_rendered_' + identity(['dataset', seed])[:32]
        header = 'data/rendered/' + dataset_id + '/manifest.json'
        shard = 'data/rendered/' + dataset_id + '/shard-00000.jsonl'
        shard_sha = identity(['opaque-derived-shard-commitment', seed])
        self.payload_metadata[shard] = {'sha256': shard_sha, 'bytes': 155}
        manifest = {'manifest_path': header, 'dataset_id': dataset_id, 'view': 'listwise',
                    'split': 'dev', 'source_records_examined': 1, 'eligible_source_records': 1,
                    'code_hashes': code, 'variant_budget': 3, 'export_version': 'manual-export-v1',
                    'training_launched': False, 'rendering_version': 'manual-render-v1',
                    'sources': [{'path': source, 'sha256': source_sha,
                                 'source_batch_manifest': batch_path,
                                 'source_batch_manifest_sha256': batch_sha,
                                 'source_code_hashes': code, 'protected_index': protected}],
                    'shards': [{'path': shard, 'sha256': shard_sha, 'bytes': 155, 'rows': 3}],
                    'rendered_rows': 3}
        header_sha = identity(['attested-header-commitment', seed])
        self.payload_metadata[header] = {'sha256': header_sha, 'bytes': 1111}
        receipt = {'version': 'gcdl-rendered-batch-1', 'seed': seed, 'split': 'dev',
                   'source_manifest': batch_path, 'source_sha256': batch_sha,
                   'source_shard_sha256': source_sha, 'specification': specification,
                   'spec_sha256': spec_sha, 'code_hashes': code,
                   'datasets': [{'view': 'listwise', 'dataset_id': dataset_id,
                                 'manifest_path': header, 'manifest_sha256': header_sha,
                                 'rendered_rows': 3, 'eligible_source_records': 1}]}
        relative = 'manifests/render-batch-dev-%d-%s.json' % (seed, spec_sha[:16])
        self.write(relative, receipt)
        self.projections[relative] = [manifest]
        self.discovery.append(relative)
        return {'path': relative, 'receipt': receipt, 'manifest': manifest}

    def archive_call(self, operation, root, relative):
        self.assertEqual(Path(root).resolve(), self.root)
        if operation == 'iter_paths':
            self.assertEqual(relative, 'manifests/render-batch-*.json')
            self.discoveries += 1
            result = list(self.discovery)
            if self.after_discovery is not None:
                self.after_discovery()
            return result
        if operation == 'read_text':
            self.assertTrue(isinstance(relative, str) and relative.startswith('manifests/'))
            self.assertNotIn('..', Path(relative).parts)
            return (self.root / relative).read_text()
        if operation == 'payload_info':
            if relative.startswith('manifests/'):
                raw = (self.root / relative).read_bytes()
                return {'sha256': sha(raw), 'bytes': len(raw)}
            self.assertFalse(relative.startswith('data/sealed/'))
            return deepcopy(self.payload_metadata[relative])
        self.fail('unexpected archive operation: ' + operation)

    def project(self, root, relative, receipt, batch):
        self.assertEqual(Path(root).resolve(), self.root)
        return deepcopy(self.projections[relative])

    @staticmethod
    def expected(cohort, status='done', pointer=None):
        receipt = cohort['receipt']
        return (receipt['source_manifest'], receipt['source_sha256'], receipt['spec_sha256'],
                receipt['seed'], receipt['split'], status,
                cohort['path'] if pointer is None else pointer)

    def insert(self, db, cohort, status='done', pointer=None):
        db.execute('INSERT INTO rendered_jobs VALUES (?,?,?,?,?,?,?)',
                   self.expected(cohort, status, pointer))
        db.commit()
        db.reset_trace()

    @staticmethod
    def rows(db):
        return db.execute('SELECT * FROM rendered_jobs ORDER BY source_manifest,source_sha256,spec_sha256').fetchall()

    def corrupt(self, cohort, key='source_sha256'):
        receipt = deepcopy(cohort['receipt'])
        receipt[key] = 'f' * 64
        self.write(cohort['path'], receipt)



class ControlFixture:
    """Independent metadata-only custody; opaque bytes exist only in RAM."""
    def __init__(self, root):
        self.root = root
        self.relative = 'data/accepted/opaque-fixture.txt'
        self.payload = b'opaque fixture bytes, no labels\n'
        self.source = 'a' * 40
        self.snapshot = 'b' * 40
        meta = {'commit': self.source, 'git_blob_sha1': remote_archive.git_blob_sha1(self.payload),
                'sha256': sha(self.payload), 'bytes': len(self.payload)}
        self.catalog = {'version': 'gcdl-archive-catalog-1', 'repository': remote_archive.REPOSITORY,
                        'programme': remote_archive.PROGRAMME, 'files': {self.relative: meta}}
        self.catalog['catalog_id'] = identity(self.catalog)
        self.catalog_path = remote_archive.catalog_path(self.catalog['catalog_id'])
        self.catalog_raw = raw_json(self.catalog)
        descriptor = {'catalog_commit': self.snapshot, 'archive_ref': remote_archive.archive_ref(self.snapshot),
                      'catalog_path': self.catalog_path, 'catalog_sha256': sha(self.catalog_raw),
                      'catalog_git_blob_sha1': remote_archive.git_blob_sha1(self.catalog_raw)}
        self.receipt = {'version': 'gcdl-archive-custody-1', 'status': 'remote_verified',
                        'repository': remote_archive.REPOSITORY, 'programme': remote_archive.PROGRAMME,
                        'catalog_id': self.catalog['catalog_id'], **descriptor,
                        'files': deepcopy(self.catalog['files']), 'local_identities': {}, 'local_ancestry': {}}
        self.receipt['receipt_id'] = identity(self.receipt)
        self.receipt_path = 'manifests/remote_custody/custody-' + self.receipt['receipt_id'] + '.json'
        self.receipt_raw = raw_json(self.receipt)
        self.index = {'version': 'gcdl-remote-archive-1', 'repository': remote_archive.REPOSITORY,
                      'programme': remote_archive.PROGRAMME,
                      'files': {self.relative: {**meta, 'catalog_id': self.catalog['catalog_id'],
                                               'receipt_id': self.receipt['receipt_id']}},
                      'catalogs': {self.catalog['catalog_id']: descriptor},
                      'receipts': {self.receipt['receipt_id']: {'path': self.receipt_path,
                                                              'sha256': sha(self.receipt_raw)}}}
        self.write(self.catalog_path, self.catalog_raw)
        self.write(self.receipt_path, self.receipt_raw)
        self.install_index(self.index)

    def write(self, relative, raw):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)

    def install_index(self, value):
        index = deepcopy(value)
        index.pop('digest', None)
        index['digest'] = identity(index)
        raw = raw_json(index)
        self.write('manifests/remote_indexes/index-' + index['digest'] + '.json', raw)
        self.write(remote_archive.INDEX_PATH, raw)
        self.current_index = index
        return index


class FixtureTransport:
    def __init__(self, fixture):
        self.fixture = fixture
        self.blob_calls = []
        self.reference_calls = []
        self.corrupt = False
        self.offline = False

    def fetch_ref(self, reference):
        self.reference_calls.append(reference)
        if self.offline:
            raise remote_archive.ArchiveError('fixture offline')
        assert reference == remote_archive.archive_ref(self.fixture.snapshot)
        return self.fixture.snapshot

    def fetch_tree(self, commit):
        f = self.fixture
        assert commit in (f.source, f.snapshot)
        entries = {remote_archive.PROGRAMME + '/' + f.relative:
                   {'type': 'blob', 'mode': '100644', 'sha': f.catalog['files'][f.relative]['git_blob_sha1'],
                    'size': len(f.payload)}}
        if commit == f.snapshot:
            entries[remote_archive.PROGRAMME + '/' + f.catalog_path] = {
                'type': 'blob', 'mode': '100644', 'sha': remote_archive.git_blob_sha1(f.catalog_raw),
                'size': len(f.catalog_raw)}
        return entries

    def fetch_blob(self, oid, max_bytes):
        self.blob_calls.append((oid, max_bytes))
        if self.offline:
            raise remote_archive.ArchiveError('fixture offline')
        f = self.fixture
        if oid == remote_archive.git_blob_sha1(f.catalog_raw):
            result = f.catalog_raw
        else:
            assert oid == f.catalog['files'][f.relative]['git_blob_sha1']
            result = b'changed opaque payload\n' if self.corrupt else f.payload
        assert len(result) <= max_bytes or self.corrupt
        return result


class LocalStructuralFactsReview(unittest.TestCase):
    def setUp(self):
        import os
        self.os = os
        temporary = tempfile.TemporaryDirectory(prefix='gcdl-local-facts-review-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'goal_conditioned_decision_data'
        self.root.mkdir()
        self.fixture = ControlFixture(self.root)
        stack = ExitStack()
        self.addCleanup(stack.close)
        forbidden = AssertionError('independent local metadata fixture forbids real network/process')
        for owner, name in ((remote_archive, '_run'), (remote_archive, 'GitHubTransport'),
                            (subprocess, 'run'), (socket, 'create_connection')):
            stack.enter_context(patch.object(owner, name, side_effect=forbidden))

    def value(self, raw=None, validator=None, root=None):
        return remote_archive._local_control_value(root or self.root,
                   self.fixture.catalog_raw if raw is None else raw,
                   remote_archive._catalog if validator is None else validator)

    def assert_counter(self, session):
        self.assertEqual(session.local_control_bytes,
                         sum(len(key[2]) for key in session.local_control_facts))
        self.assertLessEqual(session.local_control_bytes, remote_archive.CATALOG_CACHE_BYTES)
        self.assertLessEqual(len(session.local_control_facts), remote_archive.MAX_BATCH_FILES)
        self.assertTrue(all(value is True for value in session.local_control_facts.values()))

    def test_same_raw_reuses_only_two_structural_facts_and_all_original_reads(self):
        original_local, original_json = remote_archive._local, remote_archive._json
        with patch.object(remote_archive, '_local', wraps=original_local) as reads, \
             patch.object(remote_archive, '_json', wraps=original_json) as parser, \
             patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as cat, \
             patch.object(remote_archive, '_receipt', wraps=remote_archive._receipt) as custody:
            with remote_archive.read_session(self.root) as session:
                first = remote_archive.load_index(self.root)
                second = remote_archive.load_index(self.root)
                self.assertEqual(first, second)
                self.assertIsNot(first, second)
                self.assertEqual(cat.call_count, 1)
                self.assertEqual(custody.call_count, 1)
                self.assertEqual(parser.call_count, 6)
                expected = [remote_archive.INDEX_PATH,
                            'manifests/remote_indexes/index-' + first['digest'] + '.json',
                            self.fixture.catalog_path, self.fixture.receipt_path]
                self.assertEqual([call.args[1] for call in reads.call_args_list], expected * 2)
                self.assertEqual(len(session.local_control_facts), 2)
                self.assert_counter(session)
        self.assertEqual(session.local_control_facts, {})
        self.assertEqual(session.local_control_bytes, 0)

    def test_returned_index_and_nested_control_mutation_cannot_poison_facts(self):
        with remote_archive.read_session(self.root):
            first = remote_archive.load_index(self.root)
            first['files'][self.fixture.relative]['sha256'] = '0' * 64
            first['catalogs'].clear()
            second = remote_archive.load_index(self.root)
            self.assertEqual(second, self.fixture.current_index)
            catalog = self.value()
            catalog['files'].clear()
            self.assertEqual(self.value(), self.fixture.catalog)
            receipt = self.value(self.fixture.receipt_raw, remote_archive._receipt)
            receipt['local_ancestry']['forged'] = []
            self.assertEqual(self.value(self.fixture.receipt_raw, remote_archive._receipt), self.fixture.receipt)

    def test_whitespace_difference_requires_distinct_full_validator_success(self):
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                self.value()
                self.value(b' ' + self.fixture.catalog_raw + b'\n')
                self.value()
                self.assertEqual(validator.call_count, 2)
                self.assertEqual(len(session.local_control_facts), 2)
                self.assert_counter(session)

    def test_catalog_fact_never_substitutes_for_receipt_validator(self):
        with patch.object(remote_archive, '_receipt', wraps=remote_archive._receipt) as custody:
            with remote_archive.read_session(self.root) as session:
                self.value()
                for _ in range(2):
                    with self.assertRaises(remote_archive.ArchiveError):
                        self.value(validator=custody)
                self.assertEqual(custody.call_count, 2)
                self.assertEqual(len(session.local_control_facts), 1)
                self.assert_counter(session)

    def test_replaced_validator_with_same_name_does_not_hit_old_fact(self):
        original = remote_archive._catalog
        def replacement(value):
            return original(value)
        replacement.__name__ = original.__name__
        with patch.object(remote_archive, '_catalog', wraps=original) as first:
            with remote_archive.read_session(self.root) as session:
                self.value()
                with patch.object(remote_archive, '_catalog', wraps=replacement) as second:
                    self.value()
                    self.value()
                    self.assertEqual(second.call_count, 1)
                self.assertEqual(first.call_count, 1)
                self.assertEqual(len(session.local_control_facts), 2)

    def test_replaced_parser_with_same_name_requires_revalidation_and_each_hit_parses(self):
        original = remote_archive._json
        def replacement(raw):
            return original(raw)
        replacement.__name__ = original.__name__
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root):
                self.value()
                with patch.object(remote_archive, '_json', wraps=replacement) as parser:
                    self.value()
                    self.value()
                    self.assertEqual(parser.call_count, 2)
                self.assertEqual(validator.call_count, 2)

    def test_replacement_parser_cannot_reuse_success_for_a_malformed_new_object(self):
        original = remote_archive._json

        def malformed_parser(raw):
            value = original(raw)
            value['version'] = 'unqualified-catalog'
            return value

        malformed_parser.__name__ = original.__name__
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                self.value()
                with patch.object(remote_archive, '_json', malformed_parser):
                    for _ in range(2):
                        with self.assertRaisesRegex(remote_archive.ArchiveError, 'scope/shape'):
                            self.value()
                self.assertEqual(validator.call_count, 3)
                self.assertEqual(len(session.local_control_facts), 1)
                self.assertEqual(self.value(), self.fixture.catalog)
                self.assertEqual(validator.call_count, 3)
                self.assert_counter(session)

    def test_failed_structure_and_invalid_json_are_never_memoized(self):
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                for _ in range(2):
                    with self.assertRaises(remote_archive.ArchiveError):
                        self.value(b'{}\n')
                self.assertEqual(validator.call_count, 2)
                for _ in range(2):
                    with self.assertRaises(remote_archive.ArchiveError):
                        self.value(b'{invalid')
                self.assertEqual(validator.call_count, 2)
                self.assertEqual(session.local_control_facts, {})
                self.assertEqual(session.local_control_bytes, 0)

    def test_outside_session_always_validates(self):
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            self.value()
            self.value()
            self.assertEqual(validator.call_count, 2)

    def test_normal_exceptional_and_new_sessions_do_not_share_facts(self):
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with self.assertRaisesRegex(RuntimeError, 'fixture exit'):
                with remote_archive.read_session(self.root) as first:
                    self.value()
                    raise RuntimeError('fixture exit')
            self.assertEqual(first.local_control_facts, {})
            self.assertEqual(first.local_control_bytes, 0)
            with remote_archive.read_session(self.root) as second:
                self.assertEqual(second.local_control_facts, {})
                self.value()
            self.assertEqual(validator.call_count, 2)
            self.assertEqual(second.local_control_bytes, 0)

    def test_nested_same_root_preserves_facts_without_extending_deadline(self):
        import time
        deadline = time.monotonic() + 60
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root, deadline=deadline) as outer:
                self.value()
                with remote_archive.read_session(self.root, deadline=deadline + 99) as inner:
                    self.assertIs(inner, outer)
                    self.assertEqual(inner.deadline, deadline)
                    self.value()
                self.assertEqual(outer.deadline, deadline)
                self.assertEqual(validator.call_count, 1)
                self.assert_counter(outer)

    def test_expired_earlier_nested_deadline_restores_outer_facts_and_deadline(self):
        with patch.object(remote_archive.time, 'monotonic', return_value=100), \
             patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root, deadline=300) as outer:
                self.value()
                with self.assertRaises(remote_archive.ReadDeadlineExceeded):
                    with remote_archive.read_session(self.root, deadline=200) as inner:
                        self.assertIs(inner, outer)
                        self.assertEqual(inner.deadline, 200)
                        with patch.object(remote_archive.time, 'monotonic', return_value=201):
                            self.value()
                self.assertEqual(outer.deadline, 300)
                self.assertEqual(len(outer.local_control_facts), 1)
                self.assertEqual(self.value(), self.fixture.catalog)
                self.assertEqual(validator.call_count, 1)
                self.assert_counter(outer)

    def test_foreign_root_facts_and_nested_session_refuse(self):
        other = self.root.parent / 'foreign' / 'goal_conditioned_decision_data'
        other.mkdir(parents=True)
        with remote_archive.read_session(self.root):
            self.value()
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'root mismatch'):
                self.value(root=other)
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'root mismatch'):
                with remote_archive.read_session(other):
                    self.fail('foreign session admitted')
        with remote_archive.read_session(other) as session:
            self.assertEqual(session.local_control_facts, {})

    def test_warm_fact_cannot_bypass_expired_deadline_or_extend_typed_error(self):
        import time
        with remote_archive.read_session(self.root) as session:
            self.value()
            session.deadline = time.monotonic() - 1
            with self.assertRaises(remote_archive.ReadDeadlineExceeded):
                self.value()
            with patch.object(remote_archive, '_local', side_effect=AssertionError('expired read admitted')):
                with self.assertRaises(remote_archive.ReadDeadlineExceeded):
                    remote_archive.load_index(self.root)

    def test_byte_bound_lru_eviction_and_hits_preserve_exact_accounting(self):
        budget = max(len(self.fixture.catalog_raw), len(self.fixture.receipt_raw))
        with patch.object(remote_archive, 'CATALOG_CACHE_BYTES', budget), \
             patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                self.value()
                self.value()
                self.assertEqual(validator.call_count, 1)
                self.value(self.fixture.receipt_raw, remote_archive._receipt)
                self.assertEqual(len(session.local_control_facts), 1)
                self.value()
                self.assertEqual(validator.call_count, 2)
                self.assert_counter(session)

    def test_oversized_successes_validate_every_time_without_retention(self):
        with patch.object(remote_archive, 'CATALOG_CACHE_BYTES', len(self.fixture.catalog_raw) - 1), \
             patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                self.value()
                self.value()
                self.assertEqual(validator.call_count, 2)
                self.assertEqual(session.local_control_facts, {})
                self.assertEqual(session.local_control_bytes, 0)

    def test_entry_cap_is_retention_only_and_evicted_bytes_revalidate(self):
        with patch.object(remote_archive, 'MAX_BATCH_FILES', 1), \
             patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                self.value()
                self.value(b' ' + self.fixture.catalog_raw)
                self.value()
                self.assertEqual(validator.call_count, 3)
                self.assertEqual(len(session.local_control_facts), 1)
                self.assert_counter(session)

    def test_actual_128_entry_bound_promotes_hits_and_revalidates_evicted_success(self):
        self.assertEqual(remote_archive.MAX_BATCH_FILES, 128)
        bodies = [self.fixture.catalog_raw + b' ' * number for number in range(129)]
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                for body in bodies[:128]:
                    self.assertEqual(self.value(body), self.fixture.catalog)
                self.value(bodies[0])  # Keep the oldest successful body alive.
                self.value(bodies[128])
                self.assertEqual(validator.call_count, 129)
                self.assertEqual(len(session.local_control_facts), 128)
                retained = {key[2] for key in session.local_control_facts}
                self.assertIn(bodies[0], retained)
                self.assertNotIn(bodies[1], retained)
                self.value(bodies[0])
                self.assertEqual(validator.call_count, 129)
                self.value(bodies[1])
                self.assertEqual(validator.call_count, 130)
                self.assert_counter(session)

    def test_actual_4_mib_bound_does_not_cap_valid_body_semantics(self):
        self.assertEqual(remote_archive.CATALOG_CACHE_BYTES, 4 * 1024 ** 2)
        padding = b' ' * (2 * 1024 ** 2)
        first = self.fixture.catalog_raw + padding
        second = b' ' + first
        oversized = self.fixture.catalog_raw + b' ' * (
            remote_archive.CATALOG_CACHE_BYTES + 1 - len(self.fixture.catalog_raw))
        exactly_bounded = oversized[:-1]
        with patch.object(remote_archive, '_catalog', wraps=remote_archive._catalog) as validator:
            with remote_archive.read_session(self.root) as session:
                self.assertEqual(self.value(first), self.fixture.catalog)
                self.value(first)
                self.assertEqual(validator.call_count, 1)
                self.assertEqual(self.value(second), self.fixture.catalog)
                self.assertEqual(len(session.local_control_facts), 1)
                self.assertEqual(session.local_control_bytes, len(second))
                for _ in range(2):
                    self.assertEqual(self.value(oversized), self.fixture.catalog)
                self.assertEqual(validator.call_count, 4)
                self.assertEqual(session.local_control_bytes, len(second))
                self.value(first)
                self.assertEqual(validator.call_count, 5)
                self.assertEqual(self.value(exactly_bounded), self.fixture.catalog)
                self.value(exactly_bounded)
                self.assertEqual(validator.call_count, 6)
                self.assertEqual(len(session.local_control_facts), 1)
                self.assertEqual(session.local_control_bytes, 4 * 1024 ** 2)
                self.assert_counter(session)

    def test_self_rehashed_malformed_controls_still_require_structural_validation(self):
        with remote_archive.read_session(self.root) as session:
            remote_archive.load_index(self.root)
            for kind in ('catalog', 'receipt'):
                with self.subTest(kind=kind):
                    changed = deepcopy(self.fixture.index)
                    if kind == 'catalog':
                        value = deepcopy(self.fixture.catalog)
                        value['unqualified_field'] = True
                        value.pop('catalog_id')
                        value['catalog_id'] = identity(value)
                        body = raw_json(value)
                        old_descriptor = changed['catalogs'].pop(self.fixture.catalog['catalog_id'])
                        descriptor = {**old_descriptor,
                                      'catalog_path': remote_archive.catalog_path(value['catalog_id']),
                                      'catalog_sha256': sha(body),
                                      'catalog_git_blob_sha1': remote_archive.git_blob_sha1(body)}
                        changed['catalogs'][value['catalog_id']] = descriptor
                        self.fixture.write(descriptor['catalog_path'], body)
                        expected = 'scope/shape'
                    else:
                        value = deepcopy(self.fixture.receipt)
                        value['status'] = 'local-only'
                        value.pop('receipt_id')
                        value['receipt_id'] = identity(value)
                        body = raw_json(value)
                        changed['receipts'].clear()
                        relative = 'manifests/remote_custody/custody-' + value['receipt_id'] + '.json'
                        changed['receipts'][value['receipt_id']] = {'path': relative, 'sha256': sha(body)}
                        self.fixture.write(relative, body)
                        expected = 'verified archive custody receipt'
                    self.fixture.install_index(changed)
                    with self.assertRaisesRegex(remote_archive.ArchiveError, expected):
                        remote_archive.load_index(self.root)
                    self.assertEqual(len(session.local_control_facts), 2)
                    self.fixture.install_index(self.fixture.index)
                    self.assertEqual(remote_archive.load_index(self.root), self.fixture.current_index)
                    self.assert_counter(session)

    def test_warm_valid_bodies_cannot_authorize_self_rehashed_pointer_aliases(self):
        with remote_archive.read_session(self.root) as session:
            remote_archive.load_index(self.root)
            for kind in ('catalog', 'receipt'):
                with self.subTest(kind=kind):
                    alias = 'e' * 64
                    changed = deepcopy(self.fixture.index)
                    if kind == 'catalog':
                        descriptor = changed['catalogs'].pop(self.fixture.catalog['catalog_id'])
                        descriptor['catalog_path'] = remote_archive.catalog_path(alias)
                        changed['catalogs'][alias] = descriptor
                        self.fixture.write(descriptor['catalog_path'], self.fixture.catalog_raw)
                    else:
                        changed['receipts'].clear()
                        relative = 'manifests/remote_custody/custody-' + alias + '.json'
                        changed['receipts'][alias] = {'path': relative, 'sha256': sha(self.fixture.receipt_raw)}
                        self.fixture.write(relative, self.fixture.receipt_raw)
                    self.fixture.install_index(changed)
                    label = 'catalog' if kind == 'catalog' else 'custody'
                    with self.assertRaisesRegex(remote_archive.ArchiveError, label + ' pointer identity mismatch'):
                        remote_archive.load_index(self.root)
                    self.assertEqual(len(session.local_control_facts), 2)
                    self.fixture.install_index(self.fixture.index)
                    self.assertEqual(remote_archive.load_index(self.root), self.fixture.current_index)
                    self.assert_counter(session)

    def test_changed_catalog_bytes_reject_at_original_checkpoint_then_restore_succeeds(self):
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            self.fixture.write(self.fixture.catalog_path, b' ' + self.fixture.catalog_raw)
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'catalog commitment'):
                remote_archive.load_index(self.root)
            self.fixture.write(self.fixture.catalog_path, self.fixture.catalog_raw)
            self.assertEqual(remote_archive.load_index(self.root), self.fixture.current_index)

    def test_changed_receipt_bytes_reject_at_original_checkpoint_then_restore_succeeds(self):
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            self.fixture.write(self.fixture.receipt_path, self.fixture.receipt_raw + b'\n')
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'receipt hash'):
                remote_archive.load_index(self.root)
            self.fixture.write(self.fixture.receipt_path, self.fixture.receipt_raw)
            self.assertEqual(remote_archive.load_index(self.root), self.fixture.current_index)

    def test_snapshot_and_git_object_hash_remain_independently_checked(self):
        with remote_archive.read_session(self.root):
            index = remote_archive.load_index(self.root)
            snapshot = 'manifests/remote_indexes/index-' + index['digest'] + '.json'
            original = (self.root / snapshot).read_bytes()
            self.fixture.write(snapshot, original + b' ')
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'immutable snapshot'):
                remote_archive.load_index(self.root)
            self.fixture.write(snapshot, original)
            changed = deepcopy(self.fixture.index)
            changed['catalogs'][self.fixture.catalog['catalog_id']]['catalog_git_blob_sha1'] = 'd' * 40
            self.fixture.install_index(changed)
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'catalog commitment'):
                remote_archive.load_index(self.root)

    def test_catalog_id_orphan_and_index_file_bindings_never_use_structure_as_authority(self):
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            changed = deepcopy(self.fixture.index)
            changed['catalogs'][self.fixture.catalog['catalog_id']]['catalog_commit'] = 'c' * 40
            changed['catalogs'][self.fixture.catalog['catalog_id']]['archive_ref'] = remote_archive.archive_ref('c' * 40)
            self.fixture.install_index(changed)
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'orphan or unbound'):
                remote_archive.load_index(self.root)
            changed = deepcopy(self.fixture.index)
            changed['catalogs'].clear()
            self.fixture.install_index(changed)
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'orphan or unbound'):
                remote_archive.load_index(self.root)
            changed = deepcopy(self.fixture.index)
            changed['files'][self.fixture.relative]['sha256'] = 'f' * 64
            self.fixture.install_index(changed)
            with self.assertRaisesRegex(remote_archive.ArchiveError, 'payload/custody'):
                remote_archive.load_index(self.root)

    def test_identical_warm_bytes_do_not_bypass_symlink_hardlink_or_directory_refusal(self):
        path = self.root / self.fixture.catalog_path
        saved = self.root / 'saved-catalog'
        saved.write_bytes(self.fixture.catalog_raw)
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            for kind in ('symlink', 'hardlink', 'directory'):
                with self.subTest(kind=kind):
                    path.unlink()
                    if kind == 'symlink':
                        path.symlink_to(saved)
                    elif kind == 'hardlink':
                        self.os.link(saved, path)
                    else:
                        path.mkdir()
                    with self.assertRaises(remote_archive.ArchiveError):
                        remote_archive.load_index(self.root)
                    if kind == 'directory':
                        path.rmdir()
                    else:
                        path.unlink()
                    path.write_bytes(self.fixture.catalog_raw)
            self.assertEqual(remote_archive.load_index(self.root), self.fixture.current_index)

    def test_inflight_ancestry_and_inode_replacement_is_still_rejected(self):
        path = self.root / self.fixture.catalog_path
        original_inode = path.stat().st_ino
        original_read = self.os.read
        switched = []
        def replace_after_read(fd, length):
            raw = original_read(fd, length)
            if raw and not switched and self.os.fstat(fd).st_ino == original_inode:
                switched.append(True)
                directory = path.parent
                directory.rename(directory.with_name('retired-catalogs'))
                directory.mkdir()
                path.write_bytes(self.fixture.catalog_raw)
            return raw
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            with patch.object(remote_archive.os, 'read', side_effect=replace_after_read):
                with self.assertRaisesRegex(remote_archive.ArchiveError, 'identity or ancestry'):
                    remote_archive.load_index(self.root)
            self.assertTrue(switched)

    def test_warm_structure_never_satisfies_public_full_raw_verification_or_eviction(self):
        provider = FixtureTransport(self.fixture)
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            receipt = remote_archive.verify_archive(self.root, self.fixture.catalog, self.fixture.snapshot,
                                                     transport=provider)
            self.assertEqual(provider.blob_calls,
                [(remote_archive.git_blob_sha1(self.fixture.catalog_raw), len(self.fixture.catalog_raw)),
                 (self.fixture.catalog['files'][self.fixture.relative]['git_blob_sha1'], len(self.fixture.payload))])
            provider.blob_calls.clear()
            result = remote_archive.evict_verified(self.root, receipt, transport=provider)
            self.assertEqual(result['deleted'], [])
            self.assertEqual(result['deleted_bytes'], 0)
            self.assertEqual(provider.blob_calls,
                [(remote_archive.git_blob_sha1(self.fixture.catalog_raw), len(self.fixture.catalog_raw)),
                 (self.fixture.catalog['files'][self.fixture.relative]['git_blob_sha1'], len(self.fixture.payload))])
            self.assertFalse((self.root / self.fixture.relative).exists())

    def test_corrupt_or_offline_raw_remote_proof_refuses_despite_warm_structural_facts(self):
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            before = (self.root / remote_archive.INDEX_PATH).read_bytes()
            for kind in ('corrupt', 'offline'):
                provider = FixtureTransport(self.fixture)
                setattr(provider, kind, True)
                with self.assertRaises(remote_archive.ArchiveError):
                    remote_archive.verify_archive(self.root, self.fixture.catalog, self.fixture.snapshot,
                                                  transport=provider)
                with self.assertRaises(remote_archive.ArchiveError):
                    remote_archive.evict_verified(self.root, self.fixture.receipt, transport=provider)
                self.assertEqual((self.root / remote_archive.INDEX_PATH).read_bytes(), before)
                self.assertFalse((self.root / self.fixture.relative).exists())


class RestoredFaultOrderReview(RecoveryFixture):
    def test_recover_file_matches_approved_baseline_exact_bytes(self):
        self.assertEqual(sha(Path(rendered.__file__).read_bytes()),
                         'ae1e322fa40f7917ebb0f15eb01e2096ba5777c4174dfbb1c3b7f29ae5239014')

    def compare_mixed(self, upsert_failure=False):
        c = self.cohort(115)
        old_db, old_observer = self.database()
        for db in (old_db, self.db):
            self.insert(db, self.a, status='pending')
            self.insert(db, c)
            if upsert_failure:
                db.execute("CREATE TRIGGER block_b BEFORE INSERT ON rendered_jobs WHEN "
                           "new.source_manifest='manifests/batch-dev-114.json' BEGIN "
                           "SELECT RAISE(ABORT,'fresh B upsert failure'); END")
                db.commit(); db.reset_trace()
        self.corrupt(c)
        if not upsert_failure:
            self.corrupt(self.b)
        expected = self.rows(old_observer)
        fixture = ControlFixture(self.root)
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            for recover, db, observer in ((old_recover, old_db, old_observer),
                                          (rendered.recover, self.db, self.observer)):
                self.validation_calls.clear()
                with self.assertRaisesRegex(ValueError, 'source/specification'):
                    recover(db, self.root)
                self.assertEqual(self.validation_calls, [c['path']])
                self.assertEqual(db.upserts, [])
                self.assertEqual(self.rows(db), expected)
                worker.event(db, 'error', {'fixture': 'restored-mixed-fault'})
                self.assertEqual(self.rows(observer), expected)

    def test_mixed_new_corrupt_before_previous_done_corrupt_matches_old_durable_queue(self):
        self.compare_mixed()

    def test_mixed_upsert_failure_before_previous_done_corrupt_matches_old_durable_queue(self):
        self.compare_mixed(upsert_failure=True)

    def test_original_new_fault_retains_authenticated_prefix_and_caller_commit(self):
        self.corrupt(self.b)
        before = self.rows(self.observer)
        fixture = ControlFixture(self.root)
        with remote_archive.read_session(self.root):
            remote_archive.load_index(self.root)
            with self.assertRaises(ValueError):
                rendered.recover(self.db, self.root)
            self.assertEqual(self.rows(self.db), [self.expected(self.a)])
            self.assertEqual(self.rows(self.observer), before)
            worker.event(self.db, 'error', {'fixture': 'original-new-fault-prefix'})
            self.assertEqual(self.rows(self.observer), [self.expected(self.a)])

    def test_setup_commits_prior_caller_work_before_done_preflight_fault(self):
        old_db, old_observer = self.database()
        for db in (old_db, self.db):
            self.insert(db, self.b)
            db.execute('INSERT INTO rendered_jobs VALUES (?,?,?,?,?,?,?)',
                       self.expected(self.a, status='pending'))
        self.corrupt(self.b)
        before = [self.expected(self.b)]
        expected = [self.expected(self.a, status='pending'), self.expected(self.b)]
        for recover, db, observer in ((old_recover, old_db, old_observer),
                                      (rendered.recover, self.db, self.observer)):
            self.assertEqual(self.rows(observer), before)
            self.validation_calls.clear()
            with self.assertRaisesRegex(ValueError, 'source/specification'):
                recover(db, self.root)
            self.assertEqual(self.validation_calls, [self.b['path']])
            self.assertEqual(db.upserts, [])
            self.assertEqual(db.commits, 1)
            self.assertEqual(db.rollbacks, 0)
            self.assertFalse(db.in_transaction)
            self.assertEqual(self.rows(observer), expected)
            worker.event(db, 'error', {'fixture': 'prior-caller-work'})
            self.assertEqual(self.rows(observer), expected)

    def test_actual_upsert_abort_after_valid_done_preflight_keeps_original_prefix(self):
        c = self.cohort(115)
        old_db, old_observer = self.database()
        for db in (old_db, self.db):
            self.insert(db, self.a, status='pending')
            self.insert(db, c)
            db.execute("CREATE TRIGGER abort_fresh_b BEFORE INSERT ON rendered_jobs WHEN "
                       "new.source_manifest='manifests/batch-dev-114.json' BEGIN "
                       "SELECT RAISE(ABORT,'hand-attested fresh B abort'); END")
            db.commit(); db.reset_trace()
        before = [self.expected(self.a, status='pending'), self.expected(c)]
        after = [self.expected(self.a), self.expected(c)]
        for recover, db, observer in ((old_recover, old_db, old_observer),
                                      (rendered.recover, self.db, self.observer)):
            self.validation_calls.clear()
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'fresh B abort'):
                recover(db, self.root)
            self.assertEqual(self.validation_calls, [c['path'], self.a['path'], self.b['path']])
            self.assertEqual(db.upserts, [self.expected(self.a), self.expected(self.b)])
            self.assertEqual(db.commits, 1)
            self.assertEqual(db.rollbacks, 0)
            self.assertTrue(db.in_transaction)
            self.assertEqual(self.rows(observer), before)
            worker.event(db, 'error', {'fixture': 'actual-upsert-abort'})
            self.assertEqual(self.rows(observer), after)


if __name__ == '__main__':
    unittest.main()
