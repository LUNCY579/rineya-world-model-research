"""Independent projection custody tests: tiny owned fixtures and a RAM server.

These tests never contact GitHub, inspect production, read sealed targets,
change implementation files or rerun a scientific experiment.
"""
from collections import Counter
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.auditors.integrity import semantic_fingerprint
from goal_conditioned_decision_data.decision_labels import rendering
from goal_conditioned_decision_data.generators.profiles import resolve_profile
from goal_conditioned_decision_data.pipelines import remote_archive as archive, rendered, render_controls as controls
from goal_conditioned_decision_data.tests.test_surface_review import manual_record


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return sha(raw_json(value)[:-1])


def oid(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


class SnapshotServer:
    def __init__(self, root):
        self.root = root
        self.deadline = time.monotonic() + 1000
        self.refs = {}
        self.trees = {}
        self.blobs = {}
        self.calls = Counter()
        self.after_blob = None
        self.sequence = 0

    def put(self, commit, relative, raw):
        self.blobs[oid(raw)] = raw
        self.trees.setdefault(commit, {})[archive.PROGRAMME + '/' + relative] = {
            'type': 'blob', 'mode': '100644', 'sha': oid(raw), 'size': len(raw)}

    def publish(self, relative, raw):
        self.sequence += 1
        commit = hashlib.sha1(('fixture control commit %d' % self.sequence).encode()).hexdigest()
        self.put(commit, relative, raw)
        self.refs['refs/heads/archive/gcdl-001/' + commit] = commit
        return commit

    def fetch_ref(self, reference):
        self.calls['ref'] += 1
        if reference not in self.refs:
            raise archive.ArchiveError('fixture reference missing')
        return self.refs[reference]

    def fetch_tree(self, commit):
        self.calls['tree'] += 1
        return deepcopy(self.trees[commit])

    def fetch_blob(self, identifier, max_bytes):
        self.calls['blob'] += 1
        raw = self.blobs[identifier]
        if len(raw) > max_bytes:
            raise archive.ArchiveError('fixture blob bound')
        if self.after_blob:
            self.after_blob(identifier)
        return raw


class IndependentRenderedMetadataReview(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gcdl-independent-projection-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve() / archive.PROGRAMME
        self.root.mkdir()
        self.write('state/worker.lock', b'')
        self.server = SnapshotServer(self.root)
        self.code = {'manual_fixture.py': '1' * 64}
        self.index = None
        self.clear_cache()
        self.addCleanup(self.clear_cache)
        provider = patch.object(archive, 'GitHubTransport', side_effect=lambda root, **kwargs: self.server)
        provider.start(); self.addCleanup(provider.stop)
        network = patch.object(archive, '_run', side_effect=AssertionError('real transport forbidden'))
        network.start(); self.addCleanup(network.stop)
        self.receipt_path = self.add_receipt(730100)
        self.install_archive(self.render_paths(), 'a' * 40, 'b' * 40)

    def clear_cache(self):
        archive._CACHE.clear(); archive._CACHE_SIZE = 0

    def write(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def files(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob('*') if p.is_file()}

    def render_paths(self):
        return [str(p.relative_to(self.root)) for p in (self.root / 'data/rendered').rglob('*') if p.is_file()]

    def add_receipt(self, seed):
        row = manual_record(seed=seed)
        row['verification']['code_hashes'] = deepcopy(self.code)
        relative = 'data/accepted/train-%d.jsonl' % seed
        raw = raw_json(row); self.write(relative, raw)
        p = row['provenance']; profile = resolve_profile('v1')
        batch = {'version': 'gcdl-manifest-2', 'seed': seed, 'split': 'train',
                 'shard': relative, 'sha256': sha(raw), 'bytes': len(raw), 'accepted': 1,
                 'quarantined': 0, 'audit': {'verdict': 'PASS', 'accepted': 1, 'quarantined': 0},
                 'generation_profile': 'v1', 'generation_profile_evidence': {
                     k: profile[k] for k in ('id', 'generator_version', 'environment_version',
                                            'config_version', 'config_hash', 'config_file_sha256')},
                 'code_hashes': self.code,
                 'protected_index': [[row['id'], semantic_fingerprint(row), p['parent_world_id'],
                                      p['template_group_id'], 'train']]}
        self.write('manifests/batch-train-%d.json' % seed, raw_json(batch))
        db = sqlite3.connect(':memory:')
        try:
            config = {'rendered_views': ['listwise'], 'rendered_variant_budget': 3,
                      'rendered_groups_per_cycle': 1}
            with patch.object(rendered, 'source_hashes', return_value=self.code), \
                    patch.object(rendering, 'source_hashes', return_value=self.code), \
                    patch.object(rendering.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40)):
                rendered.enqueue(db, self.root, config)
                receipts = rendered.drain(db, self.root, config, time.monotonic() + 30)
            self.assertEqual(len(receipts), 1)
            return receipts[0]
        finally:
            db.close()

    def install_archive(self, paths, source, snapshot):
        files = {}
        for relative in paths:
            raw = (self.root / relative).read_bytes()
            files[relative] = {'commit': source, 'git_blob_sha1': oid(raw), 'sha256': sha(raw), 'bytes': len(raw)}
            self.server.put(source, relative, raw); self.server.put(snapshot, relative, raw)
        catalog = {'version': 'gcdl-archive-catalog-1', 'repository': archive.REPOSITORY,
                   'programme': archive.PROGRAMME, 'files': files}
        catalog['catalog_id'] = identity(catalog)
        path = 'manifests/remote_catalogs/catalog-' + catalog['catalog_id'] + '.json'
        raw = raw_json(catalog); self.write(path, raw); self.server.put(snapshot, path, raw)
        descriptor = {'catalog_commit': snapshot, 'archive_ref': 'refs/heads/archive/gcdl-001/' + snapshot,
                      'catalog_path': path, 'catalog_sha256': sha(raw), 'catalog_git_blob_sha1': oid(raw)}
        self.server.refs[descriptor['archive_ref']] = snapshot
        receipt = {'version': 'gcdl-archive-custody-1', 'status': 'remote_verified',
                   'repository': archive.REPOSITORY, 'programme': archive.PROGRAMME,
                   'catalog_id': catalog['catalog_id'], **descriptor, 'files': files,
                   'local_identities': {}, 'local_ancestry': {}}
        receipt['receipt_id'] = identity(receipt)
        receipt_path = 'manifests/remote_custody/custody-' + receipt['receipt_id'] + '.json'
        receipt_raw = raw_json(receipt); self.write(receipt_path, receipt_raw)
        index = deepcopy(self.index) if self.index else {
            'version': 'gcdl-remote-archive-1', 'repository': archive.REPOSITORY,
            'programme': archive.PROGRAMME, 'files': {}, 'catalogs': {}, 'receipts': {}}
        index.pop('digest', None)
        index['files'].update({k: {**v, 'catalog_id': catalog['catalog_id'], 'receipt_id': receipt['receipt_id']}
                               for k, v in files.items()})
        index['catalogs'][catalog['catalog_id']] = descriptor
        index['receipts'][receipt['receipt_id']] = {'path': receipt_path, 'sha256': sha(receipt_raw)}
        index['digest'] = identity(index); self.index = index
        raw = raw_json(index)
        self.write(archive.INDEX_PATH, raw)
        self.write('manifests/remote_indexes/index-' + index['digest'] + '.json', raw)

    def prepare_plan(self, relative=None):
        paths = controls.prepare(self.root, [relative or self.receipt_path])
        self.assertEqual(len(paths), 1)
        capsule = json.loads((self.root / paths[0]).read_bytes())
        plan = controls.plan_bundle(self.root, paths, transport=self.server)
        self.assertIs(plan['trusted'], False)
        commit = self.server.publish(plan['bundle_path'], (self.root / plan['bundle_path']).read_bytes())
        return paths[0], capsule, plan, commit

    def qualify(self):
        path, capsule, plan, commit = self.prepare_plan()
        authority = controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
        return capsule, authority

    def live(self, relative=None):
        relative = relative or self.receipt_path
        receipt = json.loads((self.root / relative).read_bytes())
        batch = rendered._batch(self.root, receipt['source_manifest'])
        return receipt, batch

    def untrusted_capsule(self, capsule):
        value = deepcopy(capsule); value.pop('capsule_id', None); value['capsule_id'] = identity(value)
        path = controls.PREFIX + 'capsule-' + value['capsule_id'] + '.json'
        self.write(path, raw_json(value))
        return path

    def test_qualified_projection_is_label_free_and_preserves_original_validator_result(self):
        before_receipt = (self.root / self.receipt_path).read_bytes()
        raw_result = rendered.validate_receipt(self.root, self.receipt_path, use_controls=False)
        capsule, authority = self.qualify()
        forbidden = {'input_text', 'target', 'selected_records', 'provenance', 'protected_index', 'source_code_hashes', 'code_hashes'}
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value))
            return set()
        self.assertFalse(keys(capsule) & forbidden)
        for path in self.render_paths(): (self.root / path).unlink()
        with patch.object(archive, 'read_bytes', wraps=archive.read_bytes) as reads, archive.read_session(self.root):
            result = rendered.validate_receipt(self.root, self.receipt_path)
        self.assertEqual(result, raw_result)
        self.assertFalse(any(call.args[1].startswith('data/rendered/') for call in reads.call_args_list))
        self.assertEqual((self.root / self.receipt_path).read_bytes(), before_receipt)
        self.assertEqual(json.loads((self.root / controls.POINTER).read_bytes()), authority)

    def test_rehashed_invented_count_cannot_become_authority_by_remote_publication(self):
        paths = controls.prepare(self.root, [self.receipt_path])
        capsule = json.loads((self.root / paths[0]).read_bytes())
        capsule['datasets'][0]['fields']['rendered_rows'] += 1
        capsule['datasets'][0]['shards'][0]['rows'] += 1
        plan = controls.plan_bundle(self.root, [self.untrusted_capsule(capsule)], transport=self.server)
        commit = self.server.publish(plan['bundle_path'], (self.root / plan['bundle_path']).read_bytes())
        before = self.files()
        with self.assertRaises(archive.ArchiveError):
            controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
        self.assertEqual(self.files(), before)
        self.assertFalse((self.root / controls.POINTER).exists())

    def test_closed_projection_schema_rejects_gold_sealed_paths_and_boolean_counts(self):
        paths = controls.prepare(self.root, [self.receipt_path])
        base = json.loads((self.root / paths[0]).read_bytes())
        changes = [lambda c: c.update(verified=True),
                   lambda c: c['datasets'][0]['fields'].update(target='forbidden raw target'),
                   lambda c: c['datasets'][0]['fields'].update(split='sealed'),
                   lambda c: c['datasets'][0]['fields'].update(rendered_rows=True),
                   lambda c: c['datasets'][0]['fields'].update(eligible_source_records=-1),
                   lambda c: c['datasets'][0]['fields'].update(manifest_path='data/sealed/manifest.json'),
                   lambda c: c['datasets'][0]['shards'][0].update(path='data/rendered/../foreign.jsonl')]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                value = deepcopy(base); change(value)
                path = self.untrusted_capsule(value)
                with self.assertRaises(archive.ArchiveError):
                    controls.plan_bundle(self.root, [path], transport=self.server)
        self.assertFalse((self.root / controls.POINTER).exists())

    def test_activation_always_rederives_new_capsule_with_projection_bypassed(self):
        _, _, plan, commit = self.prepare_plan()
        with patch.object(rendered, 'validate_receipt', wraps=rendered.validate_receipt) as raw_validation:
            controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
        self.assertTrue(raw_validation.call_args_list)
        self.assertTrue(all(call.kwargs.get('use_controls') is False for call in raw_validation.call_args_list))

    def test_remote_snapshot_mode_oid_size_and_raw_corruption_cannot_install_authority(self):
        _, _, plan, commit = self.prepare_plan()
        path = archive.PROGRAMME + '/' + plan['bundle_path']
        original = deepcopy(self.server.trees[commit][path])
        original_raw = self.server.blobs[original['sha']]
        for kind in ('missing_ref', 'mode', 'oid', 'size', 'raw'):
            with self.subTest(kind=kind):
                self.server.trees[commit][path] = deepcopy(original)
                self.server.blobs[original['sha']] = original_raw
                self.server.refs['refs/heads/archive/gcdl-001/' + commit] = commit
                if kind == 'missing_ref': self.server.refs.pop('refs/heads/archive/gcdl-001/' + commit)
                elif kind == 'mode': self.server.trees[commit][path]['mode'] = '120000'
                elif kind == 'oid': self.server.trees[commit][path]['sha'] = '9' * 40
                elif kind == 'size': self.server.trees[commit][path]['size'] += 1
                else: self.server.blobs[original['sha']] = b'x' * len(original_raw)
                before = self.files()
                with self.assertRaises(archive.ArchiveError):
                    controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
                self.assertEqual(self.files(), before)

    def test_snapshot_moving_after_raw_control_fetch_preserves_unattested_state(self):
        _, _, plan, commit = self.prepare_plan()
        self.server.after_blob = lambda identifier: self.server.refs.update({
            'refs/heads/archive/gcdl-001/' + commit: '9' * 40})
        before = self.files()
        with self.assertRaises(archive.ArchiveError):
            controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
        self.assertEqual(self.files(), before)

    def test_unrelated_archive_append_keeps_qualified_historical_projection(self):
        self.qualify(); receipt, batch = self.live()
        before = controls.try_projection(self.root, self.receipt_path, receipt, batch)
        old_digest = self.index['digest']
        self.write('data/backups/independent.bin', b'opaque unrelated own fixture\n')
        self.install_archive(['data/backups/independent.bin'], 'c' * 40, 'd' * 40)
        self.assertNotEqual(self.index['digest'], old_digest)
        self.assertEqual(controls.try_projection(self.root, self.receipt_path, receipt, batch), before)

    def test_reinterpreted_relevant_custody_refuses_even_with_same_payload_bytes(self):
        self.qualify(); receipt, batch = self.live()
        self.install_archive(self.render_paths(), 'c' * 40, 'd' * 40)
        with self.assertRaises(archive.ArchiveError):
            controls.try_projection(self.root, self.receipt_path, receipt, batch)

    def test_live_source_spec_and_protected_index_cannot_be_reinterpreted(self):
        self.qualify(); receipt, batch = self.live()
        for kind in ('accepted', 'protected_index', 'code', 'spec'):
            with self.subTest(kind=kind):
                changed_receipt, changed_batch = deepcopy(receipt), deepcopy(batch)
                if kind == 'accepted': changed_batch['accepted'] += 1
                elif kind == 'protected_index': changed_batch['protected_index'][0][1] = 'changed fingerprint'
                elif kind == 'code': changed_batch['code_hashes']['manual_fixture.py'] = '2' * 64
                else: changed_receipt['spec_sha256'] = '2' * 64
                with self.assertRaises(archive.ArchiveError):
                    controls.try_projection(self.root, self.receipt_path, changed_receipt, changed_batch)

    def test_warm_session_still_rehashes_local_qualified_control_bytes(self):
        _, authority = self.qualify(); receipt, batch = self.live()
        with archive.read_session(self.root):
            self.assertIsNotNone(controls.try_projection(self.root, self.receipt_path, receipt, batch))
            calls = self.server.calls.copy()
            self.write(authority['bundle_path'], b'changed local aggregate\n')
            with self.assertRaises(archive.ArchiveError):
                controls.try_projection(self.root, self.receipt_path, receipt, batch)
            self.assertEqual(self.server.calls, calls)

    def second_plan(self):
        before = set(self.render_paths())
        relative = self.add_receipt(730101)
        self.install_archive(sorted(set(self.render_paths()) - before), 'c' * 40, 'd' * 40)
        return relative, self.prepare_plan(relative)

    def test_new_aggregate_inherits_exact_previous_capsules_and_rederives_only_new(self):
        capsule, first = self.qualify()
        relative, (_, _, plan, commit) = self.second_plan()
        bundle = json.loads((self.root / plan['bundle_path']).read_bytes())
        self.assertEqual(bundle['capsules'][self.receipt_path], capsule)
        self.assertEqual(plan['new_receipts'], 1)
        self.assertEqual(plan['qualified_receipts'], 2)
        active_bytes = (self.root / first['bundle_path']).read_bytes()
        pointer_bytes = (self.root / controls.POINTER).read_bytes()
        self.assertNotEqual(plan['bundle_path'], first['bundle_path'])
        self.assertEqual((self.root / first['bundle_path']).read_bytes(), active_bytes)
        self.assertEqual((self.root / controls.POINTER).read_bytes(), pointer_bytes)
        with patch.object(controls, 'derive', wraps=controls.derive) as derive:
            second = controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
        self.assertEqual([call.args[1] for call in derive.call_args_list], [relative])
        self.assertEqual(json.loads((self.root / controls.POINTER).read_bytes()), second)
        for path in (self.receipt_path, relative):
            receipt, batch = self.live(path)
            self.assertIsNotNone(controls.try_projection(self.root, path, receipt, batch))

    def test_inherited_projection_cannot_be_changed_by_a_new_prepared_capsule(self):
        original, _ = self.qualify()
        altered = deepcopy(original)
        altered['datasets'][0]['fields']['eligible_source_records'] += 1
        path = self.untrusted_capsule(altered)
        before = self.files()
        with self.assertRaises(archive.ArchiveError):
            controls.plan_bundle(self.root, [path], transport=self.server)
        self.assertEqual(self.files(), before)

    def test_new_aggregate_cannot_remove_a_qualified_predecessor(self):
        self.qualify()
        _, (_, _, plan, _) = self.second_plan()
        bundle = json.loads((self.root / plan['bundle_path']).read_bytes())
        del bundle['capsules'][self.receipt_path]
        bundle.pop('bundle_id'); bundle['bundle_id'] = identity(bundle)
        self.write(plan['bundle_path'], raw_json(bundle))
        commit = self.server.publish(plan['bundle_path'], raw_json(bundle))
        before = self.files()
        with self.assertRaises(archive.ArchiveError):
            controls.activate(self.root, plan['bundle_path'], commit, transport=self.server)
        self.assertEqual(self.files(), before)

    def test_unqualified_replacement_slot_cannot_corrupt_active_authority(self):
        self.qualify(); original_receipt, original_batch = self.live()
        pointer = (self.root / controls.POINTER).read_bytes()
        _, (_, _, plan, _) = self.second_plan()
        self.write(plan['bundle_path'], b'incomplete untrusted replacement slot\n')
        self.assertEqual((self.root / controls.POINTER).read_bytes(), pointer)
        self.assertIsNotNone(controls.try_projection(self.root, self.receipt_path,
                                                    original_receipt, original_batch))

    def test_moved_original_payload_snapshot_refuses_without_raw_header_fetch(self):
        self.qualify(); receipt, batch = self.live()
        fetched = self.server.calls['blob']
        self.server.refs['refs/heads/archive/gcdl-001/' + 'b' * 40] = '9' * 40
        with self.assertRaises(archive.ArchiveError):
            controls.try_projection(self.root, self.receipt_path, receipt, batch)
        self.assertEqual(self.server.calls['blob'], fetched)

    def test_explicit_foreign_root_provider_refuses_even_outside_read_session(self):
        capsule, _ = self.qualify()
        path = self.untrusted_capsule(capsule)
        foreign = Path(self.directory.name).resolve() / 'other' / archive.PROGRAMME
        foreign.mkdir(parents=True)
        provider = SnapshotServer(foreign)
        provider.refs = self.server.refs
        provider.trees = self.server.trees
        provider.blobs = self.server.blobs
        before = self.files()
        with self.assertRaises(archive.ArchiveError):
            controls.plan_bundle(self.root, [path], transport=provider)
        self.assertEqual(provider.calls, Counter())
        self.assertEqual(self.files(), before)

    def cli_proof_race(self, kind):
        from goal_conditioned_decision_data.auditors import release
        from goal_conditioned_decision_data.scripts import render_controls as cli
        _, _, plan, commit = self.prepare_plan()
        changed = [False]
        operator = b'{"reason":"operator pause during control proof","utc_epoch":77}\n'

        def after_proof_blob(identifier):
            changed[0] = True
            if kind == 'pause': self.write('state/PAUSED.json', operator)

        self.server.after_blob = after_proof_blob
        before = self.files()
        with patch.object(cli, 'ROOT', self.root), \
                patch.object(release, 'current_independent_review',
                             side_effect=lambda **kwargs: not (kind == 'source' and changed[0])), \
                redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main(['activate', '--bundle', plan['bundle_path'], '--commit', commit])
        self.assertEqual(error.exception.code, 1)
        self.assertTrue(changed[0])
        self.assertFalse((self.root / controls.POINTER).exists())
        self.assertFalse(list((self.root / controls.PREFIX).glob('authority-*.json')))
        expected = {**before, 'state/PAUSED.json': operator} if kind == 'pause' else before
        self.assertEqual(self.files(), expected)

    def test_direct_cli_late_operator_pause_refuses_authority_and_preserves_new_marker(self):
        self.cli_proof_race('pause')

    def test_direct_cli_late_source_approval_failure_refuses_authority(self):
        self.cli_proof_race('source')


if __name__ == '__main__':
    unittest.main()
