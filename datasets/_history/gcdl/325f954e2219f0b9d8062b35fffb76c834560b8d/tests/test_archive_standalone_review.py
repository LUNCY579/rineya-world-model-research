"""Independent standalone helper checks, using only local hand metadata.

Load the actual archive CLI file with no package context. Fixture payloads are
opaque bytes, never scientific rows. No network, archive command, eviction,
sealed data, production action or official experiment is permitted.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.pipelines import render_controls, rendered


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/archive_cycle.py'


class IndependentArchiveStandaloneReview(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gcdl-standalone-review-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / archive.PROGRAMME
        self.root.mkdir()
        self.reads = []

    def write(self, relative, value):
        raw = value if isinstance(value, bytes) else (
            json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    @contextmanager
    def local_only(self):
        """Keep real local archive IO while rejecting every remote boundary."""
        original_read = archive.read_bytes
        original_info = archive.payload_info

        def checked_read(root, relative, **kwargs):
            self.assertEqual(Path(root).resolve(), self.root)
            self.assertFalse(str(relative).startswith('data/sealed/'))
            self.reads.append(str(relative))
            return original_read(root, relative, **kwargs)

        def checked_info(root, relative):
            self.assertEqual(Path(root).resolve(), self.root)
            self.assertFalse(str(relative).startswith('data/sealed/'))
            return original_info(root, relative)

        forbidden = AssertionError('standalone fixture forbids remote/archive actions')
        with ExitStack() as stack:
            guards = [stack.enter_context(patch.object(owner, name, side_effect=forbidden))
                      for owner, name in (
                          (archive, '_run'), (archive, 'GitHubTransport'),
                          (archive, 'evict_verified'), (subprocess, 'run'),
                          (socket, 'create_connection'))]
            stack.enter_context(patch.object(archive, 'read_bytes', side_effect=checked_read))
            stack.enter_context(patch.object(archive, 'payload_info', side_effect=checked_info))
            yield
            for guard in guards:
                guard.assert_not_called()

    def standalone(self):
        previous_path = sys.path[:]
        try:
            namespace = runpy.run_path(str(SCRIPT), run_name='gcdl_standalone_fixture')
        finally:
            sys.path[:] = previous_path
        self.assertEqual(namespace['__name__'], 'gcdl_standalone_fixture')
        self.assertIn(namespace['__package__'], (None, ''))
        self.assertIn(namespace['_publish_render_controls'].__globals__['__package__'], (None, ''))
        return namespace

    def test_standalone_empty_inventory_publisher_returns_none_without_remote_actions(self):
        with self.local_only(), \
             patch.object(render_controls, 'prepare',
                          side_effect=AssertionError('empty inventory cannot prepare controls')) as prepare:
            namespace = self.standalone()
            result = namespace['_publish_render_controls'](
                self.root, time.monotonic() + 30, None, None)
        self.assertIsNone(result)
        prepare.assert_not_called()
        self.assertEqual(list(self.root.rglob('*')), [])

    def test_standalone_matched_cohort_requires_qualified_controls_before_validation_or_eviction(self):
        source_path = 'data/accepted/dev-113.jsonl'
        opaque = self.write(source_path, b'opaque standalone fixture; not record JSON\n')
        batch_path = 'manifests/batch-dev-113.json'
        batch = {'split': 'dev', 'seed': 113, 'shard': source_path,
                 'sha256': hashlib.sha256(opaque).hexdigest(), 'bytes': len(opaque),
                 'accepted': 1, 'quarantined': 0, 'audit': {'verdict': 'PASS'}}
        self.write(batch_path, batch)
        folder = 'data/rendered/gcdl_rendered_' + '1' * 32 + '/'
        receipt_path = 'manifests/render-batch-dev-113-' + 'a' * 16 + '.json'
        completed = {'version': 'gcdl-rendered-batch-1', 'seed': 113, 'split': 'dev',
                     'source_manifest': batch_path,
                     'datasets': [{'manifest_path': folder + 'manifest.json'}]}
        self.write(receipt_path, completed)
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob('*') if path.is_file()}
        with self.local_only(), \
             patch.object(render_controls, 'try_projection', return_value=None) as projection, \
             patch.object(rendered, 'validate_receipt',
                          side_effect=AssertionError('unqualified cohort cannot validate')) as validate:
            namespace = self.standalone()
            with self.assertRaisesRegex(
                    RuntimeError,
                    '^completed rendering cohort requires qualified controls before eviction$'):
                namespace['_require_render_controls'](
                    self.root, {'files': {folder + 'manifest.json': {}}},
                    time.monotonic() + 30)
        projection.assert_called_once_with(self.root, receipt_path, completed, batch)
        validate.assert_not_called()
        self.assertIn(receipt_path, self.reads)
        self.assertIn(batch_path, self.reads)
        self.assertEqual({str(path.relative_to(self.root)): path.read_bytes()
                          for path in self.root.rglob('*') if path.is_file()}, before)


if __name__ == '__main__':
    unittest.main()
