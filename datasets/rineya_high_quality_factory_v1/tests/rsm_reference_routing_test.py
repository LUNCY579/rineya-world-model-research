"""Exposed HAND routing checks after the complete integration source freeze.

The producer and official interpreters remain separately authored. Agreement
here checks shared factory wiring and HAND regressions, not production review
or approval. Temporary corruption stays in a new test directory.
"""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factory.common import ROOT, canonical
from factory.reference import solve as reference_solve
from factory import option_resume_reference as producer_route
from factory.rsm_controls import option_resume_solve
import json


HAND = 'reviews/oracle/RSM-option-resumption-corpus-20261010T0956Z/actual-hand-candidate-attempt04.jsonl'
HAND_SHA256 = '742364bd1173a76d911a8b43bff70fb30541a46ce5ec637e2ee1477ff34a0bd5'
RESULTS = {'scope': 'Distinct producer/official routes on exposed HAND216; no production approval.', 'cases': []}


class RSMReferenceRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (ROOT / HAND).read_bytes()
        assert sha256(raw).hexdigest() == HAND_SHA256
        cls.tasks = [json.loads(line)['semantic_task'] for line in raw.splitlines()]
        assert len(cls.tasks) == 216

    @contextmanager
    def isolated_reference(self):
        parent = ROOT / 'quality_reports/RSM_CORE_INTEGRATION_HAND_EXECUTION_20261010'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='root-reference-', dir=parent) as name:
            root = Path(name)
            source = root / producer_route.REFERENCE_PATH
            source.parent.mkdir(parents=True)
            source.write_bytes((ROOT / producer_route.REFERENCE_PATH).read_bytes())
            with patch.object(producer_route, 'ROOT', root), patch.object(producer_route, '_REFERENCE', None):
                yield root, source

    def test_distinct_factory_routes_agree_on_all_hand_tasks(self):
        with self.isolated_reference():
            for task in self.tasks:
                self.assertEqual(canonical(reference_solve(task)), canonical(option_resume_solve(task)))
            self.assertEqual(producer_route.reference().__file__, str(producer_route.ROOT / producer_route.REFERENCE_PATH))
            self.assertEqual(producer_route.REFERENCE_AUTHOR_EXECUTION, '/root/relational_critic_producer')
        RESULTS['cases'].append({'case': 'distinct-routes-HAND216', 'typed_agreements': 216, 'production_approval': False})

    def test_cached_reference_rejects_source_corruption(self):
        with self.isolated_reference() as (_, source):
            reference_solve(self.tasks[0])
            source.write_bytes(source.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'frozen version'):
                reference_solve(self.tasks[0])

    def test_source_symlink_rejected_before_reference_load(self):
        with self.isolated_reference() as (root, source):
            raw = source.read_bytes()
            source.unlink()
            target = root / 'separate-reference-bytes'
            target.write_bytes(raw)
            source.symlink_to(target)
            with self.assertRaises(OSError):
                reference_solve(self.tasks[0])
            self.assertIsNone(producer_route._REFERENCE)

    def test_parent_symlink_rejected_before_reference_load(self):
        with self.isolated_reference() as (root, source):
            raw = source.read_bytes()
            source.unlink()
            parent = source.parent
            parent.rmdir()
            target = root / 'separate-reference-directory'
            target.mkdir()
            (target / source.name).write_bytes(raw)
            parent.symlink_to(target, target_is_directory=True)
            with self.assertRaises(OSError):
                reference_solve(self.tasks[0])
            self.assertIsNone(producer_route._REFERENCE)


if __name__ == '__main__':
    unittest.main()
