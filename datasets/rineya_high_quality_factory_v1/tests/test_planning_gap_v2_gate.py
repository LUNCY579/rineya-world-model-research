"""Actual frozen V2 publication routing and custody failures; no producer gold."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.common import ROOT, canonical, record, record_hash, write_json
from factory.planning_controls import (PARENT_PATH, PRIOR_PATH, checker, checker_v2,
                                       planning_checker, planning_control_report, planning_control_rows)
from oracles.independent import solve


def row():
    task = {'kind': 'graph', 'nodes': ['a', 'b'], 'edges': [['a', 'b', 1]],
            'start': 'a', 'goal': 'b', 'fuel': 1, 'plan': ['b']}
    result = record('RGM-001', 'SYNTHETIC-V2-GATE', 180, 'base', task,
                    'Public graph rules. Task JSON:\n' + canonical(task), solve(task), [], 2)
    result['generator_version'] = 'rgm-planning-gap-2.0.2'
    result['record_hash'] = record_hash(result)
    return result


def bindings(root, parent=True):
    names = list(checker_v2.SPECIFICATIONS) + [PRIOR_PATH] + ([PARENT_PATH] if parent else [])
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    manifest = {'version': checker_v2.VERSION, 'batch_name': checker_v2.BATCH,
                'track_id': 'RGM-001', 'semantics_sha256': checker_v2.SPECIFICATIONS['schemas/SEMANTICS.md'],
                'revision_of_batch': 'planning-gap-v1', 'parent_input_view_sha256': checker_v2.PARENT_SHA256,
                'public_specification_bindings': [{'path': name, 'sha256': sha}
                                                 for name, sha in checker_v2.SPECIFICATIONS.items()],
                'families': []}
    write_json(root / 'data/RGM-001/manifests/SYNTHETIC.family_controls.json', manifest)
    return manifest


class PlanningV2PublicationTests(unittest.TestCase):
    def test_marker_presence_cannot_hide_obligations_with_invalid_marker_value(self):
        candidate = row()
        candidate['generator_version'] = 'RENAMED'
        candidate['controlled_context'] = 'INVALID'
        self.assertEqual(planning_control_rows([candidate], 'RENAMED'), [candidate])

    def test_public_manifest_routes_arbitrary_fresh_generator_spelling(self):
        candidate = row()
        candidate['generator_version'] = 'independently-named-fresh-version'
        self.assertIs(planning_checker([candidate], 'RENAMED', manifest={'version': checker_v2.VERSION}), checker_v2)
        self.assertIs(planning_checker([candidate], 'legacy'), checker)

    def test_c4_and_namespaced_new_property_route_without_version_convention(self):
        for field, value in [('controlled_context', 'C4'),
                             ('skill_tags', ['property:' + checker_v2.NEW_PROPERTY])]:
            with self.subTest(field=field):
                candidate = row()
                candidate['generator_version'] = 'RENAMED'
                candidate[field] = value
                self.assertEqual(planning_control_rows([candidate], 'RENAMED'), [candidate])
                self.assertIs(planning_checker([candidate], 'RENAMED'), checker_v2)

    def test_actual_both_frozen_sources_and_seven_public_docs_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings(root)
            report = planning_control_report([row()], 'SYNTHETIC', root)
        self.assertEqual(report['checker_revision'], checker_v2.VERSION)
        self.assertEqual(report['evidence_sha256'][str(Path(checker.__file__).resolve())], checker_v2.V1_DEPENDENCY_SHA256)
        self.assertEqual(report['evidence_sha256'][str(Path(checker_v2.__file__).resolve())],
                         '69944b8e06fb984947d0d686296d0bb886aecc358ae6f7405d279d0ba5a40fdf')
        self.assertTrue(any('requires exactly 408 rows' in error for error in report['errors']))

    def test_missing_parent_blocks_before_cohort_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings(root, parent=False)
            report = planning_control_report([row()], 'SYNTHETIC', root)
        self.assertEqual(len(report['errors']), 1)
        self.assertIn('parent planning input inventory', report['errors'][0])

    def test_changed_parent_bytes_block_freshness_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings(root)
            (root / PARENT_PATH).write_text('[]\n')
            report = planning_control_report([row()], 'SYNTHETIC', root)
        self.assertIn('frozen parent planning input inventory binding mismatch', report['errors'])

    def test_unknown_manifest_version_fails_without_decoding_family_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = bindings(root)
            manifest['version'] = 'unknown-future-version'
            manifest['families'] = ['not-an-evidence-object']
            write_json(root / 'data/RGM-001/manifests/SYNTHETIC.family_controls.json', manifest)
            report = planning_control_report([row()], 'SYNTHETIC', root)
        self.assertTrue(any('exact V2 version required before evidence decoding' in error for error in report['errors']))

    def test_changed_checker_binding_blocks_before_family_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings(root)
            with patch('factory.planning_controls.V2_CHECKER_SHA256', '0' * 64):
                report = planning_control_report([row()], 'SYNTHETIC', root)
        self.assertTrue(any('frozen planning source binding mismatch' in error for error in report['errors']))

    def test_new_row_property_cannot_use_v1_metadata_checker(self):
        candidate = row()
        candidate['generator_version'] = 'RENAMED'
        candidate['skill_tags'] = ['property:' + checker_v2.NEW_PROPERTY]
        candidate['record_hash'] = record_hash(candidate)
        module = planning_checker([candidate], 'RENAMED')
        self.assertIs(module, checker_v2)
        self.assertTrue(any('false V2 row property' in error for error in module.check(candidate)))


if __name__ == '__main__':
    unittest.main()
