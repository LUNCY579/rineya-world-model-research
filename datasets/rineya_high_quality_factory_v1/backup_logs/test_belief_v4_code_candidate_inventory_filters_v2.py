#!/usr/bin/env python3
"""Scope regressions without scanning payloads, network, or production mutations."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('v4_inventory_filters', LOGS / 'prepare_belief_v4_code_candidate_inventory_v2.py')
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)


class ScopeSafety(unittest.TestCase):
    def test_exact_parent_checkpoint_and_26_raw_scope(self):
        self.assertEqual(m.PARENT, '82787689fef4e50f7a35cfee505c290b2f6a6d78')
        self.assertEqual(m.CHECKPOINT, 'belief-v4-validated-code-candidate')
        self.assertEqual(m.RAW_LOSSLESS_REQUIRED_COUNT, 26)

    def test_expert_canonical_attempts_and_exact_final_only_deferred(self):
        for name in ('reviews/expert/belief-gap-v4-canonical-review-attempt01.py',
                     'reviews/expert/belief-gap-v4-canonical-attempt02.failure.json',
                     'reviews/expert/belief-gap-v4.canonical-first-judgments.jsonl',
                     'reviews/expert/belief-gap-v4.canonical-report.md',
                     'reviews/expert/belief-gap-v4.jsonl'):
            with self.subTest(name=name):
                self.assertIsNotNone(m.deferred_reason(name))
        for name in ('reviews/expert/belief-gap-v4-input-attempt01.FINAL.json',
                     'reviews/expert/belief-gap-v4-input-attempt01.first-input-judgments.jsonl',
                     'reviews/expert/belief-v4-prospective-preparation-final.json',
                     'reviews/expert/belief-gap-v3.jsonl'):
            self.assertIsNone(m.deferred_reason(name), name)

    def test_each_red_canonical_future_prefix_deferred(self):
        for prefix in m.ADVERSARIAL_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertIsNotNone(m.deferred_reason(prefix + 'future-attempt02.json'))
        self.assertIsNotNone(m.deferred_reason('reviews/adversarial/belief-gap-v4.jsonl'))

    def test_red_initial_input_and_first_failure_always_included(self):
        for name in ('reviews/adversarial/belief-gap-v4.input_predictions.jsonl',
                     'reviews/adversarial/belief-gap-v4.input_first_judgments.jsonl',
                     'reviews/adversarial/belief-gap-v4.input_expectation_freeze.json',
                     'reviews/adversarial/belief-gap-v4.input_lineage.json',
                     'audits/belief_v4_adversarial_input_r1.py',
                     'audits/history/belief_v4_adversarial_input-first-failure.json',
                     'audits/execution/belief-gap-v4-input-adversarial-r1.json'):
            self.assertIsNone(m.deferred_reason(name), name)

    def test_every_prior2303_path_included(self):
        entries = json.loads((LOGS / 'rgm-v0003-independent-release_SPEC.json').read_bytes())['files']
        self.assertEqual(len(entries), 2303)
        for entry in entries:
            self.assertIsNone(m.deferred_reason(entry['path']), entry['path'])

    def test_all210_releases_and_public_specs_included(self):
        names = [p.relative_to(m.FACTORY).as_posix() for p in (m.FACTORY / 'data').rglob('*')
                 if p.is_file() and '/releases/' in p.as_posix()]
        self.assertEqual(len(names), 210)
        names += ['schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md',
                  'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md',
                  'research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json',
                  'backup_logs/RGM_V0003_FINAL_REVIEW_JOINED_CUSTODY.json',
                  'quality_reports/RGM_V0003_REMOTE_LOSSLESS.json']
        for name in names:
            self.assertIsNone(m.deferred_reason(name), name)

    def test_completed_official_v4_all_files_included(self):
        paths = [p for p in (m.FACTORY / m.COMPLETED_OFFICIAL_PREFIX).rglob('*') if p.is_file()]
        self.assertGreater(len(paths), 20)
        self.assertTrue((m.FACTORY / m.COMPLETED_OFFICIAL_PREFIX / 'FINAL_RECEIPT.json').is_file())
        for p in paths:
            self.assertIsNone(m.deferred_reason(p.relative_to(m.FACTORY).as_posix()))

    def test_completed_prospective_expert_red_architect_and_formal_included(self):
        names = [p.relative_to(m.FACTORY).as_posix() for base in ('reviews/expert', 'reviews/adversarial', 'reviews/adjudication')
                 for p in (m.FACTORY / base).glob('belief-v4-prospective-*') if p.is_file()]
        self.assertGreater(len(names), 15)
        formal = [p for p in (m.FACTORY / m.COMPLETED_FORMAL_PREFIX).rglob('*') if p.is_file()]
        self.assertEqual(len(formal), 12)
        names += [p.relative_to(m.FACTORY).as_posix() for p in formal]
        for name in names:
            self.assertIsNone(m.deferred_reason(name), name)

    def test_completed_rsm_feasibility_and_separate_simulator_included(self):
        paths = [p for p in (m.FACTORY / m.COMPLETED_RSM_PREFIX).rglob('*') if p.is_file()]
        self.assertGreater(len(paths), 50)
        self.assertTrue((m.FACTORY / m.COMPLETED_RSM_PREFIX / 'FINAL_RECEIPT.json').is_file())
        self.assertTrue(any('separate_simulator' in p.parts for p in paths))
        for p in paths:
            self.assertIsNone(m.deferred_reason(p.relative_to(m.FACTORY).as_posix()))

    def test_completed_producer_gate_audits_and_root_suite_included(self):
        names = ['manifests/belief-gap-v4_PRODUCER_FINAL_CUSTODY.json',
                 'data/RBM-001/candidate_renderings/belief-gap-v4.jsonl',
                 'oracles/belief_controls_v4_independent.py',
                 'tests/oracle_belief_controls_v4_test.py',
                 'factory/belief_controls.py', 'factory/pipeline.py',
                 'quality_reports/BELIEF_V4_FULL_SUITE_EXECUTION_ATTEMPT01.json',
                 'quality_reports/BELIEF_V4_MACHINE_AUDIT_EXECUTION_ATTEMPT01.json',
                 'audits/diversity/belief-gap-v4-belief-controls.json',
                 'backup_logs/BELIEF_V4_MACHINE_LOCAL_LOSSLESS_PACK_RECEIPT.json']
        for name in names:
            self.assertTrue((m.FACTORY / name).is_file(), name)
            self.assertIsNone(m.deferred_reason(name), name)

    def test_missing_or_extra_oversized_raw_is_a_blocker(self):
        m.require_lossless_count([None] * 26)
        for count in (0, 24, 25, 27):
            with self.subTest(count=count), self.assertRaises(ValueError):
                m.require_lossless_count([None] * count)


if __name__ == '__main__':
    unittest.main(verbosity=2)
