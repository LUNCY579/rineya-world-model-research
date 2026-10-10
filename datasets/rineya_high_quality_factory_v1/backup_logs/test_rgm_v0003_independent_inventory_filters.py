#!/usr/bin/env python3
"""Actual precise-scope regressions; no inventory or production mutations."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
LOGS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('inventory_filters',LOGS/'prepare_rgm_v0003_independent_release_inventory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ScopeSafety(unittest.TestCase):
    def test_official_only_exact_new_prefix_deferred(self):
        self.assertIsNotNone(m.deferred_reason(m.OFFICIAL_PREFIX+'belief_controls_v4_independent.py'))
        self.assertIsNotNone(m.deferred_reason(m.OFFICIAL_PREFIX+'nested/future-tests.json'))
        self.assertIsNone(m.deferred_reason('reviews/oracle/RBM-v4-official-OTHER/source.py'))
        self.assertIsNone(m.deferred_reason('oracles/belief_controls_v4_independent.py'))

    def test_new_adversarial_prefixes_and_future_names_deferred(self):
        for prefix in m.ADVERSARIAL_PREFIXES:
            with self.subTest(prefix=prefix):self.assertIsNotNone(m.deferred_reason(prefix+'future.py'))
        self.assertIsNone(m.deferred_reason('audits/belief_gap_v3_adversarial_audit.py'))
        self.assertIsNone(m.deferred_reason('reviews/adversarial/planning-gap-v2-renderer-r1.jsonl'))

    def test_completed_expert_all_paths_included(self):
        paths=list((m.FACTORY/'reviews/expert').glob('belief-v4-prospective-*'))
        self.assertTrue(paths)
        self.assertTrue((m.FACTORY/'reviews/expert/belief-v4-prospective-preparation-final.json').is_file())
        for path in paths:
            self.assertIsNone(m.deferred_reason(path.relative_to(m.FACTORY).as_posix()))

    def test_completed_formal_twelve_files_included(self):
        paths=[p for p in (m.FACTORY/m.COMPLETED_FORMAL_PREFIX).rglob('*') if p.is_file()]
        self.assertEqual(len(paths),12)
        for path in paths:self.assertIsNone(m.deferred_reason(path.relative_to(m.FACTORY).as_posix()))

    def test_all_previous_2095_names_included(self):
        previous=json.loads((LOGS/'rgm-v2-renderer-review-milestone_SPEC.json').read_bytes())['files']
        self.assertEqual(len(previous),2095)
        for item in previous:self.assertIsNone(m.deferred_reason(item['path']),item['path'])

    def test_all_completed_renderer_r1_names_included(self):
        names=[p.relative_to(m.FACTORY).as_posix() for base in ('reviews','audits','quality_reports') for p in (m.FACTORY/base).rglob('*') if p.is_file() and ('renderer-r1' in p.name or 'renderer_r1' in p.name)]
        self.assertGreater(len(names),30)
        for name in names:self.assertIsNone(m.deferred_reason(name),name)

    def test_all_210_release_and_public_v4_names_included(self):
        names=[p.relative_to(m.FACTORY).as_posix() for p in (m.FACTORY/'data').rglob('*') if p.is_file() and '/releases/' in p.as_posix()]
        self.assertEqual(len(names),210)
        names+=['schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md','schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4_ENCODING_ADDENDUM.md','research/curricula/BELIEF_METADATA_V4_ENCODING_ROOT_FREEZE.json','backup_logs/RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json']
        for name in names:self.assertIsNone(m.deferred_reason(name),name)


if __name__=='__main__':unittest.main(verbosity=2)
