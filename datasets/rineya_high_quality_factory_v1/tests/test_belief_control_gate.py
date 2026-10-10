"""Corrupt actual belief evidence to verify fail-closed publication gates.

Synthetic review receipts exist only inside temporary test workspaces.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.common import ROOT, load_jsonl, record_hash, write_jsonl
from factory.pipeline import audit, belief_control_rows, release


class BeliefControlGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = load_jsonl(ROOT / 'data/RBM-001/candidate_renderings/belief-gap-v1.jsonl')

    def isolated_audit(self, row):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.jsonl'
            write_jsonl(path, [row])
            with patch('factory.pipeline.candidate_paths', return_value=[path]), \
                    patch('factory.pipeline.write_jsonl'), patch('factory.pipeline.write_json'), \
                    patch('factory.pipeline.atomic'), patch('factory.pipeline.preserve_prior_audit'):
                return audit('SYNTHETIC-BELIEF-CORRUPTION')

    def test_false_usefulness_claim_fails_even_when_gold_is_correct(self):
        row = copy.deepcopy(next(r for r in self.candidates if r['controlled_context'] == 'B'))
        self.assertFalse(any(q['can_reduce_uncertainty'] for q in row['verified_target']['queries']))
        row['skill_tags'].append('useful_after_conjunctive_evidence')
        row['property_witnesses']['useful_after_conjunctive_evidence'] = {
            'query_ids': [row['semantic_task']['queries'][0]['query_id']],
            'observation_indices': [0, 1], 'prefix_indices': [1, 2],
        }
        row['record_hash'] = record_hash(row)
        report = self.isolated_audit(row)
        self.assertEqual(report['oracle_failures'], 0)
        self.assertEqual(report['metadata_failures'], 1)

    def test_deleting_control_metadata_cannot_hide_a_retained_property(self):
        row = copy.deepcopy(next(r for r in self.candidates if r['controlled_context'] == 'A'))
        for key in ['controlled_context', 'structural_family_profile', 'fact_role_assignments',
                    'query_role_assignments', 'property_witnesses', 'family_control_claims']:
            row.pop(key)
        row['skill_tags'] = ['multiworld_unobserved_invariant']
        row['record_hash'] = record_hash(row)
        self.assertEqual(len(belief_control_rows([row])), 1)
        report = self.isolated_audit(row)
        self.assertEqual(report['oracle_failures'], 0)
        self.assertEqual(report['metadata_failures'], 1)

    def test_missing_approved_rotation_blocks_release_before_directory_creation(self):
        rows = copy.deepcopy(self.candidates)
        actors = {'expert': '/root/expert_reviewer', 'adversarial': '/root/adversarial_auditor',
                  'adjudication': '/root/research_architect'}
        evidence = {role: {r['sample_id']: [{
            'sample_id': r['sample_id'], 'record_hash': r['record_hash'], 'decision': 'ACCEPT',
            'reviewer_execution_id': actor,
            'receipt_id': 'SYNTHETIC-TEST-' + role + '-' + r['sample_id'],
            'critical_defects': [], 'issues': [],
        }] for r in rows} for role, actor in actors.items()}
        evidence['expert'][rows[0]['sample_id']][0]['decision'] = 'REVISE'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = 'SYNTHETIC-BELIEF-APPROVAL-FILTER'
            path = root / 'data/RBM-001/candidate_renderings' / (batch + '.jsonl')
            write_jsonl(path, rows)
            write_jsonl(root / 'reviews/oracle' / (batch + '.jsonl'), [{
                'sample_id': r['sample_id'], 'record_hash': r['record_hash'], 'pass': True,
            } for r in rows])
            with patch('factory.pipeline.ROOT', root), \
                    patch('factory.pipeline.candidate_paths', return_value=[path]), \
                    patch('factory.pipeline.audit', return_value={
                        'split_leakage': [], 'cross_parent_semantic_duplicates': [],
                        'exact_prompt_duplicates': 0,
                    }), \
                    patch('factory.pipeline.load_receipts', side_effect=lambda role, name: evidence[role]):
                with self.assertRaisesRegex(AssertionError, 'belief family controls incomplete or false'):
                    release(batch, 'SYNTHETIC-NO-PUBLICATION', ['RBM-001'])
            self.assertFalse((root / 'data/RBM-001/releases/SYNTHETIC-NO-PUBLICATION').exists())

    def test_v2_revision_provenance_still_checks_stripped_control_metadata(self):
        row = copy.deepcopy(load_jsonl(ROOT / 'data/RBM-001/candidate_renderings/belief-gap-v2.jsonl')[0])
        for key in ['controlled_context', 'structural_family_profile', 'fact_role_assignments',
                    'query_role_assignments', 'property_witnesses', 'family_control_claims']:
            row.pop(key)
        row['skill_tags'] = []
        row['record_hash'] = record_hash(row)
        self.assertEqual(len(belief_control_rows([row])), 1)
        result = self.isolated_audit(row)
        self.assertEqual(result['oracle_failures'], 0)
        self.assertEqual(result['metadata_failures'], 1)

    def test_v2_provenance_blocks_missing_all_new_siblings(self):
        # A complete old A-F projection can pass the older contract. The
        # canonical revision must retain the new E/G and F/H obligations.
        rows = copy.deepcopy(load_jsonl(ROOT / 'data/RBM-001/candidate_renderings/belief-gap-v2.jsonl'))
        rows = [r for r in rows if r['controlled_context'] not in {'G', 'H'}]
        new_claims = {'same_prefix_empty_nonempty_counterfactual', 'contradiction_equals_mixed_outcomes'}
        for row in rows:
            row['family_control_claims'] = [c for c in row['family_control_claims']
                                            if (c.get('name') or c.get('property')) not in new_claims]
            row['record_hash'] = record_hash(row)
        from oracles.belief_controls_independent import check_batch as old_contract
        self.assertEqual(old_contract(rows)['errors'], [])
        actors = {'expert': '/root/expert_reviewer', 'adversarial': '/root/adversarial_auditor',
                  'adjudication': '/root/research_architect'}
        evidence = {role: {r['sample_id']: [{
            'sample_id': r['sample_id'], 'record_hash': r['record_hash'], 'decision': 'ACCEPT',
            'reviewer_execution_id': actor, 'receipt_id': 'SYNTHETIC-V2-' + role + '-' + r['sample_id'],
            'critical_defects': [], 'issues': [],
        }] for r in rows} for role, actor in actors.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = 'SYNTHETIC-V2-MISSING-SIBLINGS'
            path = root / 'data/RBM-001/candidate_renderings' / (batch + '.jsonl')
            write_jsonl(path, rows)
            write_jsonl(root / 'reviews/oracle' / (batch + '.jsonl'), [{
                'sample_id': r['sample_id'], 'record_hash': r['record_hash'], 'pass': True,
            } for r in rows])
            with patch('factory.pipeline.ROOT', root), \
                    patch('factory.pipeline.candidate_paths', return_value=[path]), \
                    patch('factory.pipeline.audit', return_value={
                        'split_leakage': [], 'cross_parent_semantic_duplicates': [], 'exact_prompt_duplicates': 0,
                    }), \
                    patch('factory.pipeline.load_receipts', side_effect=lambda role, name: evidence[role]):
                with self.assertRaisesRegex(AssertionError, 'belief family controls incomplete or false'):
                    release(batch, 'SYNTHETIC-NO-PUBLICATION', ['RBM-001'])
            self.assertFalse((root / 'data/RBM-001/releases/SYNTHETIC-NO-PUBLICATION').exists())


if __name__ == '__main__':
    unittest.main()
