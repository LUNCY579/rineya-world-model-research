"""Exercise publication after actual v3 evidence or whole variants are removed.

Fabricated approval fixtures are confined to temporary test workspaces.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.common import ROOT, load_jsonl, record_hash, write_jsonl
from factory.pipeline import belief_control_rows, release
from tests import test_belief_control_gate as prior_gates


class BeliefV3ControlGateTests(unittest.TestCase):
    def test_revision_still_checks_when_all_control_metadata_is_removed(self):
        row = copy.deepcopy(load_jsonl(ROOT / 'data/RBM-001/candidate_renderings/belief-gap-v3.jsonl')[0])
        for key in ['controlled_context', 'structural_family_profile', 'fact_role_assignments',
                    'query_role_assignments', 'property_witnesses', 'family_control_claims',
                    'auxiliary_scalar_variant', 'auxiliary_scalar_control']:
            row.pop(key)
        row['skill_tags'] = []
        row['record_hash'] = record_hash(row)
        self.assertEqual(len(belief_control_rows([row])), 1)
        report = prior_gates.BeliefControlGateTests.isolated_audit(self, row)
        self.assertEqual(report['oracle_failures'], 0)
        self.assertEqual(report['metadata_failures'], 1)

    def test_complete_old_projection_cannot_publish_without_transposed_variant(self):
        rows = copy.deepcopy(load_jsonl(ROOT / 'data/RBM-001/candidate_renderings/belief-gap-v3.jsonl'))
        rows = [r for r in rows if r['auxiliary_scalar_variant'] == 'identity']
        new_claims = {'auxiliary_scalar_transposition_invariance',
                      'same_observations_auxiliary_table_counterfactual',
                      'context_auxiliary_scalar_variant_crossing'}
        for row in rows:
            row.pop('auxiliary_scalar_variant')
            row.pop('auxiliary_scalar_control')
            row['family_control_claims'] = [c for c in row['family_control_claims']
                                            if (c.get('name') or c.get('property')) not in new_claims]
            row['record_hash'] = record_hash(row)
        from oracles.belief_controls_v2_independent import check_batch as old_contract
        self.assertEqual(old_contract(rows)['errors'], [])
        actors = {'expert': '/root/expert_reviewer', 'adversarial': '/root/adversarial_auditor',
                  'adjudication': '/root/research_architect'}
        evidence = {role: {r['sample_id']: [{
            'sample_id': r['sample_id'], 'record_hash': r['record_hash'], 'decision': 'ACCEPT',
            'reviewer_execution_id': actor, 'receipt_id': 'SYNTHETIC-V3-' + role + '-' + r['sample_id'],
            'critical_defects': [], 'issues': [],
        }] for r in rows} for role, actor in actors.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = 'SYNTHETIC-V3-MISSING-VARIANT'
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
