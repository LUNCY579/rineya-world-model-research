"""Actual-function regressions for independently observed publication weaknesses."""
import copy
import unittest

from factory.common import canonical, record, record_hash
from factory.pipeline import APPROVAL_ACTORS, approval_evidence, schema_validate
from factory.planning_controls import planning_control_rows
from oracles.independent import solve


class PlanningReviewCorrectionTests(unittest.TestCase):
    def row(self):
        task = {'kind': 'graph', 'nodes': ['a', 'b'], 'edges': [['a', 'b', 1]],
                'start': 'a', 'goal': 'b', 'fuel': 2, 'plan': ['b']}
        row = record('RGM-001', 'SYNTHETIC', 180, 'base', task,
                     'Public graph rules. Task JSON:\n' + canonical(task), solve(task), [], 2)
        return row

    def test_context_and_stratum_each_retain_planning_obligations_after_version_change(self):
        for field, value in [('controlled_context', 'A0'), ('primary_stratum', 'C')]:
            with self.subTest(field=field):
                row = self.row()
                row[field] = value
                self.assertEqual(planning_control_rows([row], 'RENAMED-BATCH'), [row])

    def test_prospective_batch_cannot_skip_planning_gate_with_removed_metadata(self):
        row = self.row()
        self.assertEqual(planning_control_rows([row], 'planning-gap-v2'), [row])

    def test_rehashed_privileged_metadata_in_contract_lead_rejected_by_actual_schema(self):
        row = self.row()
        schema_validate(row)
        for key in ['sample_id', 'semantic_parent_id', 'record_hash', 'review_receipt_ids', 'verified_target']:
            with self.subTest(key=key):
                corrupt = copy.deepcopy(row)
                corrupt['rendered_input'] = key + ': hidden value\n' + row['rendered_input']
                corrupt['record_hash'] = record_hash(corrupt)
                with self.assertRaisesRegex(AssertionError, 'privileged metadata in student input'):
                    schema_validate(corrupt)

    def test_exact_role_identities_required_before_release(self):
        row = self.row()
        evidence = {role: {row['sample_id']: [{'record_hash': row['record_hash'],
                    'decision': 'ACCEPT', 'reviewer_execution_id': actor,
                    'critical_defects': [], 'issues': []}]}
                    for role, actor in APPROVAL_ACTORS.items()}
        self.assertEqual(approval_evidence(row, evidence)[1], [])
        for corruption in ['unknown', 'swap']:
            altered = copy.deepcopy(evidence)
            if corruption == 'unknown':
                for role in altered:
                    altered[role][row['sample_id']][0]['reviewer_execution_id'] = '/root/other_' + role
            else:
                altered['expert'][row['sample_id']][0]['reviewer_execution_id'] = APPROVAL_ACTORS['adversarial']
                altered['adversarial'][row['sample_id']][0]['reviewer_execution_id'] = APPROVAL_ACTORS['expert']
            self.assertIn('nonindependent_approval', approval_evidence(row, altered)[1])


if __name__ == '__main__':
    unittest.main()
