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

    def test_private_metadata_labels_in_full_prompt_rejected_without_banning_prose(self):
        row = self.row()
        ordinary = copy.deepcopy(row)
        ordinary['rendered_input'] = 'Consider the difficulty of finding a path after the split in routes.\n' + row['rendered_input']
        ordinary['record_hash'] = record_hash(ordinary)
        schema_validate(ordinary)
        fields = ['semantic_family_id', 'source_provenance', 'difficulty', 'split', 'review_status',
                  'generator_version', 'oracle_version', 'generation_seed', 'skill_tags', 'track_id',
                  'target_format', 'controlled_context', 'primary_stratum', 'source_family_key',
                  'source_index', 'family_control_claims', 'property_witnesses', 'reviewed_candidate_hash']
        fields += ['producer_agent_id', 'producer_role', 'source_family_index', 'structural_difficulty']
        for key in fields:
            for label in [key + ': hidden', '**' + key.upper() + '** = hidden', '"' + key + '": "hidden"']:
                with self.subTest(field=key, label=label):
                    corrupt = copy.deepcopy(row)
                    corrupt['rendered_input'] = label + '\n' + row['rendered_input']
                    corrupt['record_hash'] = record_hash(corrupt)
                    with self.assertRaisesRegex(AssertionError, 'privileged metadata label in student input'):
                        schema_validate(corrupt)

    def test_public_payload_names_equal_to_metadata_words_remain_valid(self):
        tasks = [
            {'kind': 'graph', 'nodes': ['sample_id', 'difficulty'],
             'edges': [['sample_id', 'difficulty', 1]], 'start': 'sample_id',
             'goal': 'difficulty', 'fuel': 1, 'plan': ['difficulty']},
            {'kind': 'blocks', 'on': {'difficulty': 'TABLE', 'split': 'TABLE'},
             'goal_on': {'difficulty': 'split'}, 'plan': [{'block': 'difficulty', 'to': 'split'}]},
            {'kind': 'blocks', 'on': {'sample_id': 'TABLE', 'ordinary': 'TABLE'},
             'goal_on': {'sample_id': 'ordinary'}, 'plan': [{'block': 'sample_id', 'to': 'ordinary'}]},
        ]
        for task in tasks:
            with self.subTest(kind=task['kind'], task=task):
                row = self.row()
                row['semantic_task'] = task
                row['verified_target'] = solve(task)
                row['rendered_input'] = 'Public rules. Task JSON:\n' + canonical(task)
                row['record_hash'] = record_hash(row)
                schema_validate(row)

    def test_optional_row_metadata_and_outer_task_metadata_fail_closed(self):
        row = self.row()
        row['private_calibration_key'] = 'hidden'
        row['rendered_input'] = 'private_calibration_key: hidden\n' + row['rendered_input']
        row['record_hash'] = record_hash(row)
        with self.assertRaisesRegex(AssertionError, 'privileged metadata label'):
            schema_validate(row)
        for key in ['verified_target', 'producer_role', 'structural_difficulty']:
            with self.subTest(field=key):
                row = self.row()
                row['semantic_task'][key] = 'hidden'
                row['rendered_input'] = 'Public rules. Task JSON:\n' + canonical(row['semantic_task'])
                row['record_hash'] = record_hash(row)
                with self.assertRaisesRegex(AssertionError, 'privileged metadata key in student payload'):
                    schema_validate(row)

    def test_private_outer_task_key_casing_matches_prompt_boundary(self):
        tasks = [self.row()['semantic_task'],
                 {'kind': 'blocks', 'on': {'DIFFICULTY': 'TABLE', 'Source_Family_Index': 'TABLE'},
                  'goal_on': {'DIFFICULTY': 'Source_Family_Index'},
                  'plan': [{'block': 'DIFFICULTY', 'to': 'Source_Family_Index'}]}]
        for task in tasks:
            valid = self.row()
            valid['semantic_task'] = task
            valid['verified_target'] = solve(task)
            valid['source_family_index'] = 180
            valid['private_calibration_key'] = 'hidden'
            valid['rendered_input'] = 'Public rules. Task JSON:\n' + canonical(task)
            valid['record_hash'] = record_hash(valid)
            schema_validate(valid)
            for key in ['SOURCE_FAMILY_INDEX', 'Source_Family_Index', 'PRODUCER_ROLE', 'Producer_Role',
                        'DIFFICULTY', 'Difficulty', 'PRIVATE_CALIBRATION_KEY', 'Private_Calibration_Key']:
                with self.subTest(kind=task['kind'], key=key):
                    row = copy.deepcopy(valid)
                    row['semantic_task'][key] = 180
                    row['rendered_input'] = 'Public rules. Task JSON:\n' + canonical(row['semantic_task'])
                    row['record_hash'] = record_hash(row)
                    with self.assertRaisesRegex(AssertionError, 'privileged metadata key in student payload'):
                        schema_validate(row)


if __name__ == '__main__':
    unittest.main()
