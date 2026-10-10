"""Actual gate corruption tests. Approval fixtures are synthetic and temporary."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.common import ROOT,load_jsonl,record_hash,write_jsonl
from factory.pipeline import audit,release,memory_control_report


class MemoryControlGateTests(unittest.TestCase):
    def rows(self):
        rows=load_jsonl(ROOT/'data/REM-001/candidate_renderings/memory-gap-v2.jsonl')
        family=rows[0]['source_family_key']
        return [copy.deepcopy(r) for r in rows if r['source_family_key']==family]

    def test_false_public_property_is_a_metadata_failure(self):
        rows=self.rows();row=rows[0]
        row['skill_tags'].append('temporary_delayed_winner')
        row['record_hash']=record_hash(row)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'candidate.jsonl';write_jsonl(file,rows)
            with patch('factory.pipeline.candidate_paths',return_value=[file]), \
                 patch('factory.pipeline.write_jsonl'),patch('factory.pipeline.write_json'), \
                 patch('factory.pipeline.atomic'),patch('factory.pipeline.preserve_prior_audit'):
                result=audit('SYNTHETIC-CORRUPTION')
        self.assertEqual(result['oracle_failures'],0)
        self.assertEqual(result['metadata_failures'],1)
        self.assertEqual(result['memory_control_error_count'],1)

    def test_missing_approved_rotation_blocks_release_before_directory_creation(self):
        rows=self.rows()
        self.assertEqual(memory_control_report(rows)['errors'],[])
        # These in-memory synthetic receipts isolate post-approval filtering.
        # They never become actual review evidence or enter the real factory.
        actors={'expert':'/root/expert_reviewer','adversarial':'/root/adversarial_auditor',
                'adjudication':'/root/research_architect'}
        evidence={role:{r['sample_id']:[{'sample_id':r['sample_id'],'record_hash':r['record_hash'],
            'decision':'ACCEPT','reviewer_execution_id':actor,'receipt_id':'SYNTHETIC-TEST-'+role+'-'+r['sample_id'],
            'critical_defects':[],'issues':[]}] for r in rows} for role,actor in actors.items()}
        evidence['expert'][rows[0]['sample_id']][0]['decision']='REVISE'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch='SYNTHETIC-APPROVAL-FILTER'
            file=root/'data/REM-001/candidate_renderings'/f'{batch}.jsonl';write_jsonl(file,rows)
            write_jsonl(root/'reviews/oracle'/f'{batch}.jsonl',
                        [{'sample_id':r['sample_id'],'record_hash':r['record_hash'],'pass':True} for r in rows])
            with patch('factory.pipeline.ROOT',root),patch('factory.pipeline.candidate_paths',return_value=[file]), \
                 patch('factory.pipeline.audit',return_value={'split_leakage':[],'cross_parent_semantic_duplicates':[],
                    'exact_prompt_duplicates':0}), \
                 patch('factory.pipeline.load_receipts',side_effect=lambda role,name:evidence[role]):
                with self.assertRaisesRegex(AssertionError,'memory family controls incomplete or false'):
                    release(batch,'SYNTHETIC-NO-PUBLICATION',['REM-001'])
            self.assertFalse((root/'data/REM-001/releases/SYNTHETIC-NO-PUBLICATION').exists())


if __name__=='__main__':unittest.main()
