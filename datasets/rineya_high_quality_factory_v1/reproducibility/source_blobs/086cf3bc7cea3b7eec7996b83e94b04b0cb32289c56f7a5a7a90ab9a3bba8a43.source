"""Synthetic corruptions must fail even where Python True == 1 / False == 0."""
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factory.common import ROOT, canonical, record, record_hash, load_jsonl, write_jsonl
from factory.pipeline import audit, schema_validate, validate_release
from factory.export_validation import validate_specialized
from oracles.independent import solve


def fixture(value):
    task = {'kind':'episodic', 'events':[{'key':'a','version':2,'at':3,'op':'put','value':value}],
            'queries':[{'query_id':'q','key':'a','as_of':3}]}
    return record('REM-001','synthetic-type-regression',0,'case',task,
                  'Apply the public episodic contract. Task JSON:\n'+canonical(task),solve(task),[],1)


class TypedEvidenceTests(unittest.TestCase):
    def rechecksum(self,path,file):
        manifest=json.loads((path/'MANIFEST.json').read_text())
        manifest['files'][file.name]={'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        (path/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (path/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
            for f in sorted(path.iterdir()) if f.name!='CHECKSUMS.sha256'))

    def corrupt_selected_index(self,items,target_key):
        for item in items:
            for result in item[target_key]['results']:
                value=result['chosen_event']
                if type(value) is int and value in (0,1):
                    result['chosen_event']=bool(value)
                    return item
        self.fail('Regression fixture has no chosen event index zero or one')

    def audited(self,row,reference=None):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'candidate.jsonl'
            write_jsonl(file,[row])
            with patch('factory.pipeline.candidate_paths',return_value=[file]), \
                 patch('factory.pipeline.write_jsonl'),patch('factory.pipeline.write_json'), \
                 patch('factory.pipeline.preserve_prior_audit'):
                if reference is None:return audit('synthetic-type-regression')
                with patch('factory.pipeline.reference_solve',return_value=reference):
                    return audit('synthetic-type-regression')

    def test_valid_supported_scalar_types_pass(self):
        for value in [True,1,False,0,None,'1']:
            with self.subTest(value=canonical(value)):
                row=fixture(value);schema_validate(row)
                self.assertEqual(self.audited(row)['validation_failures'],0)

    def test_rehashed_rendered_bool_integer_changes_rejected(self):
        for value,corruption in [(True,1),(1,True),(False,0),(0,False)]:
            row=fixture(value);rendered=copy.deepcopy(row['semantic_task'])
            rendered['events'][0]['value']=corruption
            self.assertEqual(rendered,row['semantic_task'])  # Shows the former failure mode.
            row['rendered_input']='Apply the public episodic contract. Task JSON:\n'+canonical(rendered)
            row['record_hash']=record_hash(row)
            with self.subTest(value=canonical(value)),self.assertRaisesRegex(AssertionError,'rendering semantic mismatch'):
                schema_validate(row)

    def test_rehashed_gold_bool_integer_changes_rejected_by_audit(self):
        for value,corruption in [(True,1),(1,True),(False,0),(0,False)]:
            row=fixture(value);row['verified_target']['results'][0]['value']=corruption
            row['record_hash']=record_hash(row)
            with self.subTest(value=canonical(value)):
                self.assertEqual(self.audited(row)['oracle_failures'],1)

    def test_reference_type_disagreement_rejected(self):
        row=fixture(True);wrong=copy.deepcopy(row['verified_target'])
        wrong['results'][0]['value']=1
        self.assertEqual(self.audited(row,wrong)['oracle_failures'],1)

    def test_memory_derived_target_type_corruption_rejected(self):
        row=fixture(True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);file=path/'train-memory_updates.jsonl'
            item={'input':row['rendered_input'],'target':copy.deepcopy(row['verified_target'])}
            write_jsonl(file,[item]);validate_specialized(path,[row])
            item['target']['results'][0]['value']=1;write_jsonl(file,[item])
            with self.assertRaises(AssertionError):validate_specialized(path,[row])

    def test_rechecksummed_structured_export_type_corruption_rejected(self):
        # Copy immutable release bytes only into a disposable test directory.
        source=ROOT/'data/REM-001/releases/v0001'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'release';shutil.copytree(source,path)
            file=path/'train-structured.jsonl';items=load_jsonl(file)
            self.corrupt_selected_index(items,'target')
            write_jsonl(file,items)
            self.rechecksum(path,file)
            with self.assertRaisesRegex(AssertionError,'structured export mismatch'):validate_release(path)

    def test_rehashed_rechecksummed_release_gold_type_corruption_rejected(self):
        source=ROOT/'data/REM-001/releases/v0001'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'release';shutil.copytree(source,path)
            file=path/'canonical.jsonl';items=load_jsonl(file)
            changed=self.corrupt_selected_index(items,'verified_target')
            changed['record_hash']=record_hash(changed)
            write_jsonl(file,items);self.rechecksum(path,file)
            with self.assertRaisesRegex(AssertionError,'released target mismatch'):validate_release(path)


if __name__=='__main__':unittest.main()
