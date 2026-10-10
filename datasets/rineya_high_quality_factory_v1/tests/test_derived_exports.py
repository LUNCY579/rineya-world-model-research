import json
import tempfile
import unittest
from pathlib import Path
from factory.common import ROOT, load_jsonl, write_jsonl
from factory.exports import specialized
from factory.export_validation import validate_specialized


class DerivedExports(unittest.TestCase):
    def setUp(self):
        self.rows=load_jsonl(ROOT/'data/RVM-001/releases/v0001/canonical.jsonl')
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)
        for split in ['train','dev_challenge']:
            for name,items in specialized(self.rows,split).items():write_jsonl(self.path/name,items)

    def tearDown(self):self.temp.cleanup()

    def test_semantics_and_crossed_positions(self):
        counts=validate_specialized(self.path,self.rows)
        self.assertGreater(counts['train-preference.jsonl'],0)
        self.assertGreater(counts['train-process_reward.jsonl'],0)

    def test_reversed_preference_rejected(self):
        file=self.path/'train-preference.jsonl';rows=load_jsonl(file)
        rows[0]['chosen'],rows[0]['rejected']=rows[0]['rejected'],rows[0]['chosen']
        write_jsonl(file,rows)
        with self.assertRaisesRegex(AssertionError,'preference must be strict'):validate_specialized(self.path,self.rows)

    def test_corrupt_process_rejected(self):
        file=self.path/'train-process_reward.jsonl';rows=load_jsonl(file)
        rows[0]['labels']['valid']=not rows[0]['labels']['valid'];write_jsonl(file,rows)
        with self.assertRaisesRegex(AssertionError,'candidate process labels mismatch'):validate_specialized(self.path,self.rows)

    def test_malformed_label_type_rejected_by_schema(self):
        from jsonschema import Draft202012Validator,ValidationError
        schema=json.loads((ROOT/'schemas/process_reward/v1.schema.json').read_text())
        item=load_jsonl(self.path/'train-process_reward.jsonl')[0]
        for key,value in [('valid','yes'),('claim_correct',None),('progress',[]),('usefulness',{})]:
            wrong=json.loads(json.dumps(item));wrong['labels'][key]=value
            with self.assertRaises(ValidationError):Draft202012Validator(schema).validate(wrong)

    def test_id_renaming_does_not_change_semantic_pair_selection(self):
        original=specialized(self.rows,'train')['train-preference.jsonl']
        renamed=json.loads(json.dumps(self.rows))
        for row in renamed:
            ids=sorted(c['candidate_id'] for c in row['semantic_task']['candidates'])
            mapping=dict(zip(ids,reversed(ids)))
            for c in row['semantic_task']['candidates']:c['candidate_id']=mapping[c['candidate_id']]
            for c in row['verified_target']['evaluations']:c['candidate_id']=mapping[c['candidate_id']]
            row['verified_target']['ranking']=[[mapping[cid] for cid in group] for group in row['verified_target']['ranking']]
        revised=specialized(renamed,'train')['train-preference.jsonl']
        def content(items):
            result=[]
            for item in items:
                pair=[]
                for key in ['chosen','rejected']:
                    candidate=json.loads(item[key]);candidate.pop('candidate_id');pair.append(candidate)
                result.append(pair)
            return result
        self.assertEqual(content(original),content(revised))


if __name__=='__main__':unittest.main()
