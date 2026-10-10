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


if __name__=='__main__':unittest.main()
