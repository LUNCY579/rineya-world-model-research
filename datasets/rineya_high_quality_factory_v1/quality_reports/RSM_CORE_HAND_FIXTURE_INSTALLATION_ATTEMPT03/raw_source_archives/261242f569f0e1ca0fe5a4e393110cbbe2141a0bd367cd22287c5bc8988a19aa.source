import unittest
from factory.common import ROOT,load_jsonl,record_hash


class ClaimGapIntegrity(unittest.TestCase):
    def test_renderer_metadata_is_not_aliased(self):
        rows=load_jsonl(ROOT/'data/RVM-001/candidate_renderings/claim-gap-v2.jsonl')
        self.assertEqual(len(rows),24)
        for row in rows:
            self.assertEqual(row['record_hash'],record_hash(row))
            self.assertEqual(len(row['skill_tags']),len(set(row['skill_tags'])))
        for forward,reverse in zip(rows[::2],rows[1::2]):
            self.assertEqual(forward['skill_tags'],reverse['skill_tags'])
            self.assertEqual(forward['semantic_task'],reverse['semantic_task'])
            self.assertEqual(forward['verified_target'],reverse['verified_target'])


if __name__=='__main__':unittest.main()
