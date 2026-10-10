import collections
import unittest
from factory.common import ROOT, load_jsonl


class CausalRegression(unittest.TestCase):
    def setUp(self):self.rows=load_jsonl(ROOT/"data/RCM-001/candidate_renderings/pilot-v2.jsonl")

    def test_field_order_crossed_with_target_format(self):
        groups=collections.defaultdict(set)
        for r in self.rows:
            groups[(r["semantic_parent_id"],r["semantic_task"]["rule"]["op"],r["target_format"])].add(r["renderer"])
        self.assertTrue(groups)
        for renderers in groups.values():self.assertEqual(renderers,{"forward","reverse"})

    def test_difficulty_is_structural_with_explicit_shift_axes(self):
        for r in self.rows:
            self.assertLessEqual(r["difficulty"],4)
            self.assertEqual(r["difficulty"],r["structural_difficulty"])
            self.assertEqual(bool(r["challenge_axes"]),r["split"]=="dev_challenge")

    def test_counterfactual_rule_changes_only_rule(self):
        groups=collections.defaultdict(list)
        for r in self.rows:groups[r["semantic_parent_id"]].append(r)
        for rs in groups.values():
            unique=[]
            for r in rs:
                if r["semantic_task"] not in unique:unique.append(r["semantic_task"])
            self.assertEqual(len(unique),2)
            a,b=unique
            self.assertEqual({k:v for k,v in a.items() if k!="rule"},{k:v for k,v in b.items() if k!="rule"})
            self.assertEqual(b["rule"],{"op":"not","args":[a["rule"]]})


if __name__=="__main__":unittest.main()
