import collections
import unittest
from factory.common import ROOT,load_jsonl


class MatchedGoalRegression(unittest.TestCase):
    def test_development_goals_balanced_within_every_family(self):
        rows=load_jsonl(ROOT/"data/RGM-001/candidate_renderings/pilot-v3.jsonl")
        groups=collections.defaultdict(set)
        for r in rows:
            if r["split"]=="dev_challenge":groups[r["semantic_parent_id"]].add(r["verified_target"]["reaches_goal"])
        self.assertEqual(len(groups),6)
        for labels in groups.values():self.assertEqual(labels,{False,True})

    def test_shortest_plan_horizon_in_difficulty(self):
        for r in load_jsonl(ROOT/"data/RGM-001/candidate_renderings/pilot-v3.jsonl"):
            if len(r["verified_target"]["shortest_plan"] or [])>2:self.assertGreaterEqual(r["difficulty"],4)


if __name__=="__main__":unittest.main()
