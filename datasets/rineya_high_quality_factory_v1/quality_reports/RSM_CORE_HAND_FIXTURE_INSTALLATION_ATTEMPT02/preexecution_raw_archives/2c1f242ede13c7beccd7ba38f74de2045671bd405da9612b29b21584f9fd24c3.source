import collections
import unittest
from factory.common import ROOT,load_jsonl


class MetadataCrossing(unittest.TestCase):
    def test_abstention_crossed_with_every_request_id_within_family(self):
        groups=collections.defaultdict(lambda:collections.defaultdict(set))
        for r in load_jsonl(ROOT/"data/RMM-001/candidate_renderings/extended-pilot-v3.jsonl"):
            for result in r["verified_target"]["routes"]:groups[r["semantic_parent_id"]][result["request_id"]].add(result["selected_module"] is None)
        self.assertEqual(len(groups),12)
        for ids in groups.values():
            for labels in ids.values():self.assertEqual(labels,{False,True})

    def test_retrieval_status_distribution_same_for_each_query_id(self):
        groups=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
        for r in load_jsonl(ROOT/"data/REM-001/candidate_renderings/extended-pilot-v3.jsonl"):
            if r["renderer"]!="forward":continue
            for result in r["verified_target"]["results"]:groups[r["semantic_parent_id"]][result["query_id"]][result["status"]]+=1
        self.assertEqual(len(groups),12)
        for ids in groups.values():self.assertEqual(len({tuple(sorted(c.items())) for c in ids.values()}),1)


if __name__=="__main__":unittest.main()
