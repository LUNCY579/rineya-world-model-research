import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from factory.common import record_hash, load_jsonl, ROOT
from factory.pipeline import schema_validate, expected_target
from factory.reference import solve


class GateTests(unittest.TestCase):
    def setUp(self):
        self.row=load_jsonl(ROOT/"data/RCM-001/candidate_renderings/pilot-v1.jsonl")[0]

    def test_rendered_payload_is_exact_semantics(self):
        schema_validate(self.row)
        wrong=copy.deepcopy(self.row)
        wrong["rendered_input"]=wrong["rendered_input"].replace('"amount":', '"untrusted_amount":',1)
        wrong["record_hash"]=record_hash(wrong)
        with self.assertRaises(AssertionError):schema_validate(wrong)

    def test_detects_hash_tamper(self):
        wrong=copy.deepcopy(self.row);wrong["verified_target"]["after"]["x"]+=1
        with self.assertRaises(AssertionError):schema_validate(wrong)

    def test_schema_rejects_invalid_difficulty(self):
        from jsonschema import ValidationError
        wrong=copy.deepcopy(self.row);wrong["difficulty"]=99;wrong["record_hash"]=record_hash(wrong)
        with self.assertRaises(ValidationError):schema_validate(wrong)

    def test_answer_only_is_matched_to_full_oracle(self):
        self.assertEqual(expected_target(self.row,solve),self.row["verified_target"])

    def test_explicit_trace_reversed_input_semantics(self):
        trace=load_jsonl(ROOT/"data/RCM-001/candidate_renderings/pilot-v1.jsonl")[1]
        self.assertEqual(self.row["semantic_task"],trace["semantic_task"])
        self.assertEqual(self.row["verified_target"]["after"],trace["verified_target"]["after"])
        self.assertEqual(self.row["split"],trace["split"])


if __name__=="__main__":unittest.main()
