"""Owned handmade preflight tests; do not read any production data."""
from pathlib import Path
import copy
import importlib.util
import unittest

PATH=Path(__file__).resolve().parents[1]/"reviews/adjudication/adjudicate_planning_gap_v2_renderer_r1.py"
SPEC=importlib.util.spec_from_file_location("owned_r1_actual",PATH)
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)


class Preflight(unittest.TestCase):
 def setUp(self):
  self.bindings={p:{"path":p,"sha256":"a"*64,"bytes":2} for p in [A.CANDIDATE,A.MANIFEST,A.SOURCE]}
  self.candidate=[{"sample_id":"hand-one","record_hash":"a"*64}]
  self.receipts=[{"receipt_id":"hand-r1","sample_id":"hand-one","record_hash":"a"*64,
       "reviewer_role":"independent_adjudicator","reviewer_execution_id":A.IDENTITY,"decision":"ACCEPT"}]

 def test_complete_bindings_before_any_output(self):
  A.output_preflight(self.bindings,list(self.bindings),[])

 def test_missing_manifest_rejected_before_writes(self):
  del self.bindings[A.MANIFEST]
  with self.assertRaises(AssertionError):A.output_preflight(self.bindings,[A.CANDIDATE,A.MANIFEST,A.SOURCE],[])

 def test_manifest_wrong_path_rejected_before_writes(self):
  self.bindings[A.MANIFEST]["path"]="different"
  with self.assertRaises(AssertionError):A.output_preflight(self.bindings,list(self.bindings),[])

 def test_duplicate_output_paths_rejected_before_writes(self):
  with self.assertRaises(AssertionError):A.output_preflight(self.bindings,list(self.bindings),["same","same"])

 def test_frozen_source_cannot_be_output(self):
  with self.assertRaises(AssertionError):A.output_preflight(self.bindings,list(self.bindings),[A.SOURCE])

 def test_exact_receipt_join(self):
  A.receipt_preflight(self.candidate,self.receipts)

 def test_foreign_hash_rejected(self):
  self.receipts[0]["record_hash"]="b"*64
  with self.assertRaises(AssertionError):A.receipt_preflight(self.candidate,self.receipts)

 def test_wrong_actor_and_role_rejected(self):
  for key,value in [("reviewer_execution_id","/root"),("reviewer_role","producer")]:
   receipts=copy.deepcopy(self.receipts);receipts[0][key]=value
   with self.assertRaises(AssertionError):A.receipt_preflight(self.candidate,receipts)

 def test_duplicate_receipts_or_missing_record_rejected(self):
  with self.assertRaises(AssertionError):A.receipt_preflight(self.candidate,self.receipts*2)
  with self.assertRaises(AssertionError):A.receipt_preflight(self.candidate,[])


if __name__=="__main__":unittest.main()
