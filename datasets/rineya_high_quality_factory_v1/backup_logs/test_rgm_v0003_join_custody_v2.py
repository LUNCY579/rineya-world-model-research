#!/usr/bin/env python3
"""Isolated metadata corruption tests using immutable completed historical proofs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
LOGS=Path(__file__).resolve().parent
FACTORY=LOGS.parent
sp=importlib.util.spec_from_file_location('new_join',LOGS/'join_rgm_v0003_independent_release_custody_v2.py')
j=importlib.util.module_from_spec(sp);sp.loader.exec_module(j)


class JoinFailClosed(unittest.TestCase):
    def setUp(self):
        def load(name):return json.loads((FACTORY/name).read_bytes())
        self.spec=load('backup_logs/rgm-v2-renderer-review-milestone_SPEC.json')
        self.stage=load('backup_logs/rgm-v2-renderer-review-milestone_STAGE_R2.json')
        self.full=load('backup_logs/rgm-v2-renderer-review-milestone_RECOVERY_VERIFIED_RECEIPT.json')
        self.gzip=load('quality_reports/RGM_V2_RENDERER_REMOTE_LOSSLESS.json')
        self.commit='bc5198e11273cb6ea69659b6d3a8b5dc312b6f00'
        self.parent='448c36af9370460782da3c0a4ba7c5ca4d8ba71a'
        self.spec_sha='ee1c503b896b5e705c181f0fadcfe92268fd9859bb55a348eab70e90d94c24f1'
        self.stage_sha='c27836578aae3b512b632f869b55f140df974e4275526d90b40de74dbd91ce3c'
        self.full_sha='146cb0b19bc1fda50ae810823457b0e3fce70dc4dddb0a1a779d790f0e173937'
        self.tree=self.full['remote_verification']['remote_tree_sha']

    def check(self):
        return j.validate_proofs(self.spec,self.stage,self.full,self.gzip,self.commit,self.parent,self.spec_sha,self.stage_sha,self.full_sha,self.tree)

    def test_completed_historical_bound_proofs_accept_without_new_fetch_claim(self):
        coverage=self.check()
        self.assertEqual(len(coverage),2095)
        self.assertEqual(self.gzip['actual_remote_gzip_fetches'],24)

    def test_pending_or_failed_full_proof_rejected(self):
        for status in ('PUSHED_REMOTE_BYTES_VERIFICATION_PENDING','FAILED_LOCAL_ARTIFACTS_PRESERVED'):
            self.full['status']=status
            with self.assertRaises(ValueError):self.check()

    def test_commit_parent_spec_stage_bindings_rejected(self):
        saved=copy.deepcopy(self.stage)
        for field,value in [('expected_parent_sha','0'*40),('commit_sha','0'*40),('remote_commit_sha','0'*40),('spec_sha256','0'*64)]:
            self.stage=copy.deepcopy(saved);self.stage[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check()

    def test_stage_guard_false_or_data_approval_rejected(self):
        saved=copy.deepcopy(self.stage)
        for field,value in [('all_source_target_index_sha256_and_sha1_verified',False),('strict_diff_check','FAIL'),('grants_data_approval',True)]:
            self.stage=copy.deepcopy(saved);self.stage[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check()

    def test_cached_local_provenance_or_cache_transport_rejected(self):
        self.full['remote_verification']['blob_receipts'][0]['verification_source']='local_git_blob'
        with self.assertRaises(ValueError):self.check()
        self.full['remote_verification']['blob_receipts'][0]['verification_source']='actual_remote_raw_GitHub_blob'
        self.full['remote_verification']['cache_policy']='inherited cache'
        with self.assertRaises(ValueError):self.check()

    def test_missing_duplicate_and_unrelated_remote_paths_rejected(self):
        saved=copy.deepcopy(self.full)
        for mutation in ('missing','duplicate','unrelated'):
            self.full=copy.deepcopy(saved);paths=self.full['remote_verification']['blob_receipts'][0]['paths']
            if mutation=='missing':paths.pop()
            elif mutation=='duplicate':paths.append(paths[0])
            else:paths.append('unrelated/not-in-spec.txt')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_remote_size_hash_tree_or_role_rejected(self):
        saved=copy.deepcopy(self.full)
        for mutation in ('size','hash','tree','runner','private'):
            self.full=copy.deepcopy(saved)
            if mutation=='size':self.full['remote_verification']['blob_receipts'][0]['size']+=1
            elif mutation=='hash':self.full['remote_verification']['blob_receipts'][0]['sha256']='0'*64
            elif mutation=='tree':self.full['remote_verification']['remote_tree_sha']='0'*40
            elif mutation=='runner':self.full['runner_execution']='/root/producer'
            else:self.full['private_repository_evidence']['isPrivate']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_gzip_pending_old_receipt_binding_or_source_rejected(self):
        saved=copy.deepcopy(self.gzip)
        for field,value in [('status','PENDING'),('commit_sha','0'*40),('stage_receipt_sha256','0'*64),('verified_inventory_receipt_sha256','0'*64),('source_sha256','0'*64),('source_freeze_sha256','0'*64)]:
            self.gzip=copy.deepcopy(saved);self.gzip[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check()

    def test_new_actual_fetch_and_byte_for_byte_flags_required(self):
        saved=copy.deepcopy(self.gzip)
        for field in ('actual_remote_fetch_this_run','decoded_byte_for_byte_equal_to_preserved_raw'):
            self.gzip=copy.deepcopy(saved);self.gzip['actual_remote_gzip_proofs'][0][field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):self.check()

    def test_exact24_fetches_raw_coverage_and_total_required(self):
        saved=copy.deepcopy(self.gzip)
        for mutation in ('count','rawbytes','missing','duplicate','representation'):
            self.gzip=copy.deepcopy(saved)
            if mutation=='count':self.gzip['actual_remote_gzip_fetches']=23
            elif mutation=='rawbytes':self.gzip['raw_bytes_exactly_recovered']-=1
            elif mutation=='missing':self.gzip['actual_remote_gzip_proofs'].pop()
            elif mutation=='duplicate':self.gzip['actual_remote_gzip_proofs'][0]=self.gzip['actual_remote_gzip_proofs'][1]
            else:self.gzip['actual_remote_gzip_proofs'][0]['remote_gzip_path']='README.md'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check()

    def test_raw_local_roundtrip_or_evolving_source_rejected(self):
        self.full['oversized_raw_local_verification'][0]['lossless_local_roundtrip']=False
        with self.assertRaises(ValueError):self.check()
        self.full['oversized_raw_local_verification'][0]['lossless_local_roundtrip']=True
        self.full['current_source_evolution']=[{'path':'changed_source.py'}]
        with self.assertRaises(ValueError):self.check()

    def test_actual_historical_ledgers_and_missing_final_event(self):
        def lines(name):return [json.loads(x) for x in (LOGS/name).read_text().splitlines()]
        full=lines('rgm-v2-renderer-review-milestone_REMOTE_LEDGER.jsonl')
        gzip=lines('rgm-v2-renderer-review-milestone_REMOTE_LOSSLESS_LEDGER.jsonl')
        j.validate_ledgers(self.full,self.gzip,full,gzip)
        with self.assertRaises(ValueError):j.validate_ledgers(self.full,self.gzip,full[:-1],gzip)
        with self.assertRaises(ValueError):j.validate_ledgers(self.full,self.gzip,full,gzip[:-1])

    def test_explicit_proof_hash_types_rejected(self):
        self.assertEqual(j.sha64('0'*64),'0'*64)
        for value in ('0'*63,'ABCDEF'*11,'HEAD',None,True):
            with self.subTest(value=value),self.assertRaises(ValueError):j.sha64(value)

    def test_root_authorized_current_proof_filename_and_freeze_binding(self):
        self.assertEqual(j.GZIP,'quality_reports/RGM_V0003_REMOTE_LOSSLESS.json')
        self.assertEqual(j.FREEZE,'backup_logs/RGM_V0003_JOIN_CUSTODY_V2_SOURCE_FREEZE.json')

    def test_current_completed_actual_proofs_and_ledgers_accept(self):
        def load(name):return json.loads((FACTORY/name).read_bytes())
        def digest(name):
            import hashlib
            return hashlib.sha256((FACTORY/name).read_bytes()).hexdigest()
        spec=load(j.SPEC);stage=load(j.STAGE);full=load(j.FULL);gzip=load(j.GZIP)
        self.assertEqual(digest(j.FULL),'83f5ad7628b3748504a8dac73fafb045d8c7f7acdb5c7d1c481205741d092f2f')
        self.assertEqual(digest(j.GZIP),'21912e37c1e959e009a0eb80497be30a213a85d813e71d330868eff12d853de1')
        coverage=j.validate_proofs(spec,stage,full,gzip,j.COMMIT,j.PARENT,j.SPEC_SHA,j.STAGE_SHA,digest(j.FULL),'37a927fb912199351bc6dcc3683c7860403e1ce4')
        self.assertEqual(len(coverage),2303)
        with Path(full['ledger']).open() as stream:fl=[json.loads(x) for x in stream]
        with (FACTORY/j.GZIP_LEDGER).open() as stream:gl=[json.loads(x) for x in stream]
        j.validate_ledgers(full,gzip,fl,gl)
        self.assertEqual(full['remote_verification']['verified_unique_blobs'],1895)
        self.assertEqual(gzip['actual_remote_gzip_fetches'],24)
        self.assertEqual(gzip['raw_bytes_exactly_recovered'],1188634377)



if __name__=='__main__':unittest.main(verbosity=2)
