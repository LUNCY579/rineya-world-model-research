#!/usr/bin/env python3
"""Isolated corruption/binding tests; no actual network payload execution."""
import copy
import gzip
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("generic_lossless", LOGS / "verify_archival_remote_lossless.py")
r = importlib.util.module_from_spec(s);s.loader.exec_module(r)
b = r.load_bounded()
COMMIT = "a"*40
SPEC_SHA = "b"*64


def entry(name, data):
    return {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def object_id(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()


class RemoteLosslessSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="remote-lossless-test-", dir=LOGS)
        self.factory = Path(self.tmp.name);(self.factory / "backup_logs").mkdir()
        self.data = b"ordinary raw evidence\n"*100
        self.packed_data = gzip.compress(self.data,mtime=0)
        self.raw = entry("audit.json",self.data);self.packed = entry("audit.json.gz",self.packed_data)
        self.manifest = entry("audit.lossless.json",b"{}")
        self.raw_path = self.factory / "audit.json";self.raw_path.write_bytes(self.data)
        (self.factory / "audit.json.gz").write_bytes(self.packed_data)
        self.blob = {"path":b.PREFIX+"/audit.json.gz", "type":"blob", "mode":"100644", "size":len(self.packed_data), "sha":object_id(self.packed_data)}

    def tearDown(self):
        self.tmp.cleanup()  # Only this test's new owned fixtures.

    def consume(self,payload=None,packed=None,blob=None):
        return r.consume_remote(io.BytesIO(self.packed_data if payload is None else payload),self.raw_path,self.raw,
                                self.packed if packed is None else packed,self.blob if blob is None else blob,self.factory / "backup_logs")

    def bindings(self):
        files=[self.packed,self.manifest]
        raw={**self.raw,"lossless_representations":[{"gzip":self.packed,"manifest":self.manifest}]}
        spec={"approval":"provisional_checkpoint","approved_by":"/root","validation_status":"passed","files":files,"raw_size_exclusions":[raw]}
        stage={"status":"PUSHED_REMOTE_BYTES_VERIFICATION_PENDING","commit_sha":COMMIT,"remote_commit_sha":COMMIT,
               "spec_sha256":SPEC_SHA,"all_source_target_index_sha256_and_sha1_verified":True}
        receipts=[]
        for e,data in ((self.packed,self.packed_data),(self.manifest,b"{}")):
            receipts.append({"git_blob":object_id(data),"size":e["size"],"sha256":e["sha256"],"paths":[e["path"]],"verification_source":"actual_remote_raw_GitHub_blob"})
        verified={"status":"BACKED_UP","commit_sha":COMMIT,"remote_commit_sha":COMMIT,"spec_sha256":SPEC_SHA,"inventory":files,"files":2,
                  "total_bytes":sum(e["size"] for e in files),"verification_source_sha256":r.DEPENDENCY_SHA,
                  "remote_verification":{"verified_files":2,"verified_unique_blobs":2,"blob_receipts":receipts},
                  "oversized_raw_local_verification":[{**self.raw,"lossless_local_roundtrip":True}]}
        return spec,stage,verified

    def test_valid_actual_byte_consumer(self):
        p=self.consume();self.assertEqual(p["raw_bytes"],len(self.data));self.assertEqual(p["remote_git_blob"],self.blob["sha"])
        self.assertTrue(p["decoded_byte_for_byte_equal_to_preserved_raw"])

    def test_truncated_excess_sha256_and_git_sha1_rejected(self):
        for payload,packed,blob in [(self.packed_data[:-1],self.packed,self.blob),(self.packed_data+b"x",self.packed,self.blob),
                                    (self.packed_data,{**self.packed,"sha256":"0"*64},self.blob),
                                    (self.packed_data,self.packed,{**self.blob,"sha":"0"*40})]:
            with self.subTest(blob=blob["sha"]), self.assertRaises(ValueError):self.consume(payload,packed,blob)

    def test_local_raw_mismatch_cannot_use_good_local_gzip(self):
        self.raw_path.write_bytes(b"X"+self.data[1:])
        with self.assertRaises(ValueError):self.consume()
        self.assertEqual((self.factory/"audit.json.gz").read_bytes(),self.packed_data)

    def test_wrong_decoded_content_and_crc_rejected_after_compressed_binding(self):
        for payload in (gzip.compress(b"wrong"),self.packed_data[:-8]+bytes([self.packed_data[-8]^1])+self.packed_data[-7:]):
            packed=entry("audit.json.gz",payload);blob={**self.blob,"sha":object_id(payload),"size":len(payload)}
            with self.assertRaises((ValueError,gzip.BadGzipFile)):self.consume(payload,packed,blob)

    def test_preserved_raw_stat_change_rejected(self):
        path=self.raw_path;before=path.stat()
        class Mutating(io.BytesIO):
            def read(self,size=-1):
                os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
                return super().read(size)
        with self.assertRaises(ValueError):r.consume_remote(Mutating(self.packed_data),path,self.raw,self.packed,self.blob,self.factory/"backup_logs")

    def test_spool_rollover_and_remote_reads_remain_bounded(self):
        data=os.urandom(32768);self.raw_path.write_bytes(data);raw=entry("audit.json",data);payload=gzip.compress(data);packed=entry("audit.json.gz",payload)
        blob={**self.blob,"size":len(payload),"sha":object_id(payload)}
        class Limited(io.BytesIO):
            def read(self,size=-1):
                if size<0 or size>r.CHUNK:raise AssertionError("Unbounded remote read")
                return super().read(size)
        with patch.object(r,"SPOOL_MEMORY",32),patch.object(r,"CHUNK",1024):
            proof=r.consume_remote(Limited(payload),self.raw_path,raw,packed,blob,self.factory/"backup_logs")
        self.assertEqual(proof["raw_bytes"],len(data));self.assertEqual(list((self.factory/"backup_logs").iterdir()),[])

    def test_wrong_commit_stage_and_full_proof_bindings_rejected(self):
        spec,stage,verified=self.bindings()
        r.validate_bindings(spec,stage,verified,COMMIT,SPEC_SHA,b)
        for field,value in (("commit_sha","0"*40),("status","LOCAL_COMMIT_PRESERVED"),("all_source_target_index_sha256_and_sha1_verified",False)):
            with self.subTest(field=field),self.assertRaises(ValueError):r.validate_bindings(spec,{**stage,field:value},verified,COMMIT,SPEC_SHA,b)
        for field,value in (("spec_sha256","0"*64),("status","PENDING"),("verification_source_sha256","0"*64),("files",3)):
            with self.subTest(field=field),self.assertRaises(ValueError):r.validate_bindings(spec,stage,{**verified,field:value},COMMIT,SPEC_SHA,b)

    def test_missing_actual_fetch_provenance_and_coverage_rejected(self):
        spec,stage,verified=self.bindings()
        for mutation in ("provenance","omitted","duplicate"):
            q=copy.deepcopy(verified)
            if mutation=="provenance":q["remote_verification"]["blob_receipts"][0]["verification_source"]="local_git_cache"
            if mutation=="omitted":q["remote_verification"]["blob_receipts"][0]["paths"]=[]
            if mutation=="duplicate":q["remote_verification"]["blob_receipts"][1]["paths"]=[self.packed["path"]]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.validate_bindings(spec,stage,q,COMMIT,SPEC_SHA,b)

    def test_actual_tree_mode_size_blob_and_count_guards(self):
        spec,stage,verified=self.bindings();inventory,actual=r.validate_bindings(spec,stage,verified,COMMIT,SPEC_SHA,b)
        paths={self.blob["path"]:self.blob}
        with patch.object(b,"MAX_BYTES",1024):
            self.assertEqual(len(r.plan_representations(spec,inventory,actual,paths,b.PREFIX,1,b)),1)
            with self.assertRaises(ValueError):r.plan_representations(spec,inventory,actual,paths,b.PREFIX,24,b)
            for key,value in (("mode","120000"),("size",0),("sha","0"*40)):
                with self.subTest(key=key),self.assertRaises(ValueError):r.plan_representations(spec,inventory,actual,{self.blob["path"]:{**self.blob,key:value}},b.PREFIX,1,b)

    def test_fetch_route_is_actual_raw_api_and_has_no_local_fallback(self):
        calls=[]
        class Transport:
            REPOSITORY=b.REPOSITORY
            source_path=staticmethod(b.source_path)
            @staticmethod
            def process_stream(args,consume):
                calls.append(args);return consume(io.BytesIO(self.packed_data[:-1]))
        with self.assertRaises(ValueError):r.fetch_one(Transport,self.factory,self.raw,self.packed,self.blob)
        self.assertEqual(len(calls),1);args=calls[0]
        self.assertEqual(args[:2],["gh","api"]);self.assertIn("Accept: application/vnd.github.raw+json",args)
        self.assertIn("Cache-Control: no-cache",args);self.assertNotIn("--cache",args)
        self.assertTrue(args[-1].endswith("/git/blobs/"+self.blob["sha"]))


if __name__ == "__main__":unittest.main(verbosity=2)
