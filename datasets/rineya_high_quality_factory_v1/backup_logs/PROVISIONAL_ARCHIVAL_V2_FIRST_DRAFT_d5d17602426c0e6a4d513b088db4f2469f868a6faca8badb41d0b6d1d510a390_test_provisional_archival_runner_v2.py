#!/usr/bin/env python3
"""Independent actual corruption/scope tests for the new archival runner."""
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
LOGS = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("archival_runner", LOGS / "run_provisional_archival_checkpoint_v2.py")
r = importlib.util.module_from_spec(sp);sp.loader.exec_module(r)


def entry(root, name):
    p = root / name
    q = r.proof(p)
    return {"path": name, "size": q["size"], "sha256": q["sha256"]}


class ArchivalSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="archival-runner-test-", dir=LOGS)
        self.root = Path(self.tmp.name)
        self.factory = self.root / "factory";self.factory.mkdir()

    def tearDown(self):
        self.tmp.cleanup()  # Only this test's newly created fixtures.

    def write(self, root, name, data):
        p = root / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(data);return p

    def spec(self, files):
        return {"approval": "provisional_checkpoint", "approved_by": "/root", "validation_status": "passed", "files": files}

    def repo(self):
        p = self.root / "repo";p.mkdir()
        r.git(p, "init", "--quiet");r.git(p, "config", "user.name", "Archival safety test")
        r.git(p, "config", "user.email", "archival-test@example.invalid")
        self.write(p, ".gitignore", b"*.bin\n")
        r.git(p, "add", ".gitignore");r.git(p, "commit", "--quiet", "-m", "Own isolated test fixture")
        return p

    def test_path_traversal_nul_and_symlinks_rejected(self):
        self.write(self.factory, "good.txt", b"good")
        for name in ("../good.txt", "/good.txt", "./good.txt", "x\0y", "x\ny", ".git/config"):
            with self.subTest(name=name), self.assertRaises(ValueError):r.safe(self.factory, name)
        (self.factory / "link").symlink_to(self.factory / "good.txt")
        with self.assertRaises(ValueError):r.safe(self.factory, "link")

    def test_duplicate_hash_and_cap_corruption_rejected(self):
        self.write(self.factory, "sample.txt", b"abc");e = entry(self.factory, "sample.txt")
        with self.assertRaises(ValueError):r.inventory(self.factory, self.spec([e,e]))
        altered = {**e, "sha256": "0"*64}
        with self.assertRaises(ValueError):r.inventory(self.factory, self.spec([altered]))
        with patch.object(r, "MAX_BYTES", 2), self.assertRaises(ValueError):r.inventory(self.factory, self.spec([e]))

    def test_secret_crosses_stream_chunk_boundary(self):
        data = b"normal " + b"gh" + b"p_" + b"Q"*30
        with patch.object(r, "CHUNK", 8), self.assertRaises(ValueError):r.stream_proof(io.BytesIO(data), len(data))

    def test_stream_actual_git_blob_sha1_and_truncation(self):
        data = b"verified bytes"
        actual = r.stream_proof(io.BytesIO(data), len(data))
        self.assertEqual(actual["git_blob"], hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest())
        with self.assertRaises(ValueError):r.stream_proof(io.BytesIO(data[:-1]), len(data))
        with self.assertRaises(ValueError):r.stream_proof(io.BytesIO(data+b"x"), len(data))

    def test_release_guard_and_old_artifact_omission(self):
        with self.assertRaises(ValueError):r.guard_previous({"data/X/releases/v1/a": "old"}, {"data/X/releases/v1/a": {"git_blob": "new"}})
        with self.assertRaises(ValueError):r.guard_previous({"old.txt":"old"}, {})
        r.guard_previous({"data/X/releases/v1/a":"same"}, {"data/X/releases/v1/a":{"git_blob":"same"}})

    def test_copy_corruption_preserves_old_target(self):
        self.write(self.factory, "a.txt", b"original")
        frozen = r.proof(self.factory / "a.txt")
        repo = self.root / "copy";repo.mkdir()
        target = self.write(repo, r.PREFIX+"/a.txt", b"prior")
        self.write(self.factory, "a.txt", b"mutated!")
        with self.assertRaises(ValueError):r.copy_one(self.factory, repo, "a.txt", frozen)
        self.assertEqual(target.read_bytes(), b"prior")
        self.assertEqual(list(target.parent.glob(".archival-copy-*")), [])

    def lossless_fixture(self, alternate=False):
        raw = self.write(self.factory, "audit.json", b"ordinary evidence\n"*100)
        gz = self.write(self.factory, "audit.json.gz", gzip.compress(raw.read_bytes(),mtime=0))
        rp = r.proof(raw);gp = r.proof(gz)
        record = {"raw_path":"audit.json", "raw_sha256":rp["sha256"], "raw_bytes":rp["size"],
                  "gzip_path":"audit.json.gz", "gzip_sha256":gp["sha256"], "gzip_bytes":gp["size"]}
        if alternate:
            for a,b in (("raw_bytes","raw_length"),("gzip_path","compressed_path"),("gzip_bytes","compressed_length"),("gzip_sha256","compressed_sha256")):
                record[b]=record.pop(a)
            record={"items":[record]}
        self.write(self.factory, "audit.lossless.json", json.dumps(record).encode())
        checked={p:r.proof(self.factory/p) for p in ("audit.json.gz","audit.lossless.json")}
        rep={key:entry(self.factory,name) for key,name in (("gzip","audit.json.gz"),("manifest","audit.lossless.json"))}
        spec={"raw_size_exclusions":[{**entry(self.factory,"audit.json"),"lossless_representations":[rep]}]}
        return spec,checked

    def test_standard_and_nested_expert_lossless_manifests(self):
        for alternate in (False,True):
            spec,checked=self.lossless_fixture(alternate)
            with patch.object(r,"MAX_BYTES",1024):self.assertEqual(len(r.lossless(self.factory,spec,checked)),1)

    def test_unpaired_oversized_and_corrupt_gzip_rejected(self):
        spec,checked=self.lossless_fixture()
        with patch.object(r,"MAX_BYTES",1024), self.assertRaises(ValueError):r.lossless(self.factory,{},checked)
        self.write(self.factory,"audit.json.gz",gzip.compress(b"wrong raw"))
        with patch.object(r,"MAX_BYTES",1024), self.assertRaises(ValueError):r.lossless(self.factory,spec,checked)

    def test_literal_glob_and_exact_ignored_staging(self):
        repo=self.repo();name="literal[abc].txt";ignored="backup_logs/custody.bin"
        self.write(repo,r.PREFIX+"/"+name,b"literal bytes\n");self.write(repo,r.PREFIX+"/"+ignored,b"exact custody bytes")
        checked={n:r.proof(repo/r.PREFIX/n) for n in (name,ignored)}
        with self.assertRaises(ValueError):r.stage_exact(repo,checked,[])
        report=r.stage_exact(repo,checked,[ignored])
        self.assertEqual(report["new_ignored_exact_paths_force_staged"],[ignored])
        self.assertEqual(set(r.index(repo)),set(checked))

    def test_already_tracked_ignored_needs_no_force(self):
        repo=self.repo();name="backup_logs/custody.bin";p=self.write(repo,r.PREFIX+"/"+name,b"old")
        r.git(repo,"--literal-pathspecs","add","--force","--",r.PREFIX+"/"+name);r.git(repo,"commit","--quiet","-m","Frozen prior tracked custody")
        p.write_bytes(b"new exact bytes")
        report=r.stage_exact(repo,{name:r.proof(p)},[])
        self.assertEqual(report["new_ignored_exact_paths_force_staged"],[])

    def test_unrelated_stage_and_whitespace_rejected(self):
        repo=self.repo();name="audit.txt";p=self.write(repo,r.PREFIX+"/"+name,b"good\n")
        self.write(repo,"unrelated.txt",b"unrelated\n");r.git(repo,"add","unrelated.txt")
        with self.assertRaises(ValueError):r.stage_exact(repo,{name:r.proof(p)},[])
        r.git(repo,"restore","--staged","unrelated.txt")
        p.write_bytes(b"bad trailing space \n")
        with self.assertRaises(RuntimeError):r.stage_exact(repo,{name:r.proof(p)},[])


if __name__ == "__main__":unittest.main(verbosity=2)
