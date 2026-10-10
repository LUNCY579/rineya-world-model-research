#!/usr/bin/env python3
"""Independent actual corruption/scope tests for the new archival runner."""
import gzip
import copy
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
from contextlib import redirect_stderr

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

    def actual_rows(self):
        adapter=json.loads((LOGS/"RGM_RENDERER_MILESTONE_CUSTODY_ADAPTER.json").read_bytes())
        return adapter["protected_git_byte_baselines"]

    def original_path(self):
        return LOGS.parent.parent/"rineya-world-model-research"

    def test_actual_direct_nested_and_join_custody_schemas(self):
        original=self.original_path()
        direct=json.loads((LOGS/"RGM_RENDERER_MILESTONE_CUSTODY_ADAPTER.json").read_bytes())
        nested=json.loads((LOGS/"RGM_V2_PROVISIONAL_CHECKPOINT_READINESS.json").read_bytes())
        joined=json.loads((LOGS/"RGM_V2_RENDERER_FINAL_REVIEW_JOINED_CUSTODY.json").read_bytes())
        normal=r.normalize_baselines(original,direct)
        self.assertEqual(normal,r.normalize_baselines(original,nested))
        self.assertEqual(normal,r.normalize_baselines(original,joined))
        self.assertEqual(len(normal),6)

    def test_literal_list_and_order_normalize_exactly(self):
        rows=self.actual_rows()
        self.assertEqual(r.normalize_baselines(self.original_path(),rows),
                         r.normalize_baselines(self.original_path(),list(reversed(rows))))

    def test_dual_schema_agreement_and_conflict_rejected(self):
        rows=self.actual_rows()
        both={"protected_git_byte_baselines":rows,"original":{"protected_git_byte_baselines_match":copy.deepcopy(rows)}}
        self.assertEqual(len(r.normalize_baselines(self.original_path(),both)),6)
        both["original"]["protected_git_byte_baselines_match"][0]["sha256"]="0"*64
        with self.assertRaises(ValueError):r.normalize_baselines(self.original_path(),both)
        both={"protected_git_byte_baselines":rows,"original":[]}
        with self.assertRaises(ValueError):r.normalize_baselines(self.original_path(),both)

    def test_missing_and_wrong_count_custody_rejected(self):
        rows=self.actual_rows()
        for value in ({},None,False,{"original":{}},rows[:-1],rows+[rows[0]]):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                r.normalize_baselines(self.original_path(),value)
        altered=copy.deepcopy(rows);del altered[0]["sha256"]
        with self.assertRaises(ValueError):r.normalize_baselines(self.original_path(),altered)

    def test_custody_record_types_and_duplicate_labels_rejected(self):
        original=self.original_path();rows=self.actual_rows()
        for wrong in ("not records",{},False):
            with self.assertRaises(ValueError):r.normalize_baselines(original,{"protected_git_byte_baselines":wrong})
        changes=[(0,None),(0,{"label":[]}),(0,{"label":True}),(0,{"label":"unknown"})]
        for i,value in changes:
            altered=copy.deepcopy(rows);altered[i]=value
            with self.subTest(value=value), self.assertRaises(ValueError):r.normalize_baselines(original,altered)
        altered=copy.deepcopy(rows);altered[1]=altered[0]
        with self.assertRaises(ValueError):r.normalize_baselines(original,altered)

    def test_custody_path_and_existence_types_rejected(self):
        rows=self.actual_rows();original=self.original_path()
        changes=[("path",str(self.root/"other/.git/config")),("path",str(original/".git/../.git/config")),
                 ("expected_exists",1),("expected_exists",None),("expected_exists",False)]
        for key,value in changes:
            altered=copy.deepcopy(rows);altered[0][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):r.normalize_baselines(original,altered)

    def test_exact_hash_and_observed_hash_rejected(self):
        rows=self.actual_rows();original=self.original_path()
        for key,value in [("sha256","0"*64),("sha256",rows[0]["sha256"].upper()),("sha256",42),
                          ("actual_sha256","0"*64),("actual_sha256",None)]:
            altered=copy.deepcopy(rows);altered[0][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):r.normalize_baselines(original,altered)

    def test_absent_baseline_cannot_gain_hash_or_file_observation(self):
        rows=self.actual_rows();original=self.original_path()
        absent=next(i for i,x in enumerate(rows) if x["label"]=="original_worktree_config")
        for key,value in [("sha256","0"*64),("actual_sha256","0"*64),("expected_exists",True),("actual_exists",True)]:
            altered=copy.deepcopy(rows);altered[absent][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):r.normalize_baselines(original,altered)

    def test_observation_flags_must_be_literal_true_or_matching_bool(self):
        rows=self.actual_rows();original=self.original_path()
        for key,value in [("matches_sparse_restore_baseline",False),("matches_sparse_restore_baseline",1),
                          ("matches_frozen_baseline",False),("matches_frozen_baseline",1),
                          ("actual_exists",False),("actual_exists",1)]:
            altered=copy.deepcopy(rows);altered[0][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):r.normalize_baselines(original,altered)

    def test_original_custody_checks_actual_fixture_bytes_and_absence(self):
        original=self.root/"original";original.mkdir()
        expectations={};rows=[]
        for label,(filename,exists,_) in r.PROTECTED_EXPECTATIONS.items():
            checksum=None
            if exists:
                payload=("exact own fixture "+label+"\n").encode()
                self.write(original,".git/"+filename,payload);checksum=hashlib.sha256(payload).hexdigest()
            expectations[label]=(filename,exists,checksum)
            row={"label":label,"path":str(original/".git"/filename),"expected_exists":exists}
            if exists:row["sha256"]=checksum
            rows.append(row)
        def own_git_metadata(root,*args):
            return {("rev-parse","HEAD"):r.ORIGINAL_HEAD,("branch","--show-current"):"main",
                    ("status","--porcelain=v1","--untracked-files=all"):""}[args]
        with patch.object(r,"PROTECTED_EXPECTATIONS",expectations),patch.object(r,"text",side_effect=own_git_metadata):
            r.original_custody(original,{"original":{"protected_git_byte_baselines_match":rows}})
            self.write(original,".git/config",b"altered actual original bytes")
            with self.assertRaises(ValueError):r.original_custody(original,rows)
            self.write(original,".git/config",b"exact own fixture common_config\n")
            self.write(original,".git/config.worktree",b"unexpected config")
            with self.assertRaises(ValueError):r.original_custody(original,rows)

    def test_expected_parent_requires_full_explicit_lowercase_commit(self):
        valid="bc5198e11273cb6ea69659b6d3a8b5dc312b6f00"
        self.assertEqual(r.parent_sha(valid),valid)
        for value in (valid[:8],valid.upper(),"HEAD","main","refs/heads/main","0"*39,None,True):
            with self.subTest(value=value), self.assertRaises(ValueError):r.parent_sha(value)

    def cli_arguments(self):
        args=[]
        for name in ("spec","source-freeze","custody-baseline","receipt","local-commit-receipt","ledger"):
            args += ["--"+name,str(self.root/(name+".json"))]
        for name in ("expected-spec-sha256","expected-freeze-sha256","expected-custody-sha256"):
            args += ["--"+name,"0"*64]
        return args+["--message","Own isolated parser test","--runner","/root"]

    def test_cli_parent_is_required_without_default(self):
        args=self.cli_arguments()
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as missing:r.parse_args(args)
        self.assertEqual(missing.exception.code,2)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):r.parse_args(args+["--expected-parent","HEAD"])
        actual=r.parse_args(args+["--expected-parent","bc5198e11273cb6ea69659b6d3a8b5dc312b6f00"])
        self.assertEqual(actual.expected_parent,"bc5198e11273cb6ea69659b6d3a8b5dc312b6f00")
        self.assertFalse(actual.execute)

    def test_bound_parent_checks_actual_git_head(self):
        repo=self.repo();actual=r.text(repo,"rev-parse","HEAD")
        r.require_head(repo,actual)
        with self.assertRaises(ValueError):r.require_head(repo,"0"*40)

    def committed_fixture(self):
        repo=self.repo();parent=r.text(repo,"rev-parse","HEAD");name="audit.txt"
        path=self.write(repo,r.PREFIX+"/"+name,b"frozen archival fixture\n")
        checked={name:r.proof(path)}
        r.stage_exact(repo,checked,[]);r.git(repo,"commit","--quiet","-m","Own committed archival fixture")
        return repo,parent,r.text(repo,"rev-parse","HEAD"),checked

    def test_commit_sole_parent_and_exact_tree_rechecked(self):
        repo,parent,commit,checked=self.committed_fixture()
        r.verify_committed_parent_tree(repo,commit,parent,checked)
        with self.assertRaises(ValueError):r.verify_committed_parent_tree(repo,commit,"0"*40,checked)
        with self.assertRaises(ValueError):r.verify_committed_parent_tree(repo,commit,parent,{})
        corrupt=copy.deepcopy(checked);corrupt["audit.txt"]["git_blob"]="0"*40
        with self.assertRaises(ValueError):r.verify_committed_parent_tree(repo,commit,parent,corrupt)

    def test_multi_parent_commit_rejected_with_actual_git_object(self):
        repo,parent,commit,checked=self.committed_fixture()
        tree=r.text(repo,"rev-parse",commit+"^{tree}")
        merge=r.text(repo,"commit-tree",tree,"-p",parent,"-p",commit,"-m","Own two parent rejection fixture")
        self.assertEqual(len(r.text(repo,"rev-list","--parents","-n","1",merge).split()),3)
        with self.assertRaises(ValueError):r.verify_committed_parent_tree(repo,merge,commit,checked)


if __name__ == "__main__":unittest.main(verbosity=2)
