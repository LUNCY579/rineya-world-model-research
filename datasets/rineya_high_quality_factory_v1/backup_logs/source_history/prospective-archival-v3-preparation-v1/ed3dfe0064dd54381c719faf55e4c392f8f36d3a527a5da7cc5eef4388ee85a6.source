#!/usr/bin/env python3
"""Independent actual corruption/scope tests for the new archival runner."""
import gzip
import base64
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
sp = importlib.util.spec_from_file_location("archival_runner", LOGS / "run_provisional_archival_checkpoint_v3.py")
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
        for name in ("spec","source-freeze","custody-baseline","exception-config","execution-config","receipt","local-commit-receipt","ledger"):
            args += ["--"+name,str(self.root/(name+".json"))]
        for name in ("expected-spec-sha256","expected-freeze-sha256","expected-custody-sha256","expected-exception-sha256","expected-execution-config-sha256"):
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


class ExactSixWarningSafety(unittest.TestCase):
    # Reuse own-fixture utilities without running duplicate inherited tests.
    setUp=ArchivalSafety.setUp
    tearDown=ArchivalSafety.tearDown
    write=ArchivalSafety.write
    repo=ArchivalSafety.repo
    original_path=ArchivalSafety.original_path
    def actual_config(self):
        return json.loads((LOGS/"PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json").read_bytes())

    def fixture_repo(self):
        repo=self.repo();checked={}
        for row in self.actual_config()["files"]:
            payload=(LOGS.parent/row["path"]).read_bytes()
            self.write(self.factory,row["path"],payload)
            path=self.write(repo,r.PREFIX+"/"+row["path"],payload);checked[row["path"]]=r.proof(path)
        path=self.write(repo,r.PREFIX+"/ordinary.txt",b"safe new checkpoint fixture\n")
        self.write(self.factory,"ordinary.txt",path.read_bytes());checked["ordinary.txt"]=r.proof(path)
        return repo,checked

    def test_actual_exact_six_warning_git_stage_and_truthful_exit(self):
        repo,checked=self.fixture_repo();report=r.stage_exact(repo,checked,[],self.actual_config(),self.factory)
        self.assertEqual(report["actual_exit_code"],2)
        self.assertEqual(report["strict_diff_check"],"NONZERO_EXIT_2_NOT_AN_UNQUALIFIED_PASS")
        self.assertEqual(report["warning_count"],6)
        self.assertEqual(report["restricted_strict_diff_check"],"PASS")
        self.assertEqual(report["restricted_changed_path_count"],1)
        self.assertEqual(r.bounded_diff(repo)["stdout"],r.expected_diagnostics(self.actual_config()))

    def test_extra_seventh_actual_warning_rejected(self):
        repo,checked=self.fixture_repo();p=repo/r.PREFIX/"ordinary.txt";p.write_bytes(b"another warning \n")
        self.write(self.factory,"ordinary.txt",p.read_bytes());checked["ordinary.txt"]=r.proof(p)
        with self.assertRaises(r.DiffCheckRejected) as error:r.stage_exact(repo,checked,[],self.actual_config(),self.factory)
        self.assertEqual(error.exception.observation["actual_exit_code"],2)
        self.assertIn(b"ordinary.txt:1:",base64.b64decode(error.exception.observation["actual_stdout_base64"]))

    def test_warning_missing_duplicate_line_path_payload_exit_and_stderr_rejected(self):
        config=self.actual_config();wanted=r.expected_diagnostics(config)
        variations=[wanted[:-1],wanted+wanted[:wanted.find(b"\n")+1],wanted.replace(b":212:",b":211:"),
                    wanted.replace(r.WARNING_STDOUT.encode(),b"other.txt"),wanted.replace(b"++        for",b"+        for"),
                    wanted.replace(b"trailing whitespace.",b"new blank line at EOF.")]
        for value in variations:
            with self.subTest(value=value[:40]),self.assertRaises(r.DiffCheckRejected):
                r.validate_diagnostics({"returncode":2,"stdout":value,"stderr":b""},config)
        for code in (0,1,3,True):
            with self.subTest(code=code),self.assertRaises(ValueError):r.validate_diagnostics({"returncode":code,"stdout":wanted,"stderr":b""},config)
        with self.assertRaises(r.DiffCheckRejected):r.validate_diagnostics({"returncode":2,"stdout":wanted,"stderr":b"warning"},config)
        with self.assertRaises(ValueError):r.validate_diagnostics({"returncode":2,"stdout":wanted.decode(),"stderr":b""},config)

    def test_config_cannot_relax_warning_hash_size_count_type_paths_or_lines(self):
        config=self.actual_config()
        for mutate in (lambda c:c.update(warning_count=7),lambda c:c.update(required_exit_code=True),
                       lambda c:c["files"][0].update(size=1248),lambda c:c["files"][0].update(sha256="0"*64),
                       lambda c:c["files"][0].update(path="other"),lambda c:c["files"][0].update(lines=[2,4,6,8]),
                       lambda c:c["files"][1].update(lines=[211]),lambda c:c.update(required_stderr_bytes=1)):
            altered=copy.deepcopy(config);mutate(altered)
            with self.assertRaises(ValueError):r.validate_exception(altered)

    def test_source_target_and_index_exception_corruption_rejected(self):
        repo,checked=self.fixture_repo();r.stage_exact(repo,checked,[],self.actual_config(),self.factory)
        actual=r.index(repo);name=r.WARNING_SOURCE
        for root,path in ((self.factory,name),(repo,r.PREFIX+"/"+name)):
            original=(root/path).read_bytes();(root/path).write_bytes(original+b"changed")
            with self.assertRaises(ValueError):r.verify_warning_sources(self.factory,repo,checked,actual,self.actual_config())
            (root/path).write_bytes(original)
        corrupt=dict(actual);corrupt[name]="0"*40
        with self.assertRaises(ValueError):r.verify_warning_sources(self.factory,repo,checked,corrupt,self.actual_config())
        corrupt=copy.deepcopy(checked);corrupt[name]["sha256"]="0"*64
        with self.assertRaises(ValueError):r.verify_warning_sources(self.factory,repo,corrupt,actual,self.actual_config())

    def test_missing_exception_staged_file_and_empty_other_scope_rejected(self):
        repo,checked=self.fixture_repo();report=r.stage_exact(repo,checked,[],self.actual_config(),self.factory)
        without=[p for p in report["staged_paths"] if p!=r.PREFIX+"/"+r.WARNING_STDOUT]
        with self.assertRaises(ValueError):r.verify_diff_exception(repo,self.factory,checked,without,self.actual_config())
        only=[r.PREFIX+"/"+row["path"] for row in self.actual_config()["files"]]
        with self.assertRaises(ValueError):r.verify_diff_exception(repo,self.factory,checked,only,self.actual_config())

    def test_full_index_outside_bytes_and_postcommit_outside_tree_rejected(self):
        repo=self.repo();parent=r.text(repo,"rev-parse","HEAD");p=self.write(repo,r.PREFIX+"/a.txt",b"new\n")
        checked={"a.txt":r.proof(p)};r.stage_exact(repo,checked,[]);r.verify_whole_index(repo,parent,checked)
        self.write(repo,".gitignore",b"altered outside prefix\n");r.git(repo,"add",".gitignore")
        with self.assertRaises(ValueError):r.verify_whole_index(repo,parent,checked)
        r.git(repo,"commit","--quiet","-m","Own rejection test outside scope commit")
        with self.assertRaises(ValueError):r.verify_committed_parent_tree(repo,r.text(repo,"rev-parse","HEAD"),parent,checked)

    def test_worktree_unstaged_untracked_ignored_or_wrong_staged_delta_rejected(self):
        repo=self.repo();parent=r.text(repo,"rev-parse","HEAD");p=self.write(repo,r.PREFIX+"/a.txt",b"new\n")
        checked={"a.txt":r.proof(p)};r.stage_exact(repo,checked,[]);r.validate_worktree_changes(repo,checked,{})
        for name in ("outside.txt","outside.bin"):
            path=self.write(repo,name,b"extra\n")
            with self.assertRaises(ValueError):r.validate_worktree_changes(repo,checked,{})
            path.unlink()
        p.write_bytes(b"changed\n")
        with self.assertRaises(ValueError):r.validate_worktree_changes(repo,checked,{})
        p.write_bytes(b"new\n")
        with self.assertRaises(ValueError):r.validate_worktree_changes(repo,{}, {})

    def test_release_count_union_duplicates_and_old_immutable_omission(self):
        cap={"path":"data/X/releases/v2/meta.json","size":4,"sha256":"1"*64}
        raw={"path":"data/X/releases/v2/data.jsonl","size":r.MAX_BYTES+1,"sha256":"2"*64}
        self.assertEqual(len(r.release_inventory({"files":[cap],"raw_size_exclusions":[raw]})),2)
        with self.assertRaises(ValueError):r.release_inventory({"files":[cap],"raw_size_exclusions":[cap]})

    def test_actual_joined32c_adapter_strict_normalization_and_mutations(self):
        sp=importlib.util.spec_from_file_location("new_bound_adapter",LOGS/"prepare_prospective_archival_v3_custody_adapter.py")
        adapter=importlib.util.module_from_spec(sp);sp.loader.exec_module(adapter)
        joined=json.loads((LOGS/"BELIEF_V4_R2_JOINED_CUSTODY.json").read_bytes())
        rows=adapter.normalized_joined(joined,self.original_path(),r)
        self.assertEqual(len(rows),6)
        for mutate in (lambda j:j.update(commit_sha="0"*40),lambda j:j.update(status="PENDING"),
                       lambda j:j["six_protected_git_baselines"][0].update(sha256="0"*64),
                       lambda j:j["six_protected_git_baselines"].pop(),
                       lambda j:j["six_protected_git_baselines"][0].update(expected_exists=1)):
            altered=copy.deepcopy(joined);mutate(altered)
            with self.assertRaises(ValueError):adapter.normalized_joined(altered,self.original_path(),r)

    def test_frozen_dependency_change_and_source_hash_rejected(self):
        p=self.write(self.factory,"frozen.txt",b"immutable")
        checksum=r.proof(p,False)["sha256"];source=r.proof(Path(r.__file__),False)["sha256"]
        r.recheck_frozen([(p,checksum)],source);p.write_bytes(b"changed")
        with self.assertRaises(ValueError):r.recheck_frozen([(p,checksum)],source)
        with self.assertRaises(ValueError):r.recheck_frozen([],"0"*64)

    def test_recovery_changed_before_import_rejected_without_import(self):
        altered=self.write(self.factory,"backup_logs/recover_remote_checkpoint.py",b"raise RuntimeError('must not execute')\n")
        with patch.object(r,"FACTORY",self.factory),self.assertRaises(ValueError):r.import_recovery_frozen()

    def test_reference_rebase_and_missing_historical_reference_rejected(self):
        # No corpus file read is needed to reject a wrong/missing exact-baseline list.
        with self.assertRaises(ValueError):r.verify_readonly_references({"readonly_reference_files":[]})
        forged=[{"path":str(r.FACTORY.parent/"rineya_complementary_research_2026-10-09"/f"fake{i}"),"size":0,"sha256":"0"*64} for i in range(65)]
        with self.assertRaises(ValueError):r.verify_readonly_references({"readonly_reference_files":forged})


    def execution_fixture(self):
        from types import SimpleNamespace
        def e(path,size=1,sha="1"*64):return {"path":path,"size":size,"sha256":sha}
        old_files=[e(f"data/X/releases/v1/file{i}") for i in range(210)]+[e(f"old/file{i}") for i in range(2475)]
        raw=[];checked={x["path"]:{"size":x["size"],"sha256":x["sha256"],"git_blob":"1"*40} for x in old_files}
        for i in range(26):
            gz=e(f"gzip{i}");mf=e(f"manifest{i}");checked[gz["path"]]={**gz,"git_blob":"2"*40};checked[mf["path"]]={**mf,"git_blob":"3"*40}
            raw.append({**e(f"raw{i}",r.MAX_BYTES+1),"lossless_representations":[{"gzip":gz,"manifest":mf}]})
        # checked includes all prior tracked files and newly declared compressed pairs.
        prior={"files":old_files,"raw_size_exclusions":copy.deepcopy(raw)}
        new=[e(f"data/RBM-001/releases/v0002/file{i}") for i in range(12)]
        checked.update({x["path"]:{"size":x["size"],"sha256":x["sha256"],"git_blob":"4"*40} for x in new})
        files=[{k:v for k,v in x.items() if k in {"path","size","sha256"}} for x in old_files+new]
        spec={"files":files,"raw_size_exclusions":raw,"expected_parent_sha":r.PROSPECTIVE_PARENT,"checkpoint_id":"own-isolated-future-scope",
              "raw_lossless_required_count":26,"inventory_unique_git_blob_groups":4}
        args=SimpleNamespace(expected_parent=r.PROSPECTIVE_PARENT,runner="/root",expected_spec_sha256="5"*64,
                             expected_freeze_sha256="6"*64,expected_custody_sha256="7"*64,expected_exception_sha256="8"*64)
        config={"schema":"prospective-archival-stage-v3-execution-v1","author_execution":r.AUTHOR,"runner_execution":"/root",
                "expected_parent":r.PROSPECTIVE_PARENT,"prior_spec_sha256":r.PRIOR_SPEC_SHA,"prior_tracked_path_count":2685,
                "prior_release_file_count":210,"expected_release_file_count":222,"readonly_reference_count":65,"grants_data_approval":False,
                "spec_sha256":args.expected_spec_sha256,"source_freeze_sha256":args.expected_freeze_sha256,
                "custody_adapter_sha256":args.expected_custody_sha256,"exception_config_sha256":args.expected_exception_sha256,
                "checkpoint_id":spec["checkpoint_id"],"release_files":r.release_inventory(spec),"raw_lossless_required_count":26,
                "historical_raw_relocations":[]}
        return config,args,spec,prior,checked

    def test_future_exact_scope_validates_and_rejects_prior_release_raw_count_parent_or_group_corruption(self):
        config,args,spec,prior,checked=self.execution_fixture()
        self.assertEqual(r.validate_execution_config(config,args,spec,prior,checked)["prior_tracked_paths"],2685)
        changes=[("expected_parent_sha","0"*40),("raw_lossless_required_count",25),("inventory_unique_git_blob_groups",3)]
        for key,value in changes:
            broken=copy.deepcopy(spec);broken[key]=value
            with self.assertRaises(ValueError):r.validate_execution_config(config,args,broken,prior,checked)
        altered=dict(checked);altered.pop("old/file0")
        with self.assertRaises(ValueError):r.validate_execution_config(config,args,spec,prior,altered)
        altered=copy.deepcopy(spec);altered["files"][0]["sha256"]="9"*64
        c=copy.deepcopy(config);c["release_files"]=r.release_inventory(altered)
        with self.assertRaises(ValueError):r.validate_execution_config(c,args,altered,prior,checked)
        altered=copy.deepcopy(spec);altered["raw_size_exclusions"][0]["lossless_representations"]=[]
        with self.assertRaises(ValueError):r.validate_execution_config(config,args,altered,prior,checked)
        broken=copy.deepcopy(config);broken["expected_release_file_count"]=True
        with self.assertRaises(ValueError):r.validate_execution_config(broken,args,spec,prior,checked)

    def test_exact_historical_audit_relocation_requires_both_raw_histories_and_old_packages(self):
        gz={"path":"old.gz","size":10,"sha256":"a"*64};mf={"path":"old.lossless.json","size":20,"sha256":"b"*64}
        old={"path":r.MUTABLE_AUDIT,"size":123141333,"sha256":r.HISTORICAL_AUDIT_SHA,"lossless_representations":[{"gzip":gz,"manifest":mf}]}
        new={r.MUTABLE_AUDIT:{**old,"sha256":r.CURRENT_AUDIT_SHA,"lossless_representations":[{"new_standard_pair":True}]}}
        for name in r.HISTORICAL_AUDIT_PATHS:new[name]={**old,"path":name,"lossless_representations":[{"standard_archive_pair":True}]}
        checked={gz["path"]:gz,mf["path"]:mf}
        relocation={"old_raw_path":old["path"],"old_raw_bytes":old["size"],"old_raw_sha256":old["sha256"],
                    "current_raw_sha256":r.CURRENT_AUDIT_SHA,"historical_raw_paths":list(r.HISTORICAL_AUDIT_PATHS)}
        config={"historical_raw_relocations":[relocation]}
        self.assertEqual(r.validate_historical_raw_custody([old],new,checked,config),[relocation])
        for mutate in (lambda n:n.pop(r.HISTORICAL_AUDIT_PATHS[0]),lambda n:n[r.HISTORICAL_AUDIT_PATHS[1]].update(sha256="0"*64),
                       lambda n:n[r.MUTABLE_AUDIT].update(sha256="0"*64)):
            altered=copy.deepcopy(new);mutate(altered)
            with self.assertRaises(ValueError):r.validate_historical_raw_custody([old],altered,checked,config)
        with self.assertRaises(ValueError):r.validate_historical_raw_custody([old],new,{},config)
        with self.assertRaises(ValueError):r.validate_historical_raw_custody([old],new,checked,{"historical_raw_relocations":[]})

    def test_pre_push_origin_branch_head_private_or_remote_drift_rejected_before_push(self):
        commit="c"*40;parent=r.PROSPECTIVE_PARENT;private={"isPrivate":True,"nameWithOwner":r.REPOSITORY,"viewerPermission":"ADMIN"}
        metadata={("rev-parse","HEAD"):commit,("branch","--show-current"):r.BRANCH,
                  ("remote","get-url","origin"):f"https://github.com/{r.REPOSITORY}.git",
                  ("status","--porcelain=v1","--untracked-files=all"):"",("config","--get","index.sparse"):"false",
                  ("sparse-checkout","list"):"tools\n"+r.PREFIX,("ls-remote","--heads","origin","refs/heads/"+r.BRANCH):parent+"\trefs/heads/"+r.BRANCH}
        def runmeta(root,*args):return metadata[args]
        with patch.object(r,"text",side_effect=runmeta),patch.object(r,"run",return_value=json.dumps(private).encode()):
            self.assertTrue(r.prepush_custody(Path("unused"),commit,parent,private))
            for key,value in [(('rev-parse','HEAD'),"0"*40),(('branch','--show-current'),"main"),
                              (('remote','get-url','origin'),"https://github.com/other/repo.git"),
                              (('config','--get','index.sparse'),"true"),
                              (('ls-remote','--heads','origin','refs/heads/'+r.BRANCH),"0"*40+" refs/heads/"+r.BRANCH)]:
                original=metadata[key];metadata[key]=value
                with self.assertRaises(ValueError):r.prepush_custody(Path("unused"),commit,parent,private)
                metadata[key]=original
        with patch.object(r,"text",side_effect=runmeta),patch.object(r,"run",return_value=json.dumps({**private,"isPrivate":False}).encode()):
            with self.assertRaises(ValueError):r.prepush_custody(Path("unused"),commit,parent,private)


if __name__ == "__main__":unittest.main(verbosity=2)
