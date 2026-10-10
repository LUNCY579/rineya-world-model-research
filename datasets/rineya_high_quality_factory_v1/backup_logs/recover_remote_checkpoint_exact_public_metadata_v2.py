#!/usr/bin/env python3
"""Verify an already-pushed frozen checkpoint without copying or pushing.

Every local Git blob and actual GitHub raw blob is streamed and hashed. Current
All current sources are held to the exact approved-release inventory; evolving
sources are rejected. Four typed metadata exceptions remain fully byte bound. Outputs are new, exclusive evidence files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
AUTHOR = "/root/backup_manager"
REPOSITORY = "LUNCY579/rineya-world-model-research"
BRANCH = "research/rineya-high-quality-dataset-v1"
PREFIX = "datasets/rineya_high_quality_factory_v1"
MAX_BYTES = 25 * 1024 * 1024
CHUNK = 64 * 1024
SECRET_PATTERNS = [
    re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?im)^\s*(?:api_key|access_token|password|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_+/=-]{16,}"),
]


# NEW narrowly bound derivative. Every original regex and transport check remains.
FACTORY=Path(__file__).resolve().parent.parent
ORIGINAL_SOURCE_SHA='3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1'
METADATA_SOURCE_SHA='a79963d367f47c54074714d24bc2a5f372f09bf725a4b2046006cbdb586e4b1d'
METADATA_CONFIG_SHA='2e18c2c156aaefcee75ad6ad92fe51c66abf864150924d5ac1cd26dbeed5aa05'
EXPECTED_SPEC_SHA='6c21c64d1cff3ce38d548ed024f6bb9da9f9c8869463d7c4cbd5038240cd97f9'
EXPECTED_PARENT='4d0b5eff1af3184fedf7fa25ada729a6cc909268'
EXPECTED_COMMIT='653ba70f1f7035a5aed137102c14a1ff29d1cf99'
STAGE_SOURCE_SHA='f0561eb5f0e2ebe16bc46f51ed1fcbb1ad74236e90fba4d9b33da166c6767715'
STAGE_CONFIG_SHA='85cfe47a1cabb952d2f00e03d5d3e10d37134a7510c7653fbc05366100a27a72'
STAGE_FREEZE_SHA='3c312ce1c2295b78b565191571e2103f4070f92694aeaae6d6953f80e34648ee'
CUSTODY_SHA='7978cab638a00c185a9d391259c6a5c77b8a5c97ea5fefdc79fa67a31e58e023'
WARNING_CONFIG_SHA='bd6639e10289bcbd7c27e3bf68241e83dbf7f0d9985400b7abf9df9ae9b9fa4b'
_METADATA=None
_LOCK=threading.RLock()
_STREAM_REPORTS={'local_git':{},'actual_remote':{}}

def import_bound(path,expected,name):
    if path.is_symlink() or hash_file(path)['sha256']!=expected:raise ValueError('Frozen dependency differs before import')
    sp=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    if path.is_symlink() or hash_file(path)['sha256']!=expected:raise ValueError('Frozen dependency changed during import')
    return module

def metadata_helper():
    global _METADATA
    with _LOCK:
        if _METADATA is None:
            _METADATA=import_bound(FACTORY/'backup_logs/exact_four_public_metadata_secret_scan_attempt02.py',METADATA_SOURCE_SHA,'frozen_exact_metadata_remote')
        if hash_file(FACTORY/'backup_logs/exact_four_public_metadata_secret_scan_attempt02.py')['sha256']!=METADATA_SOURCE_SHA:
            raise ValueError('Exact scanner changed')
        _METADATA.validate_config()
        return _METADATA

def canonical_stream_paths(original_paths):
    if type(original_paths) not in (tuple,list) or not original_paths or len(set(original_paths))!=len(original_paths):
        raise ValueError('Exact original SPEC path context required')
    helper=metadata_helper();rows=[]
    for name in original_paths:
        relative_path(name)
        row=helper.bound_row(name)
        rows.append(row)
    if any(row is not None for row in rows) and any(row is None for row in rows):
        raise ValueError('Known metadata blob cannot permit an unrelated path alias')
    if any(row is not None for row in rows):
        first=rows[0]
        if any((row['size'],row['sha256'],row['matches'])!=(first['size'],first['sha256'],first['matches']) for row in rows):
            raise ValueError('Metadata group contexts have different exact bytes')
        return first
    return None

def exact_secret_stream(handle,size,git_blob,original_paths,context):
    if type(size) is not int or size<0:raise ValueError('Secret stream needs exact frozen size')
    helper=metadata_helper();row=canonical_stream_paths(original_paths)
    actual,observations=helper.scan_stream(handle,size,SECRET_PATTERNS,row,CHUNK)
    if git_blob is not None and (not re.fullmatch(r'[0-9a-f]{40}',git_blob) or actual['git_blob']!=git_blob):
        raise ValueError('Actual Git blob object hash mismatch')
    if row is not None:
        if context not in _STREAM_REPORTS:raise ValueError('Exact metadata stream provenance required')
        with _LOCK:
            for name in original_paths:
                _STREAM_REPORTS[context][name]={'path':name,'size':actual['size'],'sha256':actual['sha256'],'git_blob':actual['git_blob'],
                    'matches':observations,'full_size_sha256_gitsha1_verified_before_exception_acceptance':True}
    return {'size':actual['size'],'sha256':actual['sha256']}

def complete_metadata_stream_report(context):
    expected={row['path']:row for row in metadata_helper().EXPECTED_FILES}
    with _LOCK:actual=dict(_STREAM_REPORTS[context])
    if set(actual)!=set(expected):raise ValueError('All four unique exact metadata stream contexts required: '+context)
    for name,row in expected.items():
        value=actual[name]
        if (value['size']!=row['size'] or value['sha256']!=row['sha256'] or value['matches']!=row['matches']
                or value['full_size_sha256_gitsha1_verified_before_exception_acceptance'] is not True):
            raise ValueError('Incomplete/changed metadata stream proof')
    return {'status':'PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS','context':context,
        'allowed_exact_match_detections':4,'unbound_secret_detections':0,'configuration_sha256':METADATA_CONFIG_SHA,
        'scanner_source_sha256':METADATA_SOURCE_SHA,'existing_regexes_unchanged':True,'four_exception_files':[actual[n] for n in sorted(actual)]}

def typed_equal(a,b):
    if type(a) is not type(b):return False
    if type(b) is dict:return set(a)==set(b) and all(typed_equal(a[k],b[k]) for k in b)
    if type(b) is list:return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a==b


def validate_metadata_report(report,stream=False):
    helper=metadata_helper();expected={row['path']:row for row in helper.EXPECTED_FILES}
    if (type(report) is not dict or report.get('status')!='PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS'
            or type(report.get('allowed_exact_match_detections')) is not int or report['allowed_exact_match_detections']!=4
            or type(report.get('unbound_secret_detections')) is not int or report['unbound_secret_detections']!=0
            or report.get('configuration_sha256')!=METADATA_CONFIG_SHA or report.get('existing_regexes_unchanged') is not True):
        raise ValueError('Metadata report truth/config/count differs')
    rows=report.get('four_exception_files')
    if type(rows) is not list or len(rows)!=4 or len({v.get('path') for v in rows})!=4 or {v.get('path') for v in rows}!=set(expected):
        raise ValueError('All four unique metadata reports required')
    flag='full_size_sha256_gitsha1_verified_before_exception_acceptance' if stream else 'full_size_and_sha_verified_before_acceptance'
    for value in rows:
        row=expected[value['path']]
        if (type(value.get('size')) is not int or value['size']!=row['size'] or value.get('sha256')!=row['sha256']
                or not typed_equal(value.get('matches'),row['matches']) or value.get(flag) is not True):
            raise ValueError('Metadata report full-byte/span proof differs')
        if stream and (not re.fullmatch(r'[0-9a-f]{40}',value.get('git_blob','')) or report.get('scanner_source_sha256')!=METADATA_SOURCE_SHA):
            raise ValueError('Metadata report actual Git/source proof missing')
    return report


def validate_exact_scope(spec):
    files=spec.get('files',[]);raws=spec.get('raw_size_exclusions',[])
    by={e['path']:e for e in files}
    if len(files)!=3158 or sum(e['size'] for e in files)!=1263090770 or len(raws)!=31 or sum(e['size'] for e in raws)!=1724728576:
        raise ValueError('Exact approved-release scope/count/bytes differs')
    if sum(len(e['lossless_representations']) for e in raws)!=32 or sum(e['size']*len(e['lossless_representations']) for e in raws)!=1766951239:
        raise ValueError('Exact local representation coverage differs')
    for row in metadata_helper().EXPECTED_FILES:
        if by.get(row['path'])!={k:row[k] for k in ('path','size','sha256')}:raise ValueError('Known metadata SPEC binding missing/changed')
    return True


def check_new_source_freeze(args):
    if args.workers!=2:raise ValueError('Exactly two actual remote workers required')
    if args.allow_evolving_source:raise ValueError('This held exact checkpoint forbids evolving sources')
    if args.runner!='/root' or args.commit!=EXPECTED_COMMIT or args.expected_spec_sha256!=EXPECTED_SPEC_SHA:
        raise ValueError('Exact root/commit/spec binding required')
    if args.metadata_config.resolve()!=FACTORY/'backup_logs/EXACT_FOUR_PUBLIC_ID_SECRET_EXCEPTION_CONFIG.json' or args.metadata_config.is_symlink():
        raise ValueError('Only exact original metadata config path allowed')
    if args.expected_metadata_config_sha256!=METADATA_CONFIG_SHA or hash_file(args.metadata_config)['sha256']!=METADATA_CONFIG_SHA:
        raise ValueError('Metadata config binding differs')
    if (source_path(FACTORY,args.source_freeze.absolute().relative_to(FACTORY).as_posix())!=args.source_freeze.absolute()
            or not args.source_freeze.absolute().is_relative_to(FACTORY/'backup_logs')
            or args.source_freeze.is_symlink() or hash_file(args.source_freeze)['sha256']!=args.expected_freeze_sha256):
        raise ValueError('Fresh source freeze binding differs')
    freeze=json.loads(args.source_freeze.read_bytes())
    if (freeze['source']!={'path':Path(__file__).relative_to(FACTORY).as_posix(),**hash_file(Path(__file__))}
            or freeze['author_execution']!=AUTHOR or freeze.get('frozen_before_first_import_test_execution') is not True):
        raise ValueError('Recovery source not frozen')
    for row in freeze['frozen_inputs']:
        if hash_file(source_path(FACTORY,row['path']))!={'size':row['size'],'sha256':row['sha256']}:
            raise ValueError('Frozen preparation input changed')
    required={'backup_logs/belief-v4-approved-release-r1_STAGE.json','backup_logs/belief-v4-approved-release-r1_LOCAL_COMMIT.json',
        'backup_logs/belief-v4-approved-release-r1_STAGE_LEDGER.jsonl'}
    if not required<={p.absolute().relative_to(FACTORY).as_posix() for p in args.prior_evidence}:
        raise ValueError('Actual stage/localcommit/ledger bindings required')
    metadata_helper().validate_config()
    return freeze

def validate_stage_observation(stage,m):
    expected={'status':'PUSHED_REMOTE_BYTES_VERIFICATION_PENDING','commit_sha':EXPECTED_COMMIT,'remote_commit_sha':EXPECTED_COMMIT,
        'expected_parent_sha':EXPECTED_PARENT,'spec_sha256':EXPECTED_SPEC_SHA,'source_sha256':STAGE_SOURCE_SHA,'source_freeze_sha256':STAGE_FREEZE_SHA,
        'files':3158,'total_bytes':1263090770,'all_source_target_index_sha256_and_sha1_verified':True,
        'whole_index_and_outside_parent_guard':'PASS','protected_original_baselines_verified':True,'actual_release_coverage_verified':222,
        'readonly_reference_count_verified':65,'restricted_strict_diff_check':'PASS','restricted_changed_path_count':482,'excluded_exact_path_count':2}
    for key,value in expected.items():
        if not typed_equal(stage.get(key),value):raise ValueError('Actual stage binding/guard differs: '+key)
    validate_metadata_report(stage.get('secret_scan',{}))
    import base64
    result={'returncode':stage.get('actual_exit_code'),'stdout':base64.b64decode(stage['actual_stdout_base64'],validate=True),
            'stderr':base64.b64decode(stage['actual_stderr_base64'],validate=True)}
    config=json.loads((FACTORY/'backup_logs/PROSPECTIVE_ARCHIVAL_V3_EXACT_EXCEPTION.json').read_bytes())
    actual=m.validate_diagnostics(result,config)
    if any(not typed_equal(stage.get(k),v) for k,v in actual.items()):raise ValueError('Exact six-warning diagnostics differ')
    return {'unrestricted_exit':2,'exception':'EXACT_BOUND_SIX_WARNINGS_MATCH','restricted_strict_pass_paths':482}


def full_checkpoint_guards(worktree,commit,spec,local_paths):
    m=import_bound(FACTORY/'backup_logs/run_provisional_archival_checkpoint_v4_attempt02.py',STAGE_SOURCE_SHA,'frozen_stage_for_remote_guards')
    checked={e['path']:{'size':e['size'],'sha256':e['sha256'],'git_blob':local_paths[PREFIX+'/'+e['path']]['sha']} for e in spec['files']}
    prior=json.loads((FACTORY/'backup_logs/belief-v4-validated-code-candidate-r2_SPEC.json').read_bytes())
    config_path=FACTORY/'backup_logs/BELIEF_V4_APPROVED_RELEASE_R1_STAGE_V4_ATTEMPT02_EXECUTION_CONFIG.json'
    custody_path=FACTORY/'backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json'
    if hash_file(config_path)['sha256']!=STAGE_CONFIG_SHA or hash_file(custody_path)['sha256']!=CUSTODY_SHA:
        raise ValueError('Frozen stage config/custody differs')
    config=json.loads(config_path.read_bytes());custody=json.loads(custody_path.read_bytes())
    from types import SimpleNamespace
    args=SimpleNamespace(expected_spec_sha256=EXPECTED_SPEC_SHA,expected_freeze_sha256=STAGE_FREEZE_SHA,
        expected_custody_sha256=CUSTODY_SHA,expected_exception_sha256=WARNING_CONFIG_SHA,expected_parent=EXPECTED_PARENT,runner='/root')
    stage=json.loads((FACTORY/'backup_logs/belief-v4-approved-release-r1_STAGE.json').read_bytes())
    validate_stage_observation(stage,m)
    m.validate_execution_config(config,args,spec,prior,checked)
    m.verify_committed_parent_tree(worktree,commit,EXPECTED_PARENT,checked)
    m.verify_whole_index(worktree,EXPECTED_PARENT,checked)
    m.guard_previous(m.tree(worktree,EXPECTED_PARENT),checked)
    m.verify_readonly_references(config);m.verify_actual_release_coverage(FACTORY,spec,prior)
    m.original_custody(FACTORY.parent/'rineya-world-model-research',custody);m.dedicated_custody(worktree,commit)
    if m.whole_index(worktree)!=m.whole_tree(worktree,commit):raise ValueError('Actual full committed tree/index differs')
    return {'all3158_paths_full_tree_index':True,'all2685_parent_paths_preserved':True,'all222_immutable_release_files':True,
        'all65_readonly_reference_hashes':True,'original_six_git_byte_guards':True,'original_main_and_dedicated_clean':True,
        'unrestricted_diff_exit':2,'frozen_exact_trailing_whitespace_warnings':6,'restricted_strict_paths':482,
        'sole_parent_sha':EXPECTED_PARENT,'stage_source_sha256':STAGE_SOURCE_SHA,'stage_config_sha256':STAGE_CONFIG_SHA}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError("Inventory path must be a string")
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or str(path) != value
            or any(part in {"..", ".git"} for part in path.parts)
            or "\n" in value or "\r" in value):
        raise ValueError("Unsafe inventory path")
    return path


def source_path(factory, value):
    factory = factory.resolve()
    path = relative_path(value)
    result = factory.joinpath(*path.parts)
    if any(factory.joinpath(*path.parts[:i]).is_symlink()
           for i in range(1, len(path.parts) + 1)):
        raise ValueError(f"Symlink source rejected: {value}")
    if not result.resolve().is_relative_to(factory):
        raise ValueError(f"Source escapes factory: {value}")
    return result


def hash_stream(handle, size=None, git_blob=None, scan_secrets=False, original_paths=None, context=None):
    if scan_secrets and original_paths is not None:
        return exact_secret_stream(handle,size,git_blob,original_paths,context)
    sha = hashlib.sha256()
    git = hashlib.sha1() if git_blob else None
    if git:
        git.update(f"blob {size}\0".encode())
    count, tail = 0, b""
    while True:
        data = handle.read(CHUNK)
        if not data:
            break
        count += len(data)
        if size is not None and count > size:
            raise ValueError("Payload exceeds frozen size")
        sha.update(data)
        if git:
            git.update(data)
        if scan_secrets:
            if any(pattern.search(tail + data) for pattern in SECRET_PATTERNS):
                raise ValueError("Potential secret detected; content withheld")
            tail = (tail + data)[-4096:]
    if size is not None and count != size:
        raise ValueError("Payload is truncated or has the wrong frozen size")
    if git and git.hexdigest() != git_blob:
        raise ValueError("Actual Git blob object hash mismatch")
    return {"size": count, "sha256": sha.hexdigest()}


def hash_file(path, scan_secrets=False):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular source file: {path.name}")
    before = path.stat()
    with path.open("rb") as handle:
        result = hash_stream(handle, scan_secrets=scan_secrets)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                               after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Source changed during hash: {path.name}")
    return result


def process_stream(args, consume, cwd=None, timeout=180):
    # Do not capture payloads or credentials in memory, files, or error output.
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    timer = threading.Timer(timeout, proc.kill)
    timer.daemon = True
    timer.start()
    try:
        result = consume(proc.stdout)
        if proc.wait(timeout=10):
            raise RuntimeError(f"{args[0]} exited unsuccessfully; error body withheld")
        return result
    finally:
        timer.cancel()
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def command(args, cwd=None):
    def read(handle):
        result = bytearray()
        while True:
            part = handle.read(CHUNK)
            if not part:
                break
            result.extend(part)
            if len(result) > 16 * 1024 * 1024:
                raise ValueError("Metadata response exceeds bounded limit")
        return bytes(result)
    return process_stream(args, read, cwd)


def git(worktree, *args):
    return command(["git", *args], worktree).decode().strip()


def api(endpoint):
    return json.loads(command(["gh", "api", "-H", "Cache-Control: no-cache", endpoint]))


def check_entry(entry, maximum=True):
    relative_path(entry["path"])
    if (type(entry.get("size")) is not int or entry["size"] < 0
            or (maximum and entry["size"] > MAX_BYTES)
            or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))):
        raise ValueError(f"Invalid frozen size/hash: {entry['path']}")


def check_source(factory, entry, allow_evolving):
    factory = factory.resolve()
    path = source_path(factory, entry["path"])
    actual = metadata_helper().proof(path,SECRET_PATTERNS) if path.is_file() else None
    if actual is not None:actual={k:actual[k] for k in ("size","sha256")}
    expected = {"size": entry["size"], "sha256": entry["sha256"]}
    if actual == expected:
        return None
    if not allow_evolving or path.suffix != ".py":
        raise ValueError(f"Current inventoried source differs: {entry['path']}")
    archived = source_path(factory, f"reproducibility/source_blobs/{entry['sha256']}.source")
    archive_proof = None
    if archived.exists():
        if hash_file(archived, True) != expected:
            raise ValueError(f"Content-addressed source differs: {entry['path']}")
        archive_proof = str(archived.relative_to(factory))
    return {"path": entry["path"], "expected": expected, "current": actual,
            "frozen_source_archive": archive_proof,
            "status": "EVOLVED_PYTHON_SOURCE_FROZEN_COMMIT_BYTES_REQUIRED"}


def check_custody(worktree, commit):
    if git(worktree, "branch", "--show-current") != BRANCH:
        raise ValueError("Dedicated backup branch mismatch")
    if git(worktree, "rev-parse", "HEAD") != commit:
        raise ValueError("Dedicated HEAD differs from recovery commit")
    if git(worktree, "remote", "get-url", "origin") != f"https://github.com/{REPOSITORY}.git":
        raise ValueError("Dedicated origin mismatch")
    if git(worktree, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Dedicated backup worktree is dirty")
    ref = api(f"repos/{REPOSITORY}/git/ref/heads/{BRANCH}")
    if ref.get("object", {}).get("sha") != commit:
        raise ValueError("Actual GitHub branch differs from recovery commit")
    remote = git(worktree, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}")
    if remote.split() != [commit, f"refs/heads/{BRANCH}"]:
        raise ValueError("Git transport branch disagrees with GitHub API")


def write_exclusive(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--prior-evidence", type=Path, action="append", default=[])
    parser.add_argument("--allow-evolving-source", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--metadata-config", type=Path, required=True)
    parser.add_argument("--expected-metadata-config-sha256", required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    source_freeze=check_new_source_freeze(args)
    factory = Path(__file__).resolve().parent.parent
    worktree = factory.parent / ".rineya-dataset-git-worktree"
    logs = factory / "backup_logs"
    for path in [args.output, args.ledger]:
        if not path.resolve().is_relative_to(logs) or path.exists() or path.is_symlink():
            raise ValueError("Recovery outputs must be new paths under backup_logs")
    if args.output.resolve() == args.ledger.resolve():
        raise ValueError("Receipt and ledger must be distinct")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("Expected commit must be a complete SHA1")
    frozen_spec_sha = hash_file(args.spec)["sha256"]
    if frozen_spec_sha != args.expected_spec_sha256:
        raise ValueError("Frozen spec SHA256 differs from explicit expected hash")
    spec = json.loads(args.spec.read_bytes())
    if (spec.get("validation_status") != "passed" or not spec.get("approved_by")
            or spec.get("approval") not in {"reviewed_release", "provisional_checkpoint", "code_review_milestone"}):
        raise ValueError("Original checkpoint approval/validation evidence unavailable")
    validate_exact_scope(spec)
    inventory = spec["files"]
    if not inventory or len({item["path"] for item in inventory}) != len(inventory):
        raise ValueError("Frozen inventory is empty or repeats paths")
    for entry in inventory:
        check_entry(entry)
    evidence = [{"path": str(path.resolve()), **hash_file(path)} for path in args.prior_evidence]
    receipt = {"checkpoint_id": spec["checkpoint_id"], "status": "PENDING",
               "helper_author_execution": AUTHOR, "author_execution": AUTHOR,
               "runner_execution": args.runner, "repository": REPOSITORY, "branch": BRANCH,
               "approval": spec["approval"], "approved_by": spec["approved_by"],
               "started_at_utc": now(), "spec_sha256": frozen_spec_sha,
               "verification_source_sha256": hash_file(Path(__file__))["sha256"],
               "source_freeze_sha256":args.expected_freeze_sha256,"original_verifier_source_sha256":ORIGINAL_SOURCE_SHA,
               "metadata_scanner_source_sha256":METADATA_SOURCE_SHA,"metadata_exception_config_sha256":METADATA_CONFIG_SHA,
               "prior_evidence": evidence, "inventory": inventory,
               "files": len(inventory), "total_bytes": sum(e["size"] for e in inventory),
               "commit_sha": args.commit,
               "scope": "Recovery of exact already-pushed checkpoint only; no copy, checkout restoration, commit, push, approval, or modification of prior receipts/helper/spec/data",
               "self_inventory_boundary": "New recovery code, ledger and receipt are local evidence until a later exact checkpoint; they are not claimed present in the recovered commit"}
    with args.ledger.open("x", encoding="utf-8") as ledger:
        def log(event):
            ledger.write(json.dumps({"at_utc": now(), **event}, separators=(",", ":")) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
        try:
            log({"phase": "start", "spec_sha256": frozen_spec_sha, "commit": args.commit,
                 "author_execution": AUTHOR, "runner_execution": args.runner})
            check_custody(worktree, args.commit)
            metadata = api(f"repos/{REPOSITORY}")
            if (metadata.get("full_name") != REPOSITORY or metadata.get("private") is not True
                    or metadata.get("permissions", {}).get("admin") is not True):
                raise ValueError("Actual repository is not private with ADMIN access")
            receipt["private_repository_evidence"] = {"nameWithOwner": REPOSITORY, "isPrivate": True,
                                                        "viewerPermission": "ADMIN", "checked_at_utc": now()}
            commit_bytes = command(["git", "cat-file", "commit", args.commit], worktree)
            if hashlib.sha1(f"commit {len(commit_bytes)}\0".encode() + commit_bytes).hexdigest() != args.commit:
                raise ValueError("Local actual commit object hash mismatch")
            tree_sha = git(worktree, "rev-parse", f"{args.commit}^{{tree}}")
            remote_commit = api(f"repos/{REPOSITORY}/git/commits/{args.commit}")
            if ([p["sha"] for p in remote_commit.get("parents",[])]!=[EXPECTED_PARENT]
                    or remote_commit.get("sha") != args.commit or remote_commit.get("tree", {}).get("sha") != tree_sha):
                raise ValueError("Actual remote commit/tree mismatch")
            remote_tree = api(f"repos/{REPOSITORY}/git/trees/{tree_sha}?recursive=1")
            if remote_tree.get("truncated") or remote_tree.get("sha") != tree_sha:
                raise ValueError("Remote tree is truncated or mismatched")
            remote_paths = {item["path"]: item for item in remote_tree["tree"] if item["type"] == "blob"}
            local_paths = {}
            for record in command(["git", "ls-tree", "-r", "-z", args.commit, "--", PREFIX], worktree).split(b"\0"):
                if record:
                    meta, path = record.split(b"\t", 1)
                    mode, kind, blob = meta.decode().split()
                    local_paths[path.decode()] = {"mode": mode, "type": kind, "sha": blob}
            scoped={name[len(PREFIX)+1:] for name in remote_paths if name.startswith(PREFIX+'/')}
            if scoped!={e['path'] for e in inventory}:raise ValueError('Actual remote prefix coverage differs')
            groups, evolved = {}, []
            for entry in inventory:
                path = f"{PREFIX}/{entry['path']}"
                local, remote = local_paths.get(path), remote_paths.get(path)
                if (not local or not remote or local["type"] != "blob"
                        or local["mode"] not in {"100644", "100755"}
                        or any(local[key] != remote[key] for key in ["type", "mode", "sha"])
                        or remote.get("size") != entry["size"]):
                    raise ValueError(f"Frozen local/remote path or mode or size mismatch: {path}")
                change = check_source(factory, entry, args.allow_evolving_source)
                if change:
                    evolved.append(change)
                groups.setdefault(local["sha"], []).append(entry)
            receipt["capped_source_secret_scan_before"]=validate_metadata_report(metadata_helper().report(require_complete=True))
            receipt["checkpoint_custody_guards_before"]=full_checkpoint_guards(worktree,args.commit,spec,local_paths)
            for blob, entries in groups.items():
                if len({(entry["size"], entry["sha256"]) for entry in entries}) != 1:
                    raise ValueError("Shared blob inventory hashes disagree")
            if len(groups)!=2474:raise ValueError('Exact unique blob scope differs')
            raw_proofs = []
            by_path = {entry["path"]: entry for entry in inventory}
            for entry in spec.get("raw_size_exclusions", []):
                check_entry(entry, False)
                raw = source_path(factory, entry["path"])
                if hash_file(raw, True) != {"size": entry["size"], "sha256": entry["sha256"]}:
                    raise ValueError(f"Frozen oversized raw differs: {entry['path']}")
                representations = entry["lossless_representations"]
                if not representations:
                    raise ValueError("Oversized raw has no actual lossless representation")
                for representation in representations:
                    for key in ["gzip", "manifest"]:
                        frozen = representation[key]
                        if by_path.get(frozen["path"]) != frozen:
                            raise ValueError("Lossless evidence omitted from exact tracked inventory")
                    compressed = source_path(factory, representation["gzip"]["path"])
                    with gzip.open(compressed, "rb") as handle:
                        decoded = hash_stream(handle, entry["size"], scan_secrets=True)
                    if decoded["sha256"] != entry["sha256"]:
                        raise ValueError("Lossless gzip does not decode to exact frozen raw")
                raw_proofs.append({"path": entry["path"], "size": entry["size"],
                                   "sha256": entry["sha256"], "lossless_local_roundtrip": True})
            for index, (blob, entries) in enumerate(sorted(groups.items()), 1):
                expected = entries[0]
                actual = process_stream(["git", "cat-file", "blob", blob],
                                        lambda handle: hash_stream(handle, expected["size"], blob, True, [e["path"] for e in entries], "local_git"), worktree)
                if actual["sha256"] != expected["sha256"]:
                    raise ValueError(f"Frozen local Git blob SHA256 mismatch: {blob}")
                log({"phase": "local_git_blob_verified", "git_blob": blob, **actual})
                if index % 100 == 0:
                    print(json.dumps({"phase": "local_git_blobs", "verified": index, "total": len(groups)}), flush=True)
            receipt["actual_local_git_metadata_exception_scan"]=validate_metadata_report(complete_metadata_stream_report("local_git"),True)
            receipt["current_source_evolution"] = evolved
            receipt["current_source_matching_files"] = len(inventory) - len(evolved)
            receipt["local_git_verified_unique_blobs"] = len(groups)
            receipt["oversized_raw_local_verification"] = raw_proofs
            print(json.dumps({"phase": "FULL_LOCAL_FROZEN_SPEC_VERIFIED", "files": len(inventory),
                              "bytes": receipt["total_bytes"], "evolved_python_sources": len(evolved)}), flush=True)
            log({"phase": "full_local_frozen_spec_verified", "files": len(inventory), "evolved_python_sources": evolved})
            if args.local_only:
                receipt["status"] = "LOCAL_FROZEN_INVENTORY_VERIFIED_REMOTE_PENDING"
                receipt["remote_commit_sha"] = None
            else:
                def fetch(pair):
                    blob, entries = pair
                    expected = entries[0]
                    for attempt in range(3):
                        try:
                            actual = process_stream(["gh", "api", "-H", "Accept: application/vnd.github.raw+json",
                                                     "-H", "Cache-Control: no-cache",
                                                     f"repos/{REPOSITORY}/git/blobs/{blob}"],
                                                    lambda handle: hash_stream(handle, expected["size"], blob, True, [e["path"] for e in entries], "actual_remote"))
                            if actual["sha256"] != expected["sha256"]:
                                raise ValueError("Actual remote SHA256 mismatch")
                            return {"git_blob": blob, **actual, "paths": [e["path"] for e in entries],
                                    "verification_source": "actual_remote_raw_GitHub_blob"}
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(attempt + 1)
                verified = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for item in pool.map(fetch, sorted(groups.items())):
                        verified.append(item)
                        log({"phase": "actual_remote_blob_verified", **item})
                        if len(verified) % 25 == 0:
                            print(json.dumps({"phase": "actual_remote_blobs", "verified": len(verified), "total": len(groups)}), flush=True)
                receipt["remote_verification"] = {"remote_tree_sha": tree_sha, "verified_files": len(inventory),
                    "verified_unique_blobs": len(verified), "blob_receipts": verified,
                    "remote_path_coverage": "Every exact frozen inventory path matched local and actual remote tree, mode, size and actual raw blob bytes",
                    "http_accept": "application/vnd.github.raw+json", "cache_policy": "no gh --cache; Cache-Control: no-cache; every blob fetched this run"}
                receipt["actual_remote_metadata_exception_scan"]=validate_metadata_report(complete_metadata_stream_report("actual_remote"),True)
                receipt["status"] = "BACKED_UP"
                receipt["remote_commit_sha"] = args.commit
            check_custody(worktree, args.commit)
            if hash_file(args.spec)["sha256"] != frozen_spec_sha:
                raise ValueError("Frozen spec changed during recovery")
            final_evolved = [change for entry in inventory
                             if (change := check_source(factory, entry, args.allow_evolving_source))]
            receipt["current_source_evolution_at_finish"] = final_evolved
            for path, expected in zip(args.prior_evidence, evidence):
                if hash_file(path) != {"size": expected["size"], "sha256": expected["sha256"]}:
                    raise ValueError("Prior failure evidence changed during recovery")
            receipt["capped_source_secret_scan_after"]=validate_metadata_report(metadata_helper().report(require_complete=True))
            receipt["checkpoint_custody_guards_after"]=full_checkpoint_guards(worktree,args.commit,spec,local_paths)
            if check_new_source_freeze(args)!=source_freeze:raise ValueError("Frozen preparation changed")
            receipt["local_raw_representation_decode_iterations"]=sum(len(e["lossless_representations"]) for e in spec.get("raw_size_exclusions",[]))
            receipt["local_raw_representation_decoded_bytes"]=sum(e["size"]*len(e["lossless_representations"]) for e in spec.get("raw_size_exclusions",[]))
            receipt["secret_scan_truth"]="PASS_WITH_EXACT_PUBLIC_METADATA_FALSE_POSITIVE_EXCEPTIONS"
            receipt["finished_at_utc"] = now()
            receipt["ledger"] = str(args.ledger.resolve())
            log({"phase": "verification_complete", "status": receipt["status"]})
            write_exclusive(args.output, receipt)
            print(json.dumps({"status": receipt["status"], "files": len(inventory),
                              "commit_sha": args.commit, "receipt": str(args.output.resolve())}), flush=True)
        except Exception as error:
            receipt.update(status="FAILED_ACTUAL_REMOTE_VERIFICATION_EVIDENCE_PRESERVED",error_type=type(error).__name__,error=str(error),finished_at_utc=now())
            if not args.output.exists():write_exclusive(args.output,receipt)
            try:
                log({"phase": "FAILED_NEW_EVIDENCE_PRESERVED", "error_type": type(error).__name__, "error": str(error)})
            except OSError:
                pass
            raise


if __name__ == "__main__":
    main()
