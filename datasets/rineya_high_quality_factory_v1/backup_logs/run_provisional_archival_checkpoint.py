#!/usr/bin/env python3
"""Copy, stage, commit and nonforce-push a frozen exact inventory.

This NEW runner never certifies remote payload persistence. Run the unchanged
bounded recover_remote_checkpoint.py with two workers after a successful push.
All payload reads/copies use 64 KiB chunks; metadata responses are capped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading

AUTHOR = "/root/backup_manager"
FACTORY = Path(__file__).resolve().parent.parent
PREFIX = "datasets/rineya_high_quality_factory_v1"
BRANCH = "research/rineya-high-quality-dataset-v1"
REPOSITORY = "LUNCY579/rineya-world-model-research"
PARENT = "448c36af9370460782da3c0a4ba7c5ca4d8ba71a"
ORIGINAL_HEAD = "a223884d55ff47baf3890c7d4a82054d07223215"
RECOVERY_SHA = "3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1"
OLD_HELPER_SHA = "07ef424f92d903c265c6cc06622733dcdeabf5bd03d3c13d98f793c2ddcc2d4b"
CHUNK = 64 * 1024
MAX_BYTES = 25 * 1024 * 1024
RESERVE = 512 * 1024 * 1024
ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "PYTHONDONTWRITEBYTECODE": "1"}
sys.dont_write_bytecode = True
SECRET_PATTERNS = [
    re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?im)^\s*(?:api_key|access_token|password|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_+/=-]{16,}"),
]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(args, cwd=None, allowed=(0,), data=None):
    if data is not None and len(data) > 16 * 1024 * 1024:
        raise ValueError("Metadata input exceeds bounded limit")
    p = subprocess.Popen(args, cwd=cwd, stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=ENV)
    timer = threading.Timer(180, p.kill);timer.daemon = True;timer.start()
    writer = None
    if data is not None:
        def feed():
            try:p.stdin.write(data);p.stdin.close()
            except BrokenPipeError:pass
        writer = threading.Thread(target=feed, daemon=True);writer.start()
    try:
        output = bytearray()
        while True:
            part = p.stdout.read(CHUNK)
            if not part:break
            output.extend(part)
            if len(output) > 16 * 1024 * 1024:
                raise ValueError("Metadata response exceeds bounded limit")
        code = p.wait(timeout=10)
        if code not in allowed:
            raise RuntimeError(f"{args[0]} command failed with exit {code}; output withheld")
        return bytes(output)
    finally:
        timer.cancel();p.stdout.close()
        if p.poll() is None:p.kill()
        p.wait()
        if writer:writer.join(timeout=1)


def git(worktree, *args, **kwargs):
    return run(["git", *args], worktree, **kwargs)


def text(worktree, *args):
    return git(worktree, *args).decode().strip()


def safe(root, relative, must_exist=True):
    if (not isinstance(relative, str) or not relative or "\0" in relative
            or "\n" in relative or "\r" in relative):
        raise ValueError("Invalid literal inventory path")
    p = Path(relative)
    if p.is_absolute() or p.as_posix() != relative or any(x in {"..", ".git"} for x in p.parts):
        raise ValueError("Unsafe inventory path")
    root = root.resolve()
    result = root / p
    for i in range(1, len(p.parts) + 1):
        if root.joinpath(*p.parts[:i]).is_symlink():
            raise ValueError("Symlink inventory path rejected")
    if not result.resolve().is_relative_to(root):
        raise ValueError("Inventory path escapes its root")
    if result.name in {".env", ".DS_Store"} or result.suffix in {".key", ".pem", ".pyc"}:
        raise ValueError("Sensitive/generated path rejected")
    if must_exist and not result.is_file():
        raise ValueError("Missing regular inventory file")
    return result


def stream_proof(handle, size, scan=True, sink=None):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    count, tail = 0, b""
    while True:
        data = handle.read(CHUNK)
        if not data:
            break
        count += len(data)
        if count > size:
            raise ValueError("Payload exceeds frozen size")
        sha.update(data)
        blob.update(data)
        if scan:
            if any(p.search(tail + data) for p in SECRET_PATTERNS):
                raise ValueError("Potential secret; content withheld")
            tail = (tail + data)[-4096:]
        if sink:
            sink.write(data)
    if count != size:
        raise ValueError("Payload size differs from frozen size")
    return {"size": count, "sha256": sha.hexdigest(), "git_blob": blob.hexdigest()}


def proof(path, scan=True):
    before = path.stat()
    with path.open("rb") as h:
        out = stream_proof(h, before.st_size, scan)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, f) != getattr(after, f) for f in fields):
        raise ValueError("Source changed during streaming validation")
    return out


def equal(entry, actual):
    return entry["size"] == actual["size"] and entry["sha256"] == actual["sha256"]


def inventory(factory, spec):
    if (spec.get("approval") not in {"provisional_checkpoint", "code_review_milestone", "reviewed_release"}
            or not spec.get("approved_by") or spec.get("validation_status") != "passed"):
        raise ValueError("Explicit validated checkpoint authorization is missing")
    entries = spec.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Exact inventory is empty")
    out = {}
    for e in entries:
        p = e.get("path")
        if p in out or type(e.get("size")) is not int or not 0 <= e["size"] <= MAX_BYTES:
            raise ValueError("Duplicate path or invalid capped size")
        if not re.fullmatch(r"[0-9a-f]{64}", e.get("sha256", "")):
            raise ValueError("Invalid SHA256")
        actual = proof(safe(factory, p))
        if not equal(e, actual):
            raise ValueError(f"Frozen source changed: {p}")
        out[p] = actual
    return out


def declared_path(factory, value):
    p = Path(value)
    if p.is_absolute():
        value = p.relative_to(factory.resolve()).as_posix()
    return safe(factory, value)


def lossless(factory, spec, checked):
    exclusions = spec.get("raw_size_exclusions", [])
    names = [e["path"] for e in exclusions]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate oversized raw exclusion")
    actual_oversized = {p.relative_to(factory).as_posix() for p in factory.rglob("*")
                        if p.is_file() and p.stat().st_size > MAX_BYTES
                        and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual_oversized != set(names):
        raise ValueError("Oversized local evidence is missing or silently excluded")
    verified = []
    for e in exclusions:
        if e["size"] <= MAX_BYTES or e["path"] in checked:
            raise ValueError("Invalid raw cap exclusion")
        raw = safe(factory, e["path"])
        raw_proof = proof(raw)
        if not equal(e, raw_proof) or not e.get("lossless_representations"):
            raise ValueError("Frozen raw differs or has no lossless representation")
        for rep in e["lossless_representations"]:
            for key in ("gzip", "manifest"):
                r = rep[key]
                if r["path"] not in checked or not equal(r, checked[r["path"]]):
                    raise ValueError("Lossless evidence omitted from exact inventory")
            gz = safe(factory, rep["gzip"]["path"])
            manifest = json.loads(safe(factory, rep["manifest"]["path"]).read_bytes())
            records = manifest.get("artifacts", manifest.get("items", [manifest]))
            matching = False
            for rec in records:
                raw_path = declared_path(factory, rec["raw_path"])
                gz_path = declared_path(factory, rec.get("gzip_path", rec.get("compressed_path")))
                if raw_path == raw and gz_path == gz:
                    matching = (rec.get("raw_bytes", rec.get("raw_length")) == e["size"]
                                and rec["raw_sha256"] == e["sha256"]
                                and rec.get("gzip_bytes", rec.get("compressed_length")) == rep["gzip"]["size"]
                                and rec.get("gzip_sha256", rec.get("compressed_sha256")) == rep["gzip"]["sha256"])
                    if matching:
                        break
            if not matching:
                raise ValueError("Actual lossless manifest does not bind the frozen bytes")
            with raw.open("rb") as a, gzip.open(gz, "rb") as b:
                while True:
                    x, y = a.read(CHUNK), b.read(CHUNK)
                    if x != y:
                        raise ValueError("Actual gzip/raw byte comparison failed")
                    if not x:
                        break
            if not equal(e, proof(raw, False)):
                raise ValueError("Raw changed during lossless comparison")
        verified.append({"path": e["path"], "size": e["size"], "sha256": e["sha256"],
                         "secret_scan": "PASS", "byte_for_byte_lossless": True})
    return verified


def tree(worktree, commit):
    out = {}
    for rec in git(worktree, "ls-tree", "-r", "-z", commit, "--", PREFIX).split(b"\0"):
        if not rec:
            continue
        left, name = rec.split(b"\t", 1)
        mode, kind, blob = left.decode().split()
        name = name.decode()
        if not name.startswith(PREFIX + "/") or kind != "blob" or mode != "100644":
            raise ValueError("Unexpected tracked dataset mode/path/type")
        out[name[len(PREFIX) + 1:]] = blob
    return out


def guard_previous(previous, checked):
    if not previous.keys() <= checked.keys():
        raise ValueError("Previously tracked artifacts cannot be omitted")
    for name, blob in previous.items():
        if "/releases/" in "/" + name and checked[name]["git_blob"] != blob:
            raise ValueError("Immutable historical release would change")


def copy_one(factory, worktree, name, expected):
    source = safe(factory, name)
    target = safe(worktree, PREFIX + "/" + name, False)
    if target.is_file() and proof(target, False) == expected:
        return "UNCHANGED_TARGET_REVERIFIED"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".archival-copy-", dir=target.parent)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            actual = stream_proof(src, expected["size"], scan=False, sink=dst)
            dst.flush()
            os.fsync(dst.fileno())
        if actual != expected:
            raise ValueError("Frozen source changed during copy")
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
        if proof(target, False) != expected:
            raise ValueError("Copied target bytes differ")
        return "COPIED_AND_REVERIFIED"
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this runner's incomplete temporary file.


def index(worktree):
    out = {}
    for rec in git(worktree, "ls-files", "--stage", "-z", "--", PREFIX).split(b"\0"):
        if not rec:
            continue
        left, path = rec.split(b"\t", 1)
        mode, blob, stage = left.decode().split()
        path = path.decode()
        if stage != "0" or mode != "100644" or not path.startswith(PREFIX + "/"):
            raise ValueError("Unexpected index conflict/mode/path")
        out[path[len(PREFIX) + 1:]] = blob
    return out


def stage_exact(worktree, checked, allow_ignored):
    tracked = set(index(worktree))
    names = sorted(checked)
    paths = [PREFIX + "/" + n for n in names]
    ignored = {p.decode()[len(PREFIX) + 1:] for p in
               git(worktree, "check-ignore", "--no-index", "-z", "--stdin", allowed=(0, 1),
                   data=("\0".join(paths) + "\0").encode()).split(b"\0") if p}
    forced = ignored - tracked
    if forced != set(allow_ignored):
        raise ValueError("New ignored paths require exact explicit --allow-ignored-path declarations")
    for path in forced:
        if path not in checked:
            raise ValueError("Forced path outside exact inventory")
    regular = [p for p in names if p not in forced]
    for batch, force in ((regular, False), (sorted(forced), True)):
        for start in range(0, len(batch), 100):
            args = ["--literal-pathspecs", "add"] + (["--force"] if force else [])
            git(worktree, *args, "--", *[PREFIX + "/" + p for p in batch[start:start + 100]])
    staged = [x.decode() for x in git(worktree, "diff", "--cached", "--name-only", "-z").split(b"\0") if x]
    if not staged or any(not p.startswith(PREFIX + "/") or p[len(PREFIX) + 1:] not in checked for p in staged):
        raise ValueError("Empty or unrelated staged paths")
    if git(worktree, "diff", "--cached", "--diff-filter=D", "--name-only", "-z"):
        raise ValueError("No tracked deletion is authorized")
    git(worktree, "diff", "--cached", "--check")  # Strict; no whitespace exception.
    actual_index = index(worktree)
    if set(actual_index) != set(checked) or any(actual_index[p] != checked[p]["git_blob"] for p in checked):
        raise ValueError("Full exact inventory index blob IDs differ")
    return {"ignored_inventory_paths": sorted(ignored), "new_ignored_exact_paths_force_staged": sorted(forced),
            "staged_paths": staged, "strict_diff_check": "PASS"}


def exclusive(path, value):
    with path.open("x", encoding="utf-8") as h:
        json.dump(value, h, indent=2)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())


def original_custody(original, baseline):
    if (text(original, "rev-parse", "HEAD") != ORIGINAL_HEAD
            or text(original, "branch", "--show-current") != "main"
            or text(original, "status", "--porcelain=v1", "--untracked-files=all")):
        raise ValueError("Original main custody changed")
    for item in baseline["protected_git_byte_baselines"]:
        p = Path(item["path"])
        if p.exists() != item["expected_exists"]:
            raise ValueError("Protected Git baseline existence changed")
        if p.exists() and proof(p, False)["sha256"] != item["sha256"]:
            raise ValueError("Protected original/common Git bytes changed")


def dedicated_custody(worktree, expected, clean=True):
    if (text(worktree, "rev-parse", "HEAD") != expected
            or text(worktree, "branch", "--show-current") != BRANCH
            or text(worktree, "remote", "get-url", "origin") != f"https://github.com/{REPOSITORY}.git"):
        raise ValueError("Dedicated HEAD/branch/remote custody changed")
    if clean and text(worktree, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Dedicated worktree is dirty; preserve it")
    remote = text(worktree, "ls-remote", "--heads", "origin", "refs/heads/" + BRANCH)
    if remote.split() != [expected, "refs/heads/" + BRANCH]:
        raise ValueError("Remote branch is not the exact expected parent/commit")
    meta = json.loads(run(["gh", "repo", "view", REPOSITORY, "--json", "isPrivate,nameWithOwner,viewerPermission"]))
    if meta != {"isPrivate": True, "nameWithOwner": REPOSITORY, "viewerPermission": "ADMIN"}:
        raise ValueError("Actual private ADMIN repository verification failed")
    if text(worktree, "config", "--get", "index.sparse") != "false":
        raise ValueError("Expected full Git index unavailable")
    if set(text(worktree, "sparse-checkout", "list").splitlines()) != {"tools", PREFIX}:
        raise ValueError("Expected restored dedicated sparse scope unavailable")
    return meta


def frozen_inputs(args):
    logs = FACTORY / "backup_logs"
    for p in (args.receipt, args.local_commit_receipt, args.ledger):
        if p.exists() or p.is_symlink() or not p.resolve().is_relative_to(logs):
            raise ValueError("Evidence outputs must be fresh paths in backup_logs")
    if len({p.resolve() for p in (args.receipt, args.local_commit_receipt, args.ledger)}) != 3:
        raise ValueError("Evidence output paths must be distinct")
    for p in (args.spec, args.source_freeze, args.custody_baseline):
        if p.is_symlink() or not p.resolve().is_relative_to(logs):
            raise ValueError("Frozen input must be a local backup_logs file")
    expectations = ((args.spec, args.expected_spec_sha256), (args.source_freeze, args.expected_freeze_sha256),
                    (args.custody_baseline, args.expected_custody_sha256),
                    (logs / "recover_remote_checkpoint.py", RECOVERY_SHA),
                    (logs / "checkpoint_backup.py", OLD_HELPER_SHA))
    for p, expected in expectations:
        if proof(p, False)["sha256"] != expected:
            raise ValueError("Frozen input/helper checksum differs")
    freeze = json.loads(args.source_freeze.read_bytes())
    if (freeze["runner_source"]["sha256"] != proof(Path(__file__), False)["sha256"]
            or freeze.get("author_execution") != AUTHOR):
        raise ValueError("Runner lacks its exact independent source freeze")
    return [(p, expected) for p, expected in expectations]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "source-freeze", "custody-baseline", "receipt", "local-commit-receipt", "ledger"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-spec-sha256", "expected-freeze-sha256", "expected-custody-sha256", "message", "runner"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--allow-ignored-path", action="append", default=[])
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    frozen = frozen_inputs(args)
    spec = json.loads(args.spec.read_bytes());baseline = json.loads(args.custody_baseline.read_bytes())
    worktree = FACTORY.parent / ".rineya-dataset-git-worktree"
    original = FACTORY.parent / "rineya-world-model-research"
    if FACTORY.is_relative_to(worktree.resolve()) or worktree.resolve().is_relative_to(FACTORY):
        raise ValueError("Factory/worktree must be separate")
    report = {"author_execution": AUTHOR, "runner_execution": args.runner,
              "model_disclosure": "Inherited same-model independent agent execution; no model diversity claim",
              "started_at_utc": now(), "status": "PENDING_LOCAL_VALIDATION",
              "checkpoint_id": spec.get("checkpoint_id"), "spec_sha256": args.expected_spec_sha256,
              "source_sha256": proof(Path(__file__), False)["sha256"], "source_freeze_sha256": args.expected_freeze_sha256,
              "repository": REPOSITORY, "branch": BRANCH, "expected_parent_sha": PARENT,
              "approval": spec.get("approval"), "approved_by": spec.get("approved_by"), "grants_data_approval": False,
              "scope": "Exact frozen byte preservation only; no old helper edit, training, release approval, deletion, main modification, force push, or claim of remote payload verification",
              "self_inventory_boundary": "Runner execution receipts/ledger are post-inventory local evidence until a later exact checkpoint"}
    with args.ledger.open("x", encoding="utf-8") as ledger:
        def log(phase, **values):
            ledger.write(json.dumps({"at_utc": now(), "phase": phase, **values}, separators=(",", ":")) + "\n")
            ledger.flush();os.fsync(ledger.fileno())
        try:
            original_custody(original, baseline)
            report["private_repository_evidence"] = dedicated_custody(worktree, PARENT)
            checked = inventory(FACTORY, spec)
            report["raw_lossless_proofs"] = lossless(FACTORY, spec, checked)
            previous = tree(worktree, PARENT);guard_previous(previous, checked)
            required = 2 * sum(v["size"] for v in checked.values()) + RESERVE
            free = shutil.disk_usage(worktree).free
            if free < required:
                raise ValueError("Insufficient free space for 2x full inventory plus 512 MiB reserve")
            report.update(files=len(checked), total_bytes=sum(v["size"] for v in checked.values()),
                          free_bytes_at_preflight=free, required_free_bytes=required,
                          protected_original_baselines_verified=True, prior_release_guard="PASS")
            log("FULL_LOCAL_FROZEN_INPUTS_VALIDATED", files=len(checked), bytes=report["total_bytes"])
            if not args.execute:
                report["status"] = "VALIDATED_NO_COPY_COMMIT_PUSH"
                exclusive(args.receipt, report);print(json.dumps({"status": report["status"], "receipt": str(args.receipt)}));return
            for n, name in enumerate(sorted(checked), 1):
                outcome = copy_one(FACTORY, worktree, name, checked[name])
                log("source_target_verified", path=name, **checked[name], copy_outcome=outcome)
                if n % 100 == 0:
                    print(json.dumps({"phase": "verified_copy", "files": n, "total": len(checked)}), flush=True)
            original_custody(original, baseline)
            dedicated_custody(worktree, PARENT, False)
            report.update(stage_exact(worktree, checked, args.allow_ignored_path))
            groups = {}
            for name, value in checked.items():groups.setdefault(value["git_blob"], []).append(name)
            helper_spec = importlib.util.spec_from_file_location("frozen_remote", FACTORY / "backup_logs/recover_remote_checkpoint.py")
            helper = importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
            for i, (blob, names) in enumerate(sorted(groups.items()), 1):
                expected = checked[names[0]]
                actual = helper.process_stream(["git", "cat-file", "blob", blob],
                                               lambda handle: stream_proof(handle, expected["size"], scan=False), worktree)
                if actual != expected:
                    raise ValueError("Actual index object SHA256/SHA1 differs")
                log("actual_index_blob_verified", **actual, paths=names)
                if i % 100 == 0:print(json.dumps({"phase": "index_blob_verified", "blobs": i, "total": len(groups)}), flush=True)
            for name, expected in checked.items():
                if proof(safe(FACTORY, name), False) != expected or proof(safe(worktree, PREFIX + "/" + name), False) != expected:
                    raise ValueError("Frozen source or target changed before commit")
            for path, checksum in frozen:
                if proof(path, False)["sha256"] != checksum:raise ValueError("Frozen inputs changed before commit")
            if proof(Path(__file__), False)["sha256"] != report["source_sha256"]:
                raise ValueError("Source-frozen runner changed before commit")
            original_custody(original, baseline);dedicated_custody(worktree, PARENT, False)
            git(worktree, "diff", "--cached", "--check")
            report["all_source_target_index_sha256_and_sha1_verified"] = True
            git(worktree, "commit", "-m", args.message)
            commit = text(worktree, "rev-parse", "HEAD")
            if text(worktree, "rev-parse", commit + "^") != PARENT or tree(worktree, commit) != {n:v["git_blob"] for n,v in checked.items()}:
                raise ValueError("Committed exact tree/parent differs")
            report.update(status="LOCAL_COMMIT_PRESERVED", commit_sha=commit)
            exclusive(args.local_commit_receipt, report);log("LOCAL_COMMIT_PRESERVED", commit_sha=commit)
            original_custody(original, baseline)
            git(worktree, "push", "origin", "HEAD:refs/heads/" + BRANCH)
            dedicated_custody(worktree, commit)
            original_custody(original, baseline)
            report.update(status="PUSHED_REMOTE_BYTES_VERIFICATION_PENDING", remote_commit_sha=commit, finished_at_utc=now())
            report["next_actual_bounded_verification"] = ["python3", str(FACTORY / "backup_logs/recover_remote_checkpoint.py"),
                "--spec", str(args.spec), "--expected-spec-sha256", args.expected_spec_sha256,
                "--commit", commit, "--output", "NEW_RECOVERY_RECEIPT_IN_BACKUP_LOGS", "--ledger", "NEW_RECOVERY_LEDGER_IN_BACKUP_LOGS",
                "--prior-evidence", str(args.receipt), "--runner", args.runner, "--workers", "2"]
            exclusive(args.receipt, report);log("PUSHED_REMOTE_BYTES_VERIFICATION_PENDING", commit_sha=commit)
            print(json.dumps({"status": report["status"], "commit_sha": commit, "files": len(checked), "receipt": str(args.receipt)}))
        except Exception as error:
            report.update(status="FAILED_NEW_EXECUTION_LOCAL_ARTIFACTS_PRESERVED", error_type=type(error).__name__,
                          error=str(error), finished_at_utc=now())
            log("FAILED_NEW_EXECUTION_LOCAL_ARTIFACTS_PRESERVED", error_type=type(error).__name__, error=str(error))
            if not args.receipt.exists():exclusive(args.receipt, report)
            raise


if __name__ == "__main__":
    main()
