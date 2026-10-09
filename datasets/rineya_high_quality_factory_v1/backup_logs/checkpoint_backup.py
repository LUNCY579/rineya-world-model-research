#!/usr/bin/env python3
"""Back up an explicitly approved byte inventory to the private dedicated branch.

Approval spec (every path relative to the factory):
  {"checkpoint_id": "pilot-v0001", "approval": "provisional_checkpoint",
   "approved_by": "/root", "validation_status": "passed",
   "files": [{"path": "README.md", "size": 123, "sha256": "..."}]}

The helper never recursively copies, deletes destination files, switches the
original checkout, stages outside the dataset prefix, or force-pushes. Receipts
written after a push are intentionally captured by the next checkpoint.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "LUNCY579/rineya-world-model-research"
BRANCH = "research/rineya-high-quality-dataset-v1"
DESTINATION = "datasets/rineya_high_quality_factory_v1"
MAX_BYTES = 25 * 1024 * 1024
APPROVALS = {"reviewed_release", "provisional_checkpoint", "code_review_milestone"}
SECRET_PATTERNS = [
    re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?im)^\s*(?:api_key|access_token|password|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_+/=-]{16,}"),
]


def command(args: list[str], cwd: Path | None = None, *, raw: bool = False):
    result = subprocess.run(args, cwd=cwd, capture_output=True, check=False)
    if result.returncode:
        # Command errors should not contain credentials: only fixed git/gh calls
        # are used, and gh's token lookup is never printed.
        raise RuntimeError(f"{args[0]} {' '.join(args[1:3])} failed: "
                           + result.stderr.decode("utf-8", "replace").strip())
    return result.stdout if raw else result.stdout.decode("utf-8").strip()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".backup-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(factory: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or str(rel) != relative or any(piece in {"..", ".git"} for piece in rel.parts):
        raise ValueError(f"Unsafe source path: {relative}")
    if not rel.parts or "\n" in relative or "\r" in relative:
        raise ValueError("Invalid source path")
    result = factory.joinpath(*rel.parts)
    internal_parents = [factory.joinpath(*rel.parts[:index]) for index in range(1, len(rel.parts))]
    if result.is_symlink() or any(parent.is_symlink() for parent in internal_parents):
        raise ValueError(f"Symlink source rejected: {relative}")
    if not result.is_file() or not result.resolve().is_relative_to(factory):
        raise ValueError(f"Missing or out-of-scope source: {relative}")
    if result.name in {".env", ".DS_Store"} or result.suffix in {".pem", ".key", ".pyc"}:
        raise ValueError(f"Excluded sensitive or generated file: {relative}")
    return result


def validated_inventory(factory: Path, spec: dict) -> list[dict]:
    if spec.get("approval") not in APPROVALS or not spec.get("approved_by"):
        raise ValueError("Checkpoint needs explicit approval type and approving agent")
    if spec.get("validation_status") != "passed":
        raise ValueError("Checkpoint validation_status must be passed")
    entries = spec.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Checkpoint must list actual files")
    seen: set[str] = set()
    checked = []
    for entry in entries:
        relative = entry["path"]
        if relative in seen:
            raise ValueError(f"Duplicate inventory path: {relative}")
        seen.add(relative)
        source = safe_path(factory, relative)
        data = source.read_bytes()
        if len(data) > MAX_BYTES:
            raise ValueError(f"File exceeds repository 25 MiB cap: {relative}")
        if len(data) != entry.get("size") or digest(data) != entry.get("sha256"):
            raise ValueError(f"Approved source bytes changed: {relative}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Potential secret in {relative}; content withheld")
        checked.append({"path": relative, "size": len(data), "sha256": digest(data)})
    return sorted(checked, key=lambda entry: entry["path"])


def verify_worktree(worktree: Path, *, require_clean: bool = True):
    if command(["git", "branch", "--show-current"], worktree) != BRANCH:
        raise RuntimeError("Dedicated worktree is on the wrong branch")
    if command(["git", "remote", "get-url", "origin"], worktree) != f"https://github.com/{REPOSITORY}.git":
        raise RuntimeError("Unexpected origin remote")
    metadata = json.loads(command(["gh", "repo", "view", REPOSITORY, "--json", "isPrivate,viewerPermission,nameWithOwner"]))
    if metadata.get("nameWithOwner") != REPOSITORY or not metadata.get("isPrivate"):
        raise RuntimeError("Repository identity/private visibility verification failed")
    if metadata.get("viewerPermission") not in {"ADMIN", "MAINTAIN", "WRITE"}:
        raise RuntimeError("GitHub write permission unavailable")
    if require_clean and command(["git", "status", "--porcelain=v1", "--untracked-files=all"], worktree):
        raise RuntimeError("Dedicated worktree is dirty; preserve it and inspect before resuming")
    return metadata


def verify_remote_bytes(worktree: Path, commit: str, inventory: list[dict]):
    # Read the remote commit and tree from GitHub itself. A locally cached object
    # alone does not certify that GitHub stores the bytes.
    remote_commit = json.loads(command(["gh", "api", f"repos/{REPOSITORY}/git/commits/{commit}"]))
    tree_sha = remote_commit["tree"]["sha"]
    local_tree = command(["git", "rev-parse", f"{commit}^{{tree}}"], worktree)
    if tree_sha != local_tree:
        raise RuntimeError("Remote and local commit tree disagree")
    tree = json.loads(command(["gh", "api", f"repos/{REPOSITORY}/git/trees/{tree_sha}?recursive=1"]))
    if tree.get("truncated"):
        raise RuntimeError("Remote tree response truncated; cannot certify inventory")
    blobs = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    expected_by_blob: dict[str, list[dict]] = {}
    for entry in inventory:
        remote_path = f"{DESTINATION}/{entry['path']}"
        remote_blob = blobs.get(remote_path)
        if remote_blob is None or remote_blob.get("size") != entry["size"]:
            raise RuntimeError(f"Remote path/size mismatch: {remote_path}")
        expected_by_blob.setdefault(remote_blob["sha"], []).append(entry)

    def fetch_blob(pair):
        sha, entries = pair
        blob = json.loads(command(["gh", "api", f"repos/{REPOSITORY}/git/blobs/{sha}"]))
        if blob.get("encoding") != "base64":
            raise RuntimeError(f"Unsupported remote blob encoding: {sha}")
        data = base64.b64decode(blob["content"])
        checksum = digest(data)
        for entry in entries:
            if checksum != entry["sha256"] or len(data) != entry["size"]:
                raise RuntimeError(f"Remote SHA256 mismatch: {entry['path']}")
        return {"git_blob": sha, "sha256": checksum, "size": len(data),
                "paths": [entry["path"] for entry in entries]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(fetch_blob, sorted(expected_by_blob.items())))
    return {"remote_tree_sha": tree_sha, "verified_unique_blobs": len(verified),
            "verified_files": len(inventory), "blob_receipts": verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume-commit", help="Retry a preserved local checkpoint commit without recopying")
    args = parser.parse_args()
    factory = args.factory.resolve()
    worktree = (args.worktree or factory.parent / ".rineya-dataset-git-worktree").resolve()
    if factory.is_relative_to(worktree) or worktree.is_relative_to(factory):
        raise ValueError("Source factory and backup worktree must be separate")
    spec = json.loads(args.spec.read_text())
    checkpoint = spec.get("checkpoint_id", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", checkpoint):
        raise ValueError("Invalid checkpoint_id")
    receipt_path = factory / "backup_logs" / f"{checkpoint}_RECEIPT.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "BACKED_UP":
        raise ValueError("A verified checkpoint receipt is immutable; use a new checkpoint_id")
    receipt = {"checkpoint_id": checkpoint, "execution_role": "GitHub Integrity and Backup Manager",
               "helper_author_execution": "/root/backup_manager", "runner_execution": spec.get("runner_execution", spec.get("approved_by")), "repository": REPOSITORY,
               "branch": BRANCH, "approval": spec.get("approval"), "approved_by": spec.get("approved_by"),
               "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "PENDING"}
    try:
        inventory = validated_inventory(factory, spec)
        receipt["inventory"] = inventory
        receipt["files"] = len(inventory)
        receipt["total_bytes"] = sum(entry["size"] for entry in inventory)
        receipt["private_repository_evidence"] = verify_worktree(worktree)
        required_space = receipt["total_bytes"] * 2 + 32 * 1024 * 1024
        if shutil.disk_usage(worktree).free < required_space:
            raise RuntimeError("Insufficient space for safe copy and Git checkpoint")
        command(["git", "fetch", "origin", "--prune"], worktree)
        remote = command(["git", "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}"], worktree)
        if remote:
            remote_sha = remote.split()[0]
            command(["git", "merge-base", "--is-ancestor", remote_sha, "HEAD"], worktree)
            receipt["remote_before_sha"] = remote_sha
        if args.validate_only:
            print(json.dumps({"status": "VALIDATED_NO_COPY", "files": receipt["files"], "total_bytes": receipt["total_bytes"]}))
            return
        if args.resume_commit:
            commit = command(["git", "rev-parse", args.resume_commit], worktree)
            if commit != command(["git", "rev-parse", "HEAD"], worktree):
                raise RuntimeError("Resume commit must be dedicated branch HEAD")
            for entry in inventory:
                data = command(["git", "show", f"{commit}:{DESTINATION}/{entry['path']}"], worktree, raw=True)
                if digest(data) != entry["sha256"]:
                    raise RuntimeError(f"Resume commit inventory mismatch: {entry['path']}")
        else:
            for entry in inventory:
                source = safe_path(factory, entry["path"])
                target = worktree / DESTINATION / entry["path"]
                data = source.read_bytes()
                if digest(data) != entry["sha256"]:
                    raise RuntimeError(f"Source changed during copy: {entry['path']}")
                if target.exists() and "/releases/" in f"/{entry['path']}" and target.read_bytes() != data:
                    raise RuntimeError(f"Immutable release path would change: {entry['path']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".dataset-copy-", dir=target.parent)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                if digest(target.read_bytes()) != entry["sha256"]:
                    raise RuntimeError(f"Copied-byte mismatch: {entry['path']}")
            # Explicit literal path arguments avoid a broad git add and reject
            # any pre-existing changes before copying.
            command(["git", "--literal-pathspecs", "add", "--", *[f"{DESTINATION}/{entry['path']}" for entry in inventory]], worktree)
            staged = command(["git", "diff", "--cached", "--name-only", "-z"], worktree, raw=True).split(b"\0")
            if any(path and not path.decode().startswith(DESTINATION + "/") for path in staged):
                raise RuntimeError("Unexpected staged path outside dataset factory")
            if not any(staged):
                raise RuntimeError("No new checkpoint bytes to commit")
            receipt["diff_stat"] = command(["git", "diff", "--cached", "--stat"], worktree)
            receipt["diff_numstat"] = command(["git", "diff", "--cached", "--numstat"], worktree)
            command(["git", "diff", "--cached", "--check"], worktree)
            command(["git", "commit", "-m", args.message], worktree)
            commit = command(["git", "rev-parse", "HEAD"], worktree)
        receipt["commit_sha"] = commit
        receipt["status"] = "LOCAL_COMMIT_PRESERVED"
        atomic_json(receipt_path, receipt)
        command(["git", "push", "--set-upstream", "origin", f"HEAD:refs/heads/{BRANCH}"], worktree)
        remote_after = command(["git", "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}"], worktree)
        if remote_after.split()[0] != commit:
            raise RuntimeError("Remote branch SHA differs after push")
        command(["git", "fetch", "origin", f"refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}"], worktree)
        receipt["remote_verification"] = verify_remote_bytes(worktree, commit, inventory)
        receipt["remote_commit_sha"] = commit
        receipt["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        receipt["status"] = "BACKED_UP"
        atomic_json(receipt_path, receipt)
        print(json.dumps({"status": "BACKED_UP", "commit_sha": commit, "files": len(inventory), "receipt": str(receipt_path)}))
    except Exception as error:
        receipt["error"] = str(error)
        if receipt.get("status") != "LOCAL_COMMIT_PRESERVED":
            receipt["status"] = "FAILED_LOCAL_ARTIFACTS_PRESERVED"
        atomic_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Backup failed: {error}", file=sys.stderr)
        sys.exit(1)
