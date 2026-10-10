#!/usr/bin/env python3
"""Preserve manifest-required source bytes without reconstruction or release edits.

Run from the factory: PYTHONDONTWRITEBYTECODE=1 python3
reproducibility/archive_sources.py --runner /root

Reads current/archive source files and existing dedicated-branch Git objects.
It never fetches, pushes, modifies a release, or substitutes guessed source.
Content-addressed payloads remain immutable; the index describes this run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

FACTORY = Path(__file__).resolve().parent.parent
WORKTREE = FACTORY.parent / ".rineya-dataset-git-worktree"
PREFIX = "datasets/rineya_high_quality_factory_v1/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(args, *, optional=False):
    result = subprocess.run(["git", *args], cwd=WORKTREE, capture_output=True)
    if result.returncode:
        if optional:
            return None
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    return result.stdout


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".source-archive-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    args = parser.parse_args()
    if git(["branch", "--show-current"]).decode().strip() != "research/rineya-high-quality-dataset-v1":
        raise RuntimeError("Expected dedicated source-history branch unavailable")
    source_head = git(["rev-parse", "HEAD"]).decode().strip()
    manifests = []
    requirements = {}
    for manifest in sorted(FACTORY.glob("data/*/releases/*/MANIFEST.json")):
        data = manifest.read_bytes()
        parsed = json.loads(data)
        relative_manifest = manifest.relative_to(FACTORY).as_posix()
        manifests.append({"path": relative_manifest, "sha256": sha256(data),
                          "track": parsed["track"], "release": parsed["release"]})
        for relative, checksum in parsed.get("reproducibility_code_sha256", {}).items():
            source = FACTORY / relative
            if not source.resolve().is_relative_to(FACTORY) or source.is_symlink():
                raise RuntimeError(f"Unsafe manifest source path: {relative}")
            requirements.setdefault((relative, checksum), []).append(relative_manifest)

    # Snapshot bytes before recording origins, so mutable current files cannot
    # make an index claim a digest without preserving those actual bytes.
    local_by_hash = {}
    for source in sorted(FACTORY.rglob("*")):
        if not source.is_file() or source.is_symlink() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(FACTORY).as_posix()
        if source.suffix != ".py" and "schemas" not in source.relative_to(FACTORY).parts:
            continue
        data = source.read_bytes()
        local_by_hash.setdefault(sha256(data), (relative, data))

    commits = git(["rev-list", source_head, "--", PREFIX]).decode().split()
    object_cache = {}
    entries = []
    for (relative, checksum), release_manifests in sorted(requirements.items()):
        entry = {"original_path": relative, "sha256": checksum,
                 "release_manifests": sorted(release_manifests)}
        data = None
        existing_blob=FACTORY/f'reproducibility/source_blobs/{checksum}.source'
        if existing_blob.is_file():
            data=existing_blob.read_bytes()
            if sha256(data)!=checksum:
                raise RuntimeError('Existing source archive hash mismatch')
            entry['origin']={'kind':'existing_content_addressed_archive','path':existing_blob.relative_to(FACTORY).as_posix(),'captured_sha256':checksum}
        elif checksum in local_by_hash:
            local_path, data = local_by_hash[checksum]
            entry["origin"] = {"kind": "actual_local_file", "path": local_path,
                               "captured_sha256": checksum}
        else:
            for commit in commits:
                key = (commit, relative)
                if key not in object_cache:
                    payload = git(["show", f"{commit}:{PREFIX}{relative}"], optional=True)
                    object_cache[key] = (sha256(payload), payload) if payload is not None else (None, None)
                actual, payload = object_cache[key]
                if actual == checksum:
                    data = payload
                    entry["origin"] = {"kind": "dedicated_prior_commit", "commit": commit,
                                       "path": PREFIX + relative, "captured_sha256": checksum}
                    break
        if data is None:
            entry.update(status="MISSING_EXACT_ORIGINAL_BYTES", blob=None,
                         explanation="No matching actual local/archive bytes or dedicated Git-history object; no reconstruction substituted.")
        else:
            if sha256(data) != checksum:
                raise RuntimeError("Recovered payload hash changed")
            blob_relative = f"reproducibility/source_blobs/{checksum}.source"
            blob = FACTORY / blob_relative
            if blob.exists() and blob.read_bytes() != data:
                raise RuntimeError(f"Immutable content-addressed blob mismatch: {checksum}")
            if not blob.exists():
                atomic_bytes(blob, data)
            if sha256(blob.read_bytes()) != checksum:
                raise RuntimeError("Archived source hash mismatch")
            entry.update(status="PRESERVED_EXACT_BYTES", blob=blob_relative, bytes=len(data))
        entries.append(entry)

    sys.path.insert(0, str(FACTORY))
    from factory.catalog import catalog
    _, active = catalog()
    active_manifests={f"{e['path']}/MANIFEST.json" for e in active['releases'] if e['active_records']}
    for entry in entries:
        entry['referenced_by_active_release']=bool(set(entry['release_manifests'])&active_manifests)
    report = {"schema_version": "1.0", "archive_type": "actual_source_payloads_no_reconstruction",
              "archive_script_author_execution": "/root/backup_manager", "runner_execution": args.runner,
              "archive_script_improvement_execution":"/root; reuse exact existing archives and report active source coverage",
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "dedicated_history_head": source_head, "manifest_inventory": manifests,
              "requirements": len(entries), "preserved_requirements": sum(e["status"] == "PRESERVED_EXACT_BYTES" for e in entries),
              "missing_requirements": sum(e["status"] == "MISSING_EXACT_ORIGINAL_BYTES" for e in entries),
              "missing_active_requirements":sum(e['status']=='MISSING_EXACT_ORIGINAL_BYTES' and e['referenced_by_active_release'] for e in entries),
              "active_catalog_snapshot": active,
              "entries": entries,
              "custody": "Existing release files are read-only. Source blobs preserve exact hash-matching payloads. Index is a rerunnable snapshot; prior blobs are never deleted. Remote persistence requires a later exact-inventory checkpoint."}
    index=FACTORY/'reproducibility/SOURCE_ARCHIVE_INDEX.json'
    if index.exists():
        previous=index.read_bytes();old_snapshot=FACTORY/'reproducibility/archive_history'/f'{sha256(previous)}.json'
        if old_snapshot.exists() and old_snapshot.read_bytes()!=previous:raise RuntimeError('Prior source index mismatch')
        if not old_snapshot.exists():atomic_bytes(old_snapshot,previous)
    atomic_bytes(index,(json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({key: report[key] for key in ["requirements", "preserved_requirements", "missing_requirements","missing_active_requirements"]}))


if __name__ == "__main__":
    main()
