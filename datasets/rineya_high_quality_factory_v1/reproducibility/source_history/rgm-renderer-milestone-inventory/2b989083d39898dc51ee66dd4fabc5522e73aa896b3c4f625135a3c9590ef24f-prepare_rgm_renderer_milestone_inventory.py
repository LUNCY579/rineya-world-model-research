# Fresh root derivative preserving full prior inventory except explicitly unfinished NEW reviewer paths. Base SHA256 c8d052c54797f0d6970715ba640d124313e1b8c945eb7a5e6bc42e45556b08f4
# Root-authored compatibility derivative of unchanged manager inventory builder SHA256 ef3a45d6cf1b2cb53036f9d07c8f92abd4c135a395a25ef7975ff075ba5eb387
#!/usr/bin/env python3
"""Create an exact backup proposal, explicitly replacing oversized raw evidence.

This only reads factory/repository state and writes its report/spec in backup_logs.
It never creates gzip payloads, changes reviews/releases, copies, commits or pushes.
Rerun after adjudication/release/doc edits to refresh the actual inventory.
"""
from __future__ import annotations

import argparse
import fnmatch
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

FACTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FACTORY))
from backup_logs.checkpoint_backup import MAX_BYTES, SECRET_PATTERNS, atomic_json, safe_path


def snapshot(path):
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while reading: {path.relative_to(FACTORY)}")
    return {"path": path.relative_to(FACTORY).as_posix(), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}, data


def git(args, root):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


def local_declared(value):
    path = Path(value)
    if not path.is_absolute():
        path = FACTORY / path
    return safe_path(FACTORY, path.relative_to(FACTORY).as_posix())


# Defer only the actively written NEW renderer-r1 reviewer evidence; never old V2.
DEFER_PATTERNS = (
 "reviews/expert/*renderer*r1*", "reviews/adversarial/*renderer*r1*",
 "audits/rgm_planning_gap_v2_renderer_r1_adversarial*",
 "audits/execution/*renderer*r1*", "audits/diversity/planning-gap-v2-renderer-r1-input*",
 "audits/diversity/planning-gap-v2-renderer-r1-adversarial*",
 "audits/shortcut_detection/*renderer*r1*",
 "reproducibility/source_history/*renderer*r1*expert*/*",
 "reproducibility/source_history/*renderer*r1*adversarial*/*",
)

def deferred_new_review(path):
    name = path.relative_to(FACTORY).as_posix()
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in DEFER_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--approval", default="provisional_checkpoint")
    parser.add_argument("--validation-status", default="pending_final_snapshot_refresh")
    parser.add_argument("--runner", default="/root")
    parser.add_argument("--preparer", required=True,
                        help="Actual execution preparing this inventory, e.g. /root")
    args = parser.parse_args()
    if not args.checkpoint_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in args.checkpoint_id):
        raise ValueError("Invalid checkpoint identifier")
    spec_path = FACTORY / "backup_logs" / f"{args.checkpoint_id}_SPEC.json"
    report_path = FACTORY / "backup_logs" / f"{args.checkpoint_id}_INVENTORY_REPORT.json"
    receipt_path = FACTORY / "backup_logs" / f"{args.checkpoint_id}_RECEIPT.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "BACKED_UP":
        raise RuntimeError("Use a new checkpoint id; verified checkpoint already exists")

    candidates = []
    deferred = []
    excluded = []
    for path in sorted(FACTORY.rglob("*")):
        if not path.is_file():
            continue
        if path in {spec_path, report_path, receipt_path}:
            continue
        if deferred_new_review(path):
            deferred.append({"path": path.relative_to(FACTORY).as_posix(), "reason": "new_unfinished_renderer_r1_reviewer_artifact_explicitly_local", "frozen_hash_claim": False})
            continue
        entry, data = snapshot(path)
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store":
            excluded.append({**entry, "reason": "generated_cache_or_platform_metadata"})
            continue
        safe_path(FACTORY, entry["path"])
        candidates.append((entry, path))

    manifests = {}
    for entry, path in candidates:
        if path.name.endswith("compression_manifest.json") or path.name.endswith(".lossless.json"):
            metadata = json.loads(path.read_text())
            records = metadata.get("artifacts", metadata.get("items", [metadata]))
            for record in records:
                normalized = dict(record)
                # Preserve expert nested manifests as their actual original bytes;
                # support their field names alongside the standard gzip schema.
                for target, alternate in [("raw_bytes", "raw_length"),
                                          ("gzip_path", "compressed_path"),
                                          ("gzip_bytes", "compressed_length"),
                                          ("gzip_sha256", "compressed_sha256")]:
                    if target not in normalized and alternate in normalized:
                        normalized[target] = normalized[alternate]
                raw = local_declared(normalized["raw_path"])
                manifests.setdefault(raw.relative_to(FACTORY).as_posix(), []).append((path, normalized))

    files = []
    pairings = []
    blockers = []
    secret_flags = []
    for entry, path in candidates:
        fresh, data = snapshot(path)
        if fresh != entry:
            raise RuntimeError(f"Source changed during inventory: {entry['path']}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            secret_flags.append(entry["path"])
        if entry["size"] <= MAX_BYTES:
            files.append(entry)
            continue
        mappings = manifests.get(entry["path"], [])
        verified = []
        for manifest_path, metadata in mappings:
            packed = local_declared(metadata["gzip_path"])
            packed_entry, payload = snapshot(packed)
            manifest_entry, _ = snapshot(manifest_path)
            if entry["size"] != metadata["raw_bytes"] or entry["sha256"] != metadata["raw_sha256"]:
                raise RuntimeError(f"Raw manifest mismatch: {entry['path']}")
            if packed_entry["size"] != metadata["gzip_bytes"] or packed_entry["sha256"] != metadata["gzip_sha256"]:
                raise RuntimeError(f"Gzip manifest mismatch: {packed_entry['path']}")
            if packed_entry["size"] > MAX_BYTES:
                raise RuntimeError(f"Compressed payload also exceeds cap: {packed_entry['path']}")
            decoded = gzip.decompress(payload)
            if decoded != data or hashlib.sha256(decoded).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Lossless roundtrip mismatch: {entry['path']}")
            verified.append({"gzip": packed_entry, "manifest": manifest_entry,
                             "exact_raw_roundtrip_verified": True,
                             "uncompressed_secret_scan": "FAIL" if entry["path"] in secret_flags else "PASS"})
        if not verified:
            blockers.append({**entry, "reason": "oversized_raw_evidence_has_no_verified_lossless_pair"})
        excluded.append({**entry, "reason": "raw_exceeds_repository_25MiB_cap_preserved_local",
                         "lossless_representations": verified})
        pairings.append({"raw": entry, "representations": verified})

    included = {entry["path"] for entry in files}
    for pairing in pairings:
        for representation in pairing["representations"]:
            for role in ["gzip", "manifest"]:
                if representation[role]["path"] not in included:
                    raise RuntimeError("Lossless representation is absent from actual inventory")
    worktree = FACTORY.parent / ".rineya-dataset-git-worktree"
    original = FACTORY.parent / "rineya-world-model-research"
    report = {"execution_role": "Independent GitHub backup inventory preparer",
              "script_author_execution": "/root", "original_script_author_execution": "/root/backup_manager", "inventory_preparer_execution": args.preparer,
              "checkpoint_runner_execution": args.runner,
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "checkpoint_id": args.checkpoint_id, "cap_bytes": MAX_BYTES,
              "deferred_incomplete_review_paths": deferred, "declared_future_review_deferral_patterns": DEFER_PATTERNS,
              "deferral_scope": "Only NEW actively written renderer-r1 reviewer artifacts; all completed original V2/adjudication/quarantine/new producer/root audit/living reports/recovery proofs remain required. The execution guard rejects any missing previously tracked path.",
              "candidate_files_before_size_exclusion": len(candidates),
              "included_files_before_this_report": len(files),
              "included_bytes_before_this_report": sum(entry["size"] for entry in files),
              "oversized_raw_artifacts": pairings, "excluded_files": excluded,
              "secret_flags": secret_flags, "blockers": blockers,
              "free_bytes": shutil.disk_usage(FACTORY).free,
              "original_checkout": {"branch": git(["branch", "--show-current"], original),
                                    "head": git(["rev-parse", "HEAD"], original),
                                    "porcelain": git(["status", "--porcelain=v1"], original)},
              "dedicated_checkout": {"branch": git(["branch", "--show-current"], worktree),
                                      "head": git(["rev-parse", "HEAD"], worktree),
                                      "porcelain": git(["status", "--porcelain=v1"], worktree)},
              "packing_validation": "PASS" if not blockers and not secret_flags else "BLOCKED",
              "scope": "Archival snapshot only; pending samples do not acquire release approval. Raw originals unchanged. Gzip and manifest actual bytes included; remote persistence pending future checkpoint.",
              "snapshot_refresh_required": args.validation_status != "passed",
              "self_inventory_boundary": {"spec": spec_path.relative_to(FACTORY).as_posix(),
                                          "future_receipt": receipt_path.relative_to(FACTORY).as_posix(),
                                          "reason": "A specification and post-push receipt cannot hash their own final bytes; preserve in the next checkpoint."}}
    atomic_json(report_path, report)
    report_entry, _ = snapshot(report_path)
    files.append(report_entry)
    spec = {"checkpoint_id": args.checkpoint_id, "approval": args.approval,
            "approved_by": "/root", "runner_execution": args.runner,
            "inventory_preparer_execution": args.preparer,
            "inventory_script_author_execution": "/root", "original_inventory_script_author_execution": "/root/backup_manager",
            "validation_status": args.validation_status,
            "deferred_incomplete_review_paths": deferred, "declared_future_review_deferral_patterns": DEFER_PATTERNS,
            "scope_note": "Draft exact-file inventory of current factory and verified lossless evidence. Refresh after final reviews, adjudication, release, tests and living-document edits before executing checkpoint.",
            "inventory_report": report_entry,
            "raw_size_exclusions": [entry for entry in excluded if entry["reason"].startswith("raw_exceeds")],
            "excluded_generated_files": [entry for entry in excluded if entry["reason"].startswith("generated")],
            "self_inventory_boundary": report["self_inventory_boundary"],
            "files": sorted(files, key=lambda entry: entry["path"])}
    if blockers or secret_flags:
        spec["validation_status"] = "blocked_packing_or_secret_scan"
    atomic_json(spec_path, spec)
    print(json.dumps({"spec": str(spec_path), "report": str(report_path),
                      "files": len(files), "bytes": sum(entry["size"] for entry in files),
                      "oversized_raw": len(pairings), "lossless_verified": sum(bool(x["representations"]) for x in pairings),
                      "generated_exclusions": sum(e["reason"].startswith("generated") for e in excluded),
                      "blockers": len(blockers), "secret_flags": len(secret_flags),
                      "validation_status": spec["validation_status"]}))


if __name__ == "__main__":
    main()
