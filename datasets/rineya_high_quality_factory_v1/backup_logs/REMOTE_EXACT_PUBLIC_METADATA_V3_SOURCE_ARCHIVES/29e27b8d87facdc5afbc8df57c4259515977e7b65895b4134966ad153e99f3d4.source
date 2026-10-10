#!/usr/bin/env python3
"""Fetch actual remote gzip objects for a frozen, fully verified checkpoint.

Compressed payloads use an 8 MiB spool and 64 KiB reads. Decoded bytes are
compared directly against preserved local raw files. No local Git/gzip fallback
or inherited payload verification can replace a new gh API fetch.
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
import sys
import tempfile

sys.dont_write_bytecode = True
AUTHOR = "/root/backup_manager"
FACTORY = Path(__file__).resolve().parent.parent
DEPENDENCY_SHA = "a9544fd6e18daffea15be3bc65d45a8ab83239aa7632ed65618d0e7eadcd3135"
DERIVED_FROM_SHA = "7bab2c46dd1e7a52520a90617a2d910b5d6987bad8ec2489b05097045505a0aa"
CHUNK = 64 * 1024
SPOOL_MEMORY = 8 * 1024 * 1024


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bounded():
    path = FACTORY / "backup_logs/recover_remote_checkpoint_exact_public_metadata_v3.py"
    if path.is_symlink() or sha_file(path) != DEPENDENCY_SHA:
        raise ValueError("Frozen bounded verifier source changed")
    s = importlib.util.spec_from_file_location("frozen_bounded_remote", path)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    if sha_file(path)!=DEPENDENCY_SHA:raise ValueError("Recovery dependency changed during import")
    return module


def read_bound(path, expected, bounded):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Explicit expected SHA256 is invalid")
    if path.is_symlink() or not path.resolve().is_relative_to(FACTORY / "backup_logs"):
        raise ValueError("Bound input must be a regular local backup_logs file")
    bounded.source_path(FACTORY, path.absolute().relative_to(FACTORY).as_posix())
    meta = bounded.hash_file(path)
    if meta["sha256"] != expected or meta["size"] > 16 * 1024 * 1024:
        raise ValueError("Frozen metadata hash differs or exceeds bounded cap")
    return json.loads(path.read_bytes())


def validate_bindings(spec, stage, verified, commit, spec_sha, bounded):
    if (spec.get("validation_status") != "passed" or not spec.get("approved_by")
            or spec.get("approval") not in {"provisional_checkpoint", "code_review_milestone", "reviewed_release"}):
        raise ValueError("Exact checkpoint validation/authorization is missing")
    inventory = spec.get("files", [])
    if not inventory or len({e["path"] for e in inventory}) != len(inventory):
        raise ValueError("Exact inventory is empty or duplicated")
    for e in inventory:
        bounded.check_entry(e)
        if "\0" in e["path"]:
            raise ValueError("NUL inventory path rejected")
    if (stage.get("status") != "PUSHED_REMOTE_BYTES_VERIFICATION_PENDING"
            or stage.get("commit_sha") != commit or stage.get("remote_commit_sha") != commit
            or stage.get("spec_sha256") != spec_sha):
        raise ValueError("Immutable stage receipt does not bind the pushed checkpoint")
    indexed = stage.get("all_source_target_index_sha256_and_sha1_verified") is True
    legacy = len(inventory) == 1801 and stage.get("all1801_index_blobs_match_frozen_bytes") is True
    if not (indexed or legacy):
        raise ValueError("Actual precommit exact index validation receipt is missing")
    if (verified.get("status") != "BACKED_UP" or verified.get("commit_sha") != commit
            or verified.get("remote_commit_sha") != commit or verified.get("spec_sha256") != spec_sha
            or verified.get("inventory") != inventory or verified.get("files") != len(inventory)
            or verified.get("total_bytes") != sum(e["size"] for e in inventory)
            or verified.get("verification_source_sha256") != DEPENDENCY_SHA):
        raise ValueError("Completed actual remote inventory proof is not bound to this frozen checkpoint")
    for key in ("capped_source_secret_scan_before","capped_source_secret_scan_after","actual_local_git_metadata_exception_scan","actual_remote_metadata_exception_scan"):
        bounded.validate_metadata_report(verified.get(key,{}),key.startswith('actual_'),{'actual_local_git_metadata_exception_scan':'local_git','actual_remote_metadata_exception_scan':'actual_remote'}.get(key))
    remote = verified.get("remote_verification", {})
    receipts = remote.get("blob_receipts", [])
    if remote.get("verified_files") != len(inventory) or remote.get("verified_unique_blobs") != len(receipts):
        raise ValueError("Actual remote proof counts differ")
    blobs, paths, by_path = set(), set(), {e["path"]: e for e in inventory}
    actual_by_path = {}
    for b in receipts:
        if (not re.fullmatch(r"[0-9a-f]{40}", b.get("git_blob", ""))
                or b["git_blob"] in blobs or b.get("verification_source") != "actual_remote_raw_GitHub_blob"):
            raise ValueError("Remote proof is duplicated or lacks actual-fetch provenance")
        blobs.add(b["git_blob"])
        for name in b["paths"]:
            if name in paths or name not in by_path:
                raise ValueError("Actual remote proof path coverage differs")
            paths.add(name)
            if any(by_path[name][k] != b[k] for k in ("size", "sha256")):
                raise ValueError("Actual remote proof hash/size differs")
            actual_by_path[name] = b
    if paths != set(by_path):
        raise ValueError("Actual remote proof omits frozen paths")
    for key in ('actual_local_git_metadata_exception_scan','actual_remote_metadata_exception_scan'):
        for row in verified[key]['four_exception_files']:
            if actual_by_path.get(row['path'],{}).get('git_blob')!=row['git_blob']:
                raise ValueError('Metadata actual stream Git ID differs from frozen remote path coverage')
    if verified.get('current_source_evolution')!=[] or verified.get('current_source_evolution_at_finish')!=[]:
        raise ValueError('Held checkpoint does not permit evolving sources')
    prior_raws = {e["path"]: e for e in verified.get("oversized_raw_local_verification", [])}
    for e in spec.get("raw_size_exclusions", []):
        bounded.check_entry(e, False)
        old = prior_raws.get(e["path"], {})
        if (any(old.get(k) != e[k] for k in ("size", "sha256"))
                or old.get("lossless_local_roundtrip") is not True):
            raise ValueError("Completed frozen raw/secret/lossless verification is missing")
    return by_path, actual_by_path


def plan_representations(spec, inventory, actual, remote_paths, prefix, expected_raw_count, bounded):
    raws = spec.get("raw_size_exclusions", [])
    if len(raws) != expected_raw_count or len({e["path"] for e in raws}) != len(raws):
        raise ValueError("Frozen oversized raw count differs or repeats paths")
    work, seen = [], set()
    for e in raws:
        if e["size"] <= bounded.MAX_BYTES or e["path"] in inventory:
            raise ValueError("Raw cap exclusion is invalid")
        if not e.get("lossless_representations"):
            raise ValueError("Oversized raw has no frozen lossless payload")
        for rep in e["lossless_representations"]:
            for key in ("gzip", "manifest"):
                if inventory.get(rep[key]["path"]) != rep[key]:
                    raise ValueError("Lossless payload/manifest absent from exact inventory")
            packed = rep["gzip"]
            blob = remote_paths.get(prefix + "/" + packed["path"], {})
            if (blob.get("size") != packed["size"] or blob.get("mode") != "100644"
                    or blob.get("type") != "blob" or blob.get("sha") != actual[packed["path"]]["git_blob"]):
                raise ValueError("Actual remote gzip path/mode/size/blob disagrees with frozen proofs")
            key = (e["path"], packed["path"])
            if key not in seen:
                seen.add(key)
                work.append((e, packed, blob))
    return work


def manifest_reference_coverage(spec,inventory,actual,remote_paths,prefix):
    refs=[]
    for raw in spec['raw_size_exclusions']:
        for representation in raw['lossless_representations']:
            record=representation['manifest'];blob=remote_paths.get(prefix+'/'+record['path'],{})
            if (inventory.get(record['path'])!=record or actual.get(record['path'],{}).get('git_blob')!=blob.get('sha')
                    or blob.get('size')!=record['size'] or blob.get('type')!='blob' or blob.get('mode')!='100644'):
                raise ValueError('Every frozen manifest reference must match actual remote full-proof path/blob')
            refs.append({'raw_path':raw['path'],'gzip_path':representation['gzip']['path'],'manifest':record,
                'actual_remote_git_blob':blob['sha'],'actual_payload_proof_source':'Bound full inventory actual remote-byte proof; not a new gzip execution manifest fetch'})
    return refs


def consume_remote(handle, raw_path, raw, packed, blob, spool_dir):
    before = raw_path.stat()
    if before.st_size != raw["size"]:
        raise ValueError("Preserved raw has the wrong frozen size")
    with tempfile.SpooledTemporaryFile(max_size=SPOOL_MEMORY, dir=spool_dir) as spool:
        compressed_sha = hashlib.sha256()
        git_sha = hashlib.sha1(f"blob {packed['size']}\0".encode())
        compressed_size = 0
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            compressed_size += len(chunk)
            if compressed_size > packed["size"]:
                raise ValueError("Actual remote compressed payload exceeds frozen size")
            compressed_sha.update(chunk);git_sha.update(chunk);spool.write(chunk)
        if (compressed_size != packed["size"] or compressed_sha.hexdigest() != packed["sha256"]
                or git_sha.hexdigest() != blob["sha"]):
            raise ValueError("Actual remote compressed size/SHA256/GitSHA1 differs")
        spool.seek(0)
        raw_sha = hashlib.sha256()
        raw_size = 0
        with gzip.GzipFile(fileobj=spool, mode="rb") as decoded, raw_path.open("rb") as original:
            for chunk in iter(lambda: decoded.read(CHUNK), b""):
                raw_size += len(chunk)
                if raw_size > raw["size"] or chunk != original.read(len(chunk)):
                    raise ValueError("Actual remote decoded bytes differ from preserved raw")
                raw_sha.update(chunk)
            if original.read(1):
                raise ValueError("Preserved raw has additional bytes")
        if raw_size != raw["size"] or raw_sha.hexdigest() != raw["sha256"]:
            raise ValueError("Decoded frozen raw size/SHA256 differs")
    after = raw_path.stat()
    if any(getattr(before, k) != getattr(after, k) for k in
           ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("Preserved raw changed during remote decode")
    return {"raw_path": raw["path"], "raw_bytes": raw_size, "raw_sha256": raw_sha.hexdigest(),
            "remote_gzip_path": packed["path"], "remote_git_blob": blob["sha"],
            "remote_gzip_bytes": compressed_size, "remote_gzip_sha256": compressed_sha.hexdigest(),
            "actual_remote_fetch_this_run": True, "decoded_byte_for_byte_equal_to_preserved_raw": True}


def fetch_one(bounded, factory, raw, packed, blob):
    raw_path = bounded.source_path(factory, raw["path"])
    return bounded.process_stream(["gh", "api", "-H", "Accept: application/vnd.github.raw+json",
                                   "-H", "Cache-Control: no-cache",
                                   f"repos/{bounded.REPOSITORY}/git/blobs/{blob['sha']}"],
                                  lambda handle: consume_remote(handle, raw_path, raw, packed, blob, factory / "backup_logs"))


def output_paths(args, bounded):
    if args.output.resolve() == args.ledger.resolve():
        raise ValueError("Proof and ledger must be distinct")
    for p, allowed in ((args.output, (FACTORY / "quality_reports", FACTORY / "backup_logs")),
                       (args.ledger, (FACTORY / "backup_logs",))):
        if p.exists() or p.is_symlink() or not any(p.resolve().is_relative_to(root) for root in allowed):
            raise ValueError("Output must be a new owned evidence path")
        rel = p.absolute().relative_to(FACTORY).as_posix()
        if bounded.source_path(FACTORY, rel) != p.absolute():
            raise ValueError("Output path does not resolve literally")


def validate_preparation_freeze(freeze,bounded):
    if (freeze.get('author_execution')!=AUTHOR or freeze.get('frozen_before_first_import_test_execution') is not True
            or freeze.get('runner_source')!={'path':Path(__file__).relative_to(FACTORY).as_posix(),**bounded.hash_file(Path(__file__))}):
        raise ValueError('Gzip exact source/preexecution freeze differs')
    bounded.validate_frozen_inputs(freeze['frozen_inputs'])
    bounded.metadata_helper().validate_config()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "stage-receipt", "verified-receipt", "source-freeze", "output", "ledger"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-spec-sha256", "expected-stage-sha256", "expected-verified-sha256", "expected-freeze-sha256", "commit", "runner"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--expected-raw-count", type=int, required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or args.expected_raw_count < 1:
        raise ValueError("Complete actual commit/count binding is required")
    bounded = load_bounded();output_paths(args, bounded)
    if (args.runner!=AUTHOR or args.commit!=bounded.EXPECTED_COMMIT or args.expected_spec_sha256!=bounded.EXPECTED_SPEC_SHA
            or args.expected_raw_count!=31):raise ValueError('Exact runner/commit/spec/raw31 required')
    receipt = {"status": "PENDING_ACTUAL_REMOTE_GZIP_PROOF", "author_execution": AUTHOR,
               "actual_runner_execution": args.runner, "runner_execution": args.runner,
               "model_disclosure": "Inherited same model; independent execution does not imply model diversity",
               "started_at_utc": now(), "commit_sha": args.commit, "spec_sha256": args.expected_spec_sha256,
               "stage_receipt_sha256": args.expected_stage_sha256, "verified_inventory_receipt_sha256": args.expected_verified_sha256,
               "source_sha256": sha_file(Path(__file__)), "dependency_sha256": DEPENDENCY_SHA,
               "derived_from_corrected_source_sha256": DERIVED_FROM_SHA, "source_freeze_sha256": args.expected_freeze_sha256,
               "grants_data_approval": False, "overall_backup_status_claimed_by_this_gzip_proof": False,
               "scope": "New actual remote gzip fetch/decode proof joined to an already completed full inventory proof; no Git mutations or edits to prior helpers/proofs/data",
               "self_inventory_boundary": "New execution proof/ledger postdate the frozen checkpoint and remain local until a later exact inventory"}
    with args.ledger.open("x", encoding="utf-8") as ledger:
        def log(phase, **values):
            ledger.write(json.dumps({"at_utc": now(), "phase": phase, **values}, separators=(",", ":")) + "\n")
            ledger.flush();os.fsync(ledger.fileno())
        try:
            log("START_BOUND_ACTUAL_REMOTE_GZIP_PROOF", commit_sha=args.commit, author=AUTHOR, runner=args.runner)
            freeze = read_bound(args.source_freeze, args.expected_freeze_sha256, bounded)
            validate_preparation_freeze(freeze,bounded)
            spec = read_bound(args.spec, args.expected_spec_sha256, bounded)
            bounded.validate_exact_scope(spec)
            stage = read_bound(args.stage_receipt, args.expected_stage_sha256, bounded)
            verified = read_bound(args.verified_receipt, args.expected_verified_sha256, bounded)
            inventory, actual = validate_bindings(spec, stage, verified, args.commit, args.expected_spec_sha256, bounded)
            worktree = FACTORY.parent / ".rineya-dataset-git-worktree"
            bounded.check_custody(worktree, args.commit)
            private = bounded.api(f"repos/{bounded.REPOSITORY}")
            if (private.get("full_name") != bounded.REPOSITORY or private.get("private") is not True
                    or private.get("permissions", {}).get("admin") is not True):
                raise ValueError("Actual GitHub repository is not private ADMIN")
            remote = bounded.api(f"repos/{bounded.REPOSITORY}/git/commits/{args.commit}")
            tree_sha = bounded.git(worktree, "rev-parse", args.commit + "^{tree}")
            if ([p["sha"] for p in remote.get("parents",[])]!=[bounded.EXPECTED_PARENT] or remote.get("sha") != args.commit or remote.get("tree", {}).get("sha") != tree_sha
                    or verified["remote_verification"]["remote_tree_sha"] != tree_sha):
                raise ValueError("Actual commit/tree disagrees with bound inventory proof")
            tree = bounded.api(f"repos/{bounded.REPOSITORY}/git/trees/{tree_sha}?recursive=1")
            if tree.get("truncated") or tree.get("sha") != tree_sha:
                raise ValueError("Actual remote tree is truncated or mismatched")
            remote_paths = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
            scoped = {name[len(bounded.PREFIX)+1:] for name in remote_paths if name.startswith(bounded.PREFIX + "/")}
            if scoped != set(inventory):
                raise ValueError("Actual remote dataset path coverage differs from exact inventory")
            work = plan_representations(spec, inventory, actual, remote_paths, bounded.PREFIX, args.expected_raw_count, bounded)
            manifest_refs=manifest_reference_coverage(spec,inventory,actual,remote_paths,bounded.PREFIX)
            proofs = []
            for raw, packed, blob in work:
                proof = fetch_one(bounded, FACTORY, raw, packed, blob)
                proofs.append(proof);log("ACTUAL_REMOTE_GZIP_BYTE_PROOF", **proof)
                print(json.dumps({"actual_remote_gzip_verified": len(proofs), "required": len(work)}), flush=True)
            unique_raws = {proof["raw_path"]: proof for proof in proofs}
            if set(unique_raws) != {e["path"] for e in spec["raw_size_exclusions"]}:
                raise ValueError("Actual remote decode proof omits raw artifacts")
            bounded.check_custody(worktree, args.commit)
            for path, expected in ((args.spec, args.expected_spec_sha256), (args.stage_receipt, args.expected_stage_sha256),
                                   (args.verified_receipt, args.expected_verified_sha256), (args.source_freeze, args.expected_freeze_sha256)):
                if bounded.hash_file(path)["sha256"] != expected:
                    raise ValueError("Frozen binding changed during execution")
            if sha_file(Path(__file__)) != receipt["source_sha256"] or sha_file(Path(bounded.__file__)) != DEPENDENCY_SHA:
                raise ValueError("Source-frozen runner/dependency changed during execution")
            validate_preparation_freeze(freeze,bounded)
            if (len(work)!=31 or len(manifest_refs)!=32 or len({r['manifest']['path'] for r in manifest_refs})!=31
                    or sum(e['raw_bytes'] for e in proofs)!=1724728576):raise ValueError('Frozen deduplicated gzip coverage/byte totals differ')
            receipt.update(status="PASS_ACTUAL_REMOTE_GZIP_DECODE", finished_at_utc=now(), actual_remote_tree_sha=tree_sha,
                           actual_remote_gzip_proofs=proofs, actual_remote_gzip_fetches=len(proofs),
                           oversized_raw_artifacts=len(unique_raws), raw_bytes_exactly_recovered=sum(e["raw_bytes"] for e in unique_raws.values()),
                           actual_decoded_bytes_including_multiple_representations=sum(e["raw_bytes"] for e in proofs),
                           private_admin_confirmed=True, payload_transport="Actual gh API raw Accept with Cache-Control:no-cache, every planned gzip fetched this run",
                           bounded_spool_memory_bytes=SPOOL_MEMORY, stream_chunk_bytes=CHUNK,
                           full_frozen_lossless_manifest_reference_coverage=manifest_refs,
                           frozen_representation_reference_count=len(manifest_refs),unique_manifest_path_count=len({r["manifest"]["path"] for r in manifest_refs}),
                           unique_raw_gzip_pair_count=len(work),local_fullproof_decode_iterations=verified.get("local_raw_representation_decode_iterations"),
                           local_fullproof_decoded_bytes=verified.get("local_raw_representation_decoded_bytes"),
                           inherited_full_inventory_proof_boundary="Separate bound completed full proof; its inherited fetches are not counted as this execution's new gzip fetches")
            log("COMPLETE_ACTUAL_REMOTE_GZIP_PROOF", raw_count=len(unique_raws), actual_fetches=len(proofs), status=receipt["status"])
            bounded.write_exclusive(args.output, receipt)
            print(json.dumps({"status": receipt["status"], "raw_artifacts": len(unique_raws), "new_actual_fetches": len(proofs),
                              "raw_bytes": receipt["raw_bytes_exactly_recovered"], "proof_sha256": bounded.hash_file(args.output)["sha256"]}))
        except Exception as error:
            receipt.update(status="FAILED_ACTUAL_REMOTE_GZIP_PROOF_EVIDENCE_PRESERVED", error_type=type(error).__name__, error=str(error), finished_at_utc=now())
            log("FAILED_ACTUAL_REMOTE_GZIP_PROOF_EVIDENCE_PRESERVED", error_type=type(error).__name__, error=str(error))
            if not args.output.exists():bounded.write_exclusive(args.output, receipt)
            raise


if __name__ == "__main__":
    main()
