#!/usr/bin/env python3
"""Source-frozen r2 resumption: verify existing index; root alone may commit/push.

No copy, add, reset, clean, removal or Git configuration operation is implemented.
The full diff-check MUST return its real exit 2 and only the five exact diagnostics.
The narrow byte-preservation exception is distinct from a strict diff-check pass.
Remote persistence remains pending until the separate bounded actual-byte proof.
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

sys.dont_write_bytecode = True
AUTHOR = "/root/backup_manager"
FACTORY = Path(__file__).resolve().parent.parent
LOGS = FACTORY / "backup_logs"
PREFIX = "datasets/rineya_high_quality_factory_v1"
PARENT = "82787689fef4e50f7a35cfee505c290b2f6a6d78"
SPEC_SHA = "6b310f0656a468dd65920c64f15efe82e0b592d3d76250366f9555cef46aaad5"
FAILED_SHA = "37cb266ec809b27c42ec3a4b992ef66cf992cbdf63d4d0156ae5603371c28436"
FAILED_LEDGER_SHA = "d4d3beef10861e4a50c5611ac1deba227dd0577a8023c52cf819c4f814985e21"
BASE_SHA = "df03bdebc7d9cb6b24c65f82f1e17b42e8f857a106fafc3bfd2ac3ac341a0897"
BASE_FREEZE_SHA = "e182c1fbd65f5d83edcf288555bf44a2b2b67f3264d2c1073ef086166c4444fc"
RECOVERY_SHA = "3f496d27e6fc36ecda3eabcff4593b13deaea03a04fee543e49674f843bf37b1"
OLD_SHA = "07ef424f92d903c265c6cc06622733dcdeabf5bd03d3c13d98f793c2ddcc2d4b"
CUSTODY_SHA = "2e604f1dc2dc36c29ccc7968988a6abdc08a71c63e8ebf596b39faf856fb1e23"
EXCEPTION_SHA = "71abc77279f45a864b7d493840d07cd9eb4515076521e850bdd3a5d8209640f4"
TEST_SHA = "13990b79681d988f66eca74562aa67e07f33d23bd2d8d55cd9021726221ae95c"
LINE_B64 = "ICAgICAgICBmb3IgcGF0aCBpbiAoJy9vdXRzaWRlLmpzb24nLCcuLi9vdXRzaWRlLmpzb24nLCdDOlxcb3V0c2lkZS5qc29uJywnLicpOiAK"
WARNING_PATHS = (
    f"reproducibility/source_blobs/{TEST_SHA}.source",
    f"reviews/oracle/RBM-v4-integration-20261010T080702Z/attempt01-source-blobs/{TEST_SHA}.source",
    f"reviews/oracle/RBM-v4-official-20261010T073119Z/frozen_source_blobs/{TEST_SHA}.source",
    "reviews/oracle/RBM-v4-official-20261010T073119Z/oracle_belief_controls_v4_test.py",
    "tests/oracle_belief_controls_v4_test.py",
)
CHUNK = 64 * 1024
META_CAP = 2 * 1024 * 1024
ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "PYTHONDONTWRITEBYTECODE": "1"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def checksum(path):
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as h:
        while part := h.read(CHUNK):
            sha.update(part)
    after = path.stat()
    if any(getattr(before, k) != getattr(after, k) for k in
           ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("Frozen input changed during read")
    return sha.hexdigest()


def import_frozen(path, expected, name):
    if path.is_symlink() or checksum(path) != expected:
        raise ValueError("Frozen helper hash differs before import")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if checksum(path) != expected:
        raise ValueError("Frozen helper changed during import")
    return module


def canonical_exception():
    return {
        "schema": "frozen-five-warning-byte-preservation-v1",
        "checkpoint_id": "belief-v4-validated-code-candidate-r2",
        "expected_parent": PARENT, "expected_spec_sha256": SPEC_SHA,
        "paths": list(WARNING_PATHS), "source_size": 20749,
        "source_sha256": TEST_SHA, "line_number": 212,
        "line_bytes_base64": LINE_B64, "diagnostic_type": "trailing whitespace.",
        "required_exit_code": 2,
        "required_stdout_sha256": "256b3790a01671d8b33647213342d1256f95fd776dcd738065db160efd60d55e",
        "required_stdout_bytes": 1249, "required_stderr_bytes": 0,
        "reason": "Preserve independently frozen official test/source archives byte-for-byte; no whitespace config or ignore changes.",
        "grants_data_approval": False,
    }


def validate_exception(value):
    expected = canonical_exception()
    # JSON equality alone treats bool as int. Require the complete exact type tree.
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Exception configuration keys differ")
    for key, wanted in expected.items():
        actual = value[key]
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError("Exception configuration value/type differs")
        if type(wanted) is list and any(type(p) is not str for p in actual):
            raise ValueError("Exception paths must be literal strings")
    return value


def expected_diagnostics(config):
    validate_exception(config)
    line = base64.b64decode(config["line_bytes_base64"], validate=True)
    return b"".join((PREFIX + "/" + p + ":212: trailing whitespace.\n").encode()
                    + b"+" + line for p in WARNING_PATHS)


def validate_diagnostics(result, config):
    wanted = expected_diagnostics(config)
    if (type(result) is not dict or set(result) != {"returncode", "stdout", "stderr"}
            or type(result["returncode"]) is not int
            or type(result["stdout"]) is not bytes or type(result["stderr"]) is not bytes
            or result["returncode"] != 2 or result["stderr"] != b""
            or result["stdout"] != wanted):
        raise ValueError("Diff-check is not exactly the bound five nonzero diagnostics")
    return {
        "command": ["git", "diff", "--cached", "--check"],
        "actual_exit_code": 2, "actual_stdout_bytes": len(wanted),
        "actual_stdout_sha256": hashlib.sha256(wanted).hexdigest(),
        "actual_stderr_bytes": 0, "warnings": 5,
        "strict_diff_check": "NONZERO_EXIT_2_NOT_AN_UNQUALIFIED_PASS",
        "narrow_frozen_evidence_exception": "EXACT_BOUND_FIVE_WARNINGS_MATCH",
        "exception_paths": list(WARNING_PATHS), "exception_line": 212,
        "exception_source_sha256": TEST_SHA, "exception_source_bytes": 20749,
    }


def bounded_process(args, cwd):
    p = subprocess.Popen(args, cwd=cwd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
    data = {"stdout": bytearray(), "stderr": bytearray()}
    errors = []
    def read(name, handle):
        try:
            while part := handle.read(CHUNK):
                if len(data[name]) + len(part) > META_CAP:
                    raise ValueError("Diff-check diagnostics exceed bounded cap")
                data[name].extend(part)
        except Exception as error:
            errors.append(error)
            if p.poll() is None:
                p.kill()
        finally:
            handle.close()
    threads = [threading.Thread(target=read, args=(k, getattr(p, k)), daemon=True)
               for k in ("stdout", "stderr")]
    for t in threads:
        t.start()
    try:
        code = p.wait(timeout=180)
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return {"returncode": code, **{k: bytes(v) for k, v in data.items()}}


def parse_index(raw):
    out = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        left, name = record.split(b"\t", 1)
        mode, blob, stage = left.decode().split()
        name = name.decode()
        if stage != "0" or name in out:
            raise ValueError("Index conflict or duplicate path")
        out[name] = (mode, blob)
    return out


def parse_tree(raw):
    out = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        left, name = record.split(b"\t", 1)
        mode, kind, blob = left.decode().split()
        name = name.decode()
        if kind != "blob" or name in out:
            raise ValueError("Unexpected parent tree object/path")
        out[name] = (mode, blob)
    return out


def validate_whole_index(current, parent_tree, checked):
    expected = {p: v for p, v in parent_tree.items() if not p.startswith(PREFIX + "/")}
    expected.update({PREFIX + "/" + p: ("100644", v["git_blob"]) for p, v in checked.items()})
    if current != expected:
        raise ValueError("Full index differs from frozen dataset and unchanged outside-parent paths")
    return expected


def validate_changes(staged, unstaged, untracked, ignored, deleted, checked, previous):
    wanted = {PREFIX + "/" + p for p, v in checked.items() if previous.get(p) != v["git_blob"]}
    if (not staged or len(staged) != len(set(staged)) or set(staged) != wanted
            or unstaged or untracked or ignored or deleted):
        raise ValueError("Existing staged/worktree changes are not exactly the frozen scope")
    return sorted(wanted)


def verify_warning_sources(base, factory, worktree, checked, actual_index, config):
    validate_exception(config)
    wanted_line = base64.b64decode(LINE_B64, validate=True)
    values = []
    for name in WARNING_PATHS:
        expected = checked.get(name)
        if (expected is None or expected["size"] != 20749 or expected["sha256"] != TEST_SHA
                or actual_index.get(name) != expected["git_blob"]):
            raise ValueError("Exception source/index is not the bound official test bytes")
        for root, path in ((factory, name), (worktree, PREFIX + "/" + name)):
            source = base.safe(root, path)
            if base.proof(source, False) != expected:
                raise ValueError("Exception source/target checksum changed")
            with source.open("rb") as handle:
                line = b""
                for _ in range(212):
                    line = handle.readline(64 * 1024)
                if line != wanted_line:
                    raise ValueError("Exception actual source line differs")
        values.append({"path": name, **expected, "actual_line212_base64": LINE_B64})
    return values


def strict_other_paths(worktree, staged):
    exclusions = {PREFIX + "/" + p for p in WARNING_PATHS}
    others = sorted(set(staged) - exclusions)
    for i in range(0, len(others), 100):
        result = bounded_process(["git", "--literal-pathspecs", "diff", "--cached", "--check",
                                  "--", *others[i:i + 100]], worktree)
        if result != {"returncode": 0, "stdout": b"", "stderr": b""}:
            raise ValueError("Strict diff-check failed on non-excepted staged paths")
    return {"other_staged_paths": len(others), "restricted_strict_diff_check": "PASS"}


def names(base, worktree, *args):
    return [p.decode() for p in base.git(worktree, *args).split(b"\0") if p]


def current_state(base, worktree, parent, checked, previous):
    full = parse_index(base.git(worktree, "ls-files", "--stage", "-z"))
    validate_whole_index(full, parent, checked)
    staged = names(base, worktree, "diff", "--cached", "--name-only", "-z")
    validate_changes(staged, names(base, worktree, "diff", "--name-only", "-z"),
                     names(base, worktree, "ls-files", "--others", "--exclude-standard", "-z"),
                     names(base, worktree, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"),
                     names(base, worktree, "diff", "--cached", "--diff-filter=D", "--name-only", "-z"),
                     checked, previous)
    return full, staged


def frozen_inputs(args):
    outputs = (args.receipt, args.local_commit_receipt, args.ledger)
    if len({p.resolve() for p in outputs}) != 3:
        raise ValueError("Fresh evidence aliases must be distinct")
    for p in outputs:
        if p.exists() or p.is_symlink() or not p.resolve().is_relative_to(LOGS):
            raise ValueError("Evidence outputs must be fresh backup_logs paths")
    declared = (
        (args.spec, SPEC_SHA, args.expected_spec_sha256),
        (args.custody_baseline, CUSTODY_SHA, args.expected_custody_sha256),
        (args.exception_config, EXCEPTION_SHA, args.expected_exception_sha256),
        (args.failed_stage, FAILED_SHA, args.expected_failed_stage_sha256),
        (args.failed_ledger, FAILED_LEDGER_SHA, args.expected_failed_ledger_sha256),
        (LOGS / "run_provisional_archival_checkpoint_v2.py", BASE_SHA, BASE_SHA),
        (LOGS / "PROVISIONAL_ARCHIVAL_RUNNER_V2_SOURCE_FREEZE.json", BASE_FREEZE_SHA, BASE_FREEZE_SHA),
        (LOGS / "recover_remote_checkpoint.py", RECOVERY_SHA, RECOVERY_SHA),
        (LOGS / "checkpoint_backup.py", OLD_SHA, OLD_SHA),
    )
    bindings = []
    for path, fixed, supplied in declared:
        if (supplied != fixed or path.is_symlink() or not path.resolve().is_relative_to(LOGS)
                or checksum(path) != fixed):
            raise ValueError("Exact frozen input/failure/helper binding differs")
        bindings.append((path, fixed))
    if args.expected_parent != PARENT:
        raise ValueError("This narrow resumption is bound only to the frozen 827 parent")
    if args.runner != "/root":
        raise ValueError("This resumption is prepared for actual root execution only")
    if (args.source_freeze.is_symlink() or not args.source_freeze.resolve().is_relative_to(LOGS)
            or checksum(args.source_freeze) != args.expected_freeze_sha256):
        raise ValueError("Exact new source freeze differs")
    freeze = json.loads(args.source_freeze.read_bytes())
    if (freeze.get("author_execution") != AUTHOR
            or freeze["runner_source"]["sha256"] != checksum(Path(__file__))):
        raise ValueError("New runner lacks its exact independent source freeze")
    for row in freeze["frozen_inputs"]:
        p = Path(row["path"])
        if p.is_symlink() or not p.resolve().is_relative_to(LOGS) or checksum(p) != row["sha256"]:
            raise ValueError("New preparation freeze input differs")
        bindings.append((p, row["sha256"]))
    bindings.extend(((args.source_freeze, args.expected_freeze_sha256),
                     (Path(__file__), checksum(Path(__file__)))))
    return bindings


def recheck_bindings(bindings):
    for path, sha in bindings:
        if checksum(path) != sha:
            raise ValueError("Frozen resumption input changed before mutation")


def validate_spec(spec):
    if (spec.get("checkpoint_id") != "belief-v4-validated-code-candidate-r2"
            or spec.get("expected_parent_sha") != PARENT
            or len(spec.get("files", [])) != 2685
            or sum(e["size"] for e in spec["files"]) != 1080978676
            or spec.get("inventory_unique_git_blob_groups") != 2153
            or spec.get("raw_lossless_required_count") != 26
            or len(spec.get("raw_size_exclusions", [])) != 26
            or sum(e["size"] for e in spec["raw_size_exclusions"]) != 1353229114):
        raise ValueError("Narrow exact r2 scope/counts/parent differ")


def validate_failed_receipt(failed):
    if (failed.get("status") != "FAILED_NEW_EXECUTION_LOCAL_ARTIFACTS_PRESERVED"
            or failed.get("spec_sha256") != SPEC_SHA or failed.get("expected_parent_sha") != PARENT
            or failed.get("files") != 2685 or failed.get("total_bytes") != 1080978676
            or failed.get("source_sha256") != BASE_SHA
            or failed.get("source_freeze_sha256") != BASE_FREEZE_SHA
            or failed.get("error_type") != "RuntimeError"
            or failed.get("error") != "git command failed with exit 2; output withheld"
            or failed.get("commit_sha") is not None):
        raise ValueError("Preserved first stage failure is not the bound precommit diff-check failure")


def prepush_custody(base, worktree, commit):
    base.require_head(worktree, commit)
    if (base.text(worktree, "branch", "--show-current") != base.BRANCH
            or base.text(worktree, "remote", "get-url", "origin") != f"https://github.com/{base.REPOSITORY}.git"
            or base.text(worktree, "status", "--porcelain=v1", "--untracked-files=all")
            or base.text(worktree, "ls-remote", "--heads", "origin", "refs/heads/" + base.BRANCH).split()
               != [PARENT, "refs/heads/" + base.BRANCH]):
        raise ValueError("Postcommit local custody/clean state or exact remote parent changed")
    meta = json.loads(base.run(["gh", "repo", "view", base.REPOSITORY,
                               "--json", "isPrivate,nameWithOwner,viewerPermission"]))
    if meta != {"isPrivate": True, "nameWithOwner": base.REPOSITORY, "viewerPermission": "ADMIN"}:
        raise ValueError("Actual prepush private ADMIN check failed")
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "source-freeze", "custody-baseline", "exception-config", "failed-stage",
                 "failed-ledger", "receipt", "local-commit-receipt", "ledger"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-spec-sha256", "expected-freeze-sha256", "expected-custody-sha256",
                 "expected-exception-sha256", "expected-failed-stage-sha256",
                 "expected-failed-ledger-sha256", "expected-parent", "runner", "message"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--execute", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()
    frozen = frozen_inputs(args)
    base = import_frozen(LOGS / "run_provisional_archival_checkpoint_v2.py", BASE_SHA, "frozen_df03_resume")
    recovery = import_frozen(LOGS / "recover_remote_checkpoint.py", RECOVERY_SHA, "frozen_3f_resume")
    config = validate_exception(json.loads(args.exception_config.read_bytes()))
    spec = json.loads(args.spec.read_bytes());validate_spec(spec)
    validate_failed_receipt(json.loads(args.failed_stage.read_bytes()))
    baseline = json.loads(args.custody_baseline.read_bytes())
    worktree = FACTORY.parent / ".rineya-dataset-git-worktree"
    original = FACTORY.parent / "rineya-world-model-research"
    report = {"author_execution": AUTHOR, "runner_execution": args.runner,
              "model_disclosure": "Inherited same-model independent preparation; no model diversity claim",
              "checkpoint_id": spec["checkpoint_id"], "started_at_utc": now(),
              "status": "PENDING_EXACT_ALREADY_STAGED_RESUMPTION",
              "spec_sha256": SPEC_SHA, "expected_parent_sha": PARENT,
              "source_sha256": checksum(Path(__file__)), "source_freeze_sha256": args.expected_freeze_sha256,
              "custody_adapter_sha256": CUSTODY_SHA, "exception_configuration_sha256": EXCEPTION_SHA,
              "prior_failed_stage_sha256": FAILED_SHA, "prior_failed_ledger_sha256": FAILED_LEDGER_SHA,
              "immutable_df03_helper_sha256": BASE_SHA, "grants_data_approval": False,
              "scope": "Exact already-staged r2 byte preservation; five frozen line-212 warnings only. No copy/stage/reset/clean/delete/configuration/data approval.",
              "self_inventory_boundary": "This runner/config/tests/fixtures/freeze and its new receipts are post-r2-inventory local until a later checkpoint."}
    with args.ledger.open("x", encoding="utf-8") as ledger:
        def log(phase, **values):
            ledger.write(json.dumps({"at_utc": now(), "phase": phase, **values}, separators=(",", ":")) + "\n")
            ledger.flush();os.fsync(ledger.fileno())
        try:
            base.original_custody(original, baseline)
            report["private_repository_evidence"] = base.dedicated_custody(worktree, PARENT, False)
            checked = base.inventory(FACTORY, spec)
            raw = base.lossless(FACTORY, spec, checked)
            if len(raw) != 26:
                raise ValueError("Actual 26 lossless proofs are required")
            previous = base.tree(worktree, PARENT)
            base.guard_previous(previous, checked)
            if len(previous) != 2303 or sum("/releases/" in "/" + n for n in previous) != 210:
                raise ValueError("Prior2303/210 release guard counts differ")
            parent_tree = parse_tree(base.git(worktree, "ls-tree", "-r", "-z", PARENT))
            full_index, staged = current_state(base, worktree, parent_tree, checked, previous)
            for n, name in enumerate(sorted(checked), 1):
                if base.proof(base.safe(worktree, PREFIX + "/" + name), False) != checked[name]:
                    raise ValueError("Actual already-copied target differs from source/spec")
                log("source_target_reverified_no_copy", path=name, **checked[name])
                if n % 100 == 0:
                    print(json.dumps({"phase": "source_target_reverified", "files": n}), flush=True)
            groups = {}
            for name, value in checked.items():
                groups.setdefault(value["git_blob"], []).append(name)
            if len(groups) != 2153:
                raise ValueError("Actual frozen index unique groups differ")
            for i, (blob, group) in enumerate(sorted(groups.items()), 1):
                expected = checked[group[0]]
                actual = recovery.process_stream(["git", "cat-file", "blob", blob],
                           lambda h: base.stream_proof(h, expected["size"], scan=False), worktree)
                if actual != expected:
                    raise ValueError("Actual streamed index blob SHA256/SHA1 differs")
                log("actual_index_blob_reverified", **actual, paths=group)
                if i % 100 == 0:
                    print(json.dumps({"phase": "actual_index_blob_reverified", "blobs": i}), flush=True)
            warning_bytes = verify_warning_sources(base, FACTORY, worktree, checked, base.index(worktree), config)
            diagnostics = validate_diagnostics(bounded_process(["git", "diff", "--cached", "--check"], worktree), config)
            restricted = strict_other_paths(worktree, staged)
            required = 2 * sum(v["size"] for v in checked.values()) + base.RESERVE
            free = shutil.disk_usage(worktree).free
            if free < required:
                raise ValueError("Original 2x inventory plus reserve capacity guard failed")
            report.update(files=len(checked), total_bytes=sum(v["size"] for v in checked.values()),
                          unique_actual_index_blob_groups=len(groups), raw_lossless_proofs=raw,
                          preserved_raw_bytes=sum(v["size"] for v in raw), staged_paths=staged,
                          full_index_exact_scope_verified=True, outside_dataset_parent_index_unchanged=True,
                          prior2303_paths_preserved=True, immutable210_release_bytes_preserved=True,
                          protected_original_six_baselines_verified=True,
                          all_source_target_index_sha256_and_sha1_verified=True,
                          exception_actual_five_source_target_index_proofs=warning_bytes,
                          actual_diff_check=diagnostics, restricted_other_paths_check=restricted,
                          free_bytes_at_preflight=free, required_free_bytes=required)
            log("ALL_FROZEN_ALREADY_STAGED_PROOFS_PASS_WITH_EXPLICIT_NONZERO_EXCEPTION", files=len(checked),
                bytes=report["total_bytes"], actual_diff_check_exit=2, bound_warning_count=5)
            if not args.execute:
                report.update(status="VALIDATED_ALREADY_STAGED_NO_COMMIT_PUSH", finished_at_utc=now())
                base.exclusive(args.receipt, report)
                print(json.dumps({"status": report["status"], "receipt": str(args.receipt)}));return
            # Recheck the actual inventory, target bytes and all frozen bindings immediately before commit.
            for name, expected in checked.items():
                if (base.proof(base.safe(FACTORY, name), False) != expected
                        or base.proof(base.safe(worktree, PREFIX + "/" + name), False) != expected):
                    raise ValueError("Frozen source/target changed before commit")
            recheck_bindings(frozen)
            base.original_custody(original, baseline);base.dedicated_custody(worktree, PARENT, False)
            again, again_staged = current_state(base, worktree, parent_tree, checked, previous)
            if again != full_index or again_staged != staged:
                raise ValueError("Full index/staged scope changed before commit")
            verify_warning_sources(base, FACTORY, worktree, checked, base.index(worktree), config)
            report["precommit_actual_diff_check"] = validate_diagnostics(
                bounded_process(["git", "diff", "--cached", "--check"], worktree), config)
            report["precommit_restricted_other_paths_check"] = strict_other_paths(worktree, staged)
            base.git(worktree, "commit", "-m", args.message)
            commit = base.text(worktree, "rev-parse", "HEAD")
            base.verify_committed_parent_tree(worktree, commit, PARENT, checked)
            if parse_tree(base.git(worktree, "ls-tree", "-r", "-z", commit)) != full_index:
                raise ValueError("Committed full tree differs from the exact reverified index")
            report.update(status="LOCAL_COMMIT_PRESERVED", commit_sha=commit)
            base.exclusive(args.local_commit_receipt, report);log("LOCAL_COMMIT_PRESERVED", commit_sha=commit)
            base.original_custody(original, baseline)
            report["actual_prepush_private_repository_evidence"] = prepush_custody(base, worktree, commit)
            base.git(worktree, "push", "origin", "HEAD:refs/heads/" + base.BRANCH)
            base.dedicated_custody(worktree, commit)
            base.original_custody(original, baseline)
            report.update(status="PUSHED_REMOTE_BYTES_VERIFICATION_PENDING", remote_commit_sha=commit,
                          finished_at_utc=now())
            report["next_actual_bounded_verification"] = ["python3", str(LOGS / "recover_remote_checkpoint.py"),
                "--spec", str(args.spec), "--expected-spec-sha256", SPEC_SHA, "--commit", commit,
                "--output", "NEW_RECOVERY_VERIFIED_RECEIPT", "--ledger", "NEW_RECOVERY_LEDGER",
                "--prior-evidence", str(args.receipt), "--runner", args.runner, "--workers", "2"]
            report["subsequent_remote_lossless_proof_required"] = {"new_actual_fetches": 26, "raw_bytes": 1353229114}
            base.exclusive(args.receipt, report);log("PUSHED_REMOTE_BYTES_VERIFICATION_PENDING", commit_sha=commit)
            print(json.dumps({"status": report["status"], "commit_sha": commit, "files": len(checked)}))
        except Exception as error:
            report.update(status="FAILED_FRESH_RESUMPTION_ARTIFACTS_PRESERVED", error_type=type(error).__name__,
                          error=str(error), finished_at_utc=now())
            log("FAILED_FRESH_RESUMPTION_ARTIFACTS_PRESERVED", error_type=type(error).__name__, error=str(error))
            if not args.receipt.exists():
                base.exclusive(args.receipt, report)
            raise


if __name__ == "__main__":
    main()
