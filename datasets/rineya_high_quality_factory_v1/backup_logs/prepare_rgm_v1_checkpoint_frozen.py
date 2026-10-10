"""Narrow V2 deferral wrapper for a frozen provisional RGM V1 checkpoint.

The existing inventory builder retains all byte, secret, cap, lossless and Git
checks. Only specifically named unfinished V2 implementation artifacts are
filtered. This wrapper grants no data approval and requires a freeze signal.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from backup_logs import prepare_checkpoint_inventory_items as inventory

AUTHOR = "/root/backup_manager"
EXACT_V2_PATHS = {
    "oracles/rgm_planning_controls_v2_independent.py": "Unfinished independent V2 formal oracle",
    "tests/oracle_rgm_planning_v2_test.py": "Unfinished independent V2 formal oracle test (actual observed spelling)",
}
PUBLIC_ROOTS = {"schemas", "research", "handoff"}
ARTIFACT_ROOTS = {"generators", "data", "candidates", "manifests", "reviews", "audits", "oracles", "tests"}
WAVE_TOKEN = re.compile(r"(?:^|[^A-Za-z0-9])planning[-_]gap[-_]v2(?=$|[^A-Za-z0-9])", re.IGNORECASE)
V1_TOKEN = re.compile(r"(?:^|[^A-Za-z0-9])(?:planning[-_]gap[-_]v1|RGM-(?:planning-)?v1)(?=$|[^A-Za-z0-9])", re.IGNORECASE)
ORACLE_PREFIX = re.compile(r"^RGM-(?:planning-)?v2(?=$|[-._/])")
FORMAL_TEST_TOKEN = re.compile(r"rgm_planning_controls_v2(?=$|[^A-Za-z0-9])")


def defer_reason(relative):
    """Return an explicit reason, or None; releases and public design always win."""
    path = PurePosixPath(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected canonical factory-relative path")
    if "releases" in path.parts or path.parts[0] in PUBLIC_ROOTS:
        return None
    if V1_TOKEN.search(relative) or re.search(r"root[-_]freeze", path.name, re.IGNORECASE):
        return None
    if relative in EXACT_V2_PATHS:
        return EXACT_V2_PATHS[relative]
    if path.parts[0] == "tests" and FORMAL_TEST_TOKEN.search(relative):
        return "Unfinished V2 formal oracle tests under the specifically named test prefix"
    if relative.startswith("reviews/oracle/") and ORACLE_PREFIX.search(relative[len("reviews/oracle/"):]):
        return "Unfinished V2 independent oracle execution/review evidence under the named prefix"
    if path.parts[0] in ARTIFACT_ROOTS and WAVE_TOKEN.search(relative):
        return "New planning-gap-v2/planning_gap_v2 implementation, candidate or review artifact; deferred until its own frozen checkpoint"
    return None


def deferred_snapshot(path, reason):
    """Hash actual observed bytes without forwarding private expectations."""
    if path.is_symlink():
        raise ValueError("A deferred symlink must be inspected explicitly")
    for attempt in range(3):
        before = path.stat()
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return {"path": path.relative_to(ROOT).as_posix(), "reason": reason,
                    "size": size, "sha256": digest.hexdigest(),
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "hash_semantics": "Exact observed local bytes; excluded from this checkpoint and allowed to continue evolving"}
    raise RuntimeError("Deferred file changed during all bounded hash attempts")


@contextmanager
def filter_factory_walk(deferred):
    original = Path.rglob

    def filtered(path, *args, **kwargs):
        for candidate in original(path, *args, **kwargs):
            if path.resolve() == ROOT.resolve() and candidate.is_file():
                relative = candidate.relative_to(ROOT).as_posix()
                reason = defer_reason(relative)
                if reason:
                    deferred[relative] = deferred_snapshot(candidate, reason)
                    continue
            yield candidate

    Path.rglob = filtered
    try:
        yield
    finally:
        Path.rglob = original


def option(arguments, name, default=None):
    count = arguments.count(name)
    if count > 1 or (default is None and count != 1):
        raise ValueError(f"Exactly one {name} is required")
    if name not in arguments:
        return default
    index = arguments.index(name)
    if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {name}")
    return arguments[index + 1]


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments.count("--freeze-scope") != 1:
        raise ValueError("Actual inventory preparation requires root's final freeze signal and --freeze-scope")
    arguments.remove("--freeze-scope")
    checkpoint = option(arguments, "--checkpoint-id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", checkpoint):
        raise ValueError("Invalid checkpoint identifier")
    preparer, runner = option(arguments, "--preparer"), option(arguments, "--runner")
    if option(arguments, "--approval", "provisional_checkpoint") != "provisional_checkpoint":
        raise ValueError("This archival wrapper supports provisional checkpoints only; no data approval")
    paths = {suffix: ROOT / "backup_logs" / f"{checkpoint}_{suffix}.json"
             for suffix in ["SPEC", "INVENTORY_REPORT", "FROZEN_SCOPE", "RECEIPT"]}
    if any(path.exists() for path in paths.values()):
        raise ValueError("Use a fresh checkpoint id; preserve every existing spec/report/receipt")
    builder_path = Path(inventory.__file__)
    builder_sha = hashlib.sha256(builder_path.read_bytes()).hexdigest()
    original_wrapper = ROOT / "backup_logs/prepare_belief_checkpoint_frozen.py"
    original_wrapper_sha = hashlib.sha256(original_wrapper.read_bytes()).hexdigest()
    deferred = {}
    argv_before = sys.argv
    try:
        sys.argv = [str(builder_path), *arguments]
        with filter_factory_walk(deferred):
            inventory.main()
    finally:
        sys.argv = argv_before
    if hashlib.sha256(builder_path.read_bytes()).hexdigest() != builder_sha:
        raise RuntimeError("Existing inventory builder changed during execution")
    if hashlib.sha256(original_wrapper.read_bytes()).hexdigest() != original_wrapper_sha:
        raise RuntimeError("Existing belief wrapper changed during execution")
    spec = json.loads(paths["SPEC"].read_text())
    included = {entry["path"] for entry in spec["files"]}
    if included.intersection(deferred):
        raise RuntimeError("Deferred artifacts unexpectedly present in included inventory")
    report = {"actual_author_execution": AUTHOR, "actual_runner_execution": preparer,
              "checkpoint_runner_execution": runner, "at_utc": datetime.now(timezone.utc).isoformat(),
              "model_disclosure": "Same-model agent execution; separate execution does not imply model diversity",
              "checkpoint_id": checkpoint, "scope": "Completed RGM V1 quarantine/review/gate evidence, prior factory and public V2 design; only unfinished specifically named V2 implementation artifacts deferred",
              "deferred_local_files": sorted(deferred.values(), key=lambda item: item["path"]),
              "future_deferral_policy": {"exact_paths": EXACT_V2_PATHS, "formal_test_prefix": FORMAL_TEST_TOKEN.pattern,
                  "oracle_review_prefix": ORACLE_PREFIX.pattern, "future_wave_token": WAVE_TOKEN.pattern,
                  "future_artifact_roots": sorted(ARTIFACT_ROOTS), "always_included_public_roots": sorted(PUBLIC_ROOTS)},
              "all_release_artifacts_in_scope": True, "public_v2_design_and_root_freezes_in_scope": True,
              "all_v1_artifacts_in_scope": True, "grants_data_approval": False,
              "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "unchanged_inventory_builder_sha256": builder_sha,
              "unchanged_original_belief_wrapper_sha256": original_wrapper_sha,
              "deferred_hash_boundary": "Per-file exact bytes observed during this inventory run; excluded V2 files may change later and acquire no checkpoint persistence or data approval",
              "included_checks": "Unchanged existing inventory-builder byte, secret, cap, lossless, Git and source-stability checks retained"}
    inventory.atomic_json(paths["FROZEN_SCOPE"], report)
    scope_entry, _ = inventory.snapshot(paths["FROZEN_SCOPE"])
    spec["files"].append(scope_entry)
    spec["files"].sort(key=lambda entry: entry["path"])
    spec["frozen_scope_report"] = scope_entry
    spec["frozen_scope_filter_author_execution"] = AUTHOR
    spec["frozen_scope_filter_runner_execution"] = preparer
    spec["scope_note"] = report["scope"] + "; unchanged included validation/lossless checks; archival only, no data approval"
    inventory.atomic_json(paths["SPEC"], spec)
    print(json.dumps({"frozen_files": len(spec["files"]), "bytes": sum(entry["size"] for entry in spec["files"]),
                      "deferred_local_files": len(deferred), "validation_status": spec["validation_status"],
                      "author_execution": AUTHOR, "preparer_execution": preparer, "runner_execution": runner}))


if __name__ == "__main__":
    main()
