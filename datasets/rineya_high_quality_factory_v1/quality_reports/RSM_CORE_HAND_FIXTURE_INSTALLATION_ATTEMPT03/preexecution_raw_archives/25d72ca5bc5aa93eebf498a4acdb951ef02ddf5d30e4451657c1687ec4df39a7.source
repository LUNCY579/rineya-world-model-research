"""Prospective CPU-only production runner; source preparation is not a grant.

All source/dependency/public/legacy bytes are frozen before module imports.
Actual invocation requires a later explicit root authorization. Existing
outputs, first failures and frozen attempts are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
FACTORY = HERE.parents[2]
PUBLIC_PATHS = (
    "schemas/RSM_OPTION_RESUMPTION_V1.md",
    "research/curricula/RSM-OPTION-RESUMPTION-PROSPECTIVE-20261010.md",
    "research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json",
    "schemas/RSM_OPTION_RESUMPTION_METADATA_V1.md",
    "schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R1.md",
    "schemas/RSM_OPTION_RESUMPTION_METADATA_V1_AMENDMENT_R2.md",
)
OWN_PREFIX = "generators/skills/option_resumption_pilot_v1_attempt01/"
REQUIRED_CLOSURE = set(PUBLIC_PATHS) | {
    "factory/common.py", "schemas/canonical/v1.schema.json",
    "generators/skills/prospective_option_resumption_attempt01/reference_attempt01.py",
    "generators/skills/prospective_option_resumption_attempt01/hand_fixtures_attempt01.json",
    "reviews/input_only/RSM-legacy-v0001.jsonl",
} | {OWN_PREFIX + name for name in ("generator.py", "normalization.py", "calibration.py", "runner.py", "PRODUCTION_CONFIG_ATTEMPT01.json")}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def new_atomic(path, raw):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("Symlink output component: " + str(parent))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(str(path))
    temporary.rename(path)


def reject_symlinks(path):
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("Symlink in bound source/archive path: " + str(path))


def import_exact(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-alias", required=True)
    arguments = parser.parse_args()
    if arguments.authorization != "ROOT_EXPLICIT_GENERATION_GRANT":
        raise PermissionError("Separate root generation grant required; this source preparation gives none")
    if not arguments.execution_alias or any(part in arguments.execution_alias for part in ("/", "\\", "..")):
        raise ValueError("Unsafe execution alias")
    config_raw = (HERE / "PRODUCTION_CONFIG_ATTEMPT01.json").read_bytes()
    config = json.loads(config_raw)
    freeze_raw = (HERE / "PREPARED_SOURCE_FREEZE_ATTEMPT01.json").read_bytes()
    if sha(freeze_raw) != arguments.expected_freeze_sha256:
        raise RuntimeError("Externally supplied expected preparation freeze SHA mismatch")
    freeze = json.loads(freeze_raw)
    bound_paths = [binding["path"] for binding in freeze["bindings"]]
    if len(set(bound_paths)) != len(bound_paths) or not REQUIRED_CLOSURE <= set(bound_paths):
        raise RuntimeError("Missing/duplicate mandatory source closure")
    if set(config["operative_source_paths"]) != REQUIRED_CLOSURE or tuple(config["operative_public_paths"]) != PUBLIC_PATHS:
        raise RuntimeError("Configuration omitted or changed mandatory operative closure")
    actual_python = str(Path(sys.executable).resolve())
    if actual_python not in bound_paths or Path(sys.executable).resolve() != Path(config["python_executable"]).resolve():
        raise RuntimeError("Executing resolved Python interpreter is not frozen")
    # FIRST check every own/public/reference/common/canonical/legacy byte.
    for binding in freeze["bindings"]:
        source = FACTORY / binding["path"]
        archive = HERE / binding["raw_archive"]
        reject_symlinks(source)
        reject_symlinks(archive)
        raw = source.read_bytes()
        if sha(raw) != binding["sha256"] or raw != archive.read_bytes():
            raise RuntimeError("Prepared source changed: " + binding["path"])
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destinations = config["future_core_artifact_paths"]
    for destination in destinations:
        if (output_root / destination).exists():
            raise FileExistsError("Preserve historical output: " + destination)
    if shutil.disk_usage(output_root).free < config["minimum_free_disk_bytes"]:
        raise RuntimeError("Insufficient free disk before production")
    execution_dir = HERE / "executions" / arguments.execution_alias
    if execution_dir.exists():
        raise FileExistsError("Preserve historical execution: " + str(execution_dir))
    execution_dir.mkdir(parents=True)
    archives = execution_dir / "raw_source_archives"
    archives.mkdir()
    runtime_bindings = []
    for binding in freeze["bindings"]:
        raw = (HERE / binding["raw_archive"]).read_bytes()
        target = archives / (binding["sha256"] + "-" + Path(binding["path"]).name)
        if not target.exists(): new_atomic(target, raw)
        runtime_bindings.append({**binding, "runtime_archive": str(target.relative_to(execution_dir))})
    runtime_freeze = {"status": "SOURCES_FROZEN_BEFORE_FIRST_GENERATOR_REFERENCE_COMMON_TEST_IMPORT",
                      "actual_author_execution_id": "/root/relational_critic_producer",
                      "at_utc": datetime.now(timezone.utc).isoformat(), "prepared_freeze_sha256": sha(freeze_raw),
                      "config_sha256": sha(config_raw), "bindings": runtime_bindings,
                      "python_executable": sys.executable, "python_version": sys.version,
                      "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                      "generation_grant_argument": arguments.authorization, "approval_authority": False}
    new_atomic(execution_dir / "SOURCE_RUNTIME_FREEZE.json", json_bytes(runtime_freeze))
    started_wall, started_cpu = time.monotonic(), time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (config["cpu_limit_seconds"], config["cpu_limit_seconds"] + 5))
    completed_artifacts = []
    def progress(name, evidence):
        new_atomic(execution_dir / (name + ".json"), json_bytes(evidence))
    try:
        # These are the first actual imports of the new modules/shared common.
        reference = import_exact("producer_own_e1_reference", FACTORY / config["own_reference_path"])
        common = import_exact("producer_canonical_common", FACTORY / "factory/common.py")
        normalizer = import_exact("producer_rsm_normalizer", HERE / "normalization.py")
        generator = import_exact("producer_rsm_generator", HERE / "generator.py")
        calibration = import_exact("producer_rsm_calibration", HERE / "calibration.py")
        fixture = json.loads((FACTORY / config["own_hand_fixture_path"]).read_bytes())
        public_configuration = json.loads((FACTORY / config["public_configuration_path"]).read_bytes())
        legacy_raw = (FACTORY / config["target_free_legacy_path"]).read_bytes()
        legacy = [json.loads(line) for line in legacy_raw.splitlines() if line]
        if any(any(key in row for key in ("verified_target", "gold", "target", "expected")) for row in legacy):
            raise RuntimeError("Legacy view is not target-free")
        contract = (FACTORY / config["operative_public_paths"][0]).read_text()
        source_bindings = {path: sha((FACTORY / path).read_bytes()) for path in config["operative_source_paths"]}
        progress("own_hand_calibration_before_generation", calibration.calibrate_before_generation(reference, normalizer, generator, fixture))
        corpus = generator.prepare_corpus(reference, normalizer, common, public_configuration, fixture, legacy, contract, source_bindings, progress_hook=progress)
        progress("own_actual_interface_calibration_before_candidate_write", calibration.calibrate_actual_interface(corpus, reference, generator, common, source_bindings))
        payloads = {
            destinations[0]: corpus["candidate_raw"],
            destinations[1]: ("".join(common.canonical(parent) + "\n" for parent in corpus["parents"])).encode(),
            destinations[2]: json_bytes(corpus["manifest"]),
            destinations[3]: json_bytes(corpus["diagnostics"]),
            destinations[4]: json_bytes(corpus["novelty"]),
            destinations[5]: (HERE / "generator.py").read_bytes(),
            destinations[6]: json_bytes({"status": "PENDING_PRODUCER_INTEGRITY_ONLY", "counts": corpus["counts"],
                                       "all_targets_from_own_e1": True, "actual_typed_field_order_pairs": 108,
                                       "all300_actual_manifest_pairs_resolve": True,
                                       "student_metadata_excluded": True, "data_approval": False}),
        }
        generation = {"batch": config["proposed_batch"], "generator_version": generator.VERSION,
                      "actual_author_execution_id": "/root/relational_critic_producer", "generation_seed": config["generation_seed"],
                      "source_prepared_freeze_sha256": sha(freeze_raw), "config_sha256": sha(config_raw),
                      "source_bindings": source_bindings, "own_reference_sha256": config["own_reference_sha256"],
                      "target_free_legacy_sha256": sha(legacy_raw), "counts": corpus["counts"],
                      "core_artifacts": [{"path": path, "bytes": len(raw), "sha256": sha(raw)} for path, raw in sorted(payloads.items())],
                      "status": "PENDING", "common_model_prior_context_disclosure": config["exposure_disclosure"],
                      "hand_calibration_motif_reused": True, "no_approval": True}
        payloads[destinations[7]] = json_bytes(generation)
        for path, raw in payloads.items():
            if shutil.disk_usage(output_root).free < len(raw) + config["minimum_free_disk_bytes"]:
                raise RuntimeError("Disk floor reached at output stage: " + path)
            if len(raw) > 25 * 1024 * 1024:
                raise RuntimeError("Bounded artifact exceeded25MiB; preserve first failure")
            new_atomic(output_root / path, raw)
            completed_artifacts.append({"path": path, "bytes": len(raw), "sha256": sha(raw)})
        for binding in freeze["bindings"]:
            if sha((FACTORY / binding["path"]).read_bytes()) != binding["sha256"]:
                raise RuntimeError("Frozen source changed during execution")
        status = "GENERATED216_PENDING_NO_APPROVAL"
        error = None
        exit_code = 0
    except BaseException as failure:
        status = "NEED_NEW_PUBLIC_AMENDMENT_FAILED_ATTEMPT_PRESERVED"
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
        new_atomic(execution_dir / "FIRST_FAILURE.json", json_bytes(error))
        exit_code = 1
    report = {"status": status, "actual_author_execution_id": "/root/relational_critic_producer", "error": error,
              "wall_seconds": round(time.monotonic() - started_wall, 6), "cpu_seconds": round(time.process_time() - started_cpu, 6),
              "max_rss_platform_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "disk_free_after_bytes": shutil.disk_usage(output_root).free, "completed_artifacts": completed_artifacts,
              "runtime_source_freeze_sha256": sha((execution_dir / "SOURCE_RUNTIME_FREEZE.json").read_bytes()),
              "no_approval": True, "no_automatic_retry_or_frozen_byte_rewrite": True}
    new_atomic(execution_dir / "EXECUTION.json", json_bytes(report))
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    try:
        result = main()
    except BaseException as failure:
        if isinstance(failure, SystemExit) and failure.code == 0:
            raise
        raw = json_bytes({"status": "PREFLIGHT_FIRST_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY",
                          "actual_author_execution_id": "/root/relational_critic_producer",
                          "type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc(),
                          "production_or_approval_inferred": False})
        new_atomic(HERE / "preflight_failures" / (str(time.time_ns()) + ".json"), raw)
        print(raw.decode(), file=sys.stderr)
        result = 1
    raise SystemExit(result)
