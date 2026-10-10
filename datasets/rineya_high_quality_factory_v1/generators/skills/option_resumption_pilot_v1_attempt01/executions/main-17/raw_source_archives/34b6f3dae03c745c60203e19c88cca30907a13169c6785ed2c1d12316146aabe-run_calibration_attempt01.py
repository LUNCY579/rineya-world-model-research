"""Execute frozen preparation hand tests only, retaining both seed outputs.

This has no candidate/parent/gold/control-manifest generation entry point.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FACTORY = HERE.parents[2]
PYTHON = "/opt/anaconda3/bin/python3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    if path.exists():
        raise FileExistsError(str(path))
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("Preparation artifact exceeds file bound")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    if path.exists():
        raise FileExistsError(str(path))
    temporary.rename(path)


def main():
    freeze_bytes = (HERE / "SOURCE_FREEZE_ATTEMPT01.json").read_bytes()
    freeze = json.loads(freeze_bytes)
    verified = []
    for binding in freeze["frozen_bindings"]:
        actual = (FACTORY / binding["path"]).read_bytes()
        archived = (HERE / binding["raw_archive"]).read_bytes()
        if digest(actual) != binding["sha256"] or actual != archived:
            raise RuntimeError("Frozen binding changed: " + binding["path"])
        verified.append(binding["path"])
    disk_before = shutil.disk_usage(HERE).free
    executions = []
    for seed in (17, 991):
        output_path = HERE / f"CALIBRATION_HASHSEED_{seed}_ATTEMPT01.stdout.json"
        error_path = HERE / f"CALIBRATION_HASHSEED_{seed}_ATTEMPT01.stderr.txt"
        if output_path.exists() or error_path.exists():
            raise FileExistsError("Historical calibration output already exists")
        environment = dict(os.environ)
        environment.update(PYTHONHASHSEED=str(seed), PYTHONDONTWRITEBYTECODE="1")
        before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        run = subprocess.run([PYTHON, str(HERE / "calibration_attempt01.py")],
                             cwd=HERE, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=60)
        wall = time.monotonic() - started
        after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        write_new(output_path, run.stdout)
        write_new(error_path, run.stderr)
        executions.append({"hash_seed": seed, "exit_code": run.returncode,
                           "wall_seconds": round(wall, 6),
                           "child_cpu_seconds": round(after_usage.ru_utime + after_usage.ru_stime - before_usage.ru_utime - before_usage.ru_stime, 6),
                           "child_max_rss_platform_raw": after_usage.ru_maxrss,
                           "stdout": {"path": str(output_path.relative_to(HERE)), "bytes": len(run.stdout), "sha256": digest(run.stdout)},
                           "stderr": {"path": str(error_path.relative_to(HERE)), "bytes": len(run.stderr), "sha256": digest(run.stderr)}})
        if run.returncode:
            break
    success = len(executions) == 2 and all(item["exit_code"] == 0 for item in executions)
    equality = success and executions[0]["stdout"]["sha256"] == executions[1]["stdout"]["sha256"]
    report = {"actual_author_execution_id": "/root/relational_critic_producer",
              "scope": "ISOLATED_PUBLIC_ONLY_PREPARATION_CALIBRATION",
              "status": "CALIBRATION_PASSED" if success and equality else "CALIBRATION_FAILED_PRESERVED",
              "source_freeze_sha256": digest(freeze_bytes), "verified_frozen_binding_count": len(verified),
              "python_executable": PYTHON, "python_version": sys.version,
              "executions": executions, "separate_process_hash_seed_stdout_byte_equality": bool(equality),
              "disk_free_before_bytes": disk_before, "disk_free_after_bytes": shutil.disk_usage(HERE).free,
              "production_records": 0, "production_semantic_parents": 0, "production_targets": 0,
              "production_manifests": 0, "actual_legacy_normalization_checks": 0,
              "prospective_families": 12, "prospective_correlated_rows": 216,
              "production_authorization": False, "approval_authority": False,
              "common_model_prior_context_disclosure": freeze["exposure_disclosure"],
              "independent_oracle_or_private_reviews_read": False,
              "failure_policy": "Every existing byte retained. If this failed, next source/test version requires a fresh alias; no edits to frozen source."}
    if success:
        results = json.loads((HERE / executions[0]["stdout"]["path"]).read_bytes())
        report["actual_calibration_counts"] = {key: results[key] for key in ("passed", "failed", "manual_trace_events", "legal_prefix_cuts", "hand_contexts", "malformed_cases")}
    write_new(HERE / "CALIBRATION_EXECUTION_ATTEMPT01.json",
              (json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(report, sort_keys=True))
    return 0 if success and equality else 1


if __name__ == "__main__":
    raise SystemExit(main())
