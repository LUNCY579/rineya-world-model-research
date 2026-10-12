"""Append singleton CPU10.5 pins to actual-output H3 proposer/guard freezes.

Student guard extensions require exact raw parity of new singleton warm refs
with preserved old refs. No model or sealed label is accessed by this helper.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
READOUT = "recovery-H3-actual-output-readout-v1"
GUARDS = "recovery-H3-fixed-development-guards-v1"


def source_additions():
    from dsv3wm.native_cpu105_readout import registered_config_paths
    return [Path(__file__), ROOT / "src/dsv3wm/native_cpu105_readout.py", ROOT / "src/dsv3wm/native_cpu11.py",
            ROOT / "src/dsv3wm/native.py", ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_6.md",
            ROOT / "scripts/evaluate_recovery_student_cpu105_readout.py",
            ROOT / "scripts/evaluate_recovery_student_guards_cpu105_readout.py",
            ROOT / "scripts/evaluate_recovery_student_cpu11.py", ROOT / "scripts/evaluate_recovery_student_guards_cpu11.py",
            ROOT / "scripts/evaluate_recovery_student.py", ROOT / "scripts/evaluate_recovery_student_guards.py",
            ROOT / "src/dsv3wm/__init__.py", REPO / "bwm/code/bwmn/__init__.py",
            REPO / "bwm/code/bwmn/blocks.py", REPO / "bwm/code/bwmn/flops.py",
            REPO / "bwm/code/bwmn/planbench_extract.py", *registered_config_paths()]


def _command_path(record, flag):
    command = record.get("command")
    if not isinstance(command, list) or any(type(value) is not str for value in command) or command.count(flag) != 1:
        raise ValueError("Unambiguous warm guard command paths required")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError("Missing warm guard command path")
    return (REPO / command[index + 1]).resolve()


def _checked_guard(path, *, singleton):
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.native_cpu105_readout import validate_singleton_baseline
    from dsv3wm.provenance import verify_bound_inputs
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("schema") != "dsv3wm-h3-fixed-development-guards-v1"
            or record.get("status") != "development-guards-evaluated" or record.get("warm_baseline") is not True
            or record.get("artifact_arm") not in {"A0", "A2"} or record.get("oracle_feedback") is not False
            or record.get("all_generation_fixed_before_scoring") is not True):
        raise ValueError("Completed unchanged warm A0/A2 guard record required")
    if singleton:
        validate_singleton_baseline(record)
    config, data, fixtures, run, adapter, freeze = [_command_path(record, flag) for flag in
            ("--config", "--data", "--fixtures", "--run", "--adapter", "--freeze")]
    from dsv3wm.recovery_student import StudentConfig
    from dsv3wm.native_cpu105_readout import validate_readout_config
    actual_config = StudentConfig.read(config)
    validate_readout_config(actual_config)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(record.get("config")) != canonical(actual_config.dict()):
        raise ValueError("Warm guard full expanded configuration differs from its actual input")
    if _command_path(record, "--out") != path or run != adapter.parent / "run.json":
        raise ValueError("Warm guard result/run path identity mismatch")
    bindings = [config, data, fixtures, run, adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
    for actual, field in zip(bindings, ("config_sha256", "data_sha256", "fixtures_sha256", "run_sha256", "adapter_sha256", "adapter_config_sha256")):
        if sha256(actual) != record.get(field):
            raise ValueError("Warm guard config/data/artifact digest drift")
    if verify_freeze(freeze) != record.get("freeze_sha256"):
        raise ValueError("Warm guard execution freeze changed")
    sources = record.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Warm guard executed source hashes required")
    for name, digest in sources.items():
        source = (REPO / name).resolve()
        if sha256(source) != digest:
            raise ValueError("Warm guard executed source changed")
        bindings.append(source)
    verify_bound_inputs(freeze, bindings)
    trace = path.with_suffix(".traces.json")
    if sha256(trace) != record.get("traces_sha256"):
        raise ValueError("Warm guard raw trace digest changed")
    raw = json.loads(trace.read_text(encoding="utf-8"))
    if raw.get("status") != "generation-complete" or not isinstance(raw.get("dynamics"), list) or not isinstance(raw.get("behavior"), list):
        raise ValueError("Complete warm raw traces required")
    return record, raw, [path, trace, freeze, *bindings]


def raw_guard_parity(old_path, new_path):
    """Raw reply/conditioning/order parity only; never consult metric correctness."""
    from dsv3wm.native import sha256
    old, old_raw, old_paths = _checked_guard(old_path, singleton=False)
    new, new_raw, new_paths = _checked_guard(new_path, singleton=True)
    fields = ("artifact_arm", "adapter_sha256", "adapter_config_sha256", "run_sha256", "config", "config_sha256",
              "data_sha256", "fixtures_sha256")
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if any(canonical(old.get(field)) != canonical(new.get(field)) for field in fields):
        raise ValueError("Warm singleton parity requires identical artifact/config/data/fixture identities")
    if canonical(old_raw) != canonical(new_raw):
        raise ValueError("Warm singleton raw-output parity failed; preserve both outputs and do not launch student guards")
    if len(new_raw["behavior"]) != 6 or not new_raw["dynamics"]:
        raise ValueError("Complete warm dynamics and six behavior replies required")
    return {"warm_arm": new["artifact_arm"], "passed": True, "old_result": str(Path(old_path).resolve()),
            "new_result": str(Path(new_path).resolve()), "old_result_sha256": sha256(old_path), "new_result_sha256": sha256(new_path),
            "old_traces_sha256": old["traces_sha256"], "new_traces_sha256": new["traces_sha256"],
            "warm_adapter_sha256": new["adapter_sha256"], "warm_adapter_config_sha256": new["adapter_config_sha256"],
            "warm_run_sha256": new["run_sha256"],
            "dynamics_raw_rows": len(new_raw["dynamics"]), "behavior_raw_rows": 6,
            "scope": "exact preserved raw replies/conditioning/order; physical counters may differ; no metric-based selection"}, sorted(set(old_paths + new_paths))


def bind_student_warm_parity(source, artifact, proof):
    """Bind the parity artifact to every selected student's actual frozen warm edge."""
    from dsv3wm.native import sha256
    from dsv3wm.native_cpu105_readout import validate_readout_config
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery_student import StudentConfig
    directory = artifact.get("directory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("Selected student directory is required for warm parity binding")
    directory = (REPO / directory).resolve()
    student_run = directory / "run.json"
    verify_bound_inputs(source, [student_run])
    student = json.loads(student_run.read_text(encoding="utf-8"))
    if not isinstance(student, dict) or not isinstance(student.get("config"), dict):
        raise ValueError("Actual selected student run/configuration required")
    cfg = StudentConfig(**student["config"])
    kind = validate_readout_config(cfg)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (student.get("status") != "student-pilot-trained" or student.get("training_kind") != kind
            or artifact.get("kind") != kind or canonical(artifact.get("config")) != canonical(cfg.dict())
            or student.get("warm_arm") != cfg.warm_arm or artifact.get("warm_arm") != cfg.warm_arm
            or proof.get("warm_arm") != cfg.warm_arm or proof.get("passed") is not True):
        raise ValueError("Parity reference must match the selected student's actual kind/config/warm arm")
    warm = _command_path(student, "--warm-dir")
    edges = [(warm / "adapter/adapter_model.safetensors", "warm_adapter_sha256"),
             (warm / "adapter/adapter_config.json", "warm_adapter_config_sha256"),
             (warm / "run.json", "warm_run_sha256")]
    verify_bound_inputs(source, [path for path, _ in edges])
    identities = {}
    for path, field in edges:
        actual = sha256(path)
        if student.get(field) != actual or proof.get(field) != actual:
            raise ValueError(f"Warm singleton parity differs from selected student's actual {field}")
        identities[field] = actual
    return {"student_kind": kind, "student_directory": str(directory), "student_run_sha256": sha256(student_run),
            "warm_arm": cfg.warm_arm, **identities}


def extend(source, *, old_warm_guards=(), new_warm_guards=()):
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.native_cpu105_readout import admission_profile, REGISTERED_CONFIGS
    from dsv3wm.provenance import verify_bound_inputs
    source = Path(source).resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if (record.get("instrument") not in {READOUT, GUARDS} or record.get("confirmation_access") is not False
            or record.get("cpu11_phase_amendment", {}).get("profile", {}).get("name") != "dsv3wm-cpu11-v1"
            or not isinstance(record.get("files"), dict) or not record["files"]):
        raise ValueError("Actual-output CPU11 H3 readout/guard freeze required; teacher/training/Ledger/Block refuse")
    if "cpu105_singleton_readout_amendment" in record:
        raise ValueError("Singleton CPU10.5 extension already present; retain it unchanged")
    original_sha = verify_freeze(source)
    verify_bound_inputs(source, [ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/native_cpu11.py",
                       ROOT / "scripts/evaluate_recovery_student_cpu11.py", ROOT / "scripts/evaluate_recovery_student_guards_cpu11.py"])
    for filename, expected_sha in REGISTERED_CONFIGS.values():
        if sha256(ROOT / "configs" / filename) != expected_sha:
            raise ValueError("Registered H3 readout config bytes changed")
    parity, extra = {}, []
    if len(old_warm_guards) != len(new_warm_guards):
        raise ValueError("Each old warm guard needs one new singleton reference")
    for old, new in zip(old_warm_guards, new_warm_guards):
        if record["instrument"] != GUARDS:
            raise ValueError("Warm guard parity references belong only to guard freezes")
        report, paths = raw_guard_parity(old, new)
        if report["warm_arm"] in parity:
            raise ValueError("Duplicate warm singleton parity arm")
        parity[report["warm_arm"]] = report
        extra += paths
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Actual selected readout/guard artifacts required")
    student_artifacts = [artifact for artifact in artifacts if not str(artifact.get("kind", "")).startswith("warm-")]
    student_bindings = []
    if record["instrument"] == GUARDS:
        required = {artifact["warm_arm"] for artifact in student_artifacts}
        if set(parity) != required:
            raise ValueError("Student guards require raw-output-parity-qualified new singleton references for every warm arm")
        for artifact in student_artifacts:
            student_bindings.append(bind_student_warm_parity(source, artifact, parity[artifact["warm_arm"]]))
    elif parity:
        raise ValueError("Proposer readout geometry is unchanged; guard references are not used")
    files = dict(record["files"])
    for path in [source, *source_additions(), *extra]:
        key, digest = str(Path(path).resolve()).replace("\\", "/"), sha256(path)
        if key in files and files[key] != digest:
            raise ValueError("Extension cannot replace an original member digest")
        files[key] = digest
    record.update(files=files, cpu105_singleton_readout_amendment={
        "amended_at_utc": datetime.now(timezone.utc).isoformat(), "original_freeze": str(source).replace("\\", "/"),
        "original_freeze_sha256": original_sha, "profile": admission_profile(), "warm_singleton_raw_parity": parity,
        "selected_student_warm_bindings": student_bindings,
        "scope": "prospective singleton H3 readout admission/geometry only; logical schedule, configs, metrics and scoring unchanged"})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--old-warm-guard", action="append", default=[], type=Path)
    parser.add_argument("--new-warm-guard", action="append", default=[], type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Singleton CPU10.5 readout freeze exists; use a new path")
    record = extend(args.freeze, old_warm_guards=args.old_warm_guard, new_warm_guards=args.new_warm_guard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
