"""Append training-only CPU10.5 pins to the unchanged CPU11 H3 training freeze."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def extra_paths():
    from dsv3wm.native_cpu105_training import registered_config_paths
    return [Path(__file__), ROOT / "src/dsv3wm/native_cpu105_training.py",
            ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/native_cpu11.py",
            ROOT / "scripts/train_recovery_student.py", ROOT / "scripts/train_recovery_student_cpu11.py",
            ROOT / "scripts/train_recovery_student_cpu105.py", ROOT / "scripts/freeze_recovery_student_cpu11.py",
            ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_5.md", *registered_config_paths()]


def extend(source):
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.native_cpu105_training import admission_profile, REGISTERED_CONFIGS
    from dsv3wm.provenance import verify_bound_inputs
    source = Path(source).resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    previous = record.get("cpu11_phase_amendment", {}).get("profile", {})
    if (record.get("instrument") != "warm-verified-H3-students-controls-v1"
            or record.get("confirmation_access") is not False
            or previous.get("name") != "dsv3wm-cpu11-v1"
            or not isinstance(record.get("files"), dict) or not record["files"]):
        raise ValueError("Explicit CPU11 H3 student/control TRAINING freeze required; inference/Ledger/Block refuse")
    if "cpu105_training_amendment" in record:
        raise ValueError("CPU10.5 training extension already present; preserve it unchanged")
    original_sha = verify_freeze(source)
    prior_sources = [ROOT / "src/dsv3wm/native.py", ROOT / "src/dsv3wm/native_cpu11.py",
                     ROOT / "scripts/train_recovery_student.py", ROOT / "scripts/train_recovery_student_cpu11.py"]
    for filename, expected_sha in REGISTERED_CONFIGS.values():
        path = ROOT / "configs" / filename
        if sha256(path) != expected_sha:
            raise ValueError("Registered remaining H3 configuration bytes changed")
        prior_sources.append(path)
    verify_bound_inputs(source, prior_sources)
    files = dict(record["files"])
    for path in [source, *extra_paths()]:
        key, digest = str(path.resolve()).replace("\\", "/"), sha256(path)
        if key in files and files[key] != digest:
            raise ValueError("CPU10.5 extension cannot replace an original member digest")
        files[key] = digest
    record.update(files=files, cpu105_training_amendment={
        "amended_at_utc": datetime.now(timezone.utc).isoformat(), "original_freeze": str(source).replace("\\", "/"),
        "original_freeze_sha256": original_sha, "profile": admission_profile(),
        "scope": "remaining fixed A4/C3/C4 training admission only; every inference and other-domain floor unchanged",
        "registered_config_sha256": {kind: digest for kind, (_, digest) in REGISTERED_CONFIGS.items()}})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("CPU10.5 training freeze exists; use a new path")
    record = extend(args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
