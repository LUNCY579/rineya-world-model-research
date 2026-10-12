"""Append CPU11 Ledger/RP entrypoint pins to a verified development freeze.

Creates a new file and preserves every original member digest and registration
field. No model, simulator, dataset row or private label is opened by this helper.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

INSTRUMENTS = {"tick-compatible-ledger-development-v1", "rcwm-RP-retention-v2-B0",
               "rcwm-RP-retention-adapted-lineage-v2"}


def source_additions():
    from train_ledger_recovery_cpu11 import required_sources
    from evaluate_rcwm_retention_v2_cpu11 import source_paths
    return sorted(set([Path(__file__), *required_sources(), *source_paths()]))


def extend(source):
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.native_cpu11 import admission_profile
    source = Path(source).resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if (record.get("instrument") not in INSTRUMENTS or record.get("confirmation_access") is not False
            or not isinstance(record.get("files"), dict) or not record["files"]):
        raise ValueError("Explicit Ledger or tick-complete R/P v2 development freeze required")
    if "ledger_cpu11_admission_amendment" in record:
        raise ValueError("Ledger CPU11 extension already present; retain it unchanged")
    original_sha = verify_freeze(source)
    files = dict(record["files"])
    additions = [source, *source_additions()]
    for path in additions:
        key, digest = str(path.resolve()).replace("\\", "/"), sha256(path)
        if key in files and files[key] != digest:
            raise ValueError("Extension cannot replace an original member digest")
        files[key] = digest
    record.update(files=files, ledger_cpu11_admission_amendment={
        "version": "ledger-cpu11-admission-v1", "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_freeze": str(source).replace("\\", "/"), "original_freeze_sha256": original_sha,
        "profile": admission_profile(),
        "scope": "Ledger B0/train/DEV and paired R/P v2 admission only; data/config/model/algorithm/gates unchanged"})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Ledger CPU11 freeze exists; use a fresh path")
    record = extend(args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
