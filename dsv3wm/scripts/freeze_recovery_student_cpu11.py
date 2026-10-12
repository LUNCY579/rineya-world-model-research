"""Append unchanged CPU11 H3 entrypoint pins to a verified phase freeze."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.native_cpu11 import admission_profile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("H3 CPU11 phase freeze exists")
    digest = verify_freeze(args.freeze)
    record = json.loads(args.freeze.read_text(encoding="utf-8"))
    if record.get("confirmation_access") is not False or not str(record.get("instrument", "")).startswith(("warm-verified-H3", "recovery-H3")):
        raise ValueError("explicit H3 development phase freeze required")
    extra = [args.freeze, Path(__file__), ROOT / "src/dsv3wm/native_cpu11.py",
             ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_3.md", ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_4.md",
             *[ROOT / "scripts" / name for name in ("train_recovery_student_cpu11.py", "evaluate_recovery_student_cpu11.py",
                       "evaluate_recovery_student_guards_cpu11.py", "train_recovery_student.py",
                       "evaluate_recovery_student.py", "evaluate_recovery_student_guards.py")]]
    files = dict(record["files"])
    files.update({str(path.resolve()).replace("\\", "/"): sha256(path) for path in extra})
    record.update(files=files, cpu11_phase_amendment={"amended_at_utc": datetime.now(timezone.utc).isoformat(),
                  "original_freeze": str(args.freeze.resolve()), "original_freeze_sha256": digest,
                  "profile": admission_profile(), "scope": "H3 admission only; objective/data/decoder/scoring unchanged"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
