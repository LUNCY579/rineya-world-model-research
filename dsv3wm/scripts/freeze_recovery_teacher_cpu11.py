"""Extend unchanged H3 teacher pins with the explicit CPU11-only admission path."""
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
    from dsv3wm.recovery import RecoveryConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=ROOT / "prereg/RECOVERY_H3_TEACHER_FREEZE_V3.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("CPU11 teacher freeze exists; preserve it")
    original_sha = verify_freeze(args.freeze)
    record = json.loads(args.freeze.read_text(encoding="utf-8"))
    cfg = RecoveryConfig.read(ROOT / "configs/recovery_A0.json")
    if cfg.device != "cpu" or cfg.threads != 4 or record.get("confirmation_access") is not False:
        raise ValueError("original CPU/four-thread/development-only teacher required")
    extra = [args.freeze, Path(__file__), ROOT / "src/dsv3wm/native_cpu11.py",
             ROOT / "scripts/collect_recovery_teacher_cpu11.py", ROOT / "scripts/collect_recovery_teacher.py",
             ROOT / "prereg/RECOVERY_RESOURCE_CLARIFICATION_3.md"]
    files = dict(record["files"])
    files.update({str(path.resolve()).replace("\\", "/"): sha256(path) for path in extra})
    record.update(files=files, resource_admission_amendment={
        "amended_at_utc": datetime.now(timezone.utc).isoformat(), "original_freeze_sha256": original_sha,
        "original_freeze": str(args.freeze.resolve()), "profile": admission_profile(),
        "scope": "load-floor-only CPU recovery; model/config/pool/prediction/scoring unchanged"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
