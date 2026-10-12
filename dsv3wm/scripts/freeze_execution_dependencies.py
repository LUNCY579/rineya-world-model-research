"""Append shared first-party execution dependencies to an unchanged freeze.

Creates a new file. Verifies every original pin first and preserves the original
registration metadata and member digests. Never loads a model or reads a bank.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def extend(source):
    from dsv3wm.native import sha256, verify_freeze
    source = Path(source).resolve()
    original_sha = verify_freeze(source)
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("confirmation_access") is not False:
        raise ValueError("Only a development freeze with explicit closed confirmation access is admitted")
    additions = [source, Path(__file__), ROOT / "prereg/EXECUTION_SOURCE_CLOSURE_V1.md",
                 ROOT / "src/dsv3wm/__init__.py", REPO / "bwm/code/bwmn/__init__.py",
                 REPO / "bwm/code/bwmn/blocks.py", REPO / "bwm/code/bwmn/flops.py",
                 REPO / "bwm/code/bwmn/planbench_extract.py"]
    files = dict(record["files"])
    files.update({str(path.resolve()).replace("\\", "/"): sha256(path) for path in additions})
    record.update(files=files, execution_dependency_amendment={
        "version": "shared-first-party-source-closure-v1",
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_freeze": str(source).replace("\\", "/"),
        "original_freeze_sha256": original_sha,
        "scope": "prospective dependency pins only; no data/config/model/scoring changes"})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Execution dependency freeze exists; use a new path")
    record = extend(args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
