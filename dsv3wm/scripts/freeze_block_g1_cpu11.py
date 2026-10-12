"""Append CPU11 execution pins to the unchanged constructed Block G1 freeze."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def extend(source):
    from dsv3wm.native import sha256, verify_freeze
    from dsv3wm.native_cpu11 import admission_profile
    from evaluate_block_g1_cpu11 import cpu11_sources
    source = Path(source).resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if (record.get("instrument") != "constructed-original-delta-Block-G1-v1"
            or record.get("confirmation_access") is not False):
        raise ValueError("unchanged constructed Block G1 development freeze required")
    if "cpu11_resource_amendment" in record:
        raise ValueError("Block G1 CPU11 amendment already present; preserve its metadata")
    original_sha = verify_freeze(source)
    additions = [source, Path(__file__), *cpu11_sources(), ROOT / "scripts/freeze_block_g1.py",
                 ROOT / "src/dsv3wm/__init__.py", REPO / "bwm/code/bwmn/__init__.py",
                 REPO / "bwm/code/bwmn/blocks.py", REPO / "bwm/code/bwmn/flops.py",
                 REPO / "bwm/code/bwmn/planbench_extract.py"]
    files = dict(record["files"])
    for path in sorted(set(additions)):
        key = str(path.resolve()).replace("\\", "/")
        digest = sha256(path)
        if key in files and files[key] != digest:
            raise ValueError("existing pin cannot be replaced by CPU11 extension")
        files[key] = digest
    record.update(files=files, cpu11_resource_amendment={
        "version": "constructed-Block-G1-cpu11-v1",
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_freeze": str(source).replace("\\", "/"),
        "original_freeze_sha256": original_sha,
        "profile": admission_profile(),
        "scope": "existing sixteen constructed cases and G1 gate; admission only; model/data/prompt/parser/scoring/decoder unchanged"})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("preserve original freezes; select a new CPU11 freeze path")
    record = extend(args.freeze)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
