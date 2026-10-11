"""Export the BWM-001 records into bwm/ of this repository (private; not a release).

Source: Rineya_BWM_001_Research_Bundle/native_v1. Included: design/decision/preregistration documents, research notes,
code (bwmn/, scripts/, tests/, launchers, job and bundle specs), and run outputs that are records (summaries, metrics,
grades, selection/freeze records, per-instance test plans, logs). Excluded and listed in bwm/EXCLUDED_LARGE_FILES.json
with size and SHA-256: adapter weights, training data, prediction caches, anything over 25 MB. Not exported at all:
machine health logs (host name), scratch/failed/superseded directories. A secret/personal-data scan must pass before
anything is written. Re-running replaces bwm/ except bwm/README.md.

Usage: python tools/export_bwm.py [--src <native_v1>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "bwm"
DEFAULT_SRC = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
MAX_BYTES = 25 * 2 ** 20
DOCS = ["DESIGN.md", "DECISIONS.md", "PREREG.md", "PREREG_PLANBENCH.md", "PREREG_B3_DISTILL.md", "RESULTS_INTERIM.md", "TODO_BEFORE_RUNS.md"]
CODE_DIRS = ["bwmn", "scripts", "tests"]
RESULT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".stdout", ".log"}
SKIP_DIR_PARTS = {"_failed", "_superseded", "__pycache__", "BW_data", "BW_data_v2"}
SKIP_NAME = re.compile(r"^(health(_now)?\.(log|json|err)|orchestrator.*\.err|fcache.*\.jsonl|.*\.partial\.jsonl|"
                       r".*_train\.jsonl|.*_val\.jsonl|vlab_sft\.jsonl|vlab_rows\.jsonl|\.yield.*|\.stop.*|"
                       r"\.health_stop)$")
DATA_KEEP = {"META.json"}  # from the data directories, only the generator metadata
SECRET = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,})")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_OK = {"noreply@anthropic.com"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def plan(src: Path):
    keep, excluded = [], []
    docs = list(DOCS) + sorted(p.name for p in src.glob("PREREG_*.md") if p.name not in DOCS)  # every prereg
    for d in docs:
        if (src / d).exists():
            keep.append((src / d, Path("docs") / d))
    b3_open = (src / "runs/BW_B3_EVAL/B3_REPORT.json").exists()
    for p in sorted((src / "research").rglob("*")):  # incl. audit_evidence/, b4_exclusions/, b5_macros/ (D-75)
        if p.is_file() and "__pycache__" not in p.parts:
            rp = p.relative_to(src / "research").parts
            # never: git internals / third-party clones and papers (personal data, redistribution), weights or
            # training rows even as test stubs (D-51)
            if any(x == ".git" or x == "llmsp" or x == "pb2022" or x.startswith("dl_") for x in rp) or \
                    p.suffix.lower() == ".pdf":
                continue
            if p.suffix == ".safetensors" or p.name.endswith("_train.jsonl") or p.name == "distill_rows.jsonl" or \
                    (p.name.startswith("mix_") and p.suffix == ".jsonl"):
                excluded.append((p, "weights or training rows inside research evidence (never exported; D-51)"))
                continue
            if p.stat().st_size > MAX_BYTES:
                excluded.append((p, "larger than 25 MB"))
                continue
            keep.append((p, Path("docs/research") / p.relative_to(src / "research")))
    for d in CODE_DIRS:
        for p in sorted((src / d).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts:
                keep.append((p, Path("code") / p.relative_to(src)))
    for p in sorted((src / "runs").glob("*.sh")) + sorted((src / "runs").glob("jobs*.json")) + \
            sorted((src / "runs/bundles").glob("*.json")):
        keep.append((p, Path("code") / p.relative_to(src)))
    for name in ("LEDGER.json", "SOURCE.md", "LICENSE_LLMs-Planning_MIT.txt"):
        if (src / "bench/planbench" / name).exists():
            keep.append((src / "bench/planbench" / name, Path("bench/planbench") / name))
    for p in sorted((src / "runs").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        parts = set(rel.parts)
        if parts & {"_failed", "_superseded", "__pycache__"} or any(x.startswith(("_tmp", "_smoke", "_probe")) or
                                                                     (x.startswith("_") and "smoke" in x)
                                                                     for x in rel.parts):
            continue
        if p.name == "distill_rows.jsonl" or (p.name.startswith("mix_") and p.suffix == ".jsonl"):
            excluded.append((p, "training rows for the B3 proposer arms (never exported; D-51)"))
            continue
        if not b3_open and (("BW_DISTILL_r1" in parts and p.name == "solutions.jsonl") or
                            ("BW_B3_EVAL" in parts and p.name.startswith("plans_"))):
            excluded.append((p, "B3 embargo until runs/BW_B3_EVAL/B3_REPORT.json exists (PREREG_B3, D-64)"))
            continue
        if parts & {"BW_data", "BW_data_v2"} and p.name not in DATA_KEEP:
            excluded.append((p, "training data (regenerable with scripts/blocks_data.py)"))
            continue
        if p.name == "adapter_model.safetensors":
            excluded.append((p, "LoRA adapter weights (user decision 2026-10-08: no weights in the repo)"))
            continue
        evidence_cache = (p.name.startswith("fcache") and p.suffix == ".jsonl" and
                          parts & {"BW_TEST_B2", "BW_B3_HARD", "BW_DISTILL_r1", "BW_SELECT_v1", "BW_B3_EVAL"})
        if SKIP_NAME.match(p.name) and not evidence_cache:  # D-75: F caches of test/selection runs are evidence
            if p.name.startswith("fcache") or p.name.endswith((".jsonl",)):
                excluded.append((p, "cache / training rows (regenerable)"))
            continue
        if p.suffix not in RESULT_SUFFIXES and p.suffix != ".sh":
            continue
        if p.stat().st_size > MAX_BYTES:
            excluded.append((p, "larger than 25 MB"))
            continue
        keep.append((p, Path("results") / rel.relative_to("runs")))
    return keep, excluded


def scan(keep):
    bad = []
    for s, _ in keep:
        try:
            t = s.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if SECRET.search(t):
            bad.append((str(s), "secret-like token"))
        emails = {m.group(0) for m in EMAIL.finditer(t)} - EMAIL_OK
        emails = {e for e in emails if not e.endswith((".png", ".jpg", ".py"))}
        if emails:
            bad.append((str(s), f"email-like strings: {sorted(emails)[:3]}"))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    a = ap.parse_args()
    src = Path(a.src)
    keep, excluded = plan(src)
    bad = scan(keep)
    if bad:
        for b in bad:
            print("SCAN:", b)
        sys.exit("secret/personal-data scan failed; nothing written")
    readme = DST / "README.md"
    saved = readme.read_text(encoding="utf-8") if readme.exists() else None
    for sub in ("docs", "code", "results", "bench"):
        shutil.rmtree(DST / sub, ignore_errors=True)
    for s, rel in keep:
        d = DST / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    manifest = [{"path": str(p.relative_to(src)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p),
                 "reason": why} for p, why in excluded]
    (DST / "EXCLUDED_LARGE_FILES.json").write_text(json.dumps(manifest, indent=1))
    if saved is not None:
        readme.write_text(saved, encoding="utf-8")
    total = sum((DST / r).stat().st_size for _, r in keep)
    print(f"exported {len(keep)} files ({total / 2 ** 20:.1f} MB); excluded {len(excluded)} (manifest written)")


if __name__ == "__main__":
    main()
