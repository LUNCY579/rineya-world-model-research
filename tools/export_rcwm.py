"""Mirror the RCWM bundle into rcwm/ of this repository: code, docs, preregistrations, freeze records, decisions,
results, generations, logs and the small registered LoRA adapters. Large binaries are excluded and listed with
size and sha256 in rcwm/EXCLUDED_LARGE_FILES.json, so a local copy can be validated against the record.

  python tools/export_rcwm.py            # sync (copies new/changed files, removes files gone or now excluded)
  python tools/export_rcwm.py --dry-run  # report only

One process, no subprocesses. Never writes into the source bundle.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05")
DST = REPO / "rcwm"
MANIFEST = DST / "EXCLUDED_LARGE_FILES.json"
HASH_CACHE = REPO / "tools" / ".hash_cache.json"          # gitignored

MAX_BYTES = 25 * 1024 * 1024
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
SKIP_EXT = {".npy", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}
SKIP_NAMES = {"model.safetensors"}
SKIP_GLOBS = [".thermal_*", "*.lock", "*.lock.*", "*.pyc",
              "experiments/RCWM-T1/runs/l1_screen/arm-*/tokenizer.json",
              "experiments/RCWM-T1/runs/l1_screen/arm-*/vocab.json",
              "experiments/RCWM-T1/runs/l1_screen/arm-*/merges.txt",
              "experiments/RCWM-T1/runs/t2/smoke_cpu/adapter/*"]
# Registered / replication LoRA adapters (PREREG-2 artifact is t2_s0, sha 92bc7c74...): allowed up to 50 MB.
ALLOW_LARGE = ["experiments/RCWM-T1/runs/t2/t2_s[0-9]/adapter/adapter_model.safetensors"]
ALLOW_MAX = 50 * 1024 * 1024


def rel(p: Path) -> str:
    return p.relative_to(SRC).as_posix()


def classify(p: Path, size: int) -> str:
    r = rel(p)
    if any(fnmatch.fnmatch(r, g) or fnmatch.fnmatch(p.name, g) for g in SKIP_GLOBS):
        return "skip"
    if any(fnmatch.fnmatch(r, g) for g in ALLOW_LARGE):
        return "include" if size <= ALLOW_MAX else "large"
    if p.suffix.lower() in SKIP_EXT or p.name in SKIP_NAMES or size > MAX_BYTES:
        return "large"
    return "include"


def sha256(p: Path, cache: dict) -> str:
    st = p.stat()
    key = f"{rel(p)}|{st.st_size}|{int(st.st_mtime)}"
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    cache[key] = h.hexdigest()
    return cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cache = json.loads(HASH_CACHE.read_text(encoding="utf-8")) if HASH_CACHE.exists() else {}
    want, large = {}, []
    for dp, dn, fn in os.walk(SRC):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            p = Path(dp) / f
            try:
                size = p.stat().st_size
            except OSError:
                continue
            c = classify(p, size)
            if c == "include":
                want[rel(p)] = p
            elif c == "large":
                large.append(p)
    copied = removed = 0
    if not a.dry_run:
        DST.mkdir(parents=True, exist_ok=True)
        for r, p in want.items():
            d = DST / r
            if d.exists() and d.stat().st_size == p.stat().st_size and int(d.stat().st_mtime) >= int(p.stat().st_mtime):
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
            copied += 1
        keep = set(want) | {MANIFEST.relative_to(DST).as_posix()}
        for dp, dn, fn in os.walk(DST):
            for f in fn:
                d = Path(dp) / f
                r = d.relative_to(DST).as_posix()
                if r not in keep:
                    d.unlink()
                    removed += 1
        man = {"record": "RCWM_EXCLUDED_LARGE_FILES", "source_root": str(SRC), "generated_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "rule": f"excluded: extensions {sorted(SKIP_EXT)}, names {sorted(SKIP_NAMES)}, files > {MAX_BYTES // 1048576} MB "
                       f"(allow-listed adapters up to {ALLOW_MAX // 1048576} MB)",
               "files": [{"path": rel(p), "bytes": p.stat().st_size, "sha256": sha256(p, cache)} for p in sorted(large)]}
        MANIFEST.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
        HASH_CACHE.write_text(json.dumps(cache) + "\n", encoding="utf-8")
    tot = sum(p.stat().st_size for p in want.values())
    print(f"[export] include {len(want)} files ({tot / 1048576:.1f} MB); large excluded {len(large)} "
          f"({sum(p.stat().st_size for p in large) / 1048576:.0f} MB); copied {copied}; removed {removed}"
          f"{' (dry run)' if a.dry_run else ''}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
