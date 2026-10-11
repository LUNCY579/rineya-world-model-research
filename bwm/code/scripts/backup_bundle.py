"""Full local backup of native_v1 (code, records, runs incl. adapter weights and training data, release, bench).

GitHub holds records and code only (D-51: no weights, training rows or caches), so this archive is the complete copy.
Writes <dest>/bwm_native_v1_<stamp>.zip plus <dest>/bwm_native_v1_<stamp>.MANIFEST.json (sha256 + size of every file),
then re-opens the zip and verifies every member against the manifest. Skips __pycache__/.pytest_cache and training
checkpoint dirs (`ckpt/`, rewritten while a job runs; the finished adapter is what matters). Optional --extra paths
(e.g. the session memory directory) are added under extra/<name>/. Large binaries are stored, text is deflated.
Usage: python -B scripts/backup_bundle.py --dest C:/Users/clclc/BusinessProject/WorldModel/_backups [--extra DIR ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "ckpt"}
STORE_SUFFIX = {".safetensors", ".npy", ".npz", ".pt", ".bin", ".zip", ".gz", ".exe", ".dll"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files_under(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield Path(dirpath) / fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    zpath = dest / f"bwm_native_v1_{stamp}.zip"
    entries = [(p, "native_v1/" + p.relative_to(ROOT).as_posix()) for p in files_under(ROOT)]
    for ex in a.extra:
        exp = Path(ex)
        entries += [(p, f"extra/{exp.name}/" + p.relative_to(exp).as_posix()) for p in files_under(exp)]
    manifest, t0 = {}, time.time()
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
        for p, arc in entries:
            try:
                digest, size = sha256(p), p.stat().st_size
                comp = zipfile.ZIP_STORED if p.suffix.lower() in STORE_SUFFIX else zipfile.ZIP_DEFLATED
                z.write(p, arc, compress_type=comp, compresslevel=1 if comp == zipfile.ZIP_DEFLATED else None)
                if sha256(p) != digest:  # file changed while archiving (live log): record the archived bytes instead
                    digest = hashlib.sha256(z.read(arc)).hexdigest()
                manifest[arc] = {"sha256": digest, "bytes": size}
            except (PermissionError, OSError) as exc:
                manifest[arc] = {"error": str(exc)}
    bad = []
    with zipfile.ZipFile(zpath) as z:
        for arc, rec in manifest.items():
            if "sha256" in rec and hashlib.sha256(z.read(arc)).hexdigest() != rec["sha256"]:
                bad.append(arc)
    meta = {"created": stamp, "root": str(ROOT), "zip": zpath.name, "zip_sha256": sha256(zpath),
            "zip_bytes": zpath.stat().st_size, "files": len(manifest),
            "errors": sorted(k for k, v in manifest.items() if "error" in v), "verify_failures": bad,
            "skipped_dirs": sorted(SKIP_DIRS), "seconds": round(time.time() - t0, 1), "manifest": manifest}
    (dest / f"bwm_native_v1_{stamp}.MANIFEST.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "manifest"}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
