"""Pinned, read-only, bytecode-free import of BWM's Blocksworld code from the live BWM bundle.

Every BWM file is read once as bytes, its sha256 is checked against PINS (fail closed: PinError) and exactly those
bytes are compiled and executed into a fresh module object. The import system is never asked to load BWM code, so no
`.pyc` can be written into (or read from) BWM's `__pycache__` directories; `sys.dont_write_bytecode` is set as well.

  B   = bwmn/blocks.py                 (registered as sys.modules['bwmn.blocks'], package 'bwmn' from its empty __init__)
  VL  = scripts/blocks_val_large.py    (its module-level sys.path.insert is undone after loading)
  bw_distill.problems(n): NOT imported. bw_distill.py imports bwmn.blocks_planner, which imports bwmn.lm, which imports
      torch and BWM's GPU/health machinery at module level. Instead the `SEED` constant and the `problems` function are
      extracted from the pinned source with `ast` and executed verbatim in a namespace holding only
      (B, VL, json, random, ROOT, Path). This runs BWM's exact code for problems(), with no side effects (it only reads
      the hard PDDL files and runs/BW_val_large/instances_{val_large,mirror_hard}.json).

check_no_bytecode() compares a snapshot of every __pycache__ directory under the BWM bundle with the snapshot taken
before the first load. RCX never deletes or writes anything in the BWM tree.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True

BUNDLE_ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle")
BWM_ROOT = BUNDLE_ROOT / "native_v1"
PINS = {
    "bwmn/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # empty file
    "bwmn/blocks.py": "b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83",
    "scripts/blocks_val_large.py": "6a66d489944150b50294a75f8c07671ad722b51ede09409ddee1d2cfc91f2356",
    "scripts/bw_distill.py": "2b2d518206b2e36c252ac9b87f991a73ff701b54404758e0363670d573d49fa1",
}
LOADED_MODULE_STEMS = ("__init__", "blocks", "blocks_val_large", "bw_distill")

_STATE: dict = {}


class PinError(RuntimeError):
    """A BWM file does not match its pinned sha256 (or a foreign `bwmn` module is already loaded)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_pinned(rel: str) -> bytes:
    data = (BWM_ROOT / rel).read_bytes()
    got = sha256_bytes(data)
    if got != PINS[rel]:
        raise PinError(f"BWM pin mismatch for {rel}: got {got}, pinned {PINS[rel]}")
    return data


def pycache_snapshot(root: Path = BUNDLE_ROOT) -> dict:
    """{__pycache__ dir: sorted [(file, size, mtime_ns)]} for every __pycache__ directory under `root`."""
    snap = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "__pycache__":
            entries = []
            for f in filenames:
                st = os.stat(os.path.join(dirpath, f))
                entries.append((f, st.st_size, st.st_mtime_ns))
            snap[os.path.normpath(dirpath)] = sorted(entries)
    return snap


def diff_snapshots(before: dict, after: dict) -> dict:
    """New __pycache__ dirs, and new/changed .pyc files (all, and those named after modules RCX loads)."""
    new_dirs = sorted(set(after) - set(before))
    changed = []
    for d, entries in after.items():
        old = {e[0]: e for e in before.get(d, [])}
        changed += [os.path.join(d, e[0]) for e in entries if old.get(e[0]) != e]
    ours = [p for p in changed if os.path.basename(p).split(".")[0] in LOADED_MODULE_STEMS]
    return {"new_pycache_dirs": new_dirs, "new_or_changed_pyc": sorted(changed), "ours": sorted(ours)}


def _exec_module(name: str, rel: str, data: bytes, package: str | None = None, is_pkg: bool = False):
    path = BWM_ROOT / rel
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    mod.__package__ = package if package is not None else (name if is_pkg else "")
    if is_pkg:
        mod.__path__ = [str(path.parent)]
    sys.modules[name] = mod
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _extract_problems(B, VL):
    data = read_pinned("scripts/bw_distill.py")
    tree = ast.parse(data, filename=str(BWM_ROOT / "scripts/bw_distill.py"))
    seed_node = fn_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["SEED"]:
            seed_node = node
        if isinstance(node, ast.FunctionDef) and node.name == "problems":
            fn_node = node
    if seed_node is None or fn_node is None:
        raise PinError("bw_distill.py: SEED or problems() not found")
    ns = {"B": B, "VL": VL, "json": json, "random": random, "ROOT": BWM_ROOT, "Path": Path,
          "__name__": "rcx_bw_distill_problems", "__builtins__": builtins}
    bound = {a.arg for a in ast.walk(fn_node) if isinstance(a, ast.arg)}
    stored = {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free = loads - bound - stored - set(ns) - set(dir(builtins)) - {"SEED"}
    if free:
        raise PinError(f"bw_distill.problems uses names not provided to the extraction namespace: {sorted(free)}")
    exec(compile(ast.Module(body=[seed_node, fn_node], type_ignores=[]), str(BWM_ROOT / "scripts/bw_distill.py"),
                 "exec", dont_inherit=True), ns)
    return ns["problems"], ns["SEED"]


def load():
    """Return (B, VL) — BWM's blocks module and blocks_val_large module — loaded once, pinned, bytecode-free."""
    if "B" in _STATE:
        return _STATE["B"], _STATE["VL"]
    existing = sys.modules.get("bwmn")
    if existing is not None and not getattr(existing, "_rcx_pinned", False):
        raise PinError("a foreign 'bwmn' package is already imported in this process; refusing to mix")
    if "pycache_before" not in _STATE:
        _STATE["pycache_before"] = pycache_snapshot()
    datas = {rel: read_pinned(rel) for rel in ("bwmn/__init__.py", "bwmn/blocks.py", "scripts/blocks_val_large.py")}
    pkg = _exec_module("bwmn", "bwmn/__init__.py", datas["bwmn/__init__.py"], is_pkg=True)
    pkg._rcx_pinned = True
    B = _exec_module("bwmn.blocks", "bwmn/blocks.py", datas["bwmn/blocks.py"], package="bwmn")
    pkg.blocks = B
    saved_path = list(sys.path)
    try:
        VL = _exec_module("rcx_bwm_blocks_val_large", "scripts/blocks_val_large.py",
                          datas["scripts/blocks_val_large.py"])
    finally:
        sys.path[:] = saved_path
    if VL.B is not B:
        raise PinError("blocks_val_large did not bind the pinned bwmn.blocks")
    _STATE.update(B=B, VL=VL)
    return B, VL


def distill_problems():
    """Return (problems, SEED): BWM's bw_distill.problems function executed from the pinned source (see module doc)."""
    if "problems" not in _STATE:
        B, VL = load()
        _STATE["problems"], _STATE["distill_seed"] = _extract_problems(B, VL)
    return _STATE["problems"], _STATE["distill_seed"]


def check_no_bytecode() -> dict:
    """Diff of BWM __pycache__ state against the snapshot taken before the first load (empty lists = clean)."""
    before = _STATE.get("pycache_before")
    if before is None:
        before = _STATE["pycache_before"] = pycache_snapshot()
    return diff_snapshots(before, pycache_snapshot())


def provenance() -> dict:
    return {"bwm_root": str(BWM_ROOT), "pins": dict(PINS), "import_mode": "exec of sha-verified bytes (no import "
            "system, no bytecode)", "bw_distill": "SEED + problems() extracted with ast from pinned bytes"}
