"""Retention suite (RCWM-T1 roadmap item 4 / joint item 5c): shared constants, paths and file helpers. Torch-free.

Kit files (docs/retention/DESIGN.md is the design, the normalisation spec and the pre-stated decision rule):
  scripts/ret_common.py     this file
  scripts/ret_grade.py      typed whole-answer graders (primary) + lenient report-only graders
  scripts/ret_gen.py        frozen item generator, sealed draws (mint / verify / stats / audit), freshness harvest
  scripts/ret_run.py        GPU runner (dev: Supa only; confirm: Titan-Sparse + T2 s0 OFF/ON in one process, Supa)
  scripts/ret_analysis.py   dev selection, the registered confirmation analysis, synthetic dry run
  scripts/ret_test_graders.py, scripts/ret_test_tiny.py   CPU tests (no GPU, tiny random Qwen3 only)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "retention"
DRAWS = DOCS / "draws"
RUNS = ROOT / "runs" / "retention"
FREEZE = DOCS / "RET_FREEZE.json"
SELECTION = DOCS / "DEV_SELECTION.json"              # surface diversity OFF; see selection_rel()
DESIGN = DOCS / "DESIGN.md"

# ------------------------------------------------------------------------------------------------ fixed design
CATEGORIES = ("arith_word", "unit_conv", "date_time", "count_sort", "pct_ratio", "kv_recall", "unknown_twin",
              "format_follow")
LEVELS = (1, 2, 3)
N_PER_CELL = 75                     # items per category (x level in the dev draw)
DEV_SEED = 20261121                 # dev draw, surface diversity OFF (not used by any earlier RCWM-T1 builder)
DEV_SEED_SD = 20261125              # dev draw of the surface-diversity variant (pre-registered; DECISION POINT 1)
DEV_SEEDS = {False: DEV_SEED, True: DEV_SEED_SD}
MAX_NEW_TOKENS = 48
DEV_TARGET_ACC = 0.70               # dev rule: the highest level with Supa >= 0.70 ...
DEV_MIN_ACC = 0.40                  # ... else the highest with >= 0.40; no level >= 0.40 -> category excluded
MARGIN = 5                          # per-category net >= -5 of 75 (roadmap): a REPORTED point-estimate flag only
CATEGORY_MARGIN = 11                # gate: a category fails if net < -11 (more than 11 of 75 lost net), per grader
POOLED_MARGIN_RATE = 0.04           # gate: the pooled net fails if < -0.04 x selected items (-24 of 600), per grader
ALPHA = 0.05                        # gate: Holm-adjusted exact McNemar (two-sided) with lost > gained, per grader
# CATEGORY_MARGIN and POOLED_MARGIN_RATE were set by simulation before any model output (DESIGN section 7, DRYRUN.json):
# single-grader false NO-GO <= ~0.03 at 5-30% churn and reference accuracy 0.30-0.80 (<= ~0.01 at <= 20% churn).
RESERVED_PREFIX = "HSKILL_CONFIRM_V2"   # never opened or hashed (DECISIONS I-28); == l1_make_ood.RESERVED_PREFIX

# Tokenizer identity (DESIGN section 6): byte-identical across Titan-Sparse and Supa 1.7 (checked 2026-10-09);
# rcwmt.runtime.TOKENIZER_DIR is the Titan-Sparse directory. The chat template is the rendered tok.chat_template.
TOKENIZER_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                     r"Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
TOKENIZER_PINS = {"tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                  "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                  "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                  "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"}
CHAT_TEMPLATE_SHA = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"

# every kit file the freeze binds (code, design text, pre-freeze test records)
FROZEN_CODE = ("scripts/ret_common.py", "scripts/ret_grade.py", "scripts/ret_gen.py", "scripts/ret_run.py",
               "scripts/ret_analysis.py", "scripts/ret_test_graders.py", "scripts/ret_test_tiny.py")
FROZEN_DOCS = ("docs/retention/DESIGN.md",)
TEST_RECORDS = ("docs/retention/GRADER_TEST.json", "docs/retention/TINY_TEST.json", "docs/retention/DRYRUN.json",
                "docs/retention/RET_PLAN.json", "docs/retention/DEV_DRAW_PREVIEW.json",
                "docs/retention/DEV_DRAW_PREVIEW_SD.json")


def selection_rel(surface_diversity: bool) -> str:
    """The dev selection that governs draws of one surface-diversity setting (one file per setting)."""
    return "docs/retention/DEV_SELECTION_SD.json" if surface_diversity else "docs/retention/DEV_SELECTION.json"


def tokenizer_hashes(d: Path | None = None) -> dict:
    d = Path(d or TOKENIZER_DIR)
    return {f: (sha256_file(d / f) if (d / f).is_file() else None) for f in sorted(TOKENIZER_PINS)}


def tokenizer_problem(d: Path | None = None) -> str | None:
    h = tokenizer_hashes(d)
    bad = sorted(f for f, v in h.items() if v != TOKENIZER_PINS[f])
    return f"tokenizer files differ from the pins: {bad}" if bad else None


class RetStop(Exception):
    """A deterministic refusal (exit 3). persist=True: the runner writes it to STOPPED.json (it concerns the run)."""

    def __init__(self, msg: str, persist: bool = False):
        super().__init__(msg)
        self.persist = persist


def need(cond: bool, msg: str, persist: bool = False) -> None:
    if not cond:
        raise RetStop(f"[retention] {msg}", persist)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_create(path: Path, text: str) -> None:
    """Create-only text write (x mode, fsync) via a .partial file renamed into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RetStop(f"[retention] {path} exists (create-only)")
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)                     # refuses an existing target on Windows


def write_json_create(path: Path, obj) -> None:
    write_create(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def code_sha(root: Path | None = None, rels=FROZEN_CODE + FROZEN_DOCS) -> dict:
    root = Path(root or ROOT)
    return {r: (sha256_file(root / r) if (root / r).is_file() else None) for r in rels}


# ------------------------------------------------------------------------------------------------ text normalisation
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "−": "-", "–": "-",
                         "—": "-", " ": " "})


def norm_text(s: str) -> str:
    """Freshness normal form: NFKC, typographic quotes/dashes to ASCII, casefold, whitespace collapsed."""
    s = unicodedata.normalize("NFKC", str(s)).translate(_QUOTES)
    return " ".join(s.casefold().split())


def split_sentences(normed: str) -> list:
    """Sentences of a norm_text() string: split after . ? ! or : followed by whitespace, and at newlines (already
    collapsed by norm_text, so only the punctuation rule applies). Deterministic and used identically on draws and
    on every harvested string."""
    out, start = [], 0
    n = len(normed)
    i = 0
    while i < n:
        c = normed[i]
        if c in ".?!:" and (i + 1 == n or normed[i + 1] == " "):
            seg = normed[start:i + 1].strip()
            if seg:
                out.append(seg)
            start = i + 1
        i += 1
    tail = normed[start:].strip()
    if tail:
        out.append(tail)
    return out


def user_text(messages: list) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") == "user")


# ------------------------------------------------------------------------------------------------ freeze record
def verify_freeze(root: Path | None = None, freeze_path: Path | None = None) -> tuple[bool, list, dict | None]:
    """docs/retention/RET_FREEZE.json against the files: every frozen code / doc / test-record hash must match."""
    root = Path(root or ROOT)
    fz = read_json(freeze_path or (root / "docs" / "retention" / "RET_FREEZE.json"))
    if fz is None:
        return False, ["RET_FREEZE.json does not exist or is unreadable (the kit is a draft)"], None
    errs = []
    for rel, h in sorted((fz.get("files") or {}).items()):
        p = root / rel
        if not p.is_file():
            errs.append(f"{rel} missing")
        elif sha256_file(p) != h:
            errs.append(f"{rel} differs from the freeze")
    missing = sorted(set(FROZEN_CODE + FROZEN_DOCS + TEST_RECORDS) - set(fz.get("files") or {}))
    if missing:
        errs.append(f"not hashed in the freeze record: {missing}")
    return not errs, errs, fz
