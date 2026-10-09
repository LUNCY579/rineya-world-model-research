"""PREREG-3 wide freshness scan (descriptive pre-freeze record; no model, no tokenizer, no torch).

  python -B scripts/p3_wide_scan.py     # writes docs/prereg3/WIDE_SCAN.json (refused once PREREG-3 is frozen)

The builder (scripts/p3_make_bank.py) enforces freshness against 16 pinned sources and every multi-event prompt in the
records tree. This scan widens the net to every *.json / *.jsonl file under C:/Users/clclc/BusinessProject that contains
a ledger event (both the strings "amount and "account), as the PREREG-3 review's independent scan did for the superseded
revision-1 bank. From each file it harvests single-step {before, event} objects (also JSON-in-string prompts),
multi-event {before|s0, events} prompts and bank parents {s0, candidates: [{events}], twin}, rolls the multi-event ones
out with rcwmt.env.step, and counts the (state, event) payloads and FDS operative contents equal to a transition of
data/bank_p3_oor.jsonl.

Never opened: files named HSKILL_CONFIRM_V2* (by name; DECISIONS I-28) and runs/t2/*/gen_* (pilot model outputs; by
path): ripgrep (2 threads), which does the content prefilter, excludes them by glob before reading, and Python
re-checks every listed path before opening it. Also excluded: the PREREG-3 bank, docs/prereg3 (incl. the superseded v1
bank), .git, .claude, node_modules, __pycache__, venv/.venv; files over 60 MB (ripgrep --max-filesize); .json files
over 30 MB are listed, not parsed (RAM budget); .jsonl files are streamed line by line. A malformed object ends only
its own document's harvest and is counted (malformed_objects_skipped).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import l1_make_ood as L1  # noqa: E402
import p3_common as C  # noqa: E402
import p3_make_bank as MB  # noqa: E402

BUSINESS = MB.BUSINESS
OUT = C.DOCS / "WIDE_SCAN.json"
JSON_MAX_BYTES = 30_000_000
FREEZE = ROOT / "PREREG_3_FREEZE.json"
RG_GLOBS = ("*.json", "*.jsonl", "!**/HSKILL_CONFIRM_V2*", "!**/runs/t2/**/gen_*", "!**/.git/**", "!**/.claude/**",
            "!**/node_modules/**", "!**/__pycache__/**", "!**/venv/**", "!**/.venv/**", "!**/docs/prereg3/**",
            "!**/data/bank_p3_oor*")


def _peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)


def _rel(p: Path) -> str:
    return str(p.relative_to(BUSINESS)).replace("\\", "/")


def never_open(p: Path) -> str | None:
    """The reason a path must never be opened, decided from the path alone (re-check of the ripgrep globs)."""
    if p.name.startswith(L1.RESERVED_PREFIX):
        return "reserved HSKILL_CONFIRM_V2* (by name)"
    low = [x.lower() for x in p.parts]
    if p.name.startswith("gen_") and any(low[k] == "runs" and low[k + 1] == "t2" for k in range(len(low) - 1)):
        return "runs/t2/*/gen_* model outputs (by path)"
    rp = p.resolve()
    if rp in (C.BANK.resolve(), C.SOURCE.resolve()) or C.DOCS.resolve() in rp.parents:
        return "the PREREG-3 bank / docs/prereg3 itself"
    return None


def rg_list(pattern: str) -> tuple[set, list]:
    rg = shutil.which("rg")
    if not rg:
        raise SystemExit("ripgrep (rg) is required for the content prefilter")
    cmd = [rg, "-l", "-j", "2", "--no-messages", "--no-ignore", "--hidden", "--max-filesize", "60M", "-F", pattern]
    for g in RG_GLOBS:
        cmd += ["-g", g]
    cmd.append(str(BUSINESS))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode not in (0, 1):
        raise SystemExit(f"rg failed ({res.returncode}): {res.stderr[:500]}")
    return {Path(x) for x in res.stdout.splitlines() if x.strip()}, cmd[1:-1]


def reserved_names_seen() -> dict:
    """Names only (no file is opened): how many reserved / gen files the globs keep out, for the record."""
    cnt = Counter()
    for d, dirs, names in os.walk(BUSINESS):
        dirs[:] = [x for x in dirs if x not in {".git", ".claude", "node_modules", "__pycache__", "venv", ".venv"}]
        for n in names:
            if n.endswith(".json") or n.endswith(".jsonl"):
                why = never_open(Path(d) / n)
                if why:
                    cnt[why] += 1
    return dict(cnt)


def items_of(doc):
    """(state, event) transitions of every ledger payload in a parsed document."""
    G = L1._fds()
    stack = [doc]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            st = cur.get("before", cur.get("s0"))
            if isinstance(cur.get("before"), dict) and isinstance(cur.get("event"), dict):
                yield cur["before"], cur["event"]
            if isinstance(st, dict) and isinstance(cur.get("events"), list):
                yield from MB._rollout_items(st, cur["events"])
            if isinstance(cur.get("s0"), dict) and isinstance(cur.get("candidates"), list):
                for c in cur["candidates"] + ([cur["twin"]] if isinstance(cur.get("twin"), dict) else []):
                    if isinstance(c, dict) and isinstance(c.get("events"), list):
                        yield from MB._rollout_items(cur["s0"], c["events"])
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, str) and ('"before"' in cur or '"s0"' in cur):
            for o in G._objects_in(cur):
                yield o["before"], o["event"]
            if '"events"' in cur:
                i, j = cur.find("{"), cur.rfind("}")
                if 0 <= i < j:
                    try:
                        stack.append(json.loads(cur[i:j + 1]))
                    except (json.JSONDecodeError, ValueError):
                        pass


def safe_items(doc, errors: Counter):
    """items_of, but a malformed object ends only the current document's harvest (counted), never the scan."""
    it = items_of(doc)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:  # noqa: BLE001  (descriptive scan: count and continue with the next document)
            errors[type(e).__name__] += 1
            return


def docs_of(p: Path):
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"amount' in line or '\\"amount' in line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        try:
            yield json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return


def main() -> int:
    if FREEZE.exists():
        raise SystemExit("PREREG-3 is frozen: WIDE_SCAN.json is a frozen input and is not rewritten")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    t0 = time.time()
    rows, chains = C.load_bank()
    pay = MB.bank_payloads(rows, chains)
    ops = MB.bank_opkeys(rows, chains)
    with_amount, rg_args = rg_list('"amount')
    with_account, _ = rg_list('"account')
    files = sorted(with_amount & with_account)
    t_rg = round(time.time() - t0, 1)
    print(f"ripgrep prefilter: {len(files)} files contain a ledger event ({t_rg}s)", flush=True)
    tot, hits, refused, big, errors = Counter(), {}, [], [], Counter()
    for fi, p in enumerate(files):
        if fi % 100 == 0:
            print(f"  file {fi}/{len(files)} ({round(time.time() - t0)}s)", flush=True)
        if never_open(p):
            refused.append(_rel(p))                 # must stay empty: the globs already exclude these
            continue
        if p.suffix == ".json" and p.stat().st_size > JSON_MAX_BYTES:
            big.append(_rel(p))
            continue
        n = po = oo = 0
        for doc in docs_of(p):
            for b, e in safe_items(doc, errors):
                try:
                    if not isinstance(b, dict) or not isinstance(e, dict) or "kind" not in e:
                        continue
                    n += 1
                    po += L1.payload(b, e) in pay
                    if e.get("kind") in ("add", "scale", "transfer") and e.get("account") in b:
                        oo += L1.opkey(C.LEDGER, b, e) in ops
                except Exception as ex:  # noqa: BLE001  (a malformed item is counted, not fatal)
                    errors["item:" + type(ex).__name__] += 1
                    continue
        if n:
            tot["files_with_ledger_transitions"] += 1
            tot["transitions"] += n
            tot["payload_overlap"] += po
            tot["operative_overlap"] += oo
            if po or oo:
                hits[_rel(p)] = {"transitions": n, "payload_overlap": po, "operative_overlap": oo}
    rec = {"record": "P3_WIDE_SCAN", "what": " ".join(__doc__.split("\n\n")[1].split()),
           "bank_sha256": L1.sha256_file(C.BANK), "root": str(BUSINESS), "ripgrep_args": rg_args,
           "files_with_a_ledger_event": len(files), "files_with_ledger_transitions": tot["files_with_ledger_transitions"],
           "transitions_checked": tot["transitions"], "bank_transitions": len(pay),
           "payload_overlap": tot["payload_overlap"], "operative_overlap": tot["operative_overlap"],
           "overlapping_files": hits, "json_over_30MB_not_parsed": big, "refused_paths": refused,
           "malformed_objects_skipped": dict(errors),
           "never_opened_by_name_or_path": reserved_names_seen(),
           "ok": tot["payload_overlap"] == 0 and tot["operative_overlap"] == 0 and tot["transitions"] > 0 and not refused,
           "code_sha256": {r: L1.sha256_file(ROOT / r) for r in ("scripts/p3_wide_scan.py", "scripts/p3_make_bank.py",
                                                                 "scripts/p3_common.py")},
           "torch_imported": "torch" in sys.modules, "seconds": round(time.time() - t0, 1), "peak_rss_mb": _peak_mb()}
    C.DOCS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("what", "ripgrep_args")}, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")
    if "torch" in sys.modules:
        raise SystemExit("torch was imported")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
