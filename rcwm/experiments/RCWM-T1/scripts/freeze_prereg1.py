"""Freeze PREREG-1: fill the pilot table from the dry run, hash every frozen input, write §9 and
PREREG_1_FREEZE.json. Refuses to run twice."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PR = ROOT / "PREREG_1.md"
FZ = ROOT / "PREREG_1_FREEZE.json"
if FZ.exists():
    raise SystemExit("already frozen")
dry = json.loads((ROOT / "runs" / "bank_pilot" / "analysis" / "PREREG_1_DRYRUN.json").read_text(encoding="utf-8"))
m = {f: dry["per_family"][f"{f}|k4"]["means"] for f in ("ledger_replay", "register_machine")}
L, R = m["ledger_replay"], m["register_machine"]
f3 = lambda x: f"{x:.3f}"
s = PR.read_text(encoding="utf-8")
s = s.replace("| G_chain(Titan) | see §9 | see §9 |", f"| G_chain(Titan) | {f3(L['Gtitan_chain_one'])} | {f3(R['Gtitan_chain_one'])} |")
s = s.replace("| B3 (consequence head, frozen Titan states, ~1.1k labels) | see §9 | see §9 |",
              f"| B3 (consequence head, frozen Titan states, ~1.1k labels; DEV-selected) | {f3(L['B3'])} | {f3(R['B3'])} |")
s = s.replace("| S (symbolic learner, same labels) | see §9 | see §9 |", f"| S (symbolic learner, same labels; DEV-selected) | {f3(L['S'])} | {f3(R['S'])} |")
s = s.replace("| G(Supa) best mode | see §9 | see §9 |",
              "| G(Supa) best mode | no valid pilot estimate | no valid pilot estimate |")
s = s.replace("The pilot table below is filled in from the regenerated pilot dry run, done under this instrument before freeze.",
              "Values come from the code-validation dry run on pilot TEST (`PREREG_1_DRYRUN.json`), which used the pilot generations. Titan's outputs are unaffected by the instrument changes; Supa's are not. The regenerated pilot run under the frozen instrument is non-binding and runs after the confirmation jobs, so C3 has no valid pilot estimate.")
s = s.replace("**Status:** DRAFT v2.", "**Status:** FROZEN (see §9).")
files = sorted(list((ROOT / "rcwmt").glob("*.py")) + [ROOT / "scripts" / x for x in ("run_arm.py", "confirm_analysis.py", "analyze.py", "freeze_prereg1.py")]
               + [ROOT / "data" / x for x in ("bank_pilot.json", "bank_confirm.json", "bank_s4000.json")]
               + [ROOT / "runs" / "bank_pilot" / "sparse" / x for x in ("index.json", "last.npy")])
hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
sec9 = "## 9. Freeze record\n\nFrozen at " + stamp + ". The SHA-256 of every frozen input:\n\n" + \
       "\n".join(f"- `{k}` `{v}`" for k, v in hashes.items()) + \
       "\n\nThe SHA-256 of this file as frozen is recorded in `PREREG_1_FREEZE.json`.\n"
s = s[: s.index("## 9. Freeze record")] + sec9
PR.write_text(s, encoding="utf-8")
FZ.write_text(json.dumps({"frozen_at": stamp, "prereg_sha256": hashlib.sha256(PR.read_bytes()).hexdigest(), "inputs": hashes}, indent=1) + "\n", encoding="utf-8")
print("frozen", stamp, "prereg sha", json.loads(FZ.read_text())["prereg_sha256"])
