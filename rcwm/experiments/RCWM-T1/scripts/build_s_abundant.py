"""Simulator-labelled training bank for the S@abundant secondary of PREREG-1 (no model involved).
4,000 parents per family for ledger and register, seed 777001, deduplicated against the
confirmation bank's parents."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcwmt import bank, env  # noqa: E402

out = ROOT / "data" / "bank_s4000.json"
if out.exists():
    raise SystemExit(f"{out} exists")
conf = json.loads((ROOT / "data" / "bank_confirm.json").read_text(encoding="utf-8"))
keys = {env.dumps({"f": p["family"], "s0": p["s0"], "c": [c["events"] for c in p["candidates"]]}) for p in conf["parents"]}
b = bank.build(4000, 8, 4, 777001, None, exclude_parent_keys=keys)
b["parents"] = [p for p in b["parents"] if p["family"] in ("ledger_replay", "register_machine")]
for p in b["parents"]:
    p["split"] = "train"
b["purpose"] = "PREREG-1 S@abundant secondary only (simulator labels; never used for B3/G)"
out.write_text(json.dumps(b) + "\n", encoding="utf-8")
print(out, len(b["parents"]), hashlib.sha256(out.read_bytes()).hexdigest())
