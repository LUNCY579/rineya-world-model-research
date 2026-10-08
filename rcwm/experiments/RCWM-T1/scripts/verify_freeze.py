"""Check every frozen input against PREREG_1_FREEZE.json, applying logged amendments
(PREREG_1_AMENDMENT_*.json: a file may differ from the freeze only if an amendment records its
old and new hash). Run before confirm_analysis.py."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
fz = json.loads((ROOT / "PREREG_1_FREEZE.json").read_text(encoding="utf-8"))
expected = dict(fz["inputs"])
for a in sorted(ROOT.glob("PREREG_1_AMENDMENT_*.json")):
    am = json.loads(a.read_text(encoding="utf-8"))
    assert expected.get(am["file"]) == am["old_sha256"], f"{a.name}: old hash does not match the chain"
    expected[am["file"]] = am["new_sha256"]
    print(f"applied {a.name}: {am['file']} ({am['type']})")
bad = [k for k, v in expected.items() if hashlib.sha256((ROOT / k).read_bytes()).hexdigest() != v]
pr = hashlib.sha256((ROOT / "PREREG_1.md").read_bytes()).hexdigest() == fz["prereg_sha256"]
print("PREREG_1.md unchanged:", pr, "| inputs differing from freeze+amendments:", bad or "none")
sys.exit(0 if pr and not bad else 1)
