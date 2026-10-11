"""Second audit pass on the split pools (CX D-34): drop wordings containing an inflection of a real status-like word.

The first audit (split_pool.py) caught status words only as whole words. Reading P2 showed inflections such as
'posting'/'post' (anchor 'posted') and 'rejection' ('rejected'). Re-splitting would move already-read P2 wordings into
P3, so this pass only *removes* items, from P2, P3 and SPARE alike, by code; it prints counts, never a P3 wording.

    python -B experiments/RCX-E01R/refine_pool.py
Outputs (create-only): pools/P2_DEV_v2.json, pools/P3_SEALED_v2.json.secret, pools/SPARE_v2.json.secret,
pools/POOL_REFINE.json (counts by cell, sha256 of every input and output).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcx.rcl import wording_audit as WA  # noqa: E402

POOLS = Path(__file__).resolve().parent / "pools"
INPUTS = {"P2": "P2_DEV.json", "P3": "P3_SEALED.json.secret", "SPARE": "SPARE.json.secret"}
OUTPUTS = {"P2": "P2_DEV_v2.json", "P3": "P3_SEALED_v2.json.secret", "SPARE": "SPARE_v2.json.secret"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    outs = [POOLS / n for n in list(OUTPUTS.values()) + ["POOL_REFINE.json"]]
    if any(p.exists() for p in outs):
        raise SystemExit("refine outputs exist (create-only)")
    split_meta = json.loads((POOLS / "POOL_SPLIT.json").read_text(encoding="utf-8"))
    want = {"P2": split_meta["p2_sha256"], "P3": split_meta["p3_sha256"], "SPARE": split_meta["spare_sha256"]}
    vocab = WA.real_status_words()
    meta = {"decision": "CX D-34", "rule": "drop a wording with any word sharing >= 4 leading letters with a real "
                                          "status-like word (wording_audit.stem_hits)",
            "inputs": {}, "outputs": {}, "dropped_by_cell": {}, "kept_by_cell": {}}
    texts = {}
    for k, name in INPUTS.items():
        raw = (POOLS / name).read_bytes()
        if sha(raw) != want[k]:
            raise SystemExit(f"{name}: sha differs from POOL_SPLIT.json")
        doc = json.loads(raw)
        kept = [t for t in doc["templates"] if not WA.stem_hits(t["text"], vocab=vocab)]
        dropped = [t for t in doc["templates"] if WA.stem_hits(t["text"], vocab=vocab)]
        meta["inputs"][k] = {"file": name, "sha256": want[k], "n": len(doc["templates"])}
        meta["dropped_by_cell"][k] = dict(Counter(f"{t['pol']}/{t['form']}" for t in dropped))
        meta["kept_by_cell"][k] = dict(Counter(f"{t['pol']}/{t['form']}" for t in kept))
        doc = dict(doc, templates=kept, refined_from=want[k])
        texts[k] = (json.dumps(doc, indent=1) if k == "P2" else json.dumps(doc)) + "\n"
        meta["outputs"][k] = {"file": OUTPUTS[k], "sha256": sha(texts[k].encode()), "n": len(kept)}
    meta["code_sha256"] = {"refine_pool.py": sha(Path(__file__).read_bytes()),
                           "wording_audit.py": sha((ROOT / "rcx/rcl/wording_audit.py").read_bytes())}
    for k, text in texts.items():
        with open(POOLS / OUTPUTS[k], "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    with open(POOLS / "POOL_REFINE.json", "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: {"n_in": meta["inputs"][k]["n"], "n_out": meta["outputs"][k]["n"],
                          "dropped": meta["dropped_by_cell"][k]} for k in INPUTS}, indent=1))
    print("P3_v2 sha256", meta["outputs"]["P3"]["sha256"])


if __name__ == "__main__":
    main()
