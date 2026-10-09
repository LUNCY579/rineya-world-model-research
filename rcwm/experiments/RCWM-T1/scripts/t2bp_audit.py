"""RCWM-T2b prerequisite screen: training-data audit. Exits 1 (refuses) unless every gate holds. No model, no torch.

  python -B scripts/t2bp_audit.py               # the canonical files data/t2bp_train_<arm>.jsonl
  python -B scripts/t2bp_audit.py --variant sd  # the surface-diversity files data/t2bp_train_sd_<arm>.jsonl
  python -B scripts/t2bp_audit.py --record      # also writes docs/t2b/prereq/AUDIT[_sd].json

Gates (t2bp_common.audit_rows + t2bp_make_data.audit_files):
  * on the training rows, each audited heuristic - copy-input, apply-regardless (= "(before&~delete)|add ignoring
    preconditions", checked identical on every row), integer before+add, OR before|add - scores 0.50 +- 0.02;
  * the type composition equals the design; >= 250 valid minimal pairs (same before and event except exactly one bit of
    pre_on or pre_off; the applicable member's gold differs from the inapplicable member's; the flip is the
    inapplicable member's only violation); int-add-wrong (strict, A4) >= 150; applicable-delete >= 200; int-add-wrong
    (R11 definition) >= 200; pre_on-missing and pre_off-present inapplicable rows >= 300 each; no row violates both;
  * golds == rcwmt.env.step; payloads and operative contents unique; 1,500 rows, all 8-bit;
  * every rendering holds the same canonical rows in the same order; prompts == the arm's frame (D-int canonical
    prompts == T2's frame); every target parses back to its gold with the arm's strict parser (cmp/calc/flag exact);
    the filler target has exactly the trace target's token count under Titan's tokenizer on every row;
  * file sha256 == the SOURCE record.
Descriptive (recorded, not gated): (before&~delete)+add and before&~delete accuracies; cheap-feature cue ceilings; the
shortcut policies of t2bp_common.SHORTCUT_POLICIES (precondition-aware integer composites - Titan's own reading R11 is
pa_int_add - single-precondition rules, precondition-count / add=0 / add-bit-set cues) and the number of training rows
that separate each precondition-aware composite from the bitwise rule; minimal-pair twin artefacts.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2bp_common as C  # noqa: E402
import t2bp_make_data as MD  # noqa: E402


def audit(variant: str = "") -> dict:
    files = C.TRAIN_FILES_SD if variant == "sd" else C.TRAIN_FILES
    src_path = C.TRAIN_SOURCE_SD if variant == "sd" else C.TRAIN_SOURCE
    src = json.loads(src_path.read_text(encoding="utf-8"))
    res = MD.audit_files({a: files[a] for a in C.ARMS}, MD.counter(), source=src)
    res["variant"] = variant or "canonical"
    res["files"] = {files[a].name: C.sha256(files[a]) for a in C.ARMS}
    res["torch_imported"] = "torch" in sys.modules
    if variant == "sd":
        canon = MD.canonical_from_arm_rows(C.load_jsonl(C.TRAIN_FILES["dint"]))
        sd = MD.canonical_from_arm_rows(C.load_jsonl(files["dint"]))
        res["checks"]["surface-diversity rows == the canonical rows"] = canon == sd
        surf = [r["surface"] for r in C.load_jsonl(files["dint"])]
        res["checks"]["no held-out surface in training"] = all(
            s and s["para"] in C.TRAIN_SURFACE_IDX and s["names"] in C.TRAIN_SURFACE_IDX for s in surf)
        res["ok"] = res["ok"] and all(res["checks"].values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("", "sd"), default="")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    res = audit(a.variant)
    core = res["core"]
    print(f"audit ({res['variant']}): ok={res['ok']}")
    print("  heuristic accuracy on the training rows:", {k: round(v, 4) for k, v in core["heuristic_accuracy"].items()})
    print("  gate:", core["gate"], core["balance_gate"])
    print("  types:", core["types"])
    print("  quotas:", core["quotas"], "min", core["quota_min"])
    print("  checks:", res["checks"])
    print("  cue ceilings (descriptive):", core["cue_ceilings"])
    print("  shortcut policies on the training rows (descriptive):", core["shortcut_accuracy_descriptive"])
    print("  rows separating the precondition-aware composites from the rule:",
          core["rows_separating_precondition_aware_composites_from_the_rule"])
    print("  minimal-pair twin artefacts:", core["minimal_pair_twin_artefacts"])
    if a.record:
        out = C.write_doc("AUDIT_sd.json" if a.variant == "sd" else "AUDIT.json", json.dumps(res, indent=1, default=str) + "\n")
        print(f"wrote {out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
