"""RCWM-T2b prerequisite screen: training-data audit. Exits 1 (refuses) unless every gate holds. No model, no torch.

  python -B scripts/t2bp_audit.py                      # the PRIMARY build (revision 3, C1): data/t2bp_train_sd_<arm>.jsonl
  python -B scripts/t2bp_audit.py --variant canonical  # the canonical files data/t2bp_train_<arm>.jsonl (the row source)
  python -B scripts/t2bp_audit.py --record             # also writes docs/t2b/prereq/AUDIT.json (sd) or AUDIT_canonical.json

Gates (t2bp_common.audit_rows + t2bp_make_data.audit_files):
  * on the training rows, each audited heuristic - copy-input, apply-regardless (= "(before&~delete)|add ignoring
    preconditions", checked identical on every row), integer before+add, OR before|add - scores 0.50 +- 0.02;
  * the type composition equals the design; >= 250 valid minimal pairs (same before and event except exactly one bit of
    pre_on or pre_off; the applicable member's gold differs from the inapplicable member's; the flip is the
    inapplicable member's only violation); int-add-wrong (strict, A4) >= 150; applicable-delete >= 200; int-add-wrong
    (R11 definition) >= 200; pre_on-missing and pre_off-present inapplicable rows >= 300 each; no row violates both;
  * golds == rcwmt.env.step; payloads and operative contents unique; 1,500 rows, all 8-bit;
  * every rendering holds the same canonical rows in the same order; prompts == the arm's frame for the row's surface
    (D-int prompts in the canonical surface == T2's frame); every target parses back to its gold with the arm's strict
    parser (cmp/calc/flag exact);
  * amendment 4b: every filler target is the constant-fill target (FILL_K units per action kind), never shorter than
    the row's trace target, and the fill length carries zero information about the answer (one value per kind;
    predicting applicability / change / after / type from it equals the majority share; mutual information 0);
  * surface proof (C2): every prompt's surface, READ BACK FROM ITS TEXT, equals the recorded surface and holds no
    held-out factor (paraphrase 4, field names 4, the reversed top-level key order, any of the 3 held-out event key
    orders); no prompt contains text only a held-out surface produces (paraphrase 4's wording, " bits wide", the
    held-out field names as JSON keys); the events list has one element in every prompt (no list order to augment);
  * sd only: the rows and targets equal the canonical files'; the recorded surfaces use training indices only;
  * file sha256 == the SOURCE record.
Descriptive (recorded, not gated): (before&~delete)+add and before&~delete accuracies; cheap-feature cue ceilings; the
shortcut policies of t2bp_common.SHORTCUT_POLICIES and the rows separating each precondition-aware composite from the
bitwise rule; minimal-pair twin artefacts; the surface diversity actually used; the filler residual (filler minus trace
tokens) and what the revision-2 row-matched fill length would have revealed.
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


def audit(variant: str = "sd") -> dict:
    sd = variant == "sd"
    files = C.TRAIN_FILES_SD if sd else C.TRAIN_FILES
    src_path = C.TRAIN_SOURCE_SD if sd else C.TRAIN_SOURCE
    src = json.loads(src_path.read_text(encoding="utf-8"))
    res = MD.audit_files({a: files[a] for a in C.ARMS}, MD.counter(), source=src)
    res["variant"] = variant
    res["files"] = {files[a].name: C.sha256(files[a]) for a in C.ARMS}
    res["torch_imported"] = "torch" in sys.modules
    if sd:
        canon = {a: C.load_jsonl(C.TRAIN_FILES[a]) for a in C.ARMS}
        sdr = {a: C.load_jsonl(files[a]) for a in C.ARMS}
        res["checks"]["surface-diversity rows == the canonical rows"] = all(
            MD.canonical_from_arm_rows(sdr[a]) == MD.canonical_from_arm_rows(canon[a]) for a in C.ARMS)
        res["checks"]["surface-diversity targets == the canonical targets (literal order, unchanged)"] = all(
            [r["target"] for r in sdr[a]] == [r["target"] for r in canon[a]] for a in C.ARMS)
        res["checks"]["recorded surfaces use training indices and training key orders only"] = all(
            r["surface"] and not C.heldout_factors(r["surface"]) for a in C.ARMS for r in sdr[a])
        res["checks"]["the sd SOURCE names the canonical row source it read"] = (
            src.get("canonical_dint_sha256") == C.sha256(C.TRAIN_FILES["dint"]))
        res["ok"] = res["ok"] and all(res["checks"].values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("sd", "canonical"), default="sd")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    res = audit(a.variant)
    core = res["core"]
    print(f"audit ({res['variant']}): ok={res['ok']}")
    print("  heuristic accuracy on the training rows:", {k: round(v, 4) for k, v in core["heuristic_accuracy"].items()})
    print("  gate:", core["gate"], core["balance_gate"])
    print("  types:", core["types"])
    print("  quotas:", core["quotas"], "min", core["quota_min"])
    print("  checks:", json.dumps(res["checks"], indent=1))
    print("  surface diversity used:", {k: v for k, v in res["surface"]["stats"].items() if k != "heldout_fragments_scanned"})
    f4 = res.get("filler_4b") or {}
    print("  amendment 4b filler:", {k: f4.get(k) for k in ("distinct_fill_lengths_per_kind", "fill_length_lookup_accuracy",
                                                             "majority_share", "mutual_information_bits",
                                                             "rev2_row_matched_length_would_reveal_descriptive",
                                                             "residual_tokens_descriptive")})
    print("  cue ceilings (descriptive):", core["cue_ceilings"])
    print("  shortcut policies on the training rows (descriptive):", core["shortcut_accuracy_descriptive"])
    if a.record:
        out = C.write_doc("AUDIT.json" if a.variant == "sd" else "AUDIT_canonical.json",
                          json.dumps(res, indent=1, default=str) + "\n")
        print(f"wrote {out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
