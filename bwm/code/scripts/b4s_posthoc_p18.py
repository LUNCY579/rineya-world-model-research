"""Post-hoc sensitivity note for the B4 futility screen (BWM P-18 / I-23; agreed with RCX 2026-10-12 22:20).

Not part of the registration and never changes the registered GO/NO-GO, which uses all 40 instances. Runs only after
`b4_screen.py analyze` has written runs/BW_B4_SCREEN/B4S_REPORT.json, and writes once:
runs/BW_B4_SCREEN/POSTHOC_P18_SENSITIVITY.{md,json}. It states whether seeds_fresh/0 (a block-renamed suffix row of a
Pa_v2 training problem, shared by both arms) is concordant or discordant between arm a and R0, and, if discordant, the
net (a_only - R0_only) with and without it. Outcome per instance = the VAL verdict ("val", else "success"), as in
b4_screen.py. Usage: python -B scripts/b4s_posthoc_p18.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "BW_B4_SCREEN"
A_PLANS = OUT / "a_beam" / "plans.jsonl"
R0_PLANS = ROOT / "runs" / "BW_SEEDS" / "R0_beam" / "plans.jsonl"
REPORT = OUT / "B4S_REPORT.json"
EXCLUDED = "seeds_fresh/0"
WORDING = ("Post-hoc sensitivity (BWM P-18, found after item 7 and before any screen output): excluding seeds_fresh/0, "
           "a block-renamed suffix of a Pa_v2 training row shared by both arms. The registered decision uses all 40 "
           "instances.")


def outcomes(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: bool(r.get("val", r["success"])) for r in rows}


def net(a: dict, c: dict, ids) -> dict:
    a_only = sum(a[i] and not c[i] for i in ids)
    c_only = sum(c[i] and not a[i] for i in ids)
    return {"n": len(ids), "a_only": a_only, "R0_only": c_only, "net": a_only - c_only,
            "a_success": sum(a[i] for i in ids), "R0_success": sum(c[i] for i in ids)}


def analyse(a: dict, c: dict, registered: dict | None = None) -> dict:
    if set(a) != set(c):
        raise ValueError("arm a and R0 cover different instance ids")
    ids = sorted(a)
    x = (a[EXCLUDED], c[EXCLUDED])
    concordant = x[0] == x[1]
    rec = {"what": "post-hoc sensitivity; descriptive; never changes the registered decision", "wording": WORDING,
           "instance": EXCLUDED, "a_success": x[0], "R0_success": x[1],
           "concordant": concordant, "all_40": net(a, c, ids)}
    if not concordant:
        rec["without_instance"] = net(a, c, [i for i in ids if i != EXCLUDED])
    if registered is not None:
        rec["registered_decision_quoted"] = registered
    return rec


def render(rec: dict) -> str:
    al = rec["all_40"]
    lines = ["# B4 screen: post-hoc sensitivity for seeds_fresh/0 (BWM P-18)", "", rec["wording"], "",
             f"- seeds_fresh/0: arm a {'solved' if rec['a_success'] else 'failed'}, R0 "
             f"{'solved' if rec['R0_success'] else 'failed'}: **{'concordant' if rec['concordant'] else 'DISCORDANT'}**.",
             f"- All {al['n']} (registered): a {al['a_success']}, R0 {al['R0_success']}, a_only {al['a_only']}, "
             f"R0_only {al['R0_only']}, net {al['net']:+d}."]
    if not rec["concordant"]:
        w = rec["without_instance"]
        lines.append(f"- Without seeds_fresh/0 ({w['n']}): a_only {w['a_only']}, R0_only {w['R0_only']}, "
                     f"net {w['net']:+d}.")
    else:
        lines.append("- Concordant, so excluding it leaves a_only, R0_only and the net unchanged.")
    if "registered_decision_quoted" in rec:
        lines.append(f"- Registered decision (quoted from B4S_REPORT.json, unchanged): "
                     f"{json.dumps(rec['registered_decision_quoted'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    if not REPORT.exists():
        sys.exit(f"{REPORT} missing: this note is written only after the registered analysis")
    md, js = OUT / "POSTHOC_P18_SENSITIVITY.md", OUT / "POSTHOC_P18_SENSITIVITY.json"
    if md.exists() or js.exists():
        sys.exit("the post-hoc note exists (written once)")
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    quoted = {k: rep[k] for k in ("status", "decision", "n_paired", "paired") if k in rep}
    rec = analyse(outcomes(A_PLANS), outcomes(R0_PLANS), quoted or None)
    js.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    md.write_text(render(rec), encoding="utf-8")
    print(render(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
