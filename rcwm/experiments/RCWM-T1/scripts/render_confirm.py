"""Render runs/bank_confirm/analysis/PREREG_1_RESULTS.json into RESULTS_CONFIRM.md, mechanically:
every number is copied from the JSON, nothing is recomputed or selected."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "runs" / ("bank_pilot" if "--dry" in sys.argv else "bank_confirm") / "analysis" / \
      ("PREREG_1_DRYRUN.json" if "--dry" in sys.argv else "PREREG_1_RESULTS.json")
R = json.loads(src.read_text(encoding="utf-8"))
f3 = lambda x: "n/a" if x is None else f"{x:+.3f}" if isinstance(x, float) and x < 0 else f"{x:.3f}"
lines = [f"# RCWM-T1 PREREG-1 results {'(DRY RUN on pilot TEST — not confirmatory)' if R['dry_run'] else '(confirmation)'}", "",
         f"Source: `{src.relative_to(ROOT).as_posix()}`. Rendered mechanically by `scripts/render_confirm.py`. Analysis plan: `PREREG_1.md` (frozen; `PREREG_1_FREEZE.json`).", "",
         f"**Claim confirmed (C0–C4 all PASS): {R['claim_confirmed']}**", "",
         "## Primary contrasts (L and R, k = 4, pooled with equal family weight; normalised regret, negative favours G_chain(Titan))", "",
         "| | Contrast | Pooled point | 95% CI | One-sided p | Holm p | PASS | Holds in both families | Per-family points | Floor qualifier |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for n, v in R["primary"].items():
    fam = ", ".join(f"{k[:6]} {f3(x['point'])}" for k, x in v["per_family"].items())
    fq = "" if "comparator_at_or_below_floor" not in v else ("comparator at/below answer-free floor" if v["comparator_at_or_below_floor"] else "comparator above floor")
    lines.append(f"| {n} | {v['contrast']} | {f3(v['point'])} | [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] | {v['p_one_sided']:.4g} | {v['p_holm']:.4g} | "
                 f"{'PASS' if v['PASS'] else 'FAIL'} | {v['holds_in_both_families']} | {fam} | {fq} |")
lines += ["", "## Mean normalised regret per method (confirmation parents)", ""]
methods = ["Gtitan_chain_one", "Gtitan_chain_m1", "Gtitan_e2e", "Gsupa_best", "Gsupa_e2e", "Gsupa_chain_one", "B3", "S", "S_abundant",
           "B2_bilinear@full16", "B2_bilinear@sub4", "F_touched", "F_copy_s0", "F_first_only", "F_ignore_status", "F_ignore_pre"]
lines.append("| family / k | n | " + " | ".join(methods) + " |")
lines.append("|---|---|" + "---|" * len(methods))
for key, v in R["per_family"].items():
    lines.append(f"| {key} | {v['n_test_parents']} | " + " | ".join(f3(v["means"].get(m)) if m in v["means"] else "" for m in methods) + " |")
lines += ["", "## Selections (pilot DEV only) and parse rates", ""]
for key, v in R["per_family"].items():
    lines.append(f"- {key}: B3 chosen {v['selection'].get('B3_chosen')}, S chosen {v['selection'].get('S_chosen')}, Supa best mode {v.get('Gsupa_best_mode')}, "
                 f"B3 best-on-test upper bound {f3(v['B3_best_on_test_upper_bound'])}, parse {json.dumps({k: round(x, 3) for k, x in v['parse_rate'].items()})}")
lines += ["", "## Secondary contrasts (descriptive; BH q = 0.10)", "", "| contrast | point | 95% CI | one-sided p | BH p | BH sig |", "|---|---|---|---|---|---|"]
for k, v in R["secondary"].items():
    lines.append(f"| {k} | {f3(v['point'])} | [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] | {v['p_one_sided']:.4g} | {v.get('p_bh', float('nan')):.4g} | {v.get('bh_significant')} |")
lines += ["", "## Exact-state accuracy (lenient parse)", "", "| arm | family / k / mode | exact | parse | n |", "|---|---|---|---|---|"]
for arm, d in R["exact_state"].items():
    for k, v in d.items():
        lines.append(f"| {arm} | {k} | {v['exact']:.3f} | {v['parse']:.3f} | {v['n']} |")
if isinstance(R.get("twins"), dict):
    lines += ["", "## Twin consistency (k = 4)", ""]
    for arm, d in R["twins"].items():
        for k, v in d.items():
            lines.append(f"- {arm} {k}: n {v['n']}, both exact {v['both_exact']:.3f}, predictions differ {v['predictions_differ']:.3f}")
out = ROOT / ("RESULTS_CONFIRM_DRYRUN.md" if R["dry_run"] else "RESULTS_CONFIRM.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", out)
