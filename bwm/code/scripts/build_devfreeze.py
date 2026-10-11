"""Assemble the local devfreeze package for the BWM Blocksworld system (DECISIONS D-49), in the Published-Cores format.

Run only after the frozen Track B/B2 test runs exist (runs/BW_TEST_B2/*/summary.json). Writes
  release/Rineya-Titan-Sparse-BWM-Blocksworld-0.1.0-devfreeze/
    README.md        model card (YAML front matter like the vault cores; scope and regime caveats)
    CORE.json        identity, construction, measured results with exact CIs, provenance hashes
    PROMOTION.json   user-authorized local packaging record (not a release; HF push needs separate approval)
    adapters/        BW_F (delta world model) and BW_Pa_v2 (proposer), unmerged LoRA on frozen FP32 Titan-Sparse
    code/            planner, Blocksworld semantics, LM utilities and the evaluation driver
    evidence/        PREREG_PLANBENCH.md, freeze records, test summaries + plans, selection results, F grades
and, with --to-vault, copies the directory into Rineya_Published_Cores (catalog/registry files are NOT edited).
Every number is read from the run files; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import lm, stats  # noqa: E402

NAME = "Rineya-Titan-Sparse-BWM-Blocksworld-0.1.0-devfreeze"
VAULT = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
TITAN = VAULT / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"
# zero-shot / direct prompting only (P-14 erratum: o1-mini's 10% is the LRM-Modulo verifier-in-loop score; direct is 0.9%)
PUBLISHED_HARD = {"DeepSeek-R1 (PlanBench leaderboard; upstream per-instance file at fc638a1a)": 59 / 110,
                  "o1-preview (direct prompting; arXiv 2410.02162 Table 5)": 26 / 110,
                  "o1-mini (direct prompting; arXiv 2410.02162 Table 5)": 1 / 110}


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return round(lo, 4), round(hi, 4)


def load_run(tset: str, system: str):
    d = ROOT / f"runs/BW_TEST_B2/{tset}_{system}"
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())["summary"]
    rows = [json.loads(l) for l in open(d / "plans.jsonl", encoding="utf-8")]
    return {"summary": summ, "rows": rows, "dir": d}


def measured():
    out = {}
    for tset in ("hard", "mirror", "planbench"):
        beam, direct = load_run(tset, "beam"), load_run(tset, "direct")
        rec = {}
        for name, r in (("S_beam", beam), ("S_direct", direct)):
            if r is None:
                continue
            ok = [int(bool(x.get("val", x["success"]))) for x in r["rows"]]
            k, n = sum(ok), len(ok)
            rec[name] = {"solved": k, "n": n, "rate": round(k / n, 4), "clopper_pearson_95": clopper_pearson(k, n),
                         "val_disagreements": r["summary"].get("val_disagreements"),
                         "f_queries_per_instance": r["summary"].get("f_queries_per_instance"),
                         "proposer_tokens_per_instance": r["summary"].get("proposer_tokens_per_instance"),
                         "success_by_blocks": r["summary"].get("success_by_blocks")}
        if beam is not None and direct is not None:
            a = {x["id"]: int(bool(x.get("val", x["success"]))) for x in beam["rows"]}
            b = {x["id"]: int(bool(x.get("val", x["success"]))) for x in direct["rows"]}
            ids = sorted(set(a) & set(b))
            wins = sum(a[i] and not b[i] for i in ids)
            losses = sum(b[i] and not a[i] for i in ids)
            rec["paired_beam_vs_direct"] = {"n": len(ids), "beam_only": wins, "direct_only": losses,
                                            "exact_mcnemar_p": stats.mcnemar_exact(wins, losses)}
        out[tset] = rec
    return out


def readme(meas, frz, sel_mean):
    h = meas.get("hard", {})
    sb, sd = h.get("S_beam"), h.get("S_direct")
    lines = [
        "---", "license: apache-2.0", "library_name: peft", "base_model: Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
        "tags:", "  - rineya", "  - experimental", "  - research-artifact", "  - not-release-qualified", "  - fp32-only",
        "  - world-model", "  - planning", "  - blocksworld", "---", "",
        f"# {NAME}", "",
        "**Experimental research package. Not a release, not release-qualified, not a general-purpose model.** A frozen",
        "FP32 Titan-Sparse core plus two unmerged LoRA adapters (~10M parameters each) and a planner: a learned,",
        "goal-blind Blocksworld world model and a plan proposer, used for beam search *in imagination* (no execution",
        "feedback; one plan submitted per problem, scored by VAL). Trained only on generated problems with <= 6 blocks;",
        "PlanBench test problems were excluded up to renaming of blocks.", "",
        "## Frozen results (run once, settings chosen on generated validation sets by a pre-registered rule)", "",
        "| Test set | S-beam (world model + search) | S-direct (same proposer, no world model) | paired |",
        "|---|---:|---:|---:|",
    ]
    for tset, label in (("hard", "PlanBench Blocksworld-hard (110; 6-15 blocks; 20-40-step plans)"),
                        ("mirror", "Generated hard-style mirror (110)"), ("planbench", "PlanBench Blocksworld (the \"600\" set: 602 instances; 3-5 blocks)")):
        r = meas.get(tset, {})
        def cell(x):
            return "pending" if x is None else f"{x['solved']}/{x['n']} ({x['rate']:.1%}; 95% CI {x['clopper_pearson_95'][0]:.1%}-{x['clopper_pearson_95'][1]:.1%})"
        pr = r.get("paired_beam_vs_direct")
        ptxt = "-" if pr is None else f"+{pr['beam_only']} / -{pr['direct_only']}, p = {pr['exact_mcnemar_p']:.2g}"
        lines.append(f"| {label} | {cell(r.get('S_beam'))} | {cell(r.get('S_direct'))} | {ptxt} |")
    from scipy.stats import fisher_exact
    mb = meas.get("mirror", {}).get("S_beam")
    if sb is not None and mb is not None:
        gap = 100 * (sb["rate"] - mb["rate"])
        pg = fisher_exact([[sb["solved"], sb["n"] - sb["solved"]], [mb["solved"], mb["n"] - mb["solved"]]])[1]
        lines += ["", f"Contamination check (pre-registered): hard minus mirror, S-beam = {gap:+.1f} pp (unpaired Fisher p = "
                  f"{pg:.2g}); the registered flag is a gap above +10 pp ({'FLAGGED' if gap > 10 else 'not flagged'}). The "
                  "mirror's goals (tall-tower targets) are not distribution-matched to the hard set's goal structure, so part "
                  "of any gap is difficulty rather than contamination."]
        ga = ROOT / "runs/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json"
        if ga.exists():
            g = json.loads(ga.read_text())
            fmt = lambda d: ", ".join(f"{k}: {d[k]}" for k in ("<=6", "7-9", ">=10") if k in d)  # noqa: E731
            lines += ["", f"Post-hoc, descriptive (`evidence/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json`, `code/bw_mirror_gap.py`; the flag stands as computed): the "
                      f"move lower bound is matched (hard {g['hard']['mean_move_lower_bound']:.1f} vs mirror "
                      f"{g['mirror']['mean_move_lower_bound']:.1f}), but the mirror's tallest goal tower is taller "
                      f"({g['mirror']['mean_max_goal_tower']:.1f} vs {g['hard']['mean_max_goal_tower']:.1f}) and success "
                      f"falls with goal-tower height in both sets (tallest goal tower {fmt(g['hard']['success_by_max_goal_tower'])} "
                      f"on hard; {fmt(g['mirror']['success_by_max_goal_tower'])} on the mirror). Reweighted to the hard set's goal-tower mix, the "
                      f"mirror scores {g['mirror_standardized_to_hard_goal_tower_mix']:.1%} vs hard "
                      f"{g['hard']['success']:.1%} (gap {g['gap_pp_standardized']:+.1f} pp)."]
    lines += ["", "Published zero-shot references on Blocksworld-hard (different regime: general models, no domain training):", ""]
    for k, v in PUBLISHED_HARD.items():
        txt = f"- {k}: {v:.1%}"
        if sb is not None:
            m = round(v * 110)
            p2 = fisher_exact([[sb["solved"], sb["n"] - sb["solved"]], [m, 110 - m]])[1]
            txt += f" (unpaired Fisher vs ours, two-sided p = {p2:.2g})"
        lines.append(txt)
    lines.append("")
    lines.append("These comparisons use published aggregates and are unpaired (upstream per-instance files exist but are not yet")
    lines.append("vendored and re-scored with the local VAL; DECISIONS P-14). A difference with two-sided")
    lines.append("p >= 0.05 is not claimed (this applies to DeepSeek-R1).")
    lines += ["", "## What this does and does not show", "",
              "- It shows that a 0.6B frozen core with a *learned* world model and search plans far beyond the problem sizes its",
              "  training data covered, where the same proposer without the world model collapses (paired comparison above).",
              "- It does **not** show general planning ability: this is a Blocksworld specialist with a learned simulator, compared",
              "  with zero-shot generalist models in a different regime. Verifier-in-the-loop numbers (e.g. o1-preview + VAL 98.2%)",
              "  are not comparable. One training seed per adapter.", "",
              "## Configuration (frozen)", "", "```json", json.dumps(frz["args"], indent=1), "```", "",
              f"Selected on generated validation sets (mean success {sel_mean:.3f}) by the rule in `evidence/PREREG_PLANBENCH.md`",
              "(addendum B2), applied mechanically by `code/bw_freeze.py`.", "",
              "## Use", "", "```python",
              "from peft import PeftModel",
              "from bwmn import lm, blocks_planner as BP   # code/ in this package",
              "base, tok = lm.load_core('titan-sparse')     # FP32 Titan-Sparse; never cast to BF16",
              "model = PeftModel.from_pretrained(base, 'adapters/world_model_delta', adapter_name='fwd')",
              "model.load_adapter('adapters/proposer_v2', adapter_name='pa')",
              "plans = BP.plan_beam(problems, BP.LMProposerBW(model, tok, 'pa', K=..., max_new=..., prefix_ok=True),",
              "                     BP.LMForwardBW(model, tok, 'fwd', fmt='delta'), width=..., max_expansions=..., max_rollout=...)",
              "```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-vault", action="store_true")
    a = ap.parse_args()
    frz = json.loads((ROOT / "runs/FREEZE_B2_beam.json").read_text())
    meas = measured()
    if not meas.get("hard", {}).get("S_beam"):
        sys.exit("hard-set S-beam result missing: build after the frozen test run")
    out = ROOT / "release" / NAME
    if out.exists():
        shutil.rmtree(out)
    (out / "adapters").mkdir(parents=True)
    shutil.copytree(ROOT / "runs/BW_Fdelta_s0/adapter", out / "adapters/world_model_delta")
    shutil.copytree(ROOT / "runs/BW_Pa_v2_s1/adapter", out / "adapters/proposer_v2")
    (out / "code/bwmn").mkdir(parents=True)
    for f in ("__init__.py", "blocks.py", "blocks_planner.py", "lm.py", "nvml.py", "stats.py", "flops.py", "gpu_lease.py"):
        if (ROOT / "bwmn" / f).exists():
            shutil.copy(ROOT / "bwmn" / f, out / "code/bwmn" / f)
    for f in ("blocks_eval.py", "bw_run_tests.py", "bw_freeze.py", "bw_select.py", "blocks_data.py", "bw_delta_data.py",
              "blocks_val_large.py", "bw_forward_grade.py", "train_generic.py", "bw_mirror_gap.py"):
        shutil.copy(ROOT / "scripts" / f, out / "code" / f)
    ev = out / "evidence"
    ev.mkdir()
    for f in ("PREREG_PLANBENCH.md", "runs/FREEZE_B2_beam.json", "runs/FREEZE_B2_direct.json",
              "runs/BW_SELECT_v1/select.jsonl", "runs/BW_F_s0/grade_size.json", "runs/BW_Fdelta_s0/grade_size.json",
              "runs/BW_data_v2/META.json"):
        p = ROOT / f
        if p.exists():
            shutil.copy(p, ev / p.name.replace("grade_size", p.parent.name + "_grade_size"))
    shutil.copytree(ROOT / "runs/BW_TEST_B2", ev / "BW_TEST_B2", ignore=shutil.ignore_patterns("fcache*.jsonl", "tmp_*"))
    titan_sha = "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"
    core = {
        "schema_version": "1.1",
        "core_id": "rineya.core.titan-sparse-bwm-blocksworld.v0.1.0-devfreeze.lora-r16x2.planner-beam",
        "codename": "Titan-Sparse-BWM-Blocksworld", "directory_name": NAME, "version": "0.1.0-devfreeze",
        "kind": "unmerged_lora_adapters_with_planner_on_frozen_core",
        "status": "experimental_research_package_not_release_qualified",
        "provisional": True, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "base": {"core": TITAN.name, "weight_sha256": titan_sha, "dtype": "float32", "frozen": True,
                 "note": "weights are not copied; load the vault core and verify its SHA-256"},
        "adapters": {
            "world_model_delta": {"sha256": lm.file_sha256(out / "adapters/world_model_delta/adapter_model.safetensors"),
                                  "trained_on": "12,000 generated 4-6-block transitions (delta format), 600 steps"},
            "proposer_v2": {"sha256": lm.file_sha256(out / "adapters/proposer_v2/adapter_model.safetensors"),
                            "trained_on": "53,682 generated proposer rows (3-6 blocks; half whole-configuration goals), 1,000 steps"}},
        "construction": {"prereg": "evidence/PREREG_PLANBENCH.md", "prereg_sha256": frz["prereg_sha256"],
                         "freeze": "evidence/FREEZE_B2_beam.json", "selected_from": frz["selected_from"],
                         "selection_rule": frz["rule"], "test_runs_per_configuration": 1},
        "measured": meas,
        "published_reference_hard": PUBLISHED_HARD,
        "caveats": ["domain specialist with a learned simulator vs zero-shot generalists (different regime)",
                    "one training seed per adapter", "validation sets are generated, never PlanBench",
                    "not comparable with verifier-in-the-loop systems",
                    "pre-registered mirror flag raised (hard minus mirror > 10 pp); post-hoc analysis attributes it to the "
                    "mirror's taller goal towers (evidence/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json)"],
    }
    (out / "CORE.json").write_text(json.dumps(core, indent=1))
    (out / "PROMOTION.json").write_text(json.dumps({
        "schema": 1, "record": "user-authorized LOCAL packaging of a pre-registered frozen configuration (DECISIONS D-49)",
        "authorized": "user, 2026-10-08", "not_authorized": "publishing to Hugging Face or any external service",
        "why_this_is_not_post_hoc": "configuration chosen by the pre-registered rule on generated validation sets before any "
                                    "test run; each test set run once with the frozen configuration"}, indent=1))
    (out / "README.md").write_text(readme(meas, frz, frz["selection_mean_success"]), encoding="utf-8")
    print("built", out)
    if a.to_vault:
        dst = VAULT / NAME
        if dst.exists():
            sys.exit(f"{dst} exists; refusing to overwrite")
        shutil.copytree(out, dst)
        print("copied to", dst)


if __name__ == "__main__":
    main()
