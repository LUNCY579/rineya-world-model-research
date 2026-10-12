"""POST HOC, DIFFERENT REGIME. Paired comparison of the frozen BWM systems on PlanBench Blocksworld-hard (110 instances)
with the published per-instance step-0 (direct) results of DeepSeek-R1, o1-preview and o1-mini (DECISIONS D-75.3;
closes P-14 correction (d)). CPU only, no model, no new run on the hard set (stored outputs only).

Inputs (all sha256-pinned; the script refuses to run on a mismatch):
  - published: bench/planbench/results/blocksworld_hard/<engine>/<UP_FILE>, vendored unchanged from LLMs-Planning
    fc638a1a llm_planning_analysis/results_backprompting/blocksworld_hard (see bench/planbench/SOURCE.md);
  - ours: plans.jsonl of B2 S-beam (runs/BW_TEST_B2/hard_beam), H2 Pa_v3 S-beam (runs/BW_B3_HARD/h2_beam), H1 Pa_v3
    S-direct (runs/BW_B3_HARD/h1_direct) and B2 S-direct (runs/BW_TEST_B2/hard_direct).

Published side: step 0 of step_wise_dict is the model's first answer to the zero-shot PDDL prompt, before any verifier
feedback (the later steps are the VAL-backprompting loop and are NOT compared). The answer text is
response_object.choices[0].message.content.strip(), exactly as upstream's llm_utils.send_query_with_feedback returns it
(R1's separate reasoning_content is not part of the answer, as upstream).

Two extractors, both scored by the local VAL (bwmn.blocks.val_check on bench/planbench/blocksworld_hard/generated):
  - pb_pddl (primary): a port of upstream's own extractor for PDDL-format tasks, llm_utils.save_gpt3_response at
    fc638a1a (git blob 62f36232...). back_prompting_parallel.get_repeated_verification_pddl (blob 2a1e302c...) calls it
    when pddl=True; text_to_plan / text_to_plan_blocksworld (the natural-language parser, ported in
    bwmn/planbench_extract.py) is used only for natural-language tasks. Rule: keep every line that starts with '(' (no
    leading whitespace), skip lines containing '()' or 'no actions', lowercase and strip. It must reproduce the
    published step-0 llm_plan byte for byte (checked; the script fails otherwise). Its lines are scored twice: as
    tuples through val_check, and verbatim as upstream writes the plan file (same VAL binary); both must agree.
  - bwm_lenient (sensitivity): built on bwmn/planbench_extract.py helpers (answer_text, _normalize, _LIST_MARK,
    _STEP_MARK), with that module's policies where they apply to PDDL text: code-fence lines skipped; list / step
    markers and markdown emphasis stripped; every action on a line is read, in order, in PDDL "(verb args)" or call
    "verb(args)" notation, verbs = the hard domain's action names; objects kept as written (no repair: unknown objects
    and wrong arities stay and VAL rejects them); if the answer has more than one block of action lines separated by
    prose, only the LAST block is used (planbench_extract's "last plan region" rule: narration followed by a recap is
    not double counted). The module's colour-based natural-language extractors (extract_plan, text_to_plan_blocksworld)
    are run as an applicability probe only: PDDL-letter answers name no colours, so they extract nothing.

Statistics: exact two-sided McNemar on discordant pairs (bwmn.stats.mcnemar_exact, cross-checked with
scipy.stats.binomtest); Holm within the family of 4 BWM systems x 3 published systems = 12 contrasts (primary
extractor); the same family under bwm_lenient is reported as a sensitivity family with its own Holm; Clopper-Pearson
95% CIs. Compute is report-only: tokens for every system; an ESTIMATED dense-only FLOPs figure for R1 (2 x 37e9 active
parameters x tokens processed); no FLOPs for o1 (undisclosed); BWM FLOPs are left to the item-3 ledger (they need
F-query token counts, i.e. the tokenizer, which this CPU no-model track does not load). No iso-FLOP claim.

Writes runs/HARD_PUBLISHED_PAIRED/{REPORT.json, REPORT.md, per_instance.jsonl}.
Usage: python -B scripts/hard_published_paired.py
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics as st
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import planbench_extract as PX  # noqa: E402
from bwmn import stats as ST  # noqa: E402
from scipy.stats import beta, binomtest  # noqa: E402

LABEL = "POST HOC, DIFFERENT REGIME"
HARD = ROOT / "bench" / "planbench" / "blocksworld_hard"
DOMAIN = HARD / "generated_domain.pddl"
RES = ROOT / "bench" / "planbench" / "results" / "blocksworld_hard"
UPSTREAM_COMMIT = "fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284"
UP_DIR = "llm_planning_analysis/results_backprompting/blocksworld_hard"
UP_FILE = ("task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_"
           "history_feedback_type_1_tv0.0_tg1.0.json")
# engine dir -> (label, sha256 of the vendored file, git blob id of the same path at fc638a1a)
PUBLISHED = {
    "deepseek-r1-api": ("DeepSeek-R1", "332b0f8ec03f337b4f320e4986b3d758c0fbb6efd223c7cfef05534e8202740e",
                        "22ca8ced06058f04532cfe4a74f57459e1014046"),
    "o1-preview_chat": ("o1-preview", "b8fdc9ba2b7cbd3f1f26a3e4aabe2be5ac108a1499916ef91809a88b1a51478c",
                        "87ccabe011fabdcb752af3a1142f96b83b24cc79"),
    "o1-mini_chat": ("o1-mini", "928497901fc3e4be5b78fe26a5afdfa2939dec6c5b6e291c5cd67211fd71bc44",
                     "7f2e7e2bf825e1dc4a1e43ab0dd4f6e25f6eb208"),
}
# label -> (plans.jsonl, summary.json, registered entry)
OURS = {
    "B2 S-beam": ("runs/BW_TEST_B2/hard_beam/plans.jsonl", "runs/BW_TEST_B2/hard_beam/summary.json", "P-08/P-10"),
    "H2 Pa_v3 S-beam": ("runs/BW_B3_HARD/h2_beam/plans.jsonl", "runs/BW_B3_HARD/h2_beam/summary.json", "P-14"),
    "H1 Pa_v3 S-direct": ("runs/BW_B3_HARD/h1_direct/plans.jsonl", "runs/BW_B3_HARD/h1_direct/summary.json", "P-13"),
    "B2 S-direct": ("runs/BW_TEST_B2/hard_direct/plans.jsonl", "runs/BW_TEST_B2/hard_direct/summary.json",
                    "P-08/P-10"),
}
EXPECTED_OURS = {"B2 S-beam": 73, "H2 Pa_v3 S-beam": 88, "H1 Pa_v3 S-direct": 13, "B2 S-direct": 5}
AUDIT_STEP0 = ROOT / "research" / "audit_evidence" / "bwm_h2_scratch" / "upstream_step0.json"  # H2 audit, simulator
R1_ACTIVE_PARAMS = 37e9  # DeepSeek-V3/R1 MoE: activated parameters per token (developer's published figure)
OUT = ROOT / "runs" / "HARD_PUBLISHED_PAIRED"
IDS = list(range(1, 111))
# `git ls-tree -r fc638a1a` lines for the hard instances and hard result files, taken offline from the blob-less partial
# clone research/audit_evidence/bwm_item9_scratch/llmsp (trees only; no blob fetch). Used to show that the local hard
# instances are byte-identical to upstream's (git blob sha1 of the local file == upstream tree entry).
UPSTREAM_TREE = OUT / "upstream_tree_hard.txt"
UPSTREAM_INSTANCE_BLOBS: dict = {}
if UPSTREAM_TREE.exists():
    for _ln in UPSTREAM_TREE.read_text().splitlines():
        _meta, _path = _ln.split("\t")
        _m = re.search(r"blocksworld_hard/generated/instance-(\d+)\.pddl$", _path)
        if _m:
            UPSTREAM_INSTANCE_BLOBS[int(_m.group(1))] = _meta.split()[2]


def blob_of_instance(i: int) -> str | None:
    return UPSTREAM_INSTANCE_BLOBS.get(i)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob(p: Path) -> str:
    b = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def problem_path(i: int) -> Path:
    return HARD / "generated" / f"instance-{i}.pddl"


# ----------------------------------------------------------------------------- extractors

def pb_pddl_lines(response: str) -> list[str]:
    """Port of LLMs-Planning llm_planning_analysis/utils/llm_utils.py save_gpt3_response at fc638a1a (MIT License,
    Copyright (c) 2024 Valmeekam Karthik; bench/planbench/LICENSE_LLMs-Planning_MIT.txt). Change: returns the lines
    instead of writing the plan file (upstream writes '\\n'.join(plan) and returns the same string)."""
    plan = []
    for line in response.split('\n'):
        if not line.startswith('('):
            continue
        if '()' in line:
            continue
        if 'no actions' in line:
            continue
        plan.append(line.lower().strip())
    return plan


def lines_to_tuples(lines: list[str]) -> list[tuple] | None:
    """'(unstack h d)' -> ('unstack', 'h', 'd'); val_check writes '(' + ' '.join(t) + ')', i.e. the same line up to
    whitespace. None if a line is not of the form '(...)' (then only the verbatim VAL verdict exists)."""
    out = []
    for ln in lines:
        if not (ln.startswith("(") and ln.endswith(")")):
            return None
        out.append(tuple(ln[1:-1].split()))
    return out


_VERBS = list(PX.domain_actions(DOMAIN))  # hard domain action names, file order
_VALT = "|".join(re.escape(v) for v in sorted(_VERBS, key=len, reverse=True))
_ACT_RE = re.compile(rf"\(\s*({_VALT})\b([^()]*)\)|\b({_VALT})\s*\(([^()]*)\)")


def bwm_lenient(content: str) -> tuple[list[tuple], dict]:
    ans, src = PX.answer_text(content or "")
    runs: list[list[tuple]] = []
    cur: list[tuple] = []
    for raw in PX._normalize(ans).split("\n"):
        s = raw.strip()
        if not s or s.startswith("```"):
            continue  # blank and code-fence lines neither carry actions nor end a block
        s = re.sub(r"[*`]+", "", s).lstrip("#").strip()
        s = PX._LIST_MARK.sub("", s, count=1)
        s = PX._STEP_MARK.sub("", s, count=1)
        acts = []
        for m in _ACT_RE.finditer(s):
            verb, args = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            acts.append((verb,) + tuple(args.replace(",", " ").split()))
        if acts:
            cur += acts
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    plan = runs[-1] if runs else []
    return plan, {"answer_source": src, "n_blocks_of_action_lines": len(runs),
                  "n_actions_all_blocks": sum(len(r) for r in runs)}


# ----------------------------------------------------------------------------- VAL

_VAL_CACHE: dict = {}
N_VAL_CALLS = Counter()


def val_tuples(i: int, plan: list[tuple]) -> bool:
    key = ("t", i, tuple(plan))
    if key not in _VAL_CACHE:
        N_VAL_CALLS["val_check"] += 1
        _VAL_CACHE[key] = bool(B.val_check(problem_path(i), plan, domain_path=DOMAIN))
    return _VAL_CACHE[key]


def val_verbatim(i: int, text: str) -> bool:
    """Same binary and criterion as bwmn.blocks.val_check, but the plan file holds the text exactly as upstream writes
    it ('\\n'.join(lines), no trailing newline)."""
    key = ("v", i, text)
    if key not in _VAL_CACHE:
        N_VAL_CALLS["verbatim"] += 1
        with tempfile.TemporaryDirectory() as d:
            pp = Path(d) / "plan.txt"
            pp.write_text(text)
            out = subprocess.run([str(B.VAL_EXE), str(DOMAIN), str(problem_path(i)), str(pp)], capture_output=True,
                                 text=True, timeout=60).stdout
        _VAL_CACHE[key] = "Plan valid" in out
    return _VAL_CACHE[key]


# ----------------------------------------------------------------------------- statistics

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def rate(vec: dict) -> dict:
    k, n = sum(vec[i] for i in IDS), len(IDS)
    return {"solved": k, "n": n, "rate": round(k / n, 4), "clopper_pearson_95": clopper_pearson(k, n)}


def contrast(a: dict, b: dict) -> dict:
    both = sum(a[i] and b[i] for i in IDS)
    only_a = sum(a[i] and not b[i] for i in IDS)
    only_b = sum(b[i] and not a[i] for i in IDS)
    p = ST.mcnemar_exact(only_a, only_b)
    p_scipy = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    assert abs(p - p_scipy) <= 1e-12 + 1e-9 * p, (p, p_scipy)
    return {"bwm": sum(a[i] for i in IDS), "published": sum(b[i] for i in IDS), "both": both, "bwm_only": only_a,
            "published_only": only_b, "neither": len(IDS) - both - only_a - only_b,
            "diff_pp": round(100 * (only_a - only_b) / len(IDS), 1), "mcnemar_exact_p": p}


def family(ours: dict, pub: dict, keys: list[tuple]) -> dict:
    res = {f"{o} vs {pl}": contrast(ours[o], pub[pl]) for o, pl in keys}
    adj = ST.holm({k: v["mcnemar_exact_p"] for k, v in res.items()})
    for k, v in res.items():
        v["holm_p"] = adj[k]
        v["holm_sig_0.05"] = adj[k] < 0.05
    return res


def summ(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return {"n": len(xs), "median": round(st.median(xs), 1), "mean": round(st.mean(xs), 1), "min": min(xs),
            "max": max(xs)}


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problems = {i: B.parse_problem(problem_path(i)) for i in IDS}
    n_blocks = {i: len(problems[i][0]) for i in IDS}
    info = json.loads((HARD / "info.json").read_text(encoding="utf-8"))
    opt_len = {i: info[f"instance-{i}.pddl"]["length"] for i in IDS}

    # ---------------- published files: pins, prompts, extraction, VAL
    provenance, pub_vec, pub_rows, extractor_checks, tokens, published_meta = {}, {}, {}, {}, {}, {}
    prompts_by_engine = {}
    for eng, (lab, pin, blob) in PUBLISHED.items():
        f = RES / eng / UP_FILE
        got = sha256(f)
        if got != pin or git_blob(f) != blob:
            sys.exit(f"REFUSE: {rel(f)} sha256 {got} / blob {git_blob(f)} does not match the pin {pin} / {blob}")
        provenance[lab] = {"path": rel(f), "sha256": got, "bytes": f.stat().st_size, "git_blob_sha1": blob,
                           "upstream": f"{UP_DIR}/{eng}/{UP_FILE} @ {UPSTREAM_COMMIT}"}
        d = json.loads(f.read_text(encoding="utf-8"))
        recs = {int(r["instance_id"]): r for r in d["instances"]}
        assert sorted(recs) == IDS, (lab, "instance ids")
        published_meta[lab] = {"engine": d.get("engine"), "task": d.get("task"), "domain": d.get("domain"),
                               "prompt_type_field": d.get("prompt_type"),
                               "steps_recorded": dict(Counter(len(r["step_wise_dict"]) for r in recs.values())),
                               "final_act_correct_verifier_loop_NOT_COMPARED": sum(bool(r.get("act_correct"))
                                                                                   for r in recs.values())}
        vec = {"pb_pddl": {}, "bwm_lenient": {}}
        rows = {}
        chk = Counter()
        tok = {"prompt": [], "completion": [], "reasoning": [], "time_s": []}
        prompts_by_engine[lab] = {}
        for i in IDS:
            r = recs[i]
            s0 = r["step_wise_dict"]["0"]
            ro = s0["response_object"]
            msg = ro["choices"][0]["message"]
            content = (msg.get("content") or "")
            response = content.strip()
            prompts_by_engine[lab][i] = r["messages"][0]["content"]
            # published step-0 verdict (upstream VAL): sound_verifier_json when recorded; o1-preview omits it at step 0
            # for answers that were already correct (single step, act_correct True)
            svj = s0.get("sound_verifier_json")
            if svj is not None:
                pub_v, pub_src = bool(svj["validation_info"]["is_valid_plan"]), "sound_verifier_json"
            else:
                assert len(r["step_wise_dict"]) == 1 and r.get("act_correct") is True, (lab, i)
                pub_v, pub_src = True, "single_step_act_correct"
            chk[f"published_verdict_from_{pub_src}"] += 1
            # extractor 1: upstream port; must reproduce the published llm_plan
            lines = pb_pddl_lines(response)
            if "\n".join(lines) != s0.get("llm_plan"):
                sys.exit(f"REFUSE: pb_pddl does not reproduce the published llm_plan ({lab}, instance {i})")
            chk["pb_pddl_reproduces_llm_plan"] += 1
            tup = lines_to_tuples(lines)
            v_verb = val_verbatim(i, "\n".join(lines))
            if tup is None:
                chk["pb_pddl_lines_not_tuple_convertible"] += 1
                v_pb = v_verb
            else:
                v_pb = val_tuples(i, tup)
                chk["pb_pddl_tuple_vs_verbatim_agree"] += v_pb == v_verb
            # extractor 2: bwm_lenient
            plan2, inf2 = bwm_lenient(content)
            v_bw = val_tuples(i, plan2)
            chk[f"bwm_answer_source_{inf2['answer_source']}"] += 1
            chk["bwm_multi_block_answers"] += inf2["n_blocks_of_action_lines"] > 1
            chk["extractors_same_plan"] += (tup == plan2)
            chk["extractors_same_verdict"] += (v_pb == v_bw)
            chk["local_val_pb_pddl_agrees_with_published_verdict"] += (v_pb == pub_v)
            # applicability probe of the colour-based natural-language extractors in bwmn/planbench_extract.py
            nl1 = PX.text_to_plan_blocksworld(response)[0]
            nl2 = PX.extract_plan(content, PX.CANON)[0]
            chk["nl_text_to_plan_blocksworld_nonempty"] += bool(nl1)
            chk["nl_extract_plan_nonempty"] += bool(nl2)
            vec["pb_pddl"][i], vec["bwm_lenient"][i] = v_pb, v_bw
            rows[i] = {"published_verdict": pub_v, "published_verdict_source": pub_src, "pb_pddl_val": v_pb,
                       "pb_pddl_len": len(lines), "bwm_lenient_val": v_bw, "bwm_lenient_len": len(plan2),
                       "bwm_lenient_blocks": inf2["n_blocks_of_action_lines"], "finish_reason":
                           ro["choices"][0].get("finish_reason")}
            u = ro.get("usage") or {}
            tok["prompt"].append(u.get("prompt_tokens"))
            tok["completion"].append(u.get("completion_tokens"))
            tok["reasoning"].append((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            tok["time_s"].append(round(float(s0["time_taken"]), 1) if s0.get("time_taken") is not None else None)
        pub_vec[lab], pub_rows[lab], extractor_checks[lab] = vec, rows, dict(chk)
        published_meta[lab]["step0_finish_reason"] = dict(Counter(rows[i]["finish_reason"] for i in IDS))
        tokens[lab] = {k: summ(v) for k, v in tok.items()}
        if lab == "DeepSeek-R1":
            proc = [p + c for p, c in zip(tok["prompt"], tok["completion"])]
            fl = [2 * R1_ACTIVE_PARAMS * x for x in proc]
            tokens[lab]["ESTIMATED_dense_flops_per_instance"] = {
                "median": float(f"{st.median(fl):.3g}"), "mean": float(f"{st.mean(fl):.3g}"),
                "formula": "2 x 37e9 activated parameters x (prompt + completion tokens); dense matmuls only, no "
                           "attention term, no MoE routing; ESTIMATED, the provider's real serving compute is unknown"}

    # same instance, same prompt across engines (pairing check)
    pair_chk = Counter()
    for i in IDS:
        txt = problem_path(i).read_text()
        prob = "(define" + txt.split("(define")[1][:-1].strip() + "\n)"  # upstream's per_instance_output_pddl
        ps = [prompts_by_engine[lab][i] for lab in prompts_by_engine]
        pair_chk["prompt_identical_across_engines"] += len(set(ps)) == 1
        pair_chk["prompt_contains_local_problem"] += all(prob.strip() in p for p in ps)

    # ---------------- ours: re-VAL every stored plan, compare with the recorded verdict
    ours_vec, ours_rows, ours_prov, ours_cost = {}, {}, {}, {}
    for lab, (pp, sp, reg) in OURS.items():
        p = ROOT / pp
        recs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        by = {int(r["id"].split("-")[-1]): r for r in recs}
        assert sorted(by) == IDS and len(recs) == 110, (lab, "ids")
        vec, rows, agree, nb_ok = {}, {}, 0, 0
        for i in IDS:
            r = by[i]
            plan = [tuple(a.split()) for a in (r.get("plan") or [])]
            v = val_tuples(i, plan)
            agree += v == bool(r["val"])
            nb_ok += r.get("n_blocks") == n_blocks[i]
            vec[i] = bool(r["val"])
            rows[i] = {"val": bool(r["val"]), "fail": r.get("fail"), "verified": r.get("verified"),
                       "len": len(plan), "rounds": r.get("rounds"), "gen_tokens": r.get("gen_tokens")}
        if agree != 110:
            sys.exit(f"REFUSE: local VAL re-score disagrees with the recorded verdict for {lab} ({agree}/110)")
        if sum(vec.values()) != EXPECTED_OURS[lab]:
            sys.exit(f"REFUSE: {lab} solved {sum(vec.values())}, registered {EXPECTED_OURS[lab]}")
        ours_vec[lab], ours_rows[lab] = vec, rows
        ours_prov[lab] = {"plans": pp, "sha256": sha256(p), "registered": reg, "local_val_rescore_agrees": agree,
                          "n_blocks_field_agrees": nb_ok}
        s = json.loads((ROOT / sp).read_text(encoding="utf-8")).get("summary", {})
        ours_cost[lab] = {"proposer_tokens_per_instance_summary": s.get("proposer_tokens_per_instance"),
                          "proposer_gen_tokens": summ([rows[i]["gen_tokens"] for i in IDS]),
                          "f_queries_per_instance_summary": s.get("f_queries_per_instance")}

    # ---------------- statistics
    labs_pub = [PUBLISHED[e][0] for e in PUBLISHED]
    keys = [(o, pl) for o in OURS for pl in labs_pub]
    fam = {ex: family(ours_vec, {pl: pub_vec[pl][ex] for pl in labs_pub}, keys) for ex in ("pb_pddl", "bwm_lenient")}
    keys8 = [(o, pl) for o, pl in keys if pl != "o1-mini"]
    fam8 = family(ours_vec, {pl: pub_vec[pl]["pb_pddl"] for pl in labs_pub}, keys8)
    rates = {"bwm": {o: rate(ours_vec[o]) for o in OURS},
             "published_pb_pddl": {pl: rate(pub_vec[pl]["pb_pddl"]) for pl in labs_pub},
             "published_bwm_lenient": {pl: rate(pub_vec[pl]["bwm_lenient"]) for pl in labs_pub},
             "published_upstream_verdict": {pl: rate({i: pub_rows[pl][i]["published_verdict"] for i in IDS})
                                            for pl in labs_pub}}

    # audit (simulator-only, H2 leak/claims audit scratch) step-0 flags vs the local VAL
    audit = {}
    if AUDIT_STEP0.exists():
        a = json.loads(AUDIT_STEP0.read_text(encoding="utf-8"))
        for eng, (lab, _, _) in PUBLISHED.items():
            if eng in a:
                audit[lab] = {"audit_solved": sum(bool(a[eng][str(i)]) for i in IDS),
                              "agree_with_local_val_pb_pddl": sum(bool(a[eng][str(i)]) == pub_vec[lab]["pb_pddl"][i]
                                                                  for i in IDS)}
        audit["_source"] = {"path": rel(AUDIT_STEP0), "sha256": sha256(AUDIT_STEP0),
                            "note": "stdlib simulator over upstream llm_plan (research/audit_evidence/bwm_h2_scratch/"
                                    "up_rescore.py); the panel quoted H2 vs R1 +36/-7 and B2 vs R1 +30/-16"}

    # instances R1 solves but H2 (and B2 S-beam) do not
    r1 = pub_vec["DeepSeek-R1"]["pb_pddl"]
    def only_r1(o):
        return [{"instance": f"instance-{i}", "n_blocks": n_blocks[i], "optimal_len": opt_len[i],
                 "r1_plan_len": pub_rows["DeepSeek-R1"][i]["pb_pddl_len"], f"{o}_fail": ours_rows[o][i]["fail"],
                 f"{o}_verified": ours_rows[o][i]["verified"], f"{o}_rounds": ours_rows[o][i]["rounds"],
                 "b2_beam_val": ours_vec["B2 S-beam"][i], "r1_bwm_lenient_val": pub_vec["DeepSeek-R1"]["bwm_lenient"][i]}
                for i in IDS if r1[i] and not ours_vec[o][i]]
    r1_not_h2 = only_r1("H2 Pa_v3 S-beam")
    r1_not_b2 = only_r1("B2 S-beam")

    # strata (descriptive, post hoc)
    bins = {"6-10": [i for i in IDS if n_blocks[i] <= 10], "11-15": [i for i in IDS if n_blocks[i] >= 11]}
    strata = {}
    for bn, ids in bins.items():
        row = {"n": len(ids)}
        for o in OURS:
            row[o] = sum(ours_vec[o][i] for i in ids)
        for pl in labs_pub:
            row[pl] = sum(pub_vec[pl]["pb_pddl"][i] for i in ids)
        for o in ("H2 Pa_v3 S-beam", "B2 S-beam"):
            b = sum(ours_vec[o][i] and not r1[i] for i in ids)
            c = sum(r1[i] and not ours_vec[o][i] for i in ids)
            row[f"{o} vs DeepSeek-R1 discordant"] = {"bwm_only": b, "published_only": c,
                                                     "mcnemar_exact_p_unadjusted": ST.mcnemar_exact(b, c)}
        strata[bn] = row
    by_blocks = {}
    for nb in sorted(set(n_blocks.values())):
        ids = [i for i in IDS if n_blocks[i] == nb]
        by_blocks[str(nb)] = {"n": len(ids), **{o: sum(ours_vec[o][i] for i in ids) for o in OURS},
                              **{pl: sum(pub_vec[pl]["pb_pddl"][i] for i in ids) for pl in labs_pub}}

    # ---------------- per-instance file
    with open(OUT / "per_instance.jsonl", "w", encoding="utf-8") as fh:
        for i in IDS:
            row = {"instance": f"instance-{i}", "n_blocks": n_blocks[i], "optimal_len": opt_len[i],
                   "bwm": {o: ours_rows[o][i] for o in OURS},
                   "published": {pl: pub_rows[pl][i] for pl in labs_pub}}
            fh.write(json.dumps(row, sort_keys=False) + "\n")

    inputs = {"published": provenance, "bwm": ours_prov,
              "hard_instances": {"dir": rel(HARD / "generated"), "n": 110,
                                 "identical_to_upstream_blobs": sum(git_blob(problem_path(i)) == blob_of_instance(i)
                                                                    for i in IDS) if UPSTREAM_INSTANCE_BLOBS else None},
              "upstream_tree_listing": ({"path": rel(UPSTREAM_TREE), "sha256": sha256(UPSTREAM_TREE)}
                                        if UPSTREAM_TREE.exists() else None),
              "upstream_extractor_sources": {rel(p): {"sha256": sha256(p), "git_blob_sha1": git_blob(p)}
                                             for p in sorted((OUT / "upstream_src").glob("*.txt"))},
              "domain": {"path": rel(DOMAIN), "sha256": sha256(DOMAIN), "git_blob_sha1": git_blob(DOMAIN)},
              "info_json": {"path": rel(HARD / "info.json"), "sha256": sha256(HARD / "info.json")},
              "val_exe": {"path": rel(B.VAL_EXE), "sha256": sha256(B.VAL_EXE)},
              "code": {rel(Path(__file__)): sha256(Path(__file__)), "bwmn/blocks.py": sha256(ROOT / "bwmn/blocks.py"),
                       "bwmn/planbench_extract.py": sha256(ROOT / "bwmn/planbench_extract.py"),
                       "bwmn/stats.py": sha256(ROOT / "bwmn/stats.py")}}

    F = fam["pb_pddl"]

    def c(o, pl):
        v = F[f"{o} vs {pl}"]
        return (f"{v['bwm']} vs {v['published']}/110, paired +{v['bwm_only']}/-{v['published_only']}, exact p "
                f"{fmt_p(v['mcnemar_exact_p'])}, Holm p {fmt_p(v['holm_p'])}"
                + ("" if v["holm_sig_0.05"] else " (NOT significant after Holm)"))
    same_verdict = all(extractor_checks[pl]["extractors_same_verdict"] == 110 for pl in labs_pub)
    r1h2 = r1_not_h2
    reading = [
        f"H2 Pa_v3 S-beam vs DeepSeek-R1 step 0: {c('H2 Pa_v3 S-beam', 'DeepSeek-R1')}.",
        f"B2 S-beam vs DeepSeek-R1 step 0: {c('B2 S-beam', 'DeepSeek-R1')}; 'B2 above R1' is not established "
        "(as P-08's unpaired test).",
        f"Both S-beam systems vs o1-preview and o1-mini step 0: Holm p <= "
        f"{fmt_p(max(F[f'{o} vs {pl}']['holm_p'] for o in ('B2 S-beam', 'H2 Pa_v3 S-beam') for pl in ('o1-preview', 'o1-mini')))}.",
        f"The proposers alone are below R1: H1 {c('H1 Pa_v3 S-direct', 'DeepSeek-R1')}; B2 S-direct "
        f"{c('B2 S-direct', 'DeepSeek-R1')}.",
        f"vs o1-preview: H1 {c('H1 Pa_v3 S-direct', 'o1-preview')}; B2 S-direct {c('B2 S-direct', 'o1-preview')}. "
        "P-14 correction (b)'s unpaired 'H1 below o1-preview (Fisher p = 0.033)' becomes paired p "
        f"{fmt_p(F['H1 Pa_v3 S-direct vs o1-preview']['mcnemar_exact_p'])} unadjusted, "
        f"{fmt_p(F['H1 Pa_v3 S-direct vs o1-preview']['holm_p'])} after Holm (m = 12).",
        f"R1 solves {len(r1h2)} instances H2 does not ({', '.join(x['instance'] for x in r1h2)}); all have >= "
        f"{min(x['n_blocks'] for x in r1h2)} blocks; H2's outcome there: "
        f"{dict(Counter(x['H2 Pa_v3 S-beam_fail'] for x in r1h2))}, none verified "
        f"({sum(bool(x['H2 Pa_v3 S-beam_verified']) for x in r1h2)} verified); R1's plan is optimal-length on "
        f"{sum(x['r1_plan_len'] == x['optimal_len'] for x in r1h2)} of them.",
        "Extraction does not move any verdict: upstream's own PDDL extractor and the lenient bwm extractor give the same "
        f"VAL verdict on every published answer ({'330/330' if same_verdict else 'NOT all; see checks'}), and the local "
        "VAL reproduces upstream's recorded step-0 verdict on 330/330.",
    ]
    h2r1, b2r1 = F["H2 Pa_v3 S-beam vs DeepSeek-R1"], F["B2 S-beam vs DeepSeek-R1"]
    proposed_wording = (
        "POST HOC, different regime: on the same 110 PlanBench Blocksworld-hard instances, re-scored by the same local "
        f"VAL, the frozen BWM S-beam with the self-distilled proposer (H2) solved {h2r1['bwm']} and DeepSeek-R1's "
        f"zero-shot direct answer (published per-instance file, step 0) {h2r1['published']} (paired "
        f"+{h2r1['bwm_only']}/-{h2r1['published_only']}, exact McNemar Holm p {fmt_p(h2r1['holm_p'])}, 12 contrasts); "
        f"the original B2 S-beam solved {b2r1['bwm']} (+{b2r1['bwm_only']}/-{b2r1['published_only']}, Holm p "
        f"{fmt_p(b2r1['holm_p'])}, not significant). The regimes differ: R1 is a general model with no domain "
        "training, no learned simulator and no search; BWM is trained on generated Blocksworld problems.")

    report = {
        "label": LABEL,
        "reading": reading,
        "proposed_wording": proposed_wording,
        "decision": "D-75.3 (user decision 2026-10-11); closes P-14 correction (d)",
        "status": "POST HOC: not pre-registered; written after every BWM hard result was known. Different regime: "
                  "the published systems are general zero-shot reasoning models given PDDL text, one answer per "
                  "instance, no domain training, no learned simulator, no search; the BWM systems are a 0.6B core "
                  "with LoRA adapters trained on generated Blocksworld problems, planning without execution feedback. "
                  "Never to be worded 'beats' / 'outperforms o1/R1 at planning'.",
        "hard_set_exposure": "none added: stored outputs only, no model run on the hard set",
        "regime": {
            "published": "LLMs-Planning results_backprompting/blocksworld_hard, task zero_shot_pddl with the VAL "
                         "backprompting loop; ONLY step 0 (first answer, before any verifier feedback) is used. "
                         "Prompt: PDDL domain + problem, 'Provide your answer as a sequence of actions in PDDL format', "
                         "no examples; o1 and R1 called without a temperature argument (API defaults).",
            "bwm": {"B2 S-beam": "frozen B2: learned F + beam search over Pa_v2 proposals (P-08/P-10)",
                    "H2 Pa_v3 S-beam": "frozen B2 search with the self-distilled proposer Pa_v3 (P-14)",
                    "H1 Pa_v3 S-direct": "Pa_v3 proposer alone, greedy, M 256, no F (P-13)",
                    "B2 S-direct": "Pa_v2 proposer alone, greedy, M 256, no F (P-08/P-10)"}},
        "inputs": inputs,
        "pairing_checks": {**dict(pair_chk), "n": 110},
        "published_meta": published_meta,
        "extractors": {
            "pb_pddl": "PRIMARY. Port of upstream llm_utils.save_gpt3_response (fc638a1a; blob 62f36232), the extractor "
                       "back_prompting_parallel.get_repeated_verification_pddl uses for PDDL tasks; reproduces every "
                       "published step-0 llm_plan; scored by bwmn.blocks.val_check and verbatim.",
            "bwm_lenient": "SENSITIVITY. bwmn/planbench_extract.py helpers + PDDL/call-notation action regex; list "
                           "markers stripped; all actions per line; last block of action lines; no repair.",
            "nl_probe": "bwmn/planbench_extract.py extract_plan and text_to_plan_blocksworld (colour-based natural-"
                        "language parsers) are not applicable to PDDL-letter answers; probe counts below.",
            "checks": extractor_checks,
            "val_calls": dict(N_VAL_CALLS)},
        "rates": rates,
        "family_primary_pb_pddl_holm_m12": fam["pb_pddl"],
        "family_sensitivity_bwm_lenient_holm_m12": fam["bwm_lenient"],
        "secondary_holm_m8_r1_o1preview_only_pb_pddl": {k: {"mcnemar_exact_p": v["mcnemar_exact_p"],
                                                            "holm_p": v["holm_p"]} for k, v in fam8.items()},
        "audit_simulator_confirmation": audit,
        "r1_solves_h2_not": r1_not_h2,
        "r1_solves_b2_beam_not": [x["instance"] for x in r1_not_b2],
        "strata_blocks_descriptive_post_hoc": strata,
        "by_block_count": by_blocks,
        "compute_report_only": {
            "published_tokens_step0": tokens,
            "bwm": ours_cost,
            "note": "Tokens are not comparable units across a 0.6B dense model and API models. R1 FLOPs are an "
                    "ESTIMATE (dense-only, activated parameters). o1 FLOPs are undisclosed: tokens only. BWM FLOPs "
                    "(proposer + F queries, with the attention term) are left to the item-3 ledger; no FLOPs ratio "
                    "and no iso-FLOP claim is made here."},
    }
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (OUT / "REPORT.md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"rates": {k: {kk: vv["solved"] for kk, vv in v.items()} for k, v in rates.items()},
                      "primary": {k: [v["bwm_only"], v["published_only"], fmt_p(v["mcnemar_exact_p"]),
                                      fmt_p(v["holm_p"])] for k, v in fam["pb_pddl"].items()},
                      "checks": extractor_checks, "pairing": dict(pair_chk), "val_calls": dict(N_VAL_CALLS),
                      "r1_not_h2": [x["instance"] for x in r1_not_h2], "audit": audit}, indent=1))


def render_md(r: dict) -> str:
    L = []
    w = L.append
    w(f"# Blocksworld-hard: BWM vs published per-instance results ({r['label']})")
    w("")
    w(f"**{r['label']}.** {r['status']}")
    w("")
    w(f"- Decision: {r['decision']}. Hard-set exposure: {r['hard_set_exposure']}.")
    w(f"- Published side: {r['regime']['published']}")
    w("- BWM side: " + "; ".join(f"{k}: {v}" for k, v in r["regime"]["bwm"].items()) + ".")
    pc = r["pairing_checks"]
    w(f"- Pairing: the step-0 prompt is identical across the three published files on {pc['prompt_identical_across_engines']}"
      f"/110 instances and contains the local problem file's text on {pc['prompt_contains_local_problem']}/110; "
      f"the local instances are byte-identical to upstream's on {r['inputs']['hard_instances']['identical_to_upstream_blobs']}/110.")
    w("")
    w("## Success on the same 110 instances (local VAL)")
    w("")
    w("| System | Regime | Solved | Rate | Clopper-Pearson 95% |")
    w("|---|---|---|---|---|")
    for o, v in r["rates"]["bwm"].items():
        w(f"| {o} | BWM, frozen | {v['solved']}/110 | {v['rate']:.3f} | {v['clopper_pearson_95'][0]:.3f}-{v['clopper_pearson_95'][1]:.3f} |")
    for pl, v in r["rates"]["published_pb_pddl"].items():
        v2 = r["rates"]["published_bwm_lenient"][pl]
        v3 = r["rates"]["published_upstream_verdict"][pl]
        w(f"| {pl} step 0 (pb_pddl) | published, zero-shot | {v['solved']}/110 | {v['rate']:.3f} | "
          f"{v['clopper_pearson_95'][0]:.3f}-{v['clopper_pearson_95'][1]:.3f} |")
        w(f"| {pl} step 0 (bwm_lenient) | published, zero-shot | {v2['solved']}/110 | {v2['rate']:.3f} | "
          f"{v2['clopper_pearson_95'][0]:.3f}-{v2['clopper_pearson_95'][1]:.3f} |")
    w("")
    w("Upstream's own recorded step-0 verdicts: " + "; ".join(
        f"{pl} {v['solved']}/110" for pl, v in r["rates"]["published_upstream_verdict"].items()) + ".")
    w("Verifier-in-loop final results (later backprompting steps; NOT compared, a different regime again): " + "; ".join(
        f"{pl} {m['final_act_correct_verifier_loop_NOT_COMPARED']}/110 (steps recorded per instance: "
        f"{m['steps_recorded']})" for pl, m in r["published_meta"].items()) + ". The R1 file records only step 0 for "
      "every instance, so its 'final' equals its step 0; o1-mini's 11/110 is the 10% LRM-Modulo figure of P-14 "
      "correction (a), its direct step 0 is 1/110.")
    w("Step-0 finish reasons: " + "; ".join(f"{pl} {m['step0_finish_reason']}" for pl, m in r["published_meta"].items())
      + " (no truncated published answer).")
    w("")
    w("## Reading (POST HOC, DIFFERENT REGIME)")
    w("")
    for line in r["reading"]:
        w(f"- {line}")
    w("")
    w(f"## Paired exact McNemar, Holm over 12 contrasts ({r['label']})")
    w("")
    w("Primary extractor pb_pddl (upstream's own). bwm_only / published_only are the discordant pairs.")
    w("")
    w("| Contrast | BWM | Published | Both | BWM only | Published only | Neither | Diff (pp) | p (exact) | Holm p | Lenient: BWM only / Pub only, Holm p |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in r["family_primary_pb_pddl_holm_m12"].items():
        s = r["family_sensitivity_bwm_lenient_holm_m12"][k]
        w(f"| {k} | {v['bwm']} | {v['published']} | {v['both']} | {v['bwm_only']} | {v['published_only']} | "
          f"{v['neither']} | {v['diff_pp']:+.1f} | {fmt_p(v['mcnemar_exact_p'])} | {fmt_p(v['holm_p'])} | "
          f"{s['bwm_only']} / {s['published_only']}, {fmt_p(s['holm_p'])} |")
    w("")
    w("Secondary (panel item 4 family: R1 and o1-preview only, Holm m = 8, pb_pddl): " + "; ".join(
        f"{k} Holm {fmt_p(v['holm_p'])}" for k, v in r["secondary_holm_m8_r1_o1preview_only_pb_pddl"].items()) + ".")
    w("")
    w("## Extractors and VAL")
    w("")
    for k in ("pb_pddl", "bwm_lenient", "nl_probe"):
        w(f"- **{k}**: {r['extractors'][k]}")
    w("")
    w("| Check | " + " | ".join(r["extractors"]["checks"]) + " |")
    w("|---|" + "---|" * len(r["extractors"]["checks"]))
    allk = sorted({k for v in r["extractors"]["checks"].values() for k in v})
    for k in allk:
        w(f"| {k} | " + " | ".join(str(v.get(k, 0)) for v in r["extractors"]["checks"].values()) + " |")
    w("")
    w(f"VAL calls: {r['extractors']['val_calls']} (cached per instance and plan). BWM plans re-scored with the same "
      f"val_check: " + "; ".join(f"{o} {p['local_val_rescore_agrees']}/110 agree with the recorded verdict"
                                 for o, p in r["inputs"]["bwm"].items()) + ".")
    if r["audit_simulator_confirmation"]:
        a = r["audit_simulator_confirmation"]
        w("")
        w("Earlier audit (simulator over upstream llm_plan, H2 leak/claims audit): " + "; ".join(
            f"{k} {v['audit_solved']}/110, per-instance agreement with local VAL {v['agree_with_local_val_pb_pddl']}/110"
            for k, v in a.items() if not k.startswith("_")) + ".")
    w("")
    w(f"## Instances DeepSeek-R1 solves (step 0) but H2 does not ({len(r['r1_solves_h2_not'])})")
    w("")
    w("| Instance | Blocks | Optimal len | R1 plan len | H2 fail | H2 verified | H2 rounds | B2 S-beam | R1 under bwm_lenient |")
    w("|---|---|---|---|---|---|---|---|---|")
    for x in r["r1_solves_h2_not"]:
        w(f"| {x['instance']} | {x['n_blocks']} | {x['optimal_len']} | {x['r1_plan_len']} | "
          f"{x['H2 Pa_v3 S-beam_fail']} | {x['H2 Pa_v3 S-beam_verified']} | {x['H2 Pa_v3 S-beam_rounds']} | "
          f"{'solved' if x['b2_beam_val'] else 'failed'} | {'valid' if x['r1_bwm_lenient_val'] else 'invalid'} |")
    w("")
    w(f"R1 solves but B2 S-beam does not ({len(r['r1_solves_b2_beam_not'])}): " + ", ".join(r["r1_solves_b2_beam_not"]) + ".")
    w("")
    w("## By block count (descriptive, post hoc)")
    w("")
    cols = list(next(iter(r["by_block_count"].values())).keys())
    w("| Blocks | " + " | ".join(cols) + " |")
    w("|---|" + "---|" * len(cols))
    for nb, row in r["by_block_count"].items():
        w(f"| {nb} | " + " | ".join(str(row[c]) for c in cols) + " |")
    for bn, row in r["strata_blocks_descriptive_post_hoc"].items():
        d1 = row["H2 Pa_v3 S-beam vs DeepSeek-R1 discordant"]
        d2 = row["B2 S-beam vs DeepSeek-R1 discordant"]
        w(f"- {bn} blocks (n {row['n']}): H2 {row['H2 Pa_v3 S-beam']}, B2 S-beam {row['B2 S-beam']}, R1 "
          f"{row['DeepSeek-R1']}, o1-preview {row['o1-preview']}, o1-mini {row['o1-mini']}. H2 vs R1 +{d1['bwm_only']}/"
          f"-{d1['published_only']} (unadjusted p {fmt_p(d1['mcnemar_exact_p_unadjusted'])}); B2 S-beam vs R1 "
          f"+{d2['bwm_only']}/-{d2['published_only']} (unadjusted p {fmt_p(d2['mcnemar_exact_p_unadjusted'])}).")
    w("")
    w("## Compute (report-only; tokens are not comparable units)")
    w("")
    for pl, t in r["compute_report_only"]["published_tokens_step0"].items():
        parts = [f"{k} median {v['median']} (mean {v['mean']})" for k, v in t.items() if isinstance(v, dict) and "median" in v and k != "ESTIMATED_dense_flops_per_instance"]
        w(f"- {pl} step 0: " + "; ".join(parts) + ".")
        if "ESTIMATED_dense_flops_per_instance" in t:
            e = t["ESTIMATED_dense_flops_per_instance"]
            w(f"  - ESTIMATED FLOPs per instance: median {e['median']:.2g}, mean {e['mean']:.2g} ({e['formula']}).")
    for o, c in r["compute_report_only"]["bwm"].items():
        g = c["proposer_gen_tokens"]
        w(f"- {o}: proposer generated tokens median {g.get('median')} (mean {g.get('mean')}); F queries per instance "
          f"{c['f_queries_per_instance_summary']}.")
    w(f"- {r['compute_report_only']['note']}")
    w("")
    w("## Wording")
    w("")
    w(f"- Allowed (proposed; the user decides): \"{r['proposed_wording']}\" Always with the regime sentence and the "
      "extractor note.")
    w("- Not allowed: 'beats' / 'outperforms o1/R1 at planning', any iso-compute claim, or citing the verifier-loop "
      "numbers as zero-shot.")
    w("")
    w("## Inputs")
    w("")
    for pl, p in r["inputs"]["published"].items():
        w(f"- {pl}: `{p['path']}` sha256 {p['sha256']} (git blob {p['git_blob_sha1']}; {p['upstream']}).")
    for o, p in r["inputs"]["bwm"].items():
        w(f"- {o}: `{p['plans']}` sha256 {p['sha256']} ({p['registered']}).")
    w(f"- Domain `{r['inputs']['domain']['path']}` sha256 {r['inputs']['domain']['sha256']}; VAL "
      f"`{r['inputs']['val_exe']['path']}` sha256 {r['inputs']['val_exe']['sha256']}.")
    for k, v in r["inputs"]["code"].items():
        w(f"- Code `{k}` sha256 {v}.")
    w("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
