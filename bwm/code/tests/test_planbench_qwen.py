"""Tests for the same-harness Qwen comparator (D-75.4): scripts/planbench_qwen.py, the Blocksworld-hard / PDDL part of
bwmn/planbench_extract.py and bwmn/flops.py. CPU only; no model is loaded and torch/transformers are never imported
(asserted). Every CPU-lane test runs in a temporary directory (RINEYA_LANE_DIR), never on the real %TEMP% lane files.

  python -B -m pytest tests/test_planbench_qwen.py -q -p no:cacheprovider

Read-only inputs (tests that need a missing one are skipped): the sha-pinned upstream Blocksworld-hard result files
(bench/planbench/results/blocksworld_hard or research/audit_evidence/bwm_h2_scratch/dl_upstream_hard), the local VAL
binary, the H2/B2 run artifacts (runs/BW_B3_HARD/h2_beam, runs/BW_TEST_B2/hard_beam), the Titan-Sparse and Qwen3-4B
tokenizer/config/safetensors headers, and RCX's rcx/lane.py (copied to tmp_path and imported from there).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_HEAVY_BEFORE = {m for m in ("torch", "transformers") if m in sys.modules}

from bwmn import blocks as B  # noqa: E402
from bwmn import flops as FL  # noqa: E402
from bwmn import planbench_extract as X  # noqa: E402

_spec = importlib.util.spec_from_file_location("planbench_qwen_under_test", ROOT / "scripts" / "planbench_qwen.py")
PQ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PQ)


def _has_upstream() -> bool:
    try:
        return all(X.hard_result_path(m).exists() for m in X.HARD_MODELS)
    except (FileNotFoundError, ValueError):
        return False


HAS_UPSTREAM = _has_upstream()
HAS_VAL = B.VAL_EXE.exists()
QWEN_SNAPS = sorted(p for p in PQ.QWEN3_4B_SNAPSHOTS.glob("*") if (p / "config.json").exists()) \
    if PQ.QWEN3_4B_SNAPSHOTS.exists() else []
HAS_QWEN = len(QWEN_SNAPS) == 1
TITAN = PQ.MODEL_DIRS["titan-sparse"]
HAS_TITAN = (TITAN / "tokenizer.json").exists() and (TITAN / "config.json").exists()
H2 = PQ.ISO_REGISTRY["H2_hard"]
HAS_H2 = (ROOT / H2["run"] / "plans.jsonl").exists() and (ROOT / H2["freeze"]).exists()
RCX_LANE = Path(r"C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/rcx/lane.py")
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "BWM_DEVICE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "RINEYA_LANE_DIR")

needs_upstream = pytest.mark.skipif(not HAS_UPSTREAM, reason="upstream hard result files not present")
needs_val = pytest.mark.skipif(not HAS_VAL, reason="local VAL binary not present")
needs_qwen = pytest.mark.skipif(not HAS_QWEN, reason="Qwen3-4B snapshot not present")


@pytest.fixture(autouse=True)
def lane_tmp(tmp_path, monkeypatch):
    """Every test: lane files in tmp_path/lane; the process environment restored afterwards (main() sets CPU-only
    variables)."""
    for k in ENV_KEYS:
        if k in os.environ:
            monkeypatch.setenv(k, os.environ[k])
        else:
            monkeypatch.delenv(k, raising=False)
    d = tmp_path / "lane"
    d.mkdir()
    monkeypatch.setenv("RINEYA_LANE_DIR", str(d))
    return d


@pytest.fixture(scope="module")
def hard():
    if not HAS_UPSTREAM:
        pytest.skip("upstream hard result files not present")
    return X.load_hard_instances()


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-B", "-c", "pass"])
    p.wait()
    if PQ.pid_alive(p.pid):
        pytest.skip("pid reused immediately")
    return p.pid


# ----------------------------------------------------------------------------- module hygiene

def test_import_has_no_env_side_effects_and_no_torch():
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "BWM_DEVICE")}
    code = ("import importlib.util, os, sys; s = importlib.util.spec_from_file_location('m', r'%s'); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(os.environ.get('CUDA_VISIBLE_DEVICES'), os.environ.get('BWM_DEVICE'), 'torch' in sys.modules, "
            "'transformers' in sys.modules)") % (ROOT / "scripts" / "planbench_qwen.py")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=120,
                         cwd=str(ROOT)).stdout.split()
    assert out == ["None", "None", "False", "False"], out


def test_model_dirs_match_lm_models_source():
    """MODEL_DIRS replaces importing bwmn.lm (torch) for paths; it must equal lm.MODELS (read from the source)."""
    src = (ROOT / "bwmn" / "lm.py").read_text(encoding="utf-8")
    assert '"titan-sparse": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"' in src
    assert PQ.MODEL_DIRS["titan-sparse"] == PQ.CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"
    assert 'Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")' in src
    assert str(PQ.MODEL_DIRS["qwen3-0.6b"]).replace("\\", "/").split("snapshots/")[1] in src


# ----------------------------------------------------------------------------- extractors

@pytest.mark.parametrize("txt,want", PQ.UNIT_CASES)
def test_nl_extractor_unit_cases(txt, want):
    got, _ = X.extract_plan(txt, {"red": "a", "blue": "b", "orange": "c", "yellow": "d"})
    assert got == want


@pytest.mark.parametrize("txt", [
    # o1-preview generated_basic/instance-243 style: bold step title (no closing punctuation) + Markdown hard break
    "1. **Unstack the blue block from on top of the red block**\\\n   *Action:* Unstack the blue block from the red "
    "block.\\\n   *Result:* Holding blue.\n\n2. **Put down the blue block**\\\n   *Action:* Put down the blue block.",
    # o1-preview instance-408 style: the title's period is hidden by the trailing '\'
    "1. **Unstack the blue block from on top of the red block.**\\\n   *Action:* Unstack blue block from red block.\n"
    "2. **Put down the blue block onto the table.**\\\n   *Action:* Put down blue block.",
    # deepseek-r1 generated_basic/instance-2 style: bold title, two trailing spaces, indented italic commentary
    "1. **unstack blue from red**  \n   *(Blue is clear. Unstack blue from red. Now holding blue.)*  \n"
    "2. **put down blue**  \n   *(Put down blue on the table.)*",
])
def test_nl_bold_title_commentary_is_not_a_wrapped_line(txt):
    """Verifier fix (track D review): an indented unmarked line under a one-line bold step title is commentary on that
    step, not a wrapped continuation of it; before the fix every action was counted twice (11 of the published PB-600
    responses: R1 8, o1-preview 2, gemini-2.0-flash-thinking 1; all now match the published VAL verdict)."""
    got, _ = X.extract_plan(txt, {"red": "a", "blue": "b", "orange": "c", "yellow": "d"})
    assert got == [("unstack", "b", "a"), ("put-down", "b")]


@pytest.mark.parametrize("txt,objs,want", PQ.PDDL_CASES)
def test_pddl_extractor_unit_cases(txt, objs, want):
    got, _ = X.extract_pddl_plan(txt, list(objs))
    assert got == want


def test_pddl_extractor_info():
    plan, info = X.extract_pddl_plan("(unstack d b) ; d clear\n(put-down d table)\n(pickup a)\n(stack a z)", "abcd")
    assert plan == [("unstack", "d", "b"), ("pick-up", "a"), ("stack", "a", "z")]
    assert info["dropped_arity"] == 1 and info["normalized_names"] == 1 and info["unknown_objects"] == ["z"]
    assert info["issues"] and "put-down d table" in info["issues"][0]
    assert X.extract_pddl_plan("<think>\n(unstack d b)", "abcd", thinking=True) == (
        [], X.extract_pddl_plan("<think>\n(unstack d b)", "abcd", thinking=True)[1])
    assert X.extract_pddl_plan("(unstack d b)", "abcd", thinking=True)[0] == []  # thinking mode, no </think>
    assert X.extract_pddl_plan("x</think>\n(unstack d b)", "abcd", thinking=True)[0] == [("unstack", "d", "b")]
    # a bare line naming non-objects is prose, not an action
    assert X.extract_pddl_plan("stack them\nunstack d b", "abcd")[0] == [("unstack", "d", "b")]
    assert X.extract_pddl_plan("unstack x y", "abcd")[0] == []


def test_save_gpt3_response_port_semantics():
    txt = "(unstack d b)\n  (put-down d)\n(PICK-UP A) \n() \nno actions\n(no actions)\n1. (stack a b)\n(stack a d)"
    assert X.save_gpt3_response_text(txt) == "(unstack d b)\n(pick-up a)\n(stack a d)"
    assert X.upstream_pddl_text("  \n(unstack d b)\n") == "(unstack d b)"  # stripped first, as upstream
    assert X.upstream_pddl_text("<think>(a b)</think>\n(unstack d b)") == "(unstack d b)"
    assert X.pddl_text_actions("(unstack d b)\n(pick-up a)") == [("unstack", "d", "b"), ("pick-up", "a")]


# ----------------------------------------------------------------------------- Blocksworld-hard instances

def test_hard_instances(hard):
    assert [i["id"] for i in hard] == [f"hard/instance-{n}" for n in range(1, 111)]
    assert {i["n_blocks"] for i in hard} == set(range(6, 16))
    assert min(i["opt_len"] for i in hard) >= 20 and max(i["opt_len"] for i in hard) <= 40
    dom = X.HARD_DOMAIN.read_text()
    for i in hard:
        assert i["fmt"] == "pddl" and i["domain"] == X.HARD_DOMAIN
        assert i["query"] == X.hard_query(Path(i["path"]).read_text(), dom)
        assert i["query"].startswith(X.HARD_QUERY_HEAD) and i["query"].endswith("\n\n[PLAN]")
    # the same problem the planner side uses (scripts/blocks_eval.py hard loader: parse_problem on the same file)
    p = X.HARD_DIR / "generated" / "instance-7.pddl"
    assert B.parse_problem(p) == (hard[6]["objs"], hard[6]["init"], hard[6]["goal"])


@needs_upstream
def test_hard_queries_identical_in_all_upstream_files(hard):
    q = {i["instance_id"]: i["query"] for i in hard}
    for m in X.HARD_MODELS:
        for r in X.load_hard_results(m)["instances"]:
            assert r["messages"][0] == {"role": "user", "content": q[int(r["instance_id"])]} or (
                r["messages"][0]["role"] == "user" and r["messages"][0]["content"] == q[int(r["instance_id"])])


@needs_upstream
def test_upstream_port_reproduces_published_llm_plan_330():
    n = same = 0
    for m in X.HARD_MODELS:
        for r in X.load_hard_results(m)["instances"]:
            content, llm_plan, verdict = X.hard_step0(r)
            n += 1
            same += X.upstream_pddl_text(content) == llm_plan
            assert verdict is not None
    assert (n, same) == (330, 330)


@needs_upstream
def test_published_verdict_counts():
    got = {m: sum(bool(X.hard_step0(r)[2]) for r in X.load_hard_results(m)["instances"]) for m in X.HARD_MODELS}
    assert got == {"deepseek-r1-api": 59, "o1-preview_chat": 26, "o1-mini_chat": 1}


@needs_upstream
def test_hard_result_sha_pinning(tmp_path, monkeypatch):
    m = "o1-mini_chat"
    src = X.hard_result_path(m)
    # a modified vendored copy is refused, never skipped over
    bad = tmp_path / "res" / "blocksworld_hard" / m / X.HARD_TASK
    bad.parent.mkdir(parents=True)
    bad.write_bytes(src.read_bytes() + b" ")
    monkeypatch.setattr(X, "RES", tmp_path / "res")
    with pytest.raises(ValueError, match="sha256"):
        X.hard_result_path(m)
    # no vendored copy: the audit-evidence copy is used if its sha matches
    shutil.rmtree(tmp_path / "res")
    if (X.HARD_AUDIT_DIR / f"{m}.json").exists():
        assert X.hard_result_path(m) == X.HARD_AUDIT_DIR / f"{m}.json"
    monkeypatch.setattr(X, "HARD_AUDIT_DIR", tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError):
        X.hard_result_path(m)


@needs_upstream
@needs_val
def test_local_val_on_upstream_text_matches_published_verdicts(hard):
    by = {i["instance_id"]: i for i in hard}
    rng = random.Random(20261011)
    checked = 0
    for m in X.HARD_MODELS:
        recs = X.load_hard_results(m)["instances"]
        pick = [r for r in recs if X.hard_step0(r)[2]][:4] + rng.sample(recs, 6)
        for r in pick:
            content, llm_plan, verdict = X.hard_step0(r)
            inst = by[int(r["instance_id"])]
            assert X.val_check_text(inst["path"], X.upstream_pddl_text(content), X.HARD_DOMAIN) == verdict
            prim, _ = X.extract_pddl_plan(content, inst["objs"])
            v = bool(B.val_check(inst["path"], prim, X.HARD_DOMAIN)) if prim else False
            assert v == verdict
            assert (B.execute(inst["init"], inst["goal"], prim)["success"] if prim else False) == v
            checked += 1
    assert checked == (4 + 6) + (4 + 6) + (1 + 6)  # o1-mini has a single valid step-0 plan


@needs_val
def test_score_pddl_two_scorers(hard):
    inst = hard[0]
    ref = X.hard_info()["instance-1.pddl"]["plan"].strip()
    good = "<think>\nplan it\n</think>\n\n```\n" + ref + "\n```"
    s = PQ.score_variants(inst, good, inst["cmap"], True)
    assert s["val_valid"] and s["sim_valid"] and s["pb_val_valid"] and s["pb_md_val_valid"] is None
    numbered = "\n".join(f"{k + 1}. {ln}" for k, ln in enumerate(ref.split("\n")))
    s = PQ.score_variants(inst, numbered, inst["cmap"], False)
    assert s["val_valid"] and not s["pb_val_valid"] and s["pb_plan"] == []  # upstream reads only lines starting '('
    broken = "\n".join(ref.split("\n")[1:])
    s = PQ.score_variants(inst, broken, inst["cmap"], False)
    assert not s["val_valid"] and not s["sim_valid"] and not s["pb_val_valid"]


def test_systematic_subset(hard):
    sub, table = X.systematic_subset(hard, 40)
    assert len(sub) == 40 and PQ.ids_sha(sub) == PQ.HARD_SUBSET_40_SHA
    assert X.systematic_subset(hard, 40)[0] == sub  # deterministic, no seed
    assert set(table) == set(range(6, 16)) and sum(d for _, d in table.values()) == 40
    for size, drawn in table.values():
        assert abs(drawn - size * 40 / 110) < 1.5
    assert [i["id"] for i in X.systematic_subset(hard, 110)[0]] == [i["id"] for i in hard]
    with pytest.raises(ValueError):
        X.systematic_subset(hard, 111)


# ----------------------------------------------------------------------------- FLOPs

FP = {"n_nonemb": 7, "d_model": 3, "vocab": 5, "n_layers": 2, "n_heads": 2, "head_dim": 4}


def _brute_seq(fp, n_tokens, start=0):
    a = 4 * fp["n_layers"] * fp["n_heads"] * fp["head_dim"]
    return sum(2 * fp["n_nonemb"] + a * (t + 1) for t in range(start, start + n_tokens))


def test_flops_closed_form_vs_per_token_sum():
    rng = random.Random(0)
    for _ in range(50):
        p, f, g, r = rng.randint(1, 60), rng.randint(0, 5), rng.randint(0, 60), rng.choice([0, 0, rng.randint(1, 70)])
        want = _brute_seq(FP, p + f + g) + 2 * FP["d_model"] * FP["vocab"] * g + _brute_seq(FP, r)
        assert FL.flops_breakdown(FP, p, g, f, r)["total"] == want == PQ.flops(FP, p, f, g, r)


def test_shared_prefix_breakdown():
    rng = random.Random(1)
    for _ in range(30):
        P, gens = rng.randint(1, 50), [rng.randint(1, 30) for _ in range(rng.randint(1, 8))]
        want = _brute_seq(FP, P) + sum(_brute_seq(FP, g, start=P) for g in gens) \
            + 2 * FP["d_model"] * FP["vocab"] * sum(gens)
        assert FL.shared_prefix_breakdown(FP, P, gens)["total"] == want
    assert FL.shared_prefix_breakdown(FP, 40, [25]) == FL.flops_breakdown(FP, 40, 25)


def test_beam_proposer_pairs_bound():
    rng = random.Random(2)
    for _ in range(200):
        cap = rng.randint(1, 80)
        gens = [rng.randint(1, cap) for _ in range(rng.randint(1, 30))]
        assert sum(g * (g + 1) // 2 for g in gens) <= FL.beam_proposer_pairs_bound(sum(gens), cap)
    assert FL.beam_proposer_pairs_bound(160, 80) == 2 * 80 * 81 // 2


def test_max_total_tokens_inverse():
    rng = random.Random(3)
    for _ in range(100):
        p, f = rng.randint(1, 200), rng.randint(0, 4)
        cap = rng.uniform(FL.flops_breakdown(FP, p, 0, f)["total"], 10 * FL.flops_breakdown(FP, p, 500, f)["total"])
        T = FL.max_total_tokens(FP, p, cap, f)
        assert FL.flops_breakdown(FP, p, T - p - f, f)["total"] <= cap < FL.flops_breakdown(FP, p, T + 1 - p - f, f)[
            "total"]
    assert FL.max_total_tokens(FP, 100, FL.flops_breakdown(FP, 100, 0)["total"] - 1) is None


def test_iso_caps_bound_every_completion():
    fp = {"n_nonemb": 3_633_511_936, "d_model": 2560, "vocab": 151936, "n_layers": 36, "n_heads": 32,
          "head_dim": 128}
    cap, P, forced, A = float(H2["flops_per_instance"]), 601, 3, 1024
    c = PQ.iso_caps(fp, P, cap, forced, A)
    T, Bt = c["total_cap"], c["think_budget"]
    assert T == P + forced + A + Bt
    # budget forced: thinking = B, forced tokens (no LM-head rows), answer up to the reserve
    forced_cost = FL.flops_breakdown(fp, P, Bt + A, forced)["total"]
    # natural close at any t < B: every token after the prompt is generated, the answer uses the rest of T
    natural_cost = FL.flops_breakdown(fp, P, T - P, 0)["total"]
    assert forced_cost < natural_cost <= cap
    assert FL.flops_breakdown(fp, P, T + 1 - P, 0)["total"] > cap  # T is maximal
    # generate_one's phase-2 cap: total_cap - (prompt + thinking [+ forced]) keeps every completion within T tokens
    for t, f in ((Bt, forced), (0, 0), (Bt - 1, 0)):
        answer_cap = T - (P + t + f)
        assert answer_cap >= (A if f else 1) and P + t + f + answer_cap == T
    with pytest.raises(ValueError):
        PQ.iso_caps(fp, P, cap / 20, forced, A)


@needs_qwen
def test_qwen3_4b_param_count_from_config_equals_safetensors_headers():
    p = QWEN_SNAPS[0]
    assert FL.flop_params(p / "config.json")["n_nonemb"] == FL.safetensors_nonemb(p) == 3_633_511_936


@pytest.mark.skipif(not (HAS_H2 and HAS_TITAN), reason="H2 artifacts or Titan tokenizer not present")
def test_iso_registry_reproduces_from_h2_artifacts():
    led = PQ.beam_ledger(ROOT / H2["run"], ROOT / H2["freeze"])
    assert int(round(led["flops_per_instance"]["total"]["upper"])) == H2["flops_per_instance"]
    assert int(round(led["flops_per_instance"]["total"]["lower"])) == H2["lower_flops_per_instance"]
    got = {"plans.jsonl": led["plans_sha256"], "FREEZE_B3_H2.json": led["freeze_sha256"],
           "titan tokenizer.json": led["tokenizer_sha256"],
           **{f"{s}/fcache.jsonl": v["fcache_sha256"] for s, v in led["shards"].items()}}
    assert got == H2["inputs_sha256"]
    # cross-checks against recorded numbers: F-09 (22,895 unique keys across shards), P-14 (88/110)
    assert led["f_unique_keys_across_shards"] == 22895 and led["n"] == 110 and led["success"] == 88
    assert led["lora_params"] == {"fwd": 10_092_544, "pa": 10_092_544}
    assert led["flops_per_instance"]["total"]["lower"] < led["flops_per_instance"]["total"]["upper"]


@pytest.mark.skipif(not ((ROOT / "runs/BW_TEST_B2/hard_beam/plans.jsonl").exists() and HAS_TITAN),
                    reason="B2 artifacts not present")
def test_b2_ledger_matches_recorded_f_counts():
    led = PQ.beam_ledger(ROOT / "runs/BW_TEST_B2/hard_beam", ROOT / "runs/FREEZE_B2_beam.json")
    assert led["f_unique_keys_across_shards"] == 20530  # F-09
    assert led["f_unique_keys_sum_over_shards"] == 20554
    assert round(led["f_unique_keys_sum_over_shards"] / led["n"], 1) == 186.9  # E-01(b) FR-02: hard 186.9
    assert led["success"] == 73  # P-08


@needs_qwen
def test_registered_qwen_budgets_on_hard(hard):
    path = QWEN_SNAPS[0]
    prompts = PQ.hard_prompt_tokens(path, hard, True, None)
    forced = PQ.count_tokens(PQ.rust_tokenizer(path), [PQ.FORCE_THINK_END])[0]
    assert forced == 3
    t = PQ.iso_table(FL.flop_params(path / "config.json"), prompts, H2["flops_per_instance"], forced, 1024)
    assert t["prompt_tokens"] == [541, 601, 636] and t["total_cap"] == [5521, 5525, 5527]
    assert t["think_budget"] == [3864, 3900, 3953]


@needs_qwen
def test_render_qwen3_prompt_matches_chat_template():
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    tpl = json.loads((QWEN_SNAPS[0] / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, **k)
    env.globals["raise_exception"] = raise_exception
    t = env.from_string(tpl)
    q = "The following is a PDDL specification.\n[PLAN]"
    for thinking in (True, False):
        for system in (None, PQ.PLANBENCH_SYSTEM):
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
            assert t.render(messages=msgs, add_generation_prompt=True, enable_thinking=thinking) == \
                PQ.render_qwen3_prompt(q, thinking, system)


# ----------------------------------------------------------------------------- CPU lane (D-76), temp dir only

def test_lane_dir_override(lane_tmp):
    assert PQ.lane_dir() == lane_tmp and PQ.lane_lock_path() == lane_tmp / "rineya-cpu-lane.json"


def test_lane_acquire_fields_exclusive_release(lane_tmp):
    rec = PQ.lane_try_acquire("job A", 3600, log=lambda m: None)
    lock = json.loads((lane_tmp / "rineya-cpu-lane.json").read_text())
    assert set(lock) == {"holder", "job", "pid", "started", "expected_end", "token"}
    assert lock["holder"] == "bwm" and lock["pid"] == os.getpid() and lock["job"] == "job A"
    assert abs(lock["expected_end"] - lock["started"] - 3600) < 1e-6 and lock == rec
    assert PQ.lane_try_acquire("job B", 10, log=lambda m: None) is None  # O_EXCL, live holder
    assert PQ.lane_holds(rec)
    assert not PQ.lane_release({**rec, "token": "other"})  # never releases a lock that is not ours
    assert (lane_tmp / "rineya-cpu-lane.json").exists()
    assert PQ.lane_release(rec) and not (lane_tmp / "rineya-cpu-lane.json").exists()
    assert not PQ.lane_release(rec)


def test_lane_stale_dead_pid_reclaimed(lane_tmp):
    dead = _dead_pid()
    (lane_tmp / "rineya-cpu-lane.json").write_text(json.dumps(
        {"holder": "rcx", "job": "E01R", "pid": dead, "started": time.time(), "expected_end": 0, "token": "t"}))
    rec = PQ.lane_try_acquire("after stale", 60, log=lambda m: None)
    assert rec and rec["holder"] == "bwm"
    assert (lane_tmp / f"rineya-cpu-lane.json.stale-{dead}.bak").exists()


def test_lane_live_holder_kept(lane_tmp):
    lock = {"holder": "rcx", "job": "E01R", "pid": os.getpid(), "started": time.time(), "expected_end": 0, "token": "t"}
    (lane_tmp / "rineya-cpu-lane.json").write_text(json.dumps(lock))
    assert PQ.lane_try_acquire("x", 60, log=lambda m: None) is None
    assert json.loads((lane_tmp / "rineya-cpu-lane.json").read_text()) == lock
    assert not list(lane_tmp.glob("*.bak"))


def test_lane_reused_pid_is_stale(lane_tmp):
    """A live pid whose process started after the lock was written is not the holder (pid reuse)."""
    (lane_tmp / "rineya-cpu-lane.json").write_text(json.dumps(
        {"holder": "rcx", "job": "old", "pid": os.getpid(), "started": 0.0, "expected_end": 0, "token": "t"}))
    assert PQ.lane_try_acquire("x", 60, log=lambda m: None) is not None
    assert (lane_tmp / f"rineya-cpu-lane.json.stale-{os.getpid()}.bak").exists()


def test_lane_waiting_rcx_blocks_and_yields(lane_tmp):
    (lane_tmp / "rineya-cpu-lane.waiting.rcx").write_text(json.dumps({"party": "rcx", "job": "E01R", "since": 0}))
    assert PQ.lane_should_yield()
    assert PQ.lane_try_acquire("x", 60, log=lambda m: None) is None and not (lane_tmp / "rineya-cpu-lane.json").exists()
    (lane_tmp / "rineya-cpu-lane.waiting.rcx").unlink()
    assert not PQ.lane_should_yield() and PQ.lane_try_acquire("x", 60, log=lambda m: None)


def test_lane_wait_writes_marker_then_acquires(lane_tmp):
    lp = lane_tmp / "rineya-cpu-lane.json"
    lp.write_text(json.dumps({"holder": "rcx", "job": "E01R", "pid": os.getpid(), "started": time.time(),
                              "expected_end": 0, "token": "t"}))
    seen = []

    def fake_sleep(s):
        m = lane_tmp / "rineya-cpu-lane.waiting.bwm"
        seen.append(json.loads(m.read_text()) if m.exists() else None)
        if len(seen) == 2:
            lp.unlink()  # RCX releases

    rec = PQ.lane_acquire("Q4T", 100, wait=True, poll_s=0, sleep=fake_sleep, log=lambda m: None)
    assert rec["holder"] == "bwm" and len(seen) == 2
    assert all(s and s["party"] == "bwm" and s["job"] == "Q4T" for s in seen)
    assert not (lane_tmp / "rineya-cpu-lane.waiting.bwm").exists()


def test_lane_no_wait_and_max_wait(lane_tmp):
    (lane_tmp / "rineya-cpu-lane.waiting.rcx").write_text("{}")
    with pytest.raises(PQ.LaneBusy):
        PQ.lane_acquire("x", 1, wait=False, log=lambda m: None)
    assert not (lane_tmp / "rineya-cpu-lane.waiting.bwm").exists()
    with pytest.raises(PQ.LaneBusy):
        PQ.lane_acquire("x", 1, wait=True, poll_s=0, max_wait_s=-1, sleep=lambda s: None, log=lambda m: None)
    assert not (lane_tmp / "rineya-cpu-lane.waiting.bwm").exists()


def test_lane_unreadable_lock(lane_tmp):
    lp = lane_tmp / "rineya-cpu-lane.json"
    lp.write_text("")  # a writer between O_EXCL create and json.dump
    assert PQ.lane_try_acquire("x", 1, log=lambda m: None) is None and lp.read_text() == ""
    old = time.time() - 10 * PQ.LANE_UNREADABLE_S
    os.utime(lp, (old, old))
    assert PQ.lane_try_acquire("x", 1, log=lambda m: None) is not None
    assert len(list(lane_tmp.glob("rineya-cpu-lane.json.stale-unreadable-*.bak"))) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows liveness path")
def test_pid_alive_without_psutil_never_signals(monkeypatch):
    dead = _dead_pid()

    def no_kill(*a, **k):
        raise AssertionError("os.kill must not be used on Windows")

    monkeypatch.setattr(os, "kill", no_kill)
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> ImportError: OpenProcess path
    assert PQ.pid_alive(os.getpid()) and not PQ.pid_alive(dead) and not PQ.pid_alive(0) and not PQ.pid_alive("x")
    assert not PQ.pid_alive(2 ** 40)


@pytest.mark.skipif(not RCX_LANE.exists(), reason="RCX lane.py not present")
def test_lane_interoperates_with_rcx_lane_py(lane_tmp, tmp_path, monkeypatch):
    """RCX's own implementation (copied to tmp_path, imported from there; the peer file is only read) and ours agree:
    neither takes the other's live lock, RCX reads our record, RCX's release leaves our lock alone."""
    copy = tmp_path / "rcx_lane_copy.py"
    copy.write_bytes(RCX_LANE.read_bytes())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("rcx_lane_copy", copy)
    rcx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcx)
    assert rcx.lock_path() == lane_tmp / "rineya-cpu-lane.json"
    ours = PQ.lane_try_acquire("Q4N", 60, log=lambda m: None)
    assert rcx.try_acquire("E01R", os.getpid(), 10) is None
    assert rcx.read_lock()["holder"] == "bwm" and not rcx._reclaim_if_stale(rcx.read_lock())
    assert not rcx.release(os.getpid()) and PQ.lane_holds(ours)
    assert PQ.lane_release(ours)
    theirs = rcx.try_acquire("E01R", os.getpid(), 10)
    assert theirs and PQ.lane_try_acquire("Q4N", 60, log=lambda m: None) is None
    assert rcx.release(os.getpid())
    rcx.waiting_path("rcx").write_text("{}")
    assert PQ.lane_should_yield()


# ----------------------------------------------------------------------------- gates, resume, CLI

def test_native_jobs_filter():
    r = str(ROOT)
    jobs = [{"pid": 1, "cmd": "python scripts/orchestrate.py", "cwd": r},
            {"pid": 2, "cmd": "python scripts/gpu_run.py bundleI -- scripts/bundle.py runs/bundles/bundleI.json",
             "cwd": r},
            {"pid": 3, "cmd": "python scripts/health_log.py", "cwd": r},
            {"pid": 4, "cmd": "python scripts/bundle.py runs/bundles/bundleI.json", "cwd": r},
            {"pid": 5, "cmd": "python.exe scripts/orchestrate.py", "cwd": "C:/Users/clclc/AppData/Local/Temp"},
            {"pid": 6, "cmd": f"python {r}\\scripts\\blocks_eval.py --set hard", "cwd": "C:/"},
            {"pid": 7, "cmd": "python other.py", "cwd": "C:/elsewhere"}]
    assert [j["pid"] for j in PQ.native_jobs(jobs, ROOT)] == [4, 6]


def test_gen_guard_step():
    calls = []

    def check():
        calls.append(1)
        return "stop" if len(calls) == 3 else None

    g = PQ.GenGuard(10, check)
    assert not g.step(600) and not g.step(605) and not g.step(610) and not g.step(619)
    assert not g.step(620) and g.step(630) and g.reason == "stop" and len(calls) == 3
    assert g.step(631) and len(calls) == 3


def test_cache_error_narrowing():
    """Only API/shape problems of a KV-cache continuation fall back to re-prefill; memory errors and unrelated runtime
    errors propagate."""
    assert PQ._is_cache_error(TypeError("unexpected keyword past_key_values"))
    assert PQ._is_cache_error(ValueError("cache_position mismatch"))
    assert PQ._is_cache_error(RuntimeError("The size of tensor a (5) must match the size of tensor b (6)"))
    assert not PQ._is_cache_error(RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate"))
    assert not PQ._is_cache_error(RuntimeError("an illegal instruction was encountered"))
    assert not PQ._is_cache_error(MemoryError())
    assert not PQ._is_cache_error(KeyError("x"))


def _cfg():
    return {"model": "qwen3-4b", "x": 1}


def test_read_rows_resume(tmp_path):
    out = tmp_path / "r.jsonl"
    rows = [{"id": "hard/instance-1", "config": _cfg()}, {"id": "hard/instance-2", "config": _cfg()}]
    good = "".join(json.dumps(r) + "\n" for r in rows)
    out.write_text(good + '{"id": "hard/inst', encoding="utf-8")
    assert [r["id"] for r in PQ.read_rows(out, _cfg(), repair=False)] == ["hard/instance-1", "hard/instance-2"]
    assert out.read_text(encoding="utf-8").endswith('"hard/inst')  # dry run never truncates
    assert len(PQ.read_rows(out, _cfg(), repair=True)) == 2 and out.read_text(encoding="utf-8") == good
    with pytest.raises(SystemExit, match="different config"):
        PQ.read_rows(out, {"model": "qwen3-4b", "x": 2}, repair=True)
    out.write_text(json.dumps(rows[0]) + "\nnot json\n" + json.dumps(rows[1]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        PQ.read_rows(out, _cfg(), repair=True)
    out.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate"):
        PQ.read_rows(out, _cfg(), repair=True)


@pytest.mark.parametrize("argv", [
    ["--model", "qwen3-4b", "--threads", "8"],
    ["--model", "qwen3-4b", "--thinking", "on"],
    ["--model", "qwen3-4b", "--thinking", "on", "--iso", "H2_hard", "--think-budget", "100"],
    ["--model", "qwen3-4b", "--thinking", "off", "--iso", "H2_hard"],
    ["--model", "qwen3-4b", "--max-new", "512"],
    ["--model", "qwen3-4b", "--thinking", "on", "--iso", "H2_hard", "--answer-reserve", "256"],
    ["--model", "qwen3-4b", "--thinking", "on", "--iso", "H2_hard", "--iso-fraction", "0"],
])
def test_cli_rejects(argv):
    with pytest.raises(SystemExit) as e:
        PQ.main(argv + ["--dry-run"])
    assert e.value.code == 2


def test_system_prompt_resolution_and_out_names():
    import argparse
    a = argparse.Namespace(system_prompt="auto", set="hard", model="qwen3-4b", thinking="on", iso="H2_hard",
                           iso_fraction=1.0, answer_reserve=1024, think_budget=None, subset=40, max_new=1024,
                           prompt_type="zero_shot", seed=0)
    assert PQ.resolve_system(a) is None
    assert PQ.default_out(a).name == "qwen3-4b_hard_sub40_think-on_iso-H2_hardx1_A1024_max1024_s0.jsonl"
    a.set, a.thinking, a.iso, a.subset = "all", "off", None, None
    assert PQ.resolve_system(a) == PQ.PLANBENCH_SYSTEM
    assert PQ.default_out(a).name == "qwen3-4b_all_think-off_max1024_zero_shot_s0.jsonl"
    a.system_prompt = "none"
    assert PQ.default_out(a).name == "qwen3-4b_all_think-off_max1024_zero_shot_nosys_s0.jsonl"


@needs_qwen
@needs_upstream
def test_dry_run_hard_iso_loads_nothing_and_writes_nothing(tmp_path, lane_tmp, capsys):
    out = tmp_path / "q4t.jsonl"
    lock_existed = PQ.LOCK_PATH.exists()
    rc = PQ.main(["--model", "qwen3-4b", "--set", "hard", "--thinking", "on", "--iso", "H2_hard", "--subset", "40",
                  "--dry-run", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "40 selected, 0 done, 40 to run" in text
    assert "thinking budget [3864, 3900, 3940]" in text and "ids_sha256': '" + PQ.HARD_SUBSET_40_SHA in text
    assert not out.exists() and PQ.LOCK_PATH.exists() == lock_existed and not list(lane_tmp.iterdir())
    for m in ("torch", "transformers"):
        assert m in _HEAVY_BEFORE or m not in sys.modules


def test_run_config_records_budget_and_subset(hard):
    import argparse
    a = argparse.Namespace(system_prompt="auto", set="hard", model="qwen3-4b", thinking="on", iso="H2_hard",
                           iso_fraction=1.0, answer_reserve=1024, think_budget=None, subset=40, subset_seed=0,
                           max_new=1024, prompt_type="zero_shot", query_source="o1-preview_chat", seed=0, threads=4)
    if not HAS_QWEN:
        pytest.skip("Qwen3-4B snapshot not present")
    cfg = PQ.run_config(a, QWEN_SNAPS[0], hard, PQ.apply_subset(hard, a))
    assert cfg["budget"] == {"mode": "iso_flop", "iso_key": "H2_hard", "iso_flops_per_instance":
                             H2["flops_per_instance"], "iso_fraction": 1.0, "answer_reserve": 1024}
    assert cfg["subset"]["ids_sha256"] == PQ.HARD_SUBSET_40_SHA and cfg["prompt"]["system"] is None
    assert cfg["prompt"]["upstream_files_sha256"] == X.HARD_RESULT_SHA and cfg["threads"] == 4
    assert set(cfg["code_sha"]) == set(PQ.CODE_FILES) and cfg["query_source"] == "upstream_hard_fc638a1a"


def test_no_heavy_imports_after_module():
    for m in ("torch", "transformers"):
        assert m in _HEAVY_BEFORE or m not in sys.modules
