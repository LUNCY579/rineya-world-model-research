"""Stub tests for scripts/b4_screen.py (B4 futility screen runner, DRAFT; research/PREREG_B4_SCREEN_DRAFT.md).

CPU only, pure python: no torch/transformers/peft and no model. The pinned planner is executed from its sha-verified
bytes with a stub bwmn.lm; the RCX hook module is loaded through the runner's sha-verifying importer and its GNN forward
is replaced by 0.0 (the GNN file is hashed, never loaded). Read-only inputs: bwmn/blocks_planner.py, bwmn/blocks.py,
runs/FREEZE_B2_beam.json, scripts/bw_seeds.py (its SYSTEMS table, parsed with ast), scripts/bw_run_tests.py (merge),
the RCX files pinned in b4_screen.RCX_CODE_PINS, RCX's E07 instance file and, for the pre-check test only, the goal
structure of runs/BW_val_large/instances_seeds_fresh.json (no system is run on it). Every output goes to tmp_path.

  python -B -m pytest tests/test_b4_screen.py -q -p no:cacheprovider
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import shutil
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("b4_screen_under_test", ROOT / "scripts" / "b4_screen.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

HEAVY = ("torch", "transformers", "peft")
NEEDS_RCX = pytest.mark.skipif(not (S.RCX_ROOT / "rcx" / "bw" / "b4_hook.py").exists(), reason="RCX worktree absent")


@pytest.fixture
def hook_mod():
    mod, imp = S.load_rcx_hook_module()
    try:
        yield mod
    finally:
        S.unload_rcx(imp)


@pytest.fixture
def planner():
    return S.exec_pinned_planner(S.make_stub_lm())


def _no_heavy():
    return [m for m in HEAVY if m in sys.modules]


# ----------------------------------------------------------------------------- import hygiene

def test_module_import_is_light():
    assert _no_heavy() == []
    assert not any(k == "rcx" or k.startswith("rcx.") for k in sys.modules)
    assert not getattr(sys.modules.get("bwmn.lm"), "_b4s_stub", False)


# ----------------------------------------------------------------------------- statistics and the registered rule

def test_exact_pvalues_reproduce_registered_results():
    assert S.mcnemar_two_sided(68, 0) == pytest.approx(6.8e-21, rel=0.01)  # P-08
    assert round(S.mcnemar_two_sided(20, 5), 4) == 0.0041                 # P-14
    assert round(S.mcnemar_two_sided(10, 2), 3) == 0.039                  # P-13
    assert round(S.mcnemar_two_sided(6, 3), 2) == 0.51                    # P-11
    assert S.mcnemar_two_sided(0, 0) == 1.0 and S.mcnemar_two_sided(3, 3) == 1.0
    assert S.sign_p_one_sided(3, 0) == Fraction(1, 8)
    assert S.sign_p_one_sided(4, 1) == Fraction(3, 16)
    assert S.sign_p_one_sided(5, 2) == Fraction(29, 128)


def test_futility_rule_go_region():
    assert S.min_gains_for_go(7) == {l: l + 2 for l in range(8)}       # D-77: net >= 2
    for g in range(41):
        for l in range(41 - g):
            d = S.futility_decision(g, l)
            assert d["go"] == (g - l >= 2)
            if g - l <= 0:
                assert d["decision"] == "NO-GO" and d["because"].startswith("net <= 0")
    assert S.futility_decision(2, 0)["decision"] == "GO"
    assert S.futility_decision(3, 2)["decision"] == "NO-GO"
    assert S.futility_decision(5, 2)["decision"] == "GO"


def test_oc_table_structure_and_null_rates():
    t = S.oc_table(ns=(40,))
    row = {r["scenario"]: r["n40"] for r in t["rows"]}
    reg, old = list(S.OC_RULES)[:2]
    assert reg.startswith("registered: net>=2") and old.startswith("earlier draft")
    for name, r in row.items():
        assert r[reg] >= r[old]                                         # the looser rule GOes at least as often
    assert row["+9 pp, low loss"][reg] > row["+9 pp, low loss"][old]
    assert t["min_arm_only_for_go_by_control_only"][reg] == {str(l): l + 2 for l in range(8)}


def test_clopper_pearson_matches_registered_intervals():
    assert [round(x, 3) for x in S.clopper_pearson(73, 110)] == [0.567, 0.751]  # P-08
    assert [round(x, 3) for x in S.clopper_pearson(88, 110)] == [0.713, 0.870]  # P-14
    assert S.clopper_pearson(0, 40)[0] == 0.0 and S.clopper_pearson(40, 40)[1] == 1.0


# ----------------------------------------------------------------------------- the pinned planner

def test_exec_pinned_planner_binds_stub_and_restores():
    had_lm = sys.modules.get("bwmn.lm")
    stub = S.make_stub_lm()
    B, BP = S.exec_pinned_planner(stub)
    assert BP.lm is stub and callable(BP.plan_beam) and BP.B is B
    assert sys.modules.get("bwmn.lm") is had_lm
    assert sys.modules.get("bwmn.blocks_planner") is not BP
    assert _no_heavy() == []
    with pytest.raises(RuntimeError):
        stub.load_core("titan-sparse")


def test_planner_pin_refusal(tmp_path):
    p = tmp_path / "blocks_planner.py"
    p.write_bytes((ROOT / S.PLANNER_REL).read_bytes() + b"\n# modified\n")
    had_lm = sys.modules.get("bwmn.lm")
    with pytest.raises(S.PinError):
        S.exec_pinned_planner(S.make_stub_lm(), path=p)
    assert sys.modules.get("bwmn.lm") is had_lm


# ----------------------------------------------------------------------------- the pinned RCX hook

@NEEDS_RCX
def test_rcx_loader_executes_only_pinned_bytes(hook_mod):
    imp = next(m for m in sys.meta_path if isinstance(m, S.RCXPinnedImporter))
    assert set(imp.executed) == {"rcx/__init__.py", "rcx/bw/__init__.py", "rcx/bw/b4_hook.py", "rcx/bw/bwm_import.py",
                                 "rcx/bw/encode.py", "rcx/bw/space.py"}            # models.py (torch) not executed
    assert all(imp.executed[k] == S.RCX_CODE_PINS[k] for k in imp.executed)
    assert hook_mod.BWM_PLANNER_PIN == S.PLANNER_SHA256
    assert Path(hook_mod.__file__).resolve() == (S.RCX_ROOT / "rcx/bw/b4_hook.py").resolve()
    assert _no_heavy() == []
    with pytest.raises(S.PinError):
        __import__("rcx.bw.search")                                              # not pinned


@NEEDS_RCX
def test_rcx_loader_refuses_modified_file_and_cleans_up(tmp_path):
    for rel in S.RCX_CODE_PINS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S.RCX_ROOT / rel, tmp_path / rel)
    with open(tmp_path / "rcx/bw/encode.py", "ab") as f:
        f.write(b"\n")
    n_meta = len(sys.meta_path)
    with pytest.raises(S.PinError):
        S.load_rcx_hook_module(rcx_root=tmp_path)
    assert len(sys.meta_path) == n_meta
    assert not any(k == "rcx" or k.startswith("rcx.") for k in sys.modules)


def test_rcx_loader_refuses_a_foreign_rcx_package():
    sys.modules["rcx"] = types.ModuleType("rcx")
    try:
        with pytest.raises(S.PinError):
            S.load_rcx_hook_module()
    finally:
        sys.modules.pop("rcx", None)


@NEEDS_RCX
def test_stub_hook_values_counters_and_meter(hook_mod, planner):
    B, _ = planner
    Stub = S.make_stub_hook_class(hook_mod)
    h = Stub(**S.HOOK_KWARGS)
    assert bool(h) is True and h.hybrid_lambda == 1.0 and h.invalid_policy == "fallback"
    rng, g = random.Random(5), None
    while g is None or not any(f[0] == "on" for f in g[1]):
        g = B.generate(rng, 7, full_goal_frac=1.0)
    objs, init, goal = g
    assert h(init, goal) == float(hook_mod.misplaced_sanitised(init, goal))      # model term 0 + 1.0 * misplaced
    on = sorted(f for f in init if f[0] == "on")
    bad = init | {("ontable", on[0][1])}                                        # a block in two positions
    assert h(bad, goal) >= hook_mod.PENALTY
    assert h.counters["invalid_states"] == 1 and h.counters["fallbacks"] == 1
    m = S.HookMeter(Stub(**S.HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE,
                    [{"id": "x/0", "goal": goal, "objs": objs, "init": init}])
    assert bool(m) is True
    assert m(init, goal) == h(init, goal) and m(bad, goal) == h(bad, goal)
    s = m.summary()
    assert s["calls"] == 2 and s["penalty_calls"] == 1 and s["per_instance"]["x/0"]["calls"] == 2


# ----------------------------------------------------------------------------- dry-run models

def test_scripted_model_f09_pattern_and_crn(planner):
    B, _ = planner
    tok = S.FakeTok()
    m = S.ScriptedModel(B, false_legal=1.0, inconsistent=0.0, noise=0.0)
    s = frozenset({("on", "a", "b"), ("ontable", "b"), ("clear", "a"), ("handempty",)})
    m.set_adapter("fwd")
    legal = m.forward_text(B.forward_prompt_delta(s, ("unstack", "a", "b")), 40, tok)
    assert B.parse_forward_delta(legal["text"], s) == B.step(s, ("unstack", "a", "b"))
    fl = m.forward_text(B.forward_prompt_delta(s, ("pick-up", "a")), 40, tok)          # stacked block: illegal
    assert json.loads(fl["text"])["legal"] is True                                      # F's flag says "legal" ...
    assert B.parse_forward_delta(fl["text"], s) is None                                 # ... the support check rejects
    goal = frozenset({("on", "b", "a")})
    m.set_adapter("pa")
    p = B.proposer_prompt(s, goal)
    assert m.proposer_text(p, None, 80, tok) == m.proposer_text(p, None, 80, tok)
    plan = B.parse_plan(m.proposer_text(p, 7, 80, tok)["text"], ["a", "b"])[0]
    assert B.execute(s, goal, plan)["success"]                                           # noise 0: the plan is valid
    with pytest.raises(RuntimeError):
        m.set_adapter("fwd") or m.proposer_text(p, 1, 80, tok)                           # wrong adapter active


# ----------------------------------------------------------------------------- dry run end to end

@NEEDS_RCX
def test_dry_run_end_to_end(tmp_path):
    rep = S.dry_run(tmp_path / "d", sizes=(5, 6, 7, 8), val=False, inconsistent=0.0)
    assert rep["heavy_modules_newly_imported"] == [] and _no_heavy() == []
    assert rep["rcx_pycache_unchanged"]
    c = rep["checks"]
    assert c["shard_invariance"]["identical"] and c["meter_pass_through"] and c["freeze_preview_args_equal_r0"]
    a = rep["analysis"]
    assert a["n_paired"] == 4 and a["decision"]["decision"] in ("GO", "NO-GO")
    assert a["certificate_c_eq_cpen_on_control"]["holds"]                  # exact stub F: c+pen == c
    assert a["hook"]["penalty_calls"] == 0 and a["hook"]["calls"] > 0
    assert a["f_cache_cross_check"]["common_different_text"] == 0
    for arm in a["arms"].values():
        assert arm["verified_precision"] == 1.0
    assert rep["args"] == json.loads((ROOT / S.B2_BEAM_FREEZE_REL).read_text())["args"]
    for f in ("control_beam/plans.jsonl", "control_beam/fcache.jsonl", "a_stub_beam/summary.json",
              "a_stub_beam/hook_merged.json", "a_stub_beam/shard_3of4/hook.json", "DRY_RUN_REPORT.json"):
        assert (tmp_path / "d" / f).exists(), f
    assert not (ROOT / S.OUT_REL).exists() and not (ROOT / S.FREEZE_REL).exists()
    assert not any(k == "rcx" or k.startswith("rcx.") for k in sys.modules)


@NEEDS_RCX
def test_dry_run_detects_inconsistent_imagined_states(tmp_path):
    rep = S.dry_run(tmp_path / "d", sizes=(6, 7), val=False, inconsistent=0.3, shard_check=False)
    a = rep["analysis"]
    assert not a["certificate_c_eq_cpen_on_control"]["holds"]
    assert a["certificate_c_eq_cpen_on_control"]["inconsistent"] > 0
    assert a["hook"]["penalty_calls"] > 0 and a["hook"]["penalty_share_calls"] > 0


def test_dry_run_never_writes_under_runs():
    with pytest.raises(ValueError):
        S.dry_run(ROOT / "runs" / "BW_B4_SCREEN_dry")
    assert not (ROOT / "runs" / "BW_B4_SCREEN_dry").exists()


# ----------------------------------------------------------------------------- analysis

def _rows(outcomes, n_blocks=10):
    return [{"id": f"seeds_fresh/{i}", "n_blocks": n_blocks, "success": bool(o), "val": bool(o), "verified": bool(o),
             "rounds": 3, "gen_tokens": 5000, "fail": None if o else "budget", "plan": []}
            for i, o in enumerate(outcomes)]


def _pair(g, l, both, n=40):
    ctrl = [0] * g + [1] * l + [1] * both + [0] * (n - g - l - both)
    arm = [1] * g + [0] * l + [1] * both + [0] * (n - g - l - both)
    return _rows(ctrl), _rows(arm)


def test_analyze_pair_decisions_and_attribution():
    c, a = _pair(5, 1, 20)
    rep = S.analyze_pair(c, a, n_expected=40, hook={"calls": 1000, "penalty_calls": 10})
    assert rep["paired"] == {"a_only": 5, "R0_only": 1, "both": 20, "neither": 14}
    assert rep["decision"]["decision"] == "GO" and rep["hook"]["go_credited_to_critic"]
    assert rep["wording"] == S.WORDING["GO"]
    rep = S.analyze_pair(c, a, n_expected=40, hook={"calls": 1000, "penalty_calls": 51})
    assert rep["decision"]["decision"] == "GO" and not rep["hook"]["go_credited_to_critic"]
    assert rep["wording"].endswith(S.ATTRIBUTION_NOTE)
    c, a = _pair(3, 2, 20)
    assert S.analyze_pair(c, a, n_expected=40)["decision"]["decision"] == "NO-GO"
    c, a = _pair(4, 0, 20)
    rep = S.analyze_pair(c, a, n_expected=40, drop_ids=["seeds_fresh/0"])
    assert rep["n_paired"] == 39 and rep["paired"]["a_only"] == 3 and rep["decision"]["decision"] == "GO"


def test_analyze_pair_refuses_misaligned_inputs():
    c, a = _pair(2, 1, 10)
    with pytest.raises(ValueError):
        S.analyze_pair(c, a[:-1], n_expected=40)
    with pytest.raises(ValueError):
        S.analyze_pair(c, a, n_expected=39)
    with pytest.raises(ValueError):
        S.analyze_pair(c, a, n_expected=40, drop_ids=["seeds_fresh/99"])
    a[0] = dict(a[0], id="seeds_fresh/x")
    with pytest.raises(ValueError):
        S.analyze_pair(c, a, n_expected=40)


def test_stalled_subset_and_anytime(planner):
    B, _ = planner
    init = frozenset({("ontable", "a"), ("ontable", "b"), ("ontable", "c"), ("clear", "a"), ("clear", "b"),
                      ("clear", "c"), ("handempty",)})
    goal = frozenset({("on", "a", "b"), ("on", "b", "c")})
    inst = [{"id": "t/0", "objs": ["a", "b", "c"], "init": init, "goal": goal}]
    ctrl = [{"id": "t/0", "n_blocks": 3, "success": False, "val": False, "verified": False, "rounds": 7,
             "gen_tokens": 900, "fail": "dead_end", "plan": ["pick-up b", "stack b c"]}]   # 1 goal fact unmet
    arm = [{"id": "t/0", "n_blocks": 3, "success": True, "val": True, "verified": True, "rounds": 2,
            "gen_tokens": 400, "fail": None, "plan": ["pick-up b", "stack b c", "pick-up a", "stack a b"]}]
    rep = S.analyze_pair(ctrl, arm, n_expected=1, instances=inst, B=B)
    st = rep["control_failures_unmet_at_submitted_prefix"]
    assert st["distribution"] == {"1": 1} and st["stalled_at_1"] == 1 and st["a_solves_among_stalled_at_1"] == 1
    assert rep["arms"]["a"]["anytime_success_within_rounds"]["1"] == 0
    assert rep["arms"]["a"]["anytime_success_within_rounds"]["2"] == 1
    assert rep["arms"]["R0"]["failure_taxonomy"]["dead_end"] == 1


@NEEDS_RCX
def test_cache_cross_check_and_certificate(tmp_path, hook_mod, planner):
    B, _ = planner
    s = frozenset({("on", "a", "b"), ("ontable", "b"), ("clear", "a"), ("handempty",)})
    good = B.forward_target_delta(s, B.step(s, ("unstack", "a", "b"))[0], True)
    bad = json.dumps({"add": ["holding a", "clear b", "ontable a"], "del": ["clear a", "handempty", "on a b"],
                      "legal": True})                                                   # a held AND on the table

    def cache(p, text):
        p.write_text(json.dumps({"s": sorted(map(list, s)), "a": ["unstack", "a", "b"], "text": text}) + "\n")
        return p

    c1, c2 = cache(tmp_path / "c1.jsonl", good), cache(tmp_path / "c2.jsonl", good)
    c3 = cache(tmp_path / "c3.jsonl", bad)
    x = S.cross_cache([c1], [c2])
    assert x["common_keys"] == 1 and x["common_same_text"] == 1
    assert S.cross_cache([c1], [c3])["common_different_text"] == 1
    assert S.certificate_c_eq_cpen([c1], B, hook_mod)["holds"]
    cert = S.certificate_c_eq_cpen([c3], B, hook_mod)
    assert not cert["holds"] and cert["inconsistent"] == 1
    # verifier case: F-accepted put-down that omits 'ontable a'/'clear a' -> block a vanishes, hand still specified.
    # The hook (called with the goal) puts it in the penalty band; the certificate must not say "holds".
    h = frozenset({("holding", "a"), ("ontable", "b"), ("clear", "b")})
    gone = json.dumps({"add": ["handempty"], "del": ["holding a"], "legal": True})
    assert B.parse_forward_delta(gone, h)[1] is True
    assert hook_mod.is_consistent(B.parse_forward_delta(gone, h)[0], frozenset({("on", "a", "b")}))[0] is False
    c4 = tmp_path / "c4.jsonl"
    c4.write_text(json.dumps({"s": sorted(map(list, h)), "a": ["put-down", "a"], "text": gone}) + "\n")
    cert = S.certificate_c_eq_cpen([c4], B, hook_mod)
    assert not cert["holds"] and cert["reasons"].get("no_position") == 1


def test_merge_shards_uses_item7_merge(tmp_path):
    for i, outs in enumerate(([1, 0], [1])):
        d = tmp_path / f"shard_{i}of2"
        d.mkdir()
        rows = [dict(r, id=f"seeds_fresh/{2 * j + i}") for j, r in enumerate(_rows(outs))]
        S.write_outputs(d, {"f_model_queries": 10, "wall_s": 1.0}, rows)
        (d / "hook.json").write_text(json.dumps({"describe_end": {"counters": {"calls": 3}, "reasons": {}},
                                                 "meter": {"calls": 3, "penalty_calls": i, "failsafe_calls": 0,
                                                           "seconds": 0.5, "per_instance": {f"k{i}": {}}}}))
    hook = S.merge_shards(tmp_path, 2)
    summ = json.loads((tmp_path / "summary.json").read_text())["summary"]
    assert summ["n"] == 3 and summ["val_success"] == pytest.approx(2 / 3) and summ["f_model_queries"] == 20
    assert hook["calls"] == 6 and hook["penalty_calls"] == 1 and hook["counters"] == {"calls": 6}


# ----------------------------------------------------------------------------- freeze

def _r0_record_as_bw_seeds_writes_it():
    """bw_seeds.freeze for R0/beam: dict(B2 beam args, pa=Pa) with fwd_adapter=F (SYSTEMS parsed with ast)."""
    tree = ast.parse((ROOT / "scripts" / "bw_seeds.py").read_text())
    systems = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "SYSTEMS")
    f, pa = systems["R0"]
    b2 = json.loads((ROOT / S.B2_BEAM_FREEZE_REL).read_text())
    args = dict(b2["args"], pa=pa)
    args["fwd_adapter"] = f
    return {"args": args, "pa_sha256": b2["pa_sha256"], "fwd_sha256": b2["fwd_sha256"]}, b2


def test_r0_beam_is_the_b2_beam_and_freeze_record():
    r0, b2 = _r0_record_as_bw_seeds_writes_it()
    S.check_r0_is_b2(r0, b2)
    rec = S.build_freeze_record(r0, "x", pins={}, pre={"drop_ids": [], "n_after_drop": 40}, prereg={}, addendum={},
                                r0_outputs_present=[], runner_sha="y", now="t")
    assert rec["args"] == r0["args"] == b2["args"]
    assert rec["unmet_hook"]["kwargs"] == {"invalid_policy": "fallback", "hybrid_lambda": 1.0}
    assert rec["unmet_hook"]["model_file"] == S.GNN_REL and rec["set"]["n"] == 40 and rec["set"]["shards"] == 4
    for k, v in (("E", 48), ("pa", "runs/BW_Pa_v3_s1/adapter")):
        bad = {"args": dict(r0["args"], **{k: v}), "pa_sha256": r0["pa_sha256"], "fwd_sha256": r0["fwd_sha256"]}
        with pytest.raises(S.PinError):
            S.check_r0_is_b2(bad, b2)
    with pytest.raises(S.PinError):
        S.check_r0_is_b2(dict(r0, pa_sha256="0" * 64), b2)


def test_stage_freeze_refusals(tmp_path):
    ns = argparse.Namespace(prereg=str(tmp_path / "PREREG_B4_SCREEN_DRAFT.md"), addendum=str(tmp_path / "A.md"),
                            allow_draft=False, allow_after_r0=False)
    with pytest.raises(SystemExit, match="bw_seeds.py freeze"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    r0, _ = _r0_record_as_bw_seeds_writes_it()
    (tmp_path / "runs").mkdir()
    (tmp_path / S.R0_FREEZE_REL).write_text(json.dumps(r0))
    out = tmp_path / S.R0_OUT_REL / "shard_0of4"
    out.mkdir(parents=True)
    (out / "summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="before any R0 output"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    (out / "summary.json").unlink()
    (out / "progress.txt").write_text("12:00:00 round 1: active 10 -> 7; F queries 900\n")   # R0 shard mid-run
    with pytest.raises(SystemExit, match="before any R0 output"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    (out / "progress.txt").unlink()
    with pytest.raises(SystemExit, match="is a draft"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    assert not (tmp_path / S.FREEZE_REL).exists() and not (tmp_path / S.OUT_REL).exists()


# ----------------------------------------------------------------------------- pre-check (prereg 4.3)

def test_goal_chain_key():
    assert S.goal_chain_key(["a", "b", "c", "d"], [("on", "a", "b"), ("on", "b", "c")]) == "chains:4:1,3"
    assert S.goal_chain_key(["a", "b", "c"], [("on", "a", "c"), ("on", "b", "c")]) is None   # two on one block
    assert S.goal_chain_key(["a", "b"], [("ontable", "a")]) is None


@NEEDS_RCX
def test_precheck_on_registered_inputs():
    pre = S.precheck()
    assert pre["e07_rows_goal_key_agreement"] == "103/103"
    assert pre["seeds_fresh_unkeyed"] == 0 and pre["gnn_train_val_goal_classes"] == 28
    assert pre["drop_ids"] == [] and pre["n_after_drop"] == 40
