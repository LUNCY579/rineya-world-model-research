"""Stub tests for scripts/b4_screen.py (B4 futility screen runner, DRAFT; research/PREREG_B4_SCREEN_DRAFT.md; RCX
co-sign review 9dd352ee..., 13 required edits, D-80; the canary 9e9ad339... and RCX's low items, D-80 follow-up / D-81).
The registered canary file is read torch-free (schema, probes, counters); its values are checked only by the CPU
pre-flight `b4_screen.py canary`, never here.

CPU only, pure python: no torch/transformers/peft and no model. The pinned planner is executed from its sha-verified
bytes with a stub bwmn.lm; the RCX hook module is loaded through the runner's sha-verifying importer and its GNN forward
is replaced by 0.0 (the GNN file is hashed or copied, never loaded). The GPU path (run-shard -> merge -> analyze) runs
end to end on a temporary tree with only the model loading stubbed: a scripted F/proposer, a stub torch namespace
(version and CUDA view only), executor-equivalent VAL, and a hook record that marks models.py executed as the GPU
process would. Read-only inputs: bwmn/blocks_planner.py, bwmn/blocks.py, runs/FREEZE_B2_beam.json, scripts/bw_seeds.py
(its SYSTEMS table, parsed with ast), scripts/bw_run_tests.py (merge), the files pinned in b4_screen (hashed), the RCX
files pinned in b4_screen.RCX_CODE_PINS, RCX's E07 instance file and E07b VAL-B instance file (goal structure only) and,
for the pre-check and hygiene tests only, the goal structure of runs/BW_val_large/instances_seeds_fresh.json (no system
is run on it). Every output goes to tmp_path.

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
HAVE_RCX = (S.RCX_ROOT / "rcx" / "bw" / "b4_hook.py").exists()
NEEDS_RCX = pytest.mark.skipif(not HAVE_RCX, reason="RCX worktree absent")
REAL_LOADER = S.load_rcx_hook_module


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


def test_futility_rule_go_region_and_p_values_never_gate():
    assert S.min_gains_for_go(7) == {l: l + 2 for l in range(8)}       # D-77: net >= 2
    for g in range(41):
        for l in range(41 - g):
            d = S.futility_decision(g, l)
            assert d["go"] == (g - l >= 2)
            if g - l <= 0:
                assert d["decision"] == "NO-GO" and d["because"].startswith("net <= 0")
    d = S.futility_decision(2, 0)
    assert d["decision"] == "GO" and d["p_one_sided_sign"] == 0.25         # p > 0.20 and still GO: p does not gate
    assert d["reported_only"] == ["p_one_sided_sign", "p_mcnemar_two_sided"]
    assert S.futility_decision(3, 2)["decision"] == "NO-GO"
    assert S.futility_decision(5, 2)["decision"] == "GO"
    assert not hasattr(S, "GO_MAX_P_ONE_SIDED")                         # edit 3: no p-value in the frozen rule


def test_oc_table_structure_and_superseded_rule():
    t = S.oc_table(ns=(40,))
    row = {r["scenario"]: r["n40"] for r in t["rows"]}
    reg, old = list(S.OC_RULES)[:2]
    assert reg.startswith("registered: net>=2") and old.startswith("superseded by D-77")
    for name, r in row.items():
        assert r[reg] >= r[old]                                         # the looser rule GOes at least as often
    assert row["+9 pp, low loss"][reg] > row["+9 pp, low loss"][old]
    assert t["min_arm_only_for_go_by_control_only"][reg] == {str(l): l + 2 for l in range(8)}
    for name in ("+5 pp", "+9 pp, high loss"):                          # edit 4: restored rows
        assert name in row
    assert row["+5 pp"][reg] == 0.578 and row["+9 pp, high loss"][reg] == 0.755


def test_oc_dgrid_reproduces_rcx_cosign_table():
    g = S.oc_dgrid()
    rows = {r["d"]: r for r in g["rows"]}
    rcx = {0.10: (0.220, 0.870, None), 0.15: (0.266, 0.813, 0.968), 0.20: (0.296, 0.777, 0.933),
           0.30: (0.332, 0.731, 0.881)}                                  # COSIGN_REVIEW_B4_SCREEN.md, edit 4
    for d, (null, p9, p136) in rcx.items():
        assert (rows[d]["null"], rows[d]["+9 pp"], rows[d]["+13.6 pp"]) == (null, p9, p136)
    assert rows[0.227]["null"] == 0.308                                 # the H2 anchor
    assert 0.30 <= rows[0.227]["null"] <= 0.36 and rows[0.40]["null"] == 0.354
    nulls = [rows[d]["null"] for d in S.OC_D_GRID]
    assert nulls == sorted(nulls)                                       # the null false-GO rate rises with d
    assert g["null_sup_over_d"] == {"d": 1.0, "p_go": 0.4373}           # <= 0.44 at any discordance
    assert rows[0.05]["+9 pp"] is None


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


def _goal_instance(B, n=7, seed=5):
    rng, g = random.Random(seed), None
    while g is None or not any(f[0] == "on" for f in g[1]):
        g = B.generate(rng, n, full_goal_frac=1.0)
    return g


@NEEDS_RCX
def test_stub_hook_values_counters_and_meter_split(hook_mod, planner):
    B, _ = planner
    Stub = S.make_stub_hook_class(hook_mod)
    h = Stub(**S.HOOK_KWARGS)
    assert bool(h) is True and h.hybrid_lambda == 1.0 and h.invalid_policy == "fallback"
    objs, init, goal = _goal_instance(B)
    assert h(init, goal) == float(hook_mod.misplaced_sanitised(init, goal))      # model term 0 + 1.0 * misplaced
    on = sorted(f for f in init if f[0] == "on")
    bad = init | {("ontable", on[0][1])}                                        # a block in two positions
    assert h(bad, goal) >= hook_mod.PENALTY
    assert h.counters["invalid_states"] == 1 and h.counters["fallbacks"] == 1
    m = S.HookMeter(Stub(**S.HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE,
                    [{"id": "x/0", "goal": goal, "objs": objs, "init": init}], classify=hook_mod.is_consistent)
    assert bool(m) is True
    assert m(init, goal) == h(init, goal) and m(bad, goal) == h(bad, goal)
    s = m.summary()
    assert s["calls"] == 2 and s["penalty_calls"] == 1 and s["per_instance"]["x/0"]["calls"] == 2
    assert s["penalty_calls_invalid_state"] == 1 and s["penalty_calls_critic_error"] == 0
    # a consistent state that fell back (critic error) is split as critic_error
    class Err(Stub):
        def _forward(self, g):
            raise RuntimeError("model error")
    m2 = S.HookMeter(Err(**S.HOOK_KWARGS), hook_mod.PENALTY, hook_mod.FAILSAFE, classify=hook_mod.is_consistent)
    assert m2(init, goal) >= hook_mod.PENALTY and m2.summary()["penalty_calls_critic_error"] == 1
    assert m2.hook.counters["model_errors"] == 1


def _good_describe(rcx_root):
    return {"model_sha256": S.GNN_SHA256, "sidecar_sha256": S.GNN_SIDECAR_SHA256,
            "config_source": "sidecar json 'config'", "invalid_policy": "fallback", "hybrid_lambda": 1.0,
            "round_to": None, "bwm_planner_pin": S.PLANNER_SHA256,
            "code_sha256": {rel: S.RCX_CODE_PINS[rel] for rel in S.RCX_D76_FILES},
            "model_file": str(Path(rcx_root) / S.GNN_REL), "registry": {"goals": 0, "conflicts": 0},
            "counters": {"calls": 0}}


class _FakeHook:
    def __init__(self, d, truthy=True):
        self.d, self.truthy = d, truthy

    def describe(self):
        return dict(self.d)

    def __bool__(self):
        return self.truthy


def test_verify_hook_accepts_only_the_registered_call(tmp_path):                 # edit 13
    good = _good_describe(tmp_path)
    assert S.verify_hook(_FakeHook(good), tmp_path) == good
    for k, v in (("model_sha256", "0" * 64), ("sidecar_sha256", "1" * 64), ("config_source", "default RelGNNConfig()"),
                 ("invalid_policy", "raw"), ("hybrid_lambda", 0.0), ("round_to", 0.5), ("bwm_planner_pin", "x"),
                 ("model_file", str(tmp_path / "other.pt")), ("registry", {"goals": 40, "conflicts": 0})):
        with pytest.raises(S.PinError):
            S.verify_hook(_FakeHook(dict(good, **{k: v})), tmp_path)
    for rel in S.RCX_D76_FILES:
        with pytest.raises(S.PinError):
            S.verify_hook(_FakeHook(dict(good, code_sha256=dict(good["code_sha256"], **{rel: "f" * 64}))), tmp_path)
    with pytest.raises(S.PinError):
        S.verify_hook(_FakeHook(good, truthy=False), tmp_path)


def _write_canary(path, hook_factory, probes_states, **over):
    h = hook_factory()
    probes = []
    for k, (state, goal, consistent) in enumerate(probes_states):
        probes.append({"id": f"p{k:02d}", "state": sorted(list(f) for f in state), "goal": sorted(list(f) for f in goal),
                       "consistent": consistent, "expected": float(h(state, goal))})
    rec = {"what": "test canary", "tolerance_abs": 1e-5,
           "hook_call": {"path": S.GNN_REL, "invalid_policy": "fallback", "hybrid_lambda": 1.0},
           "expected_counters_after_all": dict(h.counters),
           "pins_sha256": {**S.RCX_CODE_PINS, S.GNN_REL: S.GNN_SHA256}, "probes": probes}
    rec.update(over)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return S.sha256_file(path), rec


@NEEDS_RCX
def test_check_canary(tmp_path, hook_mod, planner, monkeypatch):                 # edit 8; D-80 follow-up
    B, _ = planner
    Stub = S.make_stub_hook_class(hook_mod)
    objs, init, goal = _goal_instance(B)
    on = sorted(f for f in init if f[0] == "on")
    states = [(init, goal, True), (B.step(init, ("unstack", on[0][1], on[0][2]))[0] if B.applicable(
        init, ("unstack", on[0][1], on[0][2])) else init, goal, True), (init | {("ontable", on[0][1])}, goal, False)]
    sha, rec = _write_canary(tmp_path / "c.json", lambda: Stub(**S.HOOK_KWARGS), states, torch_version="2.6.0+cu124")
    r = S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "c.json", want_sha=sha)
    assert r["probes"] == 3 and r["max_abs_delta"] == 0.0 and r["counters_match"] and r["failures"] == []
    assert r["abs_delta_by_probe"] == {"p00": 0.0, "p01": 0.0, "p02": 0.0}
    assert r["counters_after_all"]["calls"] == 3 and r["counters_after_all"]["invalid_states"] == 1
    assert r["canary_torch_version"] == "2.6.0+cu124" and r["host_torch_version"] is None   # torch is not loaded here
    assert r["torch_versions_match"] is None and _no_heavy() == []
    r = S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "c.json", want_sha=sha, host_torch_version="2.5.1")
    assert r["torch_versions_match"] is False and r["failures"] == []           # a mismatch is report-only: passes
    r = S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "c.json", want_sha=sha,
                       host_torch_version="2.6.0+cu124")
    assert r["torch_versions_match"] is True
    with pytest.raises(S.PinError):                                              # other bytes than registered
        S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "c.json", want_sha="0" * 64)
    monkeypatch.setattr(S, "CANARY_SHA256", None)
    with pytest.raises(S.PinError, match="placeholder"):                         # no registered canary: refuse
        S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "c.json")
    with pytest.raises(S.PinError, match="placeholder"):
        S.load_canary(tmp_path / "c.json")
    rec["probes"][0]["expected"] += 2e-5                                          # beyond 1e-5
    (tmp_path / "d.json").write_text(json.dumps(rec))
    with pytest.raises(S.CanaryError, match="p00"):
        S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "d.json", want_sha=S.sha256_file(tmp_path / "d.json"))
    for over in ({"hook_call": {"path": S.GNN_REL, "invalid_policy": "raw", "hybrid_lambda": 1.0}},
                 {"tolerance_abs": 1e-3}, {"expected_counters_after_all": {"calls": 99}},
                 {"expected_counters_after_all": None}, {"expected_counters_after_all": {}},   # RCX low item 4
                 {"probes": []}, {"pins_sha256": {"rcx/bw/encode.py": "0" * 64}}):
        sha2, _ = _write_canary(tmp_path / "e.json", lambda: Stub(**S.HOOK_KWARGS), states, **over)
        with pytest.raises(S.CanaryError):
            S.check_canary(lambda: Stub(**S.HOOK_KWARGS), tmp_path / "e.json", want_sha=sha2)
    _, rec3 = _write_canary(tmp_path / "f.json", lambda: Stub(**S.HOOK_KWARGS), states)
    rec3.pop("expected_counters_after_all")                                       # the key absent altogether
    (tmp_path / "f.json").write_text(json.dumps(rec3))
    made = []
    with pytest.raises(S.CanaryError, match="expected_counters_after_all"):
        S.check_canary(lambda: made.append(1) or Stub(**S.HOOK_KWARGS), tmp_path / "f.json",
                       want_sha=S.sha256_file(tmp_path / "f.json"))
    assert made == []                                                              # refused before any hook is built


@NEEDS_RCX
def test_real_canary_file_schema_torch_free(hook_mod):                          # RCX low item 8; D-80 follow-up
    """The registered canary (RCX commit 42cbfcc) read without torch: sha, D-76a call, tolerance, pins, the 25 probes
    (20 consistent from E07 test and E07b TEST-B, 5 flagged inconsistent) and the expected counters."""
    p = S.RCX_ROOT / S.CANARY_REL
    assert S.CANARY_SHA256 == "9e9ad3395b3b274db90f96ff47e8fee2e7d7329f045618f88643e0899a11e87a"
    c = S.load_canary(p)                                                           # sha + schema, torch-free
    assert S.sha256_file(p) == S.CANARY_SHA256 and float(c["tolerance_abs"]) == S.CANARY_TOL
    assert c["hook_call"] == {"path": S.GNN_REL, **S.HOOK_KWARGS}
    pins = {**S.RCX_CODE_PINS, S.GNN_REL: S.GNN_SHA256, S.GNN_SIDECAR_REL: S.GNN_SIDECAR_SHA256,
            S.E07_INSTANCES_REL: S.E07_INSTANCES_SHA256}
    assert c["pins_sha256"] == pins
    probes = c["probes"]
    ids = [x["id"] for x in probes]
    assert len(probes) == 25 and len(set(ids)) == 25
    assert sum(bool(x["consistent"]) for x in probes) == 20
    exp = c["expected_counters_after_all"]
    assert set(exp) == set(hook_mod.COUNTERS)
    assert {k: exp[k] for k in ("calls", "evaluations", "model_evals", "invalid_states", "fallbacks")} == \
        {"calls": 25, "evaluations": 25, "model_evals": 20, "invalid_states": 5, "fallbacks": 5}
    assert all(exp[k] == 0 for k in ("encode_errors", "model_errors", "nonfinite", "failsafe", "analysis_errors"))
    assert c["torch_version"] == "2.6.0+cu124" and c["torch_threads"] == 1
    sf = json.loads((ROOT / S.SEEDS_FRESH_REL).read_text())
    sf_keys = {json.dumps((sorted(r["objs"]), sorted(map(list, r["goal"])))) for r in sf}
    for x in probes:
        state, goal = frozenset(map(tuple, x["state"])), frozenset(map(tuple, x["goal"]))
        ok, _ = hook_mod.is_consistent(state, goal)
        assert ok is bool(x["consistent"])                                         # the flag is the hook's own
        assert (x["expected"] >= hook_mod.PENALTY) is (not x["consistent"])      # penalty band iff inconsistent
        assert x["expected"] < hook_mod.FAILSAFE
        assert x["instance_id"].split("/")[0] in ("test", "testb")               # never seeds_fresh
        assert json.dumps((sorted(x["objs"]), sorted(map(list, goal)))) not in sf_keys
    assert sum(x["instance_id"].startswith("testb/") for x in probes) == 10
    assert _no_heavy() == []


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


def test_arm_path_with_metered_default_key_reproduces_control_rows(planner, tmp_path):   # edit 13
    B, BP = planner
    inst = S.dry_instances(B, (5, 6, 7, 8))
    args = json.loads((ROOT / S.B2_BEAM_FREEZE_REL).read_text())["args"]
    _, ctrl = S.run_arm(BP, B, S.ScriptedModel(B, inconsistent=0.0), S.FakeTok(), inst, args, tmp_path / "c",
                        unmet=None, val=False)
    meter = S.HookMeter(lambda s, g: len(g - s), 1000.0, 1e6, inst)
    _, arm = S.run_arm(BP, B, S.ScriptedModel(B, inconsistent=0.0), S.FakeTok(), inst, args, tmp_path / "a",
                       unmet=meter, val=False)
    assert arm == ctrl                                                   # every field of every row
    assert meter.calls > 0 and meter.penalty_calls == 0
    assert set(meter.summary()["per_instance"]) == {it["id"] for it in inst}
    assert (tmp_path / "c" / "fcache.jsonl").read_text() == (tmp_path / "a" / "fcache.jsonl").read_text()


# ----------------------------------------------------------------------------- dry run end to end

@NEEDS_RCX
def test_dry_run_end_to_end(tmp_path):
    rep = S.dry_run(tmp_path / "d", sizes=(5, 6, 7, 8), val=False, inconsistent=0.0)
    assert rep["heavy_modules_newly_imported"] == [] and _no_heavy() == []
    assert rep["rcx_pycache_unchanged"]
    c = rep["checks"]
    assert c["shard_invariance"]["identical"] and c["meter_pass_through"] and c["freeze_preview_args_equal_r0"]
    a = rep["analysis"]
    assert a["n_paired"] == 4 and a["status"] in ("GO", "NO-GO") and a["round1_identity"]["holds"]
    assert a["certificate_c_eq_cpen_on_control"]["holds"]                  # exact stub F: c+pen == c
    assert a["arm_unpenalised_inconsistent_report_only"]["strict_inconsistent"] == 0
    assert a["hook"]["penalty_calls"] == 0 and a["hook"]["calls"] > 0
    assert a["f_cache_cross_check"]["common_different_text"] == 0
    for arm in a["arms"].values():
        assert arm["verified_precision"] == 1.0 and "unverified_val_successes" in arm
    assert "arm_failures_unmet_at_submitted_prefix" in a                  # symmetric stalled report
    assert all("no passed canary" in r for r in a["void_reasons_if_this_were_the_screen"])
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
    split = a["hook"]["penalty_split_calls"]
    assert split["invalid_state"] == a["hook"]["penalty_calls"] and split["critic_error"] == 0
    assert a["hook"]["penalty_split_evaluations"]["invalid_states"] > 0


def test_dry_run_never_writes_under_runs():
    with pytest.raises(ValueError):
        S.dry_run(ROOT / "runs" / "BW_B4_SCREEN_dry")
    assert not (ROOT / "runs" / "BW_B4_SCREEN_dry").exists()


# ----------------------------------------------------------------------------- analysis (pure)

def _rows(outcomes, n_blocks=10, rounds=3):
    return [{"id": f"seeds_fresh/{i}", "n_blocks": n_blocks, "success": bool(o), "val": bool(o), "verified": bool(o),
             "rounds": rounds, "gen_tokens": 5000, "fail": None if o else "budget", "plan": []}
            for i, o in enumerate(outcomes)]


def _pair(g, l, both, n=40):
    ctrl = [0] * g + [1] * l + [1] * both + [0] * (n - g - l - both)
    arm = [1] * g + [0] * l + [1] * both + [0] * (n - g - l - both)
    return _rows(ctrl), _rows(arm)


def test_analyze_pair_decisions_and_attribution_boundary():
    c, a = _pair(5, 1, 20)
    rep = S.analyze_pair(c, a, n_expected=40, hook={"calls": 1000, "penalty_calls": 10})
    assert rep["paired"] == {"a_only": 5, "R0_only": 1, "both": 20, "neither": 14}
    assert rep["status"] == "GO" and rep["decision"]["decision"] == "GO" and rep["hook"]["go_credited_to_critic"]
    assert rep["wording"] == S.WORDING["GO"]
    rep = S.analyze_pair(c, a, n_expected=40, hook={"calls": 1000, "penalty_calls": 50})   # exactly 5%: credited
    assert rep["hook"]["go_credited_to_critic"] and not rep["hook"]["attribution_rule_fired"]
    rep = S.analyze_pair(c, a, n_expected=40, hook={"calls": 1000, "penalty_calls": 51})   # more than 5%
    assert rep["decision"]["decision"] == "GO" and not rep["hook"]["go_credited_to_critic"]
    assert rep["wording"].endswith(S.ATTRIBUTION_NOTE)
    rep = S.analyze_pair(c, a, n_expected=40, hook={"calls": 1000, "penalty_calls": 0},
                         go_min_net=5)                                     # constants come from the frozen record
    assert rep["decision"]["decision"] == "NO-GO"
    c, a = _pair(3, 2, 20)
    assert S.analyze_pair(c, a, n_expected=40)["decision"]["decision"] == "NO-GO"
    c, a = _pair(4, 0, 20)
    rep = S.analyze_pair(c, a, n_expected=40, drop_ids=["seeds_fresh/0"])
    assert rep["n_paired"] == 39 and rep["paired"]["a_only"] == 3 and rep["decision"]["decision"] == "GO"
    ev = S.hook_report({"calls": 10, "penalty_calls": 1, "counters": {"evaluations": 8, "fallbacks": 1,
                                                                      "analysis_errors": 1, "invalid_states": 1}},
                       None, 0.05)
    assert ev["penalty_share_evaluations"] == 0.25                        # analysis errors count in the share


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


def test_round1_identity_fault_means_no_decision():                     # edit 7
    c, a = _pair(4, 0, 20)
    c[30] = dict(c[30], success=True, val=True, verified=True, rounds=1, plan=["pick-up a"], gen_tokens=100)
    a[30] = dict(c[30])
    assert S.round1_identity({r["id"]: r for r in c}, {r["id"]: r for r in a}, [r["id"] for r in c])["holds"]
    for change in ({"gen_tokens": 101}, {"rounds": 2}, {"plan": ["pick-up b"]}, {"verified": False}):
        a2 = list(a)
        a2[30] = dict(a[30], **change)
        rep = S.analyze_pair(c, a2, n_expected=40)
        assert rep["status"] == "INSTRUMENT_FAULT" and rep["decision"] is None
        assert rep["round1_identity"]["mismatched_ids"] == ["seeds_fresh/30"]
        assert rep["wording"] == S.WORDING["INSTRUMENT_FAULT"] and "re-run" in rep["wording"]
    a2 = list(a)
    a2[31] = dict(a[31], verified=True, rounds=1)                         # only the arm solved in round 1
    assert S.analyze_pair(c, a2, n_expected=40, fault_status="INSTRUMENT_FAULT_RERUN")["status"] == \
        "INSTRUMENT_FAULT_RERUN"


def test_stalled_subset_symmetric_and_anytime(planner):
    B, _ = planner
    init = frozenset({("ontable", "a"), ("ontable", "b"), ("ontable", "c"), ("clear", "a"), ("clear", "b"),
                      ("clear", "c"), ("handempty",)})
    goal = frozenset({("on", "a", "b"), ("on", "b", "c")})
    inst = [{"id": "t/0", "objs": ["a", "b", "c"], "init": init, "goal": goal},
            {"id": "t/1", "objs": ["a", "b", "c"], "init": init, "goal": goal}]
    ctrl = [{"id": "t/0", "n_blocks": 3, "success": False, "val": False, "verified": False, "rounds": 7,
             "gen_tokens": 900, "fail": "dead_end", "plan": ["pick-up b", "stack b c"]},   # 1 goal fact unmet
            {"id": "t/1", "n_blocks": 3, "success": True, "val": True, "verified": False, "rounds": 3,
             "gen_tokens": 900, "fail": "budget", "plan": ["pick-up b", "stack b c", "pick-up a", "stack a b"]}]
    arm = [{"id": "t/0", "n_blocks": 3, "success": True, "val": True, "verified": True, "rounds": 2,
            "gen_tokens": 400, "fail": None, "plan": ["pick-up b", "stack b c", "pick-up a", "stack a b"]},
           {"id": "t/1", "n_blocks": 3, "success": False, "val": False, "verified": False, "rounds": 7,
            "gen_tokens": 900, "fail": "dead_end", "plan": []}]                            # 2 goal facts unmet
    rep = S.analyze_pair(ctrl, arm, n_expected=2, instances=inst, B=B)
    st = rep["control_failures_unmet_at_submitted_prefix"]
    assert st["distribution"] == {"1": 1} and st["stalled_at_1"] == 1 and st["a_solves_among_stalled_at_1"] == 1
    assert st["other_failures"] == 0 and "true state" in st["note"]
    mirror = rep["arm_failures_unmet_at_submitted_prefix"]
    assert mirror["distribution"] == {"2": 1} and mirror["stalled_at_1"] == 0
    assert mirror["other_failures"] == 1 and mirror["R0_solves_among_other_failures"] == 1
    assert rep["arms"]["R0"]["unverified_val_successes"] == 1 and rep["arms"]["a"]["unverified_val_successes"] == 0
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


@NEEDS_RCX
def test_unpenalised_certificate_counts_a_vanished_non_goal_block(tmp_path, hook_mod, planner):
    B, _ = planner
    s = frozenset({("ontable", "b"), ("on", "a", "b"), ("clear", "a"), ("ontable", "c"), ("clear", "c"),
                   ("handempty",)})
    goal = frozenset({("on", "a", "b")})                                 # c is not a goal block
    inst = [{"id": "t/0", "objs": ["a", "b", "c"], "init": s, "goal": goal}]
    keys = [(("pick-up", "c"), {"add": [], "del": ["clear c", "ontable c"], "legal": True}),          # c vanishes
            (("unstack", "a", "b"), {"add": [], "del": ["clear a", "on a b"], "legal": True}),         # a vanishes
            (("pick-up", "a"), B.forward_target_delta(s, s, False))]                                    # illegal flag
    p = tmp_path / "f.jsonl"
    p.write_text("".join(json.dumps({"s": sorted(map(list, s)), "a": list(a),
                                     "text": t if isinstance(t, str) else json.dumps(t)}) + "\n" for a, t in keys))
    r = S.unpenalised_certificate({0: p}, inst, B, hook_mod)
    assert r["accepted_predictions"] == 2 and r["strict_inconsistent"] == 2
    assert r["penalised_by_hook"] == 1 and r["unpenalised"] == 1
    assert r["unpenalised_reasons"] == {"missing_clear": 1, "no_position": 1}     # c has no position and no clear
    assert r["registry"] == "off"


def test_void_reasons():                                                 # edit 8
    ok = {"counters": {"model_evals": 10, "calls": 12}, "canary": {"shard_0of4": {"failures": [],
                                                                                   "counters_match": True}}}
    assert S.void_reasons(ok) == []
    for k in S.VOID_COUNTERS:
        assert S.void_reasons(dict(ok, counters={"model_evals": 10, k: 1})) == [f"hook {k} = 1"]
    assert S.void_reasons(dict(ok, counters={"model_evals": 0})) == ["hook model_evals = 0"]
    assert S.void_reasons(dict(ok, canary={"shard_0of4": None})) == ["shard_0of4: no passed canary"]
    assert S.void_reasons(dict(ok, canary={"shard_0of4": ["x"]})) == ["shard_0of4: no passed canary"]  # malformed
    assert S.canary_report({"canary": {"shard_0of4": ["x"]}})["shards"]["shard_0of4"]["max_abs_delta"] is None
    per = {"shard_0of4": 7, "shard_1of4": 0}                              # RCX low item 7: per shard as well
    assert S.void_reasons(dict(ok, model_evals_by_shard=per)) == ["shard_1of4: hook model_evals = 0"]
    can = {"failures": [], "counters_match": True, "max_abs_delta": 0.0, "canary_torch_version": "2.6.0+cu124",
           "host_torch_version": "2.7.0", "torch_versions_match": False}
    assert S.void_reasons(dict(ok, canary={"shard_0of4": can})) == []    # a torch mismatch is never VOID
    cr = S.canary_report({"canary": {"shard_0of4": can}})
    assert cr["torch_version_mismatch_shards"] == ["shard_0of4"] and "not VOID" in cr["torch_version_flag"]
    assert cr["shards"]["shard_0of4"]["max_abs_delta"] == 0.0


def test_merge_shards_uses_item7_merge_after_hook_merged(tmp_path, monkeypatch):
    for i, outs in enumerate(([1, 0], [1])):
        d = tmp_path / f"shard_{i}of2"
        d.mkdir()
        rows = [dict(r, id=f"seeds_fresh/{2 * j + i}") for j, r in enumerate(_rows(outs))]
        S.write_outputs(d, {"f_model_queries": 10, "wall_s": 1.0}, rows)
        (d / "hook.json").write_text(json.dumps({"describe_end": {"counters": {"calls": 3}, "reasons": {}},
                                                 "meter": {"calls": 3, "penalty_calls": i, "failsafe_calls": 0,
                                                           "penalty_calls_invalid_state": i,
                                                           "penalty_calls_critic_error": 0,
                                                           "seconds": 0.5, "per_instance": {f"k{i}": {}}},
                                                 "canary": {"probes": 25}, "rcx_modules_executed": {"x": "y"},
                                                 "torch_version": "t"}))
    real = S._load_script("bw_run_tests")
    order = []

    def merge(out, k):
        order.append(("hook_merged_exists", (out / "hook_merged.json").exists()))
        real.merge(out, k)
    monkeypatch.setattr(S, "_load_script", lambda name: types.SimpleNamespace(merge=merge))
    hook = S.merge_shards(tmp_path, 2)
    assert order == [("hook_merged_exists", True)]                       # hook_merged.json before the marker
    summ = json.loads((tmp_path / "summary.json").read_text())["summary"]
    assert summ["n"] == 3 and summ["val_success"] == pytest.approx(2 / 3) and summ["f_model_queries"] == 20
    assert hook["calls"] == 6 and hook["penalty_calls"] == 1 and hook["counters"] == {"calls": 6}
    assert hook["model_evals_by_shard"] == {"shard_0of2": 0, "shard_1of2": 0}
    assert hook["penalty_calls_invalid_state"] == 1 and set(hook["canary"]) == {"shard_0of2", "shard_1of2"}
    (tmp_path / "shard_1of2" / "hook.json").unlink()
    with pytest.raises(S.PinError):                                      # every hook shard must have its record
        S.merge_hooks(tmp_path, 2)


# ----------------------------------------------------------------------------- registration and finalisation

_HEAD = ["# PREREG X (DRAFT): title", "", "| | |", "|---|---|", "| Status | **DRAFT** |", "| Runner sha256 | `{r}` |",
         "| Tests sha256 | `{t}` |", "| Screen prereg (final) sha256 | recorded at finalisation |", "", "## 1. Body",
         "text"]


def test_registered_shas_and_finalisation_diff():                        # edits 2 and 11
    draft = "\n".join(_HEAD).format(r="a" * 64, t="b" * 64)
    assert S.registered_shas(draft) == {"runner": "a" * 64, "tests": "b" * 64}
    with pytest.raises(S.PinError):
        S.registered_shas(draft.replace("| Tests sha256 |", "| Test sha |"))
    with pytest.raises(S.PinError):
        S.registered_shas(draft + "\n| Runner sha256 | `" + "c" * 64 + "` |")
    final = draft.replace("(DRAFT)", "").replace("**DRAFT**", "**REGISTERED** (DECISIONS D-80)") \
        .replace("recorded at finalisation", "`" + "d" * 64 + "`")
    assert S.finalisation_diff(draft, final) == {"ok": True, "changed_lines": [1, 5, 8], "outside_header": []}
    for bad in (final.replace("`" + "a" * 64 + "`", "`" + "e" * 64 + "`"),   # the runner sha row
                final.replace("\ntext", "\ntext changed"),                  # the body
                final + "\nextra line",
                final.replace(": title", ": another title", 1)):            # the title may only drop "(DRAFT)"
        assert not S.finalisation_diff(draft, bad)["ok"]
    assert S.finalisation_diff(draft, draft.replace(" (DRAFT)", "", 1))["ok"]


def test_check_versions():                                              # edit 6
    frozen = {"python": "3.11", "torch": "2.4.0+cu121", "peft": "0.12", "nvidia_driver": "555.85"}
    S.check_versions(frozen, dict(frozen), types.SimpleNamespace(__version__="2.4.0+cu121"))
    for k, v in (("peft", "0.13"), ("nvidia_driver", "560.1"), ("accelerate", "1.0")):
        with pytest.raises(S.PinError):
            S.check_versions(frozen, dict(frozen, **{k: v}))
    with pytest.raises(S.PinError):
        S.check_versions(frozen, dict(frozen), types.SimpleNamespace(__version__="2.5.0"))
    v = S.library_versions()                                            # metadata only: nothing heavy imported
    assert set(S.VERSION_PACKAGES) <= set(v) and _no_heavy() == []


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
                                seeds_outputs_present=[], runner_sha="y", now="t")
    assert rec["args"] == r0["args"] == b2["args"]
    assert rec["unmet_hook"]["kwargs"] == {"invalid_policy": "fallback", "hybrid_lambda": 1.0}
    assert rec["unmet_hook"]["model_file"] == S.GNN_REL and rec["set"]["n"] == 40 and rec["set"]["shards"] == 4
    assert "go_max_p_one_sided" not in rec["rule"] and rec["rule"]["go_min_net"] == 2      # edit 3
    assert rec["rule"]["reported_only"] == ["p_one_sided_sign", "p_mcnemar_two_sided"]
    assert rec["code_path_pins"] == S.CODE_PATH_PINS and rec["late_freeze"] is False
    for k, v in (("E", 48), ("pa", "runs/BW_Pa_v3_s1/adapter")):
        bad = {"args": dict(r0["args"], **{k: v}), "pa_sha256": r0["pa_sha256"], "fwd_sha256": r0["fwd_sha256"]}
        with pytest.raises(S.PinError):
            S.check_r0_is_b2(bad, b2)
    with pytest.raises(S.PinError):
        S.check_r0_is_b2(dict(r0, pa_sha256="0" * 64), b2)


def _ns(tmp_path, prereg="PREREG_B4_SCREEN_DRAFT.md", addendum="A.md", **kw):
    return argparse.Namespace(prereg=str(tmp_path / prereg), addendum=str(tmp_path / addendum),
                              allow_draft=kw.get("allow_draft", False), allow_after_r0=kw.get("allow_after_r0", False))


def test_stage_freeze_refusals(tmp_path):
    ns = _ns(tmp_path)
    with pytest.raises(SystemExit, match="bw_seeds.py freeze"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    r0, _ = _r0_record_as_bw_seeds_writes_it()
    (tmp_path / "runs").mkdir()
    (tmp_path / S.R0_FREEZE_REL).write_text(json.dumps(r0))
    out = tmp_path / S.R0_OUT_REL / "shard_0of4"
    out.mkdir(parents=True)
    (out / "summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="before any item-7 output"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    (out / "summary.json").unlink()
    (out / "progress.txt").write_text("12:00:00 round 1: active 10 -> 7; F queries 900\n")   # R0 shard mid-run
    with pytest.raises(SystemExit, match="before any item-7 output"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    (out / "progress.txt").unlink()
    with pytest.raises(SystemExit, match="is a draft"):
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    assert not (tmp_path / S.FREEZE_REL).exists() and not (tmp_path / S.OUT_REL).exists()


def test_freeze_guard_covers_all_item7_outputs(tmp_path):              # edit 9
    r0, _ = _r0_record_as_bw_seeds_writes_it()
    (tmp_path / "runs").mkdir()
    (tmp_path / S.R0_FREEZE_REL).write_text(json.dumps(r0))
    d = tmp_path / "runs/BW_SEEDS/R0_direct"
    d.mkdir(parents=True)
    (d / "plans.jsonl").write_text("{}\n")                               # R0_direct reveals per-instance outcomes
    assert S.bw_seeds_outputs(tmp_path) == ["runs/BW_SEEDS/R0_direct/plans.jsonl"]
    with pytest.raises(SystemExit, match="before any item-7 output"):
        S.stage_freeze(_ns(tmp_path), root=tmp_path, rcx_root=tmp_path)
    (d / "plans.jsonl").unlink()
    log = tmp_path / S.BW_SEEDS_LOG_REL
    log.write_text("wrote 6 freeze records\n")                           # the i7-freeze line only: not an output
    assert S.bw_seeds_outputs(tmp_path) == []
    log.write_text("wrote 6 freeze records\n[seeds] run R0 direct\n")   # a run has started
    assert S.bw_seeds_outputs(tmp_path) == [S.BW_SEEDS_LOG_REL]
    with pytest.raises(SystemExit, match="before any item-7 output"):
        S.stage_freeze(_ns(tmp_path), root=tmp_path, rcx_root=tmp_path)
    for marker in S.CLOSING_MARKERS:                                     # no late freeze once these exist
        (tmp_path / marker).write_text("{}")
        with pytest.raises(SystemExit, match="can no longer be frozen"):
            S.stage_freeze(_ns(tmp_path, allow_after_r0=True), root=tmp_path, rcx_root=tmp_path)
        (tmp_path / marker).unlink()
    assert S.closing_markers(tmp_path) == []
    log.write_text(log.read_text() + '{\n "systems": {},\n "release_gate_reading": "not replicated"\n}\n')
    assert S.closing_markers(tmp_path) == [f"{S.BW_SEEDS_LOG_REL} (contains 'release_gate_reading')"]  # low item 5
    with pytest.raises(SystemExit, match="can no longer be frozen"):     # item 7's reading is in the log
        S.stage_freeze(_ns(tmp_path, allow_after_r0=True), root=tmp_path, rcx_root=tmp_path)


def _registered_texts(tmp_path, runner=None, tests=None, addendum_runner=None, addendum_prereg=None):
    tests_file = tmp_path / S.TESTS_REL
    tests_file.parent.mkdir(parents=True, exist_ok=True)
    if not tests_file.exists():
        tests_file.write_text("# the registered tests (stand-in bytes)\n")
    runner = runner or S.sha256_file(Path(S.__file__))
    tests = tests or S.sha256_file(tests_file)
    (tmp_path / "PREREG_B4_SCREEN.md").write_text("\n".join(_HEAD).format(r=runner, t=tests))
    prereg_sha = addendum_prereg or S.sha256_file(tmp_path / "PREREG_B4_SCREEN.md")
    (tmp_path / "PREREG_ITEM7_ADDENDUM_A.md").write_text(f"| | |\n|---|---|\n| Screen prereg (final) sha256 | "
                                                          f"`{prereg_sha}` |\n\nrunner `{addendum_runner or runner}`\n")
    return _ns(tmp_path, prereg="PREREG_B4_SCREEN.md", addendum="PREREG_ITEM7_ADDENDUM_A.md")


def test_addendum_prereg_row():                                          # RCX low item 2
    row = "| Screen prereg (final) sha256 | `" + "a" * 64 + "` |"
    assert S.addendum_prereg_sha("| | |\n|---|---|\n" + row + "\n") == "a" * 64
    for bad in ("| Screen prereg (final) sha256 | recorded at finalisation |", row + "\n" + row, ""):
        with pytest.raises(S.PinError):
            S.addendum_prereg_sha(bad)
    draft = ROOT / "research/PREREG_ITEM7_ADDENDUM_A_DRAFT.md"
    if draft.exists():                                                   # the draft cannot be frozen against
        with pytest.raises(S.PinError):
            S.addendum_prereg_sha(draft.read_text(encoding="utf-8"))


def test_freeze_refuses_unregistered_runner_also_late(tmp_path, monkeypatch):   # edit 2; RCX low item 2
    r0, _ = _r0_record_as_bw_seeds_writes_it()
    (tmp_path / "runs").mkdir()
    (tmp_path / S.R0_FREEZE_REL).write_text(json.dumps(r0))
    d = tmp_path / S.R0_OUT_REL / "shard_0of4"
    d.mkdir(parents=True)
    (d / "progress.txt").write_text("x\n")                               # late: R0 has started
    for kw in ({"runner": "0" * 64}, {"tests": "1" * 64}, {"addendum_runner": "2" * 64},
               {"addendum_prereg": "3" * 64}):                         # the addendum names another prereg
        ns = _registered_texts(tmp_path, **kw)
        ns.allow_after_r0 = True
        with pytest.raises(SystemExit, match="freeze refused"):
            S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    ns = _registered_texts(tmp_path)
    ns.allow_after_r0 = True
    monkeypatch.setattr(S, "CANARY_SHA256", None)
    with pytest.raises(SystemExit, match="placeholder"):                 # edit 8: no canary sha, no freeze
        S.stage_freeze(ns, root=tmp_path, rcx_root=tmp_path)
    assert not (tmp_path / S.FREEZE_REL).exists() and not (tmp_path / S.OUT_REL).exists()


def _schema_canary(torch_version="2.6.0+cu124"):
    """A canary record that passes load_canary's schema check (its values are never computed: the freeze does not run
    the hook)."""
    return json.dumps({"what": "schema-only test canary", "tolerance_abs": 1e-5, "torch_version": torch_version,
                       "hook_call": {"path": S.GNN_REL, "invalid_policy": "fallback", "hybrid_lambda": 1.0},
                       "expected_counters_after_all": {"calls": 1}, "pins_sha256": dict(S.RCX_CODE_PINS),
                       "probes": [{"id": "p0", "state": [["handempty"]], "goal": [], "consistent": True,
                                   "expected": 0.0}]})


def _copy_rcx(dst, canary_text=None):
    for rel in {**S.RCX_CODE_PINS, S.GNN_REL: 0, S.GNN_SIDECAR_REL: 0, S.E07_INSTANCES_REL: 0}:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S.RCX_ROOT / rel, dst / rel)
    (dst / S.CANARY_REL).parent.mkdir(parents=True, exist_ok=True)
    (dst / S.CANARY_REL).write_text(_schema_canary() if canary_text is None else canary_text)
    return S.sha256_file(dst / S.CANARY_REL)


@NEEDS_RCX
def test_freeze_success_snapshot_and_reverify(tmp_path, monkeypatch, capsys):     # edits 2, 6, 9, 10
    root, rcx = tmp_path / "root", tmp_path / "rcx"
    (root / "runs").mkdir(parents=True)
    r0, b2 = _r0_record_as_bw_seeds_writes_it()
    (root / S.R0_FREEZE_REL).write_text(json.dumps(r0))
    shutil.copyfile(ROOT / S.B2_BEAM_FREEZE_REL, root / S.B2_BEAM_FREEZE_REL)
    bad_sha = _copy_rcx(tmp_path / "rcx_bad", '{"probes": []}')        # a canary that fails the schema check
    monkeypatch.setattr(S, "CANARY_SHA256", bad_sha)
    monkeypatch.setattr(S, "current_pins", lambda r, x, **k: {"stub": "pins"})
    monkeypatch.setattr(S, "precheck", lambda *a, **k: {"drop_ids": [], "n_after_drop": 40})
    monkeypatch.setattr(S, "check_adapters", lambda r, rec: None)
    monkeypatch.setattr(S, "runtime_versions", lambda: {"torch": "t", "nvidia_driver": "d"})
    ns = _registered_texts(root)
    with pytest.raises(SystemExit, match="freeze refused"):
        S.stage_freeze(ns, root=root, rcx_root=tmp_path / "rcx_bad")
    assert not (root / S.FREEZE_REL).exists() and not (root / S.OUT_REL).exists()
    monkeypatch.setattr(S, "CANARY_SHA256", _copy_rcx(rcx))
    assert S.stage_freeze(ns, root=root, rcx_root=rcx) == 0
    assert "WARNING (report-only, not a refusal)" in capsys.readouterr().out   # canary torch 2.6.0+cu124 vs host "t"
    rec = json.loads((root / S.FREEZE_REL).read_text())
    assert "go_max_p_one_sided" not in rec["rule"] and rec["late_freeze"] is False
    assert rec["registered"]["runner_sha256"] == S.sha256_file(Path(S.__file__)) == rec["runner_sha256"]
    assert rec["versions"] == {"torch": "t", "nvidia_driver": "d"} and rec["canary"]["sha256"] == S.CANARY_SHA256
    assert rec["canary"]["canary_torch_version"] == "2.6.0+cu124" and rec["canary"]["host_torch_version_at_freeze"] == "t"
    assert rec["canary"]["torch_versions_match_at_freeze"] is False                # recorded, not refused
    assert rec["prereg"]["sha256"] == S.sha256_file(root / "PREREG_B4_SCREEN.md")
    assert set(rec["rcx_snapshot"]["files"]) == set(S.rcx_snapshot_files())
    for rel, sha in rec["rcx_snapshot"]["files"].items():
        assert S.sha256_file(root / S.SNAPSHOT_REL / rel) == sha
    assert (root / S.OUT_REL / "PRECHECK.json").exists()
    with pytest.raises(SystemExit, match="written once"):
        S.stage_freeze(ns, root=root, rcx_root=rcx)
    assert S.load_and_verify_freeze(root)["runner_sha256"] == rec["runner_sha256"]
    with monkeypatch.context() as mp:                                    # RCX low item 8: the placeholder path
        mp.setattr(S, "CANARY_SHA256", None)
        with pytest.raises(S.PinError, match="canary"):
            S.load_and_verify_freeze(root)
        with pytest.raises(S.PinError, match="placeholder"):
            S.rcx_snapshot_files()
    mod, imp = S.load_rcx_hook_module(root / S.SNAPSHOT_REL)              # run and analyze load from the snapshot
    try:
        assert Path(mod.__file__).resolve() == (root / S.SNAPSHOT_REL / "rcx/bw/b4_hook.py").resolve()
    finally:
        S.unload_rcx(imp)
    with open(root / S.SNAPSHOT_REL / "rcx/bw/space.py", "ab") as f:
        f.write(b"\n")
    with pytest.raises(S.PinError):
        S.load_and_verify_freeze(root)


def test_current_pins_enforce_the_code_path(tmp_path):                   # edit 6
    pins = S.current_pins(ROOT, S.RCX_ROOT, with_snapshot_files=False) if HAVE_RCX else None
    if pins is not None:
        assert all(pins[rel] == sha for rel, sha in S.CODE_PATH_PINS.items())
    for rel in [S.PLANNER_REL, S.BLOCKS_REL, S.B2_BEAM_FREEZE_REL, S.ITEM7_PREREG_REL, S.SEEDS_FRESH_REL,
                *S.CODE_PATH_PINS]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    with open(tmp_path / "bwmn/lm.py", "ab") as f:
        f.write(b"\n# changed\n")
    with pytest.raises(S.PinError, match="bwmn/lm.py"):
        S.current_pins(tmp_path, S.RCX_ROOT, with_snapshot_files=False)


# ----------------------------------------------------------------------------- the GPU path, stubbed (edit 13)

INT_SIZES = (5, 5, 6, 6, 6, 7, 7, 8)
STUB_VERSIONS = {"python": "3.x", "torch": "0.0+b4s-stub", "nvidia_driver": "stub"}


def _registered_stub(hook_mod, rcx_root):
    """The dry-run stub hook whose describe() reports the registered call (the GNN forward is still 0.0)."""
    Stub = S.make_stub_hook_class(hook_mod)

    class RegStub(Stub):
        def describe(self):
            d = super().describe()
            d.update(model_file=str(Path(rcx_root) / S.GNN_REL), model_sha256=S.GNN_SHA256,
                     sidecar_sha256=S.GNN_SIDECAR_SHA256, config_source="sidecar json 'config'",
                     code_sha256={rel: S.RCX_CODE_PINS[rel] for rel in S.RCX_D76_FILES})
            return d
    return RegStub(**S.HOOK_KWARGS)


def _fake_val(B):
    def val(problem_path, plan, domain_path=None):                       # VAL semantics via the exact executor
        objs, init, goal = B.parse_problem(problem_path)
        return bool(B.execute(init, goal, list(plan))["success"])
    return val


@pytest.fixture(scope="module")
def screen_tree(tmp_path_factory):
    """R0 written as blocks_eval/bw_seeds write it, addendum A.0's table, then arm a through stage_run_shard (x4, the
    real code with model loading stubbed) and merge_shards."""
    if not HAVE_RCX:
        pytest.skip("RCX worktree absent")
    root = tmp_path_factory.mktemp("b4s_root")
    B, BP = S.exec_pinned_planner(S.make_stub_lm())
    inst = [dict(it, id=f"seeds_fresh/{i}") for i, it in enumerate(S.dry_instances(B, INT_SIZES))]
    args = dict(json.loads((ROOT / S.B2_BEAM_FREEZE_REL).read_text())["args"])
    rec = {"args": args, "precheck": {"drop_ids": [], "n_after_drop": len(inst)},
           "rule": {"go_min_net": 2, "penalty_share_max": 0.05, "reported_only": list(S.REPORTED_ONLY)},
           "prereg": {"path": "stub"}, "versions": dict(STUB_VERSIONS)}
    (root / "runs").mkdir()
    (root / S.FREEZE_REL).write_text(json.dumps(rec))
    out = root / S.OUT_REL
    out.mkdir(parents=True)
    (out / "PRECHECK.json").write_text(json.dumps({"drop_ids": [], "n_after_drop": len(inst)}))
    snap = root / S.SNAPSHOT_REL
    mod, imp = REAL_LOADER()
    try:
        probes = [(it["init"], it["goal"], True) for it in inst[:3]]
        on = sorted(f for f in inst[0]["init"] if f[0] == "on") or [("on", "x", "y")]
        probes.append((inst[0]["init"] | {("holding", on[0][1])}, inst[0]["goal"], False))
        canary_sha, _ = _write_canary(snap / S.CANARY_REL, lambda: _registered_stub(mod, snap), probes,
                                      torch_version="2.6.0+cu124")          # != the stub torch: report-only flag
    finally:
        S.unload_rcx(imp)
    captured, made, hold, roots = [], [], {}, []
    real_plan_beam = BP.plan_beam

    def spy(*a, **k):
        captured.append(k.get("unmet"))
        return real_plan_beam(*a, **k)

    def fake_loader(rcx_root=None, pins=None):
        roots.append(rcx_root)
        assert Path(rcx_root).resolve() == snap.resolve()               # run-shard loads RCX code from the snapshot
        if hold.get("imp"):
            S.unload_rcx(hold["imp"])
        m, i2 = REAL_LOADER(S.RCX_ROOT)
        i2.executed["rcx/bw/models.py"] = S.RCX_CODE_PINS["rcx/bw/models.py"]   # as the GPU process records it
        hold["imp"] = i2
        return m, i2

    def fake_make(hook_mod, rcx_root):
        h = _registered_stub(hook_mod, rcx_root)
        made.append(h)
        return h

    torch_stub = types.SimpleNamespace(__version__=STUB_VERSIONS["torch"],
                                       cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(B, "val_check", _fake_val(B))
        for i in range(S.SHARDS):                                        # R0: unmet=None, blocks_eval's records
            d = root / S.R0_OUT_REL / f"shard_{i}of{S.SHARDS}"
            part = [it for it in inst if S.shard_of(it["id"]) == i]
            summ, rows = S.run_arm(BP, B, S.ScriptedModel(B, inconsistent=0.0), S.FakeTok(), part, args, d,
                                   unmet=None, val=True)
            for r in rows:
                r.pop("expansions")
            ns = dict(S.expected_r0_shard_args(args, i), out=f"runs\\BW_SEEDS\\R0_beam\\shard_{i}of4")
            (d / "summary.json").write_text(json.dumps({"args": ns, "summary": summ}, indent=1))
            with open(d / "plans.jsonl", "w", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
        S._load_script("bw_run_tests").merge(root / S.R0_OUT_REL, S.SHARDS)
        table = {rel: {"sha256": S.sha256_file(root / rel)} for rel in S.r0_files(root)}
        (root / S.ADDENDUM_A_INPUTS_REL).write_text(json.dumps({"files": table}))
        BP.plan_beam = spy
        mp.setattr(S, "load_and_verify_freeze", lambda r: rec)
        mp.setattr(S, "load_screen_instances", lambda r: inst)
        mp.setattr(S, "load_gpu_stack", lambda r, a: (B, BP, S.ScriptedModel(B, inconsistent=0.0), S.FakeTok(),
                                                      torch_stub))
        mp.setattr(S, "runtime_versions", lambda: dict(STUB_VERSIONS))
        mp.setattr(S, "load_rcx_hook_module", fake_loader)
        mp.setattr(S, "make_registered_hook", fake_make)
        mp.setattr(S, "CANARY_SHA256", canary_sha)
        try:
            for i in range(S.SHARDS):
                assert S.stage_run_shard(argparse.Namespace(arm="a", shard=f"{i}/{S.SHARDS}"), root=root) == 0
            with pytest.raises(SystemExit, match="runs once"):
                S.stage_run_shard(argparse.Namespace(arm="a", shard=f"0/{S.SHARDS}"), root=root)
        finally:
            BP.plan_beam = real_plan_beam
            if hold.get("imp"):
                S.unload_rcx(hold["imp"])
        S.merge_shards(root / S.OUT_REL / "a_beam", S.SHARDS)
    yield {"root": root, "inst": inst, "rec": rec, "captured": captured, "made": made, "B": B, "roots": roots}


def test_stage_run_shard_passes_the_meter_and_counts_match(screen_tree):  # edit 13
    root, captured, made = screen_tree["root"], screen_tree["captured"], screen_tree["made"]
    assert len(captured) == S.SHARDS and len(made) == 2 * S.SHARDS       # per shard: the canary hook, the run hook
    assert len(screen_tree["roots"]) == S.SHARDS                         # every shard loaded from the snapshot
    for i, unmet in enumerate(captured):
        assert isinstance(unmet, S.HookMeter) and unmet.hook is made[2 * i + 1] and unmet.classify is not None
        h = json.loads((root / S.OUT_REL / f"a_beam/shard_{i}of4/hook.json").read_text())
        assert h["meter"]["calls"] == unmet.calls == h["describe_end"]["counters"]["calls"] > 0
        assert h["describe_start"]["counters"]["calls"] == 0
        assert h["canary"]["probes"] == 4 and h["canary"]["counters_match"] and h["canary"]["failures"] == []
        assert h["canary"]["canary_torch_version"] == "2.6.0+cu124"      # D-80 follow-up: both versions recorded
        assert h["canary"]["host_torch_version"] == STUB_VERSIONS["torch"] and h["canary"]["torch_versions_match"] is False
        assert h["canary"]["max_abs_delta"] == 0.0 and len(h["canary"]["abs_delta_by_probe"]) == 4
        assert "rcx/bw/models.py" in h["rcx_modules_executed"] and h["torch_version"] == STUB_VERSIONS["torch"]
        s = json.loads((root / S.OUT_REL / f"a_beam/shard_{i}of4/summary.json").read_text())
        assert s["args"] == screen_tree["rec"]["args"] and s["shard"] == f"{i}/4" and s["arm"] == "a"
        assert s["summary"]["hook_calls"] == unmet.calls
    merged = json.loads((root / S.OUT_REL / "a_beam/hook_merged.json").read_text())
    assert merged["calls"] == sum(m.calls for m in captured) and len(merged["canary"]) == S.SHARDS
    ev = [json.loads(x) for x in (root / S.ATTEMPTS_REL).read_text().splitlines()]
    assert [e["event"] for e in ev] == ["start", "end"] * S.SHARDS and not any(e.get("warm_fcache") for e in ev)
    assert not any(k == "rcx" or k.startswith("rcx.") for k in sys.modules)


def _analyze(root, monkeypatch, tree, **rec_over):
    rec = dict(tree["rec"], **rec_over)
    monkeypatch.setattr(S, "load_and_verify_freeze", lambda r: rec)
    monkeypatch.setattr(S, "load_screen_instances", lambda r: tree["inst"])
    monkeypatch.setattr(S, "N_FRESH", len(tree["inst"]))
    S.stage_analyze(argparse.Namespace(rerun=False), root=root, rcx_root=S.RCX_ROOT)
    return json.loads((root / S.OUT_REL / "B4S_REPORT.json").read_text())


def _copy_tree(tree, tmp_path):
    dst = tmp_path / "r"
    shutil.copytree(tree["root"], dst)
    return dst


def test_stage_analyze_end_to_end(screen_tree, tmp_path, monkeypatch):     # edits 1, 7, 8, 9
    root = _copy_tree(screen_tree, tmp_path)
    rep = _analyze(root, monkeypatch, screen_tree)
    assert rep["status"] in ("GO", "NO-GO") and rep["decision"]["decision"] == rep["status"]
    assert rep["round1_identity"]["holds"] and rep["n_paired"] == len(screen_tree["inst"])
    assert rep["certificate_c_eq_cpen_on_control"]["holds"] and rep["f_cache_cross_check"]["common_different_text"] == 0
    assert rep["arm_unpenalised_inconsistent_report_only"]["strict_inconsistent"] == 0
    assert rep["hook"]["penalty_calls"] == 0 and rep["hook"]["calls"] >= len(screen_tree["inst"])
    assert set(rep["hook_shards"]) == {f"shard_{i}of4" for i in range(4)}
    assert "control_failures_unmet_at_submitted_prefix" in rep and "arm_failures_unmet_at_submitted_prefix" in rep
    assert rep["attempts"]["starts_per_shard"] == {f"a {i}/4": 1 for i in range(4)}
    inputs = json.loads((root / S.OUT_REL / "ANALYSIS_INPUTS.json").read_text())
    assert set(S.r0_files(root)) <= set(inputs) and S.ADDENDUM_A_INPUTS_REL in inputs and S.ATTEMPTS_REL in inputs
    cr = rep["canary_report_only"]                                        # D-80 follow-up: report-only, never VOID
    assert set(cr["shards"]) == {f"shard_{i}of4" for i in range(4)}
    assert all(v["max_abs_delta"] == 0.0 and v["canary_torch_version"] == "2.6.0+cu124"
               and v["host_torch_version"] == STUB_VERSIONS["torch"] for v in cr["shards"].values())
    assert cr["torch_version_mismatch_shards"] == sorted(cr["shards"]) and "not VOID" in cr["torch_version_flag"]
    md = (root / S.OUT_REL / "B4S_REPORT.md").read_text()
    assert md.startswith("# B4 futility screen") and "| shard_0of4 | 0.0 | 2.6.0+cu124 |" in md
    with pytest.raises(SystemExit, match="written once"):
        S.stage_analyze(argparse.Namespace(rerun=False), root=root, rcx_root=S.RCX_ROOT)
    with pytest.raises(SystemExit, match="INSTRUMENT_FAULT"):            # no re-run without a fault report
        S.stage_run(argparse.Namespace(rerun=True), root=root)


def _edit_json(p, f):
    d = json.loads(p.read_text())
    f(d)
    p.write_text(json.dumps(d))


def _edit_rows(p, f):
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    f(rows)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


A = S.OUT_REL + "/a_beam"


def _edit_shard_hooks(r, f, shards=range(4)):
    """Edit the shard hook records, then rewrite hook_merged.json as merge_shards does (a consistent arm tree)."""
    for i in shards:
        _edit_json(r / A / f"shard_{i}of4/hook.json", f)
    (r / A / "hook_merged.json").write_text(json.dumps(S.merge_hooks(r / A, S.SHARDS), indent=1))


REFUSALS = {   # edit 1: each of these refuses (no report is written)
    "hook_merged_differs_from_shards": lambda r: _edit_json(r / A / "hook_merged.json",   # hides the shard canaries
                                                            lambda d: d.update(canary={})),
    "precheck_differs": lambda r: _edit_json(r / S.OUT_REL / "PRECHECK.json",
                                             lambda d: d.update(drop_ids=["seeds_fresh/0"])),
    "hook_json_missing": lambda r: (r / A / "shard_2of4/hook.json").unlink(),
    "hook_calls_below_n": lambda r: _edit_json(r / A / "shard_1of4/hook.json", lambda d: d["meter"].update(calls=1)),
    "arm_row_without_val": lambda r: _edit_rows(r / A / "plans.jsonl", lambda rows: rows[0].pop("val")),
    "arm_shard_args": lambda r: _edit_json(r / A / "shard_3of4/summary.json", lambda d: d["args"].update(E=48)),
    "not_the_registered_hook": lambda r: _edit_json(r / A / "shard_0of4/hook.json",
                                                    lambda d: d["describe_end"].update(hybrid_lambda=0.0)),
    "attempts_deleted": lambda r: (r / S.ATTEMPTS_REL).unlink(),         # RCX low item 3
    "attempts_end_event_missing": lambda r: (r / S.ATTEMPTS_REL).write_text(
        "".join(x + "\n" for x in (r / S.ATTEMPTS_REL).read_text().splitlines()[:-1])),
    "attempts_end_event_twice": lambda r: S.log_attempt(r, stage="run-shard", arm="a", shard="2/4", event="end"),
}


def _r0_shard_args_changed(r):
    _edit_json(r / S.R0_OUT_REL / "shard_0of4/summary.json", lambda d: d["args"].update(W=8))


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_stage_analyze_refusals(screen_tree, tmp_path, monkeypatch, case):
    root = _copy_tree(screen_tree, tmp_path)
    REFUSALS[case](root)
    with pytest.raises((SystemExit, S.PinError)):
        _analyze(root, monkeypatch, screen_tree)
    assert not (root / S.OUT_REL / "B4S_REPORT.json").exists()


def test_stage_analyze_takes_the_rule_from_the_freeze(screen_tree, tmp_path, monkeypatch):
    for rule in ({"go_min_net": 3, "penalty_share_max": 0.05}, {"go_min_net": 2, "penalty_share_max": 0.05,
                                                                 "go_max_p_one_sided": "1/5"}):
        root = _copy_tree(screen_tree, tmp_path / str(len(rule)))
        with pytest.raises(SystemExit, match="rule"):
            _analyze(root, monkeypatch, screen_tree, rule=rule)


def _first_arm_round1_mismatch(rows):
    r = next(x for x in rows if not (x["verified"] and x["rounds"] == 1))
    r.update(verified=True, rounds=1)


VOIDS = {   # edits 8 and 9: a VOID report, never a GO/NO-GO (hook cases: in the shard records, merged consistently)
    "r0_differs_from_addendum_a0": lambda r: (r / S.R0_OUT_REL / "plans.jsonl").write_text(
        (r / S.R0_OUT_REL / "plans.jsonl").read_text() + "\n"),
    "model_errors": lambda r: _edit_shard_hooks(r, lambda d: d["describe_end"]["counters"].update(model_errors=2),
                                                shards=[2]),
    "model_evals_zero": lambda r: _edit_shard_hooks(r, lambda d: d["describe_end"]["counters"].update(model_evals=0)),
    "model_evals_zero_in_one_shard": lambda r: _edit_shard_hooks(       # RCX low item 7
        r, lambda d: d["describe_end"]["counters"].update(model_evals=0), shards=[1]),
    "canary_missing": lambda r: _edit_shard_hooks(r, lambda d: d.update(canary=None), shards=[0]),
    "addendum_a0_missing": lambda r: (r / S.ADDENDUM_A_INPUTS_REL).unlink(),
    "canary_failed_event": lambda r: S.log_attempt(r, stage="run-shard", arm="a", shard="1/4", event="canary_failed",
                                                   error="probe c03"),
    # RCX low item 1: R0's shard summary changed after A.0 -> VOID, although check_r0_shards would also refuse
    "r0_shard_summary_changed_after_a0": _r0_shard_args_changed,
    # a VOID wins over refusals: model errors in a shard record and a hook_merged.json that differs from the shards
    "void_with_refusal": lambda r: (_edit_shard_hooks(r, lambda d: d["describe_end"]["counters"].update(model_errors=1),
                                                      shards=[3]),
                                    _edit_json(r / A / "hook_merged.json", lambda d: d.update(calls=0))),
    # ... also when a shard hook record is malformed (wrong JSON shape): collected as a refusal, never raised
    "void_with_malformed_hook_record": lambda r: (VOIDS["r0_differs_from_addendum_a0"](r),
                                                  _edit_json(r / A / "shard_0of4/hook.json",
                                                             lambda d: d.update(describe_end=None))),
}


@pytest.mark.parametrize("case", sorted(VOIDS))
def test_stage_analyze_void(screen_tree, tmp_path, monkeypatch, case):
    root = _copy_tree(screen_tree, tmp_path)
    VOIDS[case](root)
    rep = _analyze(root, monkeypatch, screen_tree)
    assert rep["status"] == "VOID" and rep["void_reasons"] and "decision" not in rep
    assert rep["wording"] == S.WORDING["VOID"]
    if case == "model_evals_zero_in_one_shard":
        assert rep["void_reasons"] == ["shard_1of4: hook model_evals = 0"]
    if case == "r0_shard_summary_changed_after_a0":
        assert any("sha256 differs" in x for x in rep["void_reasons"])
        assert any("'W': (8, 4)" in x for x in rep["r0_args_problems"])
    if case == "void_with_refusal":
        assert rep["void_reasons"] == ["hook model_errors = 1"]
        assert any("hook_merged.json" in x for x in rep["refusal_problems_report_only"])
    if case == "void_with_malformed_hook_record":
        assert any("sha256 differs" in x for x in rep["void_reasons"])
        assert any("hook shard records" in x for x in rep["refusal_problems_report_only"])
    if case not in ("canary_missing", "void_with_malformed_hook_record"):
        assert set(rep["canary_report_only"]["shards"]) == {f"shard_{i}of4" for i in range(4)}


def test_void_report_when_the_arm_never_ran_and_canary_failure_is_final(screen_tree, tmp_path, monkeypatch):
    for case in ("r0_differs_from_addendum_a0", "canary_failed_event"):  # run refused, so there is no arm
        r = _copy_tree(screen_tree, tmp_path / case)
        shutil.rmtree(r / A)
        VOIDS[case](r)
        rep = _analyze(r, monkeypatch, screen_tree)
        assert rep["status"] == "VOID" and any("not completed" in x for x in rep["void_reasons"])
    r = _copy_tree(screen_tree, tmp_path / "rerun_after_canary")             # no shard runs after a canary failure
    (r / A / "shard_1of4" / "summary.json").unlink()
    _edit_rows(r / S.ATTEMPTS_REL, lambda ev: ev.remove(next(e for e in ev if e["event"] == "end"
                                                             and e["shard"] == "1/4")))
    VOIDS["canary_failed_event"](r)
    monkeypatch.setattr(S, "load_and_verify_freeze", lambda root: screen_tree["rec"])
    with pytest.raises(SystemExit, match="screen VOID"):
        S.stage_run_shard(argparse.Namespace(arm="a", shard="1/4"), root=r)
    rep = _analyze(_copy_tree({"root": r}, tmp_path / "void_copy"), monkeypatch, screen_tree)
    assert rep["status"] == "VOID" and any("canary failed" in x for x in rep["void_reasons"])
    (r / S.ATTEMPTS_REL).unlink()                                        # RCX low item 3: deleting the record
    monkeypatch.setattr(S.subprocess, "call", lambda *a, **k: pytest.fail("no shard may start"))
    with pytest.raises(SystemExit, match="ATTEMPTS.jsonl missing"):      # ... does not erase the canary failure
        S.stage_run_shard(argparse.Namespace(arm="a", shard="1/4"), root=r)
    with pytest.raises(SystemExit):                                      # a refusal, never a reading
        _analyze(r, monkeypatch, screen_tree)
    assert not (r / S.OUT_REL / "B4S_REPORT.json").exists()
    r2 = _copy_tree(screen_tree, tmp_path / "canary_first_shard")          # a canary failure on the very first shard
    shutil.rmtree(r2 / A)
    (r2 / S.ATTEMPTS_REL).unlink()
    (r2 / A / "shard_0of4").mkdir(parents=True)                           # run-shard creates it before the canary
    S.log_attempt(r2, stage="run-shard", arm="a", shard="0/4", event="start", warm_fcache=False, fcache_lines=0)
    VOIDS["canary_failed_event"](r2)
    (r2 / S.ATTEMPTS_REL).unlink()
    with pytest.raises(SystemExit, match="ATTEMPTS.jsonl missing"):
        S.stage_run_shard(argparse.Namespace(arm="a", shard="0/4"), root=r2)


def test_stage_analyze_instrument_fault(screen_tree, tmp_path, monkeypatch):   # edit 7
    root = _copy_tree(screen_tree, tmp_path)
    _edit_rows(root / A / "plans.jsonl", _first_arm_round1_mismatch)
    rep = _analyze(root, monkeypatch, screen_tree)
    assert rep["status"] == "INSTRUMENT_FAULT" and rep["decision"] is None
    assert len(rep["round1_identity"]["mismatched_ids"]) == 1
    assert S._rerun_allowed(root)


def test_stage_run_guards(screen_tree, tmp_path, monkeypatch):           # edits 7 and 9, before any GPU work
    monkeypatch.setattr(S, "load_and_verify_freeze", lambda r: screen_tree["rec"])
    monkeypatch.setattr(S.subprocess, "call", lambda *a, **k: pytest.fail("no shard may start"))
    root = _copy_tree(screen_tree, tmp_path)
    assert S.stage_run(argparse.Namespace(rerun=False), root=root) == 0   # arm complete: nothing to run
    for case in ("r0_differs_from_addendum_a0", "addendum_a0_missing"):
        r = _copy_tree(screen_tree, tmp_path / case)
        shutil.rmtree(r / A)
        VOIDS[case](r)
        with pytest.raises(SystemExit, match="screen VOID"):
            S.stage_run(argparse.Namespace(rerun=False), root=r)
    r = _copy_tree(screen_tree, tmp_path / "args")
    _r0_shard_args_changed(r)
    with pytest.raises(S.PinError):
        S.stage_run(argparse.Namespace(rerun=False), root=r)


def test_r0_args_fault_is_an_instrument_fault_that_unlocks_the_rerun(screen_tree, tmp_path, monkeypatch):   # edit 7
    """R0 ran with other arguments (and A.0 hashed those outputs): run refuses before any GPU work; analyze then reports
    an INSTRUMENT_FAULT (no decision), so the pre-registered re-run of c and a in one turn can follow."""
    root = _copy_tree(screen_tree, tmp_path)
    shutil.rmtree(root / A)                                              # the arm never ran
    for i in range(S.SHARDS):
        _edit_json(root / S.R0_OUT_REL / f"shard_{i}of4/summary.json", lambda d: d["args"].update(W=8))
    table = {rel: {"sha256": S.sha256_file(root / rel)} for rel in S.r0_files(root)}
    (root / S.ADDENDUM_A_INPUTS_REL).write_text(json.dumps({"files": table}))   # A.0 hashed R0 as it ran
    monkeypatch.setattr(S, "load_and_verify_freeze", lambda r: screen_tree["rec"])
    calls = []
    monkeypatch.setattr(S.subprocess, "call", lambda cmd, **k: calls.append(cmd) or 1)
    with pytest.raises(S.PinError, match="W"):
        S.stage_run(argparse.Namespace(rerun=False), root=root)
    with pytest.raises(SystemExit, match="INSTRUMENT_FAULT"):           # no re-run before the fault is reported
        S.stage_run(argparse.Namespace(rerun=True), root=root)
    assert calls == []
    rep = _analyze(root, monkeypatch, screen_tree)
    assert rep["status"] == "INSTRUMENT_FAULT" and rep["decision"] is None and "paired" not in rep
    assert any("'W': (8, 4)" in x for x in rep["instrument_fault_reasons"])
    assert rep["wording"] == S.WORDING["INSTRUMENT_FAULT_R0_ARGS"]
    assert "Instrument fault reasons" in (root / S.OUT_REL / "B4S_REPORT.md").read_text()
    assert S._rerun_allowed(root)
    assert S.stage_run(argparse.Namespace(rerun=True), root=root) == 1   # the stub shard "fails": c is started first
    assert calls and calls[0][-4:] == ["--arm", "rerun_c", "--shard", "0/4"]
    r2 = _copy_tree(screen_tree, tmp_path / "void_first")               # R0 also differs from A.0: VOID first
    shutil.rmtree(r2 / A)
    for i in range(S.SHARDS):
        _edit_json(r2 / S.R0_OUT_REL / f"shard_{i}of4/summary.json", lambda d: d["args"].update(W=8))
    rep = _analyze(r2, monkeypatch, screen_tree)
    assert rep["status"] == "VOID" and rep["r0_args_problems"]
    r3 = _copy_tree(screen_tree, tmp_path / "after_the_arm")            # R0's args fail although arm a ran and A.0
    for i in range(S.SHARDS):                                            # matches: INSTRUMENT FAULT > refusal
        _edit_json(r3 / S.R0_OUT_REL / f"shard_{i}of4/summary.json", lambda d: d["args"].update(W=8))
    table = {rel: {"sha256": S.sha256_file(r3 / rel)} for rel in S.r0_files(r3)}
    (r3 / S.ADDENDUM_A_INPUTS_REL).write_text(json.dumps({"files": table}))
    rep = _analyze(r3, monkeypatch, screen_tree)
    assert rep["status"] == "INSTRUMENT_FAULT" and rep["decision"] is None and "paired" not in rep
    assert rep["wording"] == S.WORDING["INSTRUMENT_FAULT_R0_ARGS"] and S._rerun_allowed(r3)
    assert set(rep["canary_report_only"]["shards"]) == {f"shard_{i}of4" for i in range(4)}   # the arm ran: shown
    assert "| shard_0of4 | 0.0 | 2.6.0+cu124 |" in (r3 / S.OUT_REL / "B4S_REPORT.md").read_text()


def _stand_in_rerun_shards(root, rec):
    """Stand-ins for the re-run shards (the GPU path is tested above), with the attempt records run-shard writes."""
    out = root / S.OUT_REL
    for i in range(S.SHARDS):
        dc, da = out / f"rerun/c_beam/shard_{i}of4", out / f"rerun/a_beam/shard_{i}of4"
        shutil.copytree(root / S.R0_OUT_REL / f"shard_{i}of4", dc)
        _edit_json(dc / "summary.json", lambda d, i=i: d.update(args=rec["args"], arm="rerun_c", shard=f"{i}/4"))
        shutil.copytree(out / f"a_beam/shard_{i}of4", da)
        for arm in ("rerun_c", "rerun_a"):
            for ev in ("start", "end"):
                S.log_attempt(root, stage="run-shard", arm=arm, shard=f"{i}/4", event=ev)
    return out


def test_rerun_after_instrument_fault(screen_tree, tmp_path, monkeypatch):   # edit 7: c and a re-run, then analysed
    root = _copy_tree(screen_tree, tmp_path)
    _edit_rows(root / A / "plans.jsonl", _first_arm_round1_mismatch)
    assert _analyze(root, monkeypatch, screen_tree)["status"] == "INSTRUMENT_FAULT"
    out = _stand_in_rerun_shards(root, screen_tree["rec"])
    monkeypatch.setattr(S.subprocess, "call", lambda *a, **k: pytest.fail("no shard may start"))
    assert S.stage_run(argparse.Namespace(rerun=True), root=root) == 0
    assert (out / "rerun/c_beam/summary.json").exists() and not (out / "rerun/c_beam/hook_merged.json").exists()
    assert (out / "rerun/a_beam/hook_merged.json").exists()
    S.stage_analyze(argparse.Namespace(rerun=True), root=root, rcx_root=S.RCX_ROOT)
    rr = json.loads((out / "B4S_REPORT_RERUN.json").read_text())
    assert rr["status"] in ("GO", "NO-GO") and set(rr["arms"]) == {"c", "a"} and rr["meta"]["rerun"]
    assert (out / "ANALYSIS_INPUTS_RERUN.json").exists()


def test_rerun_identity_fault_is_terminal(screen_tree, tmp_path, monkeypatch):   # RCX low item 6: one name
    root = _copy_tree(screen_tree, tmp_path)
    _edit_rows(root / A / "plans.jsonl", _first_arm_round1_mismatch)
    assert _analyze(root, monkeypatch, screen_tree)["status"] == "INSTRUMENT_FAULT"
    out = _stand_in_rerun_shards(root, screen_tree["rec"])
    monkeypatch.setattr(S.subprocess, "call", lambda *a, **k: pytest.fail("no shard may start"))
    S.stage_run(argparse.Namespace(rerun=True), root=root)
    _edit_rows(out / "rerun/a_beam/plans.jsonl", _first_arm_round1_mismatch)
    S.stage_analyze(argparse.Namespace(rerun=True), root=root, rcx_root=S.RCX_ROOT)
    rr = json.loads((out / "B4S_REPORT_RERUN.json").read_text())
    assert rr["status"] == "INSTRUMENT_FAULT_RERUN" and rr["decision"] is None
    assert rr["wording"] == S.WORDING["INSTRUMENT_FAULT_RERUN"] and "INSTRUMENT_FAULT_RERUN" in rr["wording"]
    assert "no further re-run" in rr["wording"]
    assert S.stage_run(argparse.Namespace(rerun=True), root=root) == 0   # the re-run shards ran once: nothing starts
    with pytest.raises(SystemExit, match="written once"):
        S.stage_analyze(argparse.Namespace(rerun=True), root=root, rcx_root=S.RCX_ROOT)


# ----------------------------------------------------------------------------- pre-check and hygiene (prereg 4.2-4.3)

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
    assert pre["seeds_fresh_in_e07_test_classes"] == [f"seeds_fresh/{i}" for i in (0, 2, 3, 9, 21)]


@NEEDS_RCX
def test_hygiene_goal_classes_shared_with_valb_but_no_instance():
    """Recommended hygiene statement: seeds_fresh shares goal CLASSES (not instances) with E07 test (ids 0, 2, 3, 9,
    21) and with E07b's VAL-B selection set (ids 18, 19, 20, 22, 29); the GNN never trained on either."""
    sf = json.loads((ROOT / S.SEEDS_FRESH_REL).read_text())
    valb = json.loads((S.RCX_ROOT / "runs/RCX-E07/e07b/data/valb_instances.json").read_text())
    vk = {r["goal_key"] for r in valb}
    assert [i for i, r in enumerate(sf) if S.goal_chain_key(r["objs"], r["goal"]) in vk] == [18, 19, 20, 22, 29]
    canon = lambda r: (len(r["objs"]), sorted(map(tuple, r["init"])), sorted(map(tuple, r["goal"])))
    assert not {json.dumps(canon(r)) for r in sf} & {json.dumps(canon(r)) for r in valb}
    e07 = json.loads((S.RCX_ROOT / S.E07_INSTANCES_REL).read_text())
    assert not {json.dumps(canon(r)) for r in sf} & {json.dumps(canon(r)) for r in e07}
