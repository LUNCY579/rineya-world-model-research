"""Stub tests for scripts/item7_addendum_a.py (PREREG_ITEM7_ADDENDUM_A_DRAFT.md section A; B.6 F audit of the screen arm).

CPU only, no model, no torch. A synthetic item-7 tree (runs/BW_SEEDS/{R0,R1,R2}_{beam,direct}, FREEZE_I7_*, a small
generated stand-in for seeds_fresh) is produced in tmp_path by the pinned planner with b4_screen's scripted F/proposer
(R1 gets a share of wrong-state F answers so FREPLAY flags something); token counts use a 4-characters-per-token stand-in
and FLOP parameters a small stand-in config (the real tokenizer and adapters are only hashed, in
test_registered_pins_are_the_real_files). Read-only inputs: the pinned planner and blocks module, scripts/freplay.py
(imported read-only by the code under test), scripts/bw_run_tests.py (merge), scripts/b4_screen.py (stub models).

  python -B -m pytest tests/test_item7_addendum_a.py -q -p no:cacheprovider
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("item7_addendum_a_under_test", "scripts/item7_addendum_a.py")
S = _load("b4_screen_for_addendum_tests", "scripts/b4_screen.py")
SIZES = (5, 5, 6, 6, 6, 7, 7, 8)
FP = {"n_nonemb": 1000, "d_model": 8, "vocab": 50, "n_layers": 2, "n_heads": 2, "head_dim": 4}
MODELS = {"R0": {"noise": 0.15, "inconsistent": 0.0}, "R1": {"noise": 0.25, "inconsistent": 0.05},
          "R2": {"noise": 0.10, "inconsistent": 0.0}}


class FakeCounter:
    sha256 = "stand-in"

    def count(self, texts):
        return [-(-len(t) // 4) for t in texts]


def _write_rows(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("i7_root")
    B, BP = S.exec_pinned_planner(S.make_stub_lm())
    rng, inst = random.Random(20261012), []
    for n in SIZES:
        g = None
        while g is None:
            g = B.generate(rng, n, full_goal_frac=1.0)
        inst.append(g)
    sf = root / A.SEEDS_FRESH_REL
    sf.parent.mkdir(parents=True)
    sf.write_text(json.dumps([{"objs": o, "init": sorted(map(list, i)), "goal": sorted(map(list, g)), "n": len(o)}
                              for o, i, g in inst]))
    insts = S.load_json_set(sf, "seeds_fresh")
    b2b = json.loads((ROOT / "runs/FREEZE_B2_beam.json").read_text())["args"]
    b2d = json.loads((ROOT / "runs/FREEZE_B2_direct.json").read_text())["args"]
    merge = S._load_script("bw_run_tests").merge
    for s, kw in MODELS.items():
        fb = dict(b2b, pa=f"runs/BW_Pa_v2_{s}/adapter", fwd_adapter=f"runs/BW_Fdelta_{s}/adapter")
        fd = dict(b2d, pa=f"runs/BW_Pa_v2_{s}/adapter")
        (root / f"runs/FREEZE_I7_{s}_beam.json").write_text(json.dumps({"args": fb, "system": s}))
        (root / f"runs/FREEZE_I7_{s}_direct.json").write_text(json.dumps({"args": fd, "system": s}))
        bd = root / A.SEEDS_DIR / f"{s}_beam"
        for i in range(A.BEAM_SHARDS):
            part = [it for j, it in enumerate(insts) if j % A.BEAM_SHARDS == i]
            d = bd / f"shard_{i}of4"
            summ, rows = S.run_arm(BP, B, S.ScriptedModel(B, **kw), S.FakeTok(), part, fb, d, val=False)
            for r in rows:
                r.pop("expansions")
                r["val"] = r["success"]
            (d / "summary.json").write_text(json.dumps({"args": {"set": "seeds_fresh"}, "summary": summ}))
            _write_rows(d / "plans.jsonl", rows)
        merge(bd, A.BEAM_SHARDS)
        dd = root / A.SEEDS_DIR / f"{s}_direct"
        dd.mkdir(parents=True)
        model = S.ScriptedModel(B, **kw)
        prop = BP.LMProposerBW(model, S.FakeTok(), "pa", "actions", K=1, max_new=fd["M"], prefix_ok=False)
        eps = BP.plan_direct(insts, prop, eval_seed=0)
        summ, rows = BP.score(insts, eps)
        for r in rows:
            r["val"] = r["success"]
        (dd / "summary.json").write_text(json.dumps({"args": {"set": "seeds_fresh"}, "summary": summ}))
        _write_rows(dd / "plans.jsonl", rows)
    (root / A.RUN_DONE_REL).write_text("all systems evaluated")
    (root / A.SCREEN_FREEZE_REL).write_text("{}")
    return {"root": root, "insts": {it["id"]: it for it in insts}}


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(A, "PINS", {})
    monkeypatch.setattr(A, "load_token_counter", lambda: FakeCounter())
    monkeypatch.setattr(A, "flop_params_for", lambda root, rel: dict(FP))


def _copy(tree, tmp_path, name="r"):
    dst = tmp_path / name
    shutil.copytree(tree["root"], dst)
    return dst


def test_constants_match_the_screen_runner():
    assert A.INPUTS_REL == S.ADDENDUM_A_INPUTS_REL and A.SCREEN_FREEZE_REL == S.FREEZE_REL
    assert A.SCREEN_INPUTS_REL == S.OUT_REL + "/ANALYSIS_INPUTS.json" and A.SCREEN_ARM_REL == S.OUT_REL + "/a_beam"
    assert A.SEEDS_FRESH_REL == S.SEEDS_FRESH_REL and A.PINS[A.SEEDS_FRESH_REL] == S.SEEDS_FRESH_SHA256
    assert A.INPUTS_REL in S.CLOSING_MARKERS                             # A.0 closes the screen's freeze window


def test_hash_requires_the_screen_freeze_and_complete_outputs(tree, tmp_path):   # A.0
    r = _copy(tree, tmp_path, "a")
    (r / A.SCREEN_FREEZE_REL).unlink()
    with pytest.raises(SystemExit, match="late-freeze window"):
        A.stage_hash(argparse.Namespace(without_screen_freeze=False), root=r)
    assert A.stage_hash(argparse.Namespace(without_screen_freeze=True), root=r) == 0
    assert json.loads((r / A.INPUTS_REL).read_text())["screen_freeze_present"] is False
    r = _copy(tree, tmp_path, "b")
    (r / A.SEEDS_DIR / "R2_beam/shard_3of4/fcache.jsonl").unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        A.stage_hash(argparse.Namespace(without_screen_freeze=False), root=r)
    r = _copy(tree, tmp_path, "c")
    (r / A.RUN_DONE_REL).unlink()
    with pytest.raises(SystemExit, match="not complete"):
        A.stage_hash(argparse.Namespace(without_screen_freeze=False), root=r)
    assert not (r / A.INPUTS_REL).exists()


def test_hash_then_analyze_end_to_end(tree, tmp_path, stubs):
    r = _copy(tree, tmp_path)
    assert A.stage_hash(argparse.Namespace(without_screen_freeze=False), root=r) == 0
    tab = json.loads((r / A.INPUTS_REL).read_text())
    assert set(A.expected_files()) <= set(tab["files"]) and tab["screen_freeze_present"] is True
    for rel_, e in tab["files"].items():
        data = (r / rel_).read_bytes()
        assert e["sha256"] == A.hashlib.sha256(data).hexdigest() and e["lines"] == len(data.splitlines())
    with pytest.raises(SystemExit, match="written once"):
        A.stage_hash(argparse.Namespace(without_screen_freeze=False), root=r)
    assert A.stage_analyze(argparse.Namespace(), root=r) == 0
    rep = json.loads((r / (A.REPORT_REL + ".json")).read_text())
    n = len(SIZES)
    # A.1: FREPLAY's definitions on each system's caches
    for s in A.SYSTEMS:
        a = rep["A1_f_audit"]["sets"][f"i7_{s}_beam"]
        assert len(a["instances"]) == n and a["unique_keys"] > 0 and a["checks"]["plans_not_covered"] == []
        assert set(a["FREPLAY_R1"]) >= {"exactF_success_lower_L", "exactF_success_upper_U", "flagged"}
        assert a["FREPLAY_R2"]["R2"]["k"] > 0                            # the stub F says "legal" on illegal keys
    assert rep["A1_f_audit"]["sets"]["i7_R0_beam"]["harmful_keys"] == []   # exact stub F: nothing harmful
    r1 = rep["A1_f_audit"]["sets"]["i7_R1_beam"]
    assert r1["additional_report_only"]["harmful_by_class"]["wrong_state"] > 0
    assert r1["FREPLAY_R1"]["exactF_success_upper_U"] - r1["FREPLAY_R1"]["exactF_success_lower_L"] == \
        r1["FREPLAY_R1"]["flagged"]
    assert set(rep["A1_f_audit"]["seed_spread_report_only"]) == {"harmful", "false_illegal", "support_share_R2",
                                                                 "flag_accuracy", "delta_exact"}
    # A.2: every instance in exactly one class
    for s in A.SYSTEMS:
        b, d = rep["A2_taxonomy"]["beam"][s], rep["A2_taxonomy"]["direct"][s]
        assert sum(b["classes"].values()) == n and sum(d["classes"].values()) == n
        dist = b["unverified_failures_unmet_goal_facts"]
        assert sum(dist["distribution"].values()) == b["classes"]["dead_end"] + b["classes"]["budget"] + \
            b["classes"]["other"]
        assert dist["stalled_at_1"] == dist["distribution"].get("1", 0)
        assert d["at_cap"]["cap_tokens"] == 256
        assert sum(sum(v.values()) for v in rep["A2_taxonomy"]["crosstabs"][f"{s}: beam x direct"].values()) == n
    # A.3: exact round curve ends at the solved count; differences are R - R0
    for s in A.SYSTEMS:
        c = rep["A3_anytime"]["curves"][s]["rounds_exact"]
        vals = [c[k] for k in sorted(c, key=int)]
        assert vals == sorted(vals) and vals[-1] == rep["A2_taxonomy"]["beam"][s]["classes"]["solved"]
        assert len(c) >= 12
    d1 = rep["A3_anytime"]["differences"]["R1-R0"]["rounds_exact"]["12"]
    assert d1 == rep["A3_anytime"]["curves"]["R1"]["rounds_exact"]["12"] - \
        rep["A3_anytime"]["curves"]["R0"]["rounds_exact"]["12"]
    # A.4: labels and positivity; the exact recomputation is in test_ledger_formulas
    for s in A.SYSTEMS:
        x = rep["A4_ledger"]["systems"][s]
        assert x["beam"]["flops"]["total"] > 0 and x["direct"]["flops"]["total"] > 0
        assert x["beam_over_direct_flops_per_instance"] > 1
    assert rep["A4_ledger"]["labels"]["beam proposer"] == "ESTIMATE" and rep["meta"]["torch_loaded"] is False
    assert (r / (A.REPORT_REL + ".md")).read_text().startswith("# Item 7 addendum A")
    with pytest.raises(SystemExit, match="written once"):
        A.stage_analyze(argparse.Namespace(), root=r)
    assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))


def test_analyze_refuses_changed_or_new_outputs(tree, tmp_path, stubs):
    for case in ("changed", "new", "no_table"):
        r = _copy(tree, tmp_path, case)
        if case != "no_table":
            A.stage_hash(argparse.Namespace(without_screen_freeze=False), root=r)
        if case == "changed":
            p = r / A.SEEDS_DIR / "R0_beam/shard_1of4/fcache.jsonl"
            p.write_text(p.read_text() + "\n")
        elif case == "new":
            d = r / A.SEEDS_DIR / "R1_beam/shard_9of4"
            d.mkdir()
            (d / "plans.jsonl").write_text("{}\n")
        with pytest.raises(A.PinError):
            A.stage_analyze(argparse.Namespace(), root=r)
        assert not (r / (A.REPORT_REL + ".json")).exists()


def test_ledger_formulas(tree):
    FL = A.flops_module()
    r = tree["root"]
    args = json.loads((r / "runs/FREEZE_I7_R0_beam.json").read_text())["args"]
    led = A.beam_ledger(r, f"{A.SEEDS_DIR}/R0_beam", args, tree["insts"], FakeCounter(), FP, FP)
    want_f, keys_sum = 0, 0
    for i in range(A.BEAM_SHARDS):                                       # independent recomputation of the F term
        keys = {}
        for line in (r / A.SEEDS_DIR / f"R0_beam/shard_{i}of4/fcache.jsonl").read_text().splitlines():
            d = json.loads(line)
            keys[(frozenset(map(tuple, d["s"])), tuple(d["a"]))] = d["text"]
        keys_sum += len(keys)
        for (s, a), t in keys.items():
            p = -(-len(A.B.forward_prompt_delta(s, a)) // 4)
            g = min(-(-len(t) // 4) + 1, 40)
            want_f += FL.flops_breakdown(FP, p, g)["total"]
    assert led["flops"]["f_exact_given_tokenizer"] == want_f and led["from_caches_exact"][
        "f_unique_keys_sum_over_shards"] == keys_sum
    per = led["per_instance"]
    x = sum(v["expansions_bound"] for v in per.values()) * led["expansion_scale_to_logged_total"]
    assert x == pytest.approx(led["logged_exact"]["expansions_total"], rel=1e-9)   # scaled to the logged total
    E, W = int(args["E"]), int(args["W"])
    for v in per.values():
        assert v["expansions_bound"] == min(E + W - 1, 1 + W * (v["rounds"] - 1))
    assert led["logged_exact"]["gen_tokens_total"] == sum(v["gen_tokens"] for v in per.values())
    rows = A.rows_of(r / A.SEEDS_DIR / "R0_direct/plans.jsonl")
    dl = A.direct_ledger(r, f"{A.SEEDS_DIR}/R0_direct", tree["insts"], FakeCounter(), FP)
    want = sum(FL.flops_breakdown(FP, -(-len(A.B.proposer_prompt(tree["insts"][x["id"]]["init"],
                                                                 tree["insts"][x["id"]]["goal"])) // 4),
                                   x["gen_tokens"])["total"] for x in rows)
    assert dl["flops"]["total"] == want


def test_expansion_bound_holds_when_the_budget_binds(tmp_path):
    """A.4's x_i = min(E + W - 1, 1 + W (rounds - 1)) bounds plan_beam's expansions: the run stops at the first round
    end with >= E expansions, so a budget failure can exceed E by up to W - 1 (min(E, ...) is not a bound)."""
    B, BP = S.exec_pinned_planner(S.make_stub_lm())
    inst = S.dry_instances(B, (9, 10, 11, 12))
    args = dict(json.loads((ROOT / "runs/FREEZE_B2_beam.json").read_text())["args"], E=6)
    _, rows = S.run_arm(BP, B, S.ScriptedModel(B, noise=0.3), S.FakeTok(), inst, args, tmp_path, val=False)
    E, W = 6, int(args["W"])
    assert all(r["expansions"] <= min(E + W - 1, 1 + W * (r["rounds"] - 1)) for r in rows)
    assert any(r["fail"] == "budget" and r["expansions"] > E for r in rows)


def test_taxonomy_classes_on_hand_rows(tree):
    it = next(iter(tree["insts"].values()))
    on = sorted(f for f in it["goal"] if f[0] == "on")
    rows = [{"id": it["id"], "success": True, "val": True, "verified": False, "fail": "budget", "plan": [],
             "rounds": 3, "gen_tokens": 10},
            {"id": it["id"], "success": False, "val": False, "verified": True, "fail": None, "plan": [], "rounds": 2,
             "gen_tokens": 10}]
    assert [A.beam_class(x) for x in rows] == ["solved", "verified_but_invalid"]
    assert A.beam_class(dict(rows[1], verified=False, fail="dead_end")) == "dead_end"
    assert A.direct_class({"success": False, "val": False, "fail": "unparseable", "valid": True}) == "unparseable"
    assert A.direct_class({"success": False, "val": False, "fail": None, "valid": False}) == "invalid_action"
    assert A.direct_class({"success": False, "val": False, "fail": None, "valid": True}) == "goal_not_reached"
    t = A.beam_taxonomy(rows, tree["insts"])
    assert t["unverified_successes"] == [it["id"]] and t["classes"]["verified_but_invalid"] == 1
    bad = dict(rows[0], val=False, success=False, verified=False, plan=[f"stack {on[0][1]} {on[0][2]}"])
    assert A.beam_taxonomy([bad], tree["insts"])["unverified_failures_unmet_goal_facts"]["distribution"] == \
        {"invalid_prefix": 1}


def test_faudit_screen(tree, tmp_path, stubs):                           # B.6
    r = _copy(tree, tmp_path)
    arm = r / A.SCREEN_ARM_REL
    shutil.copytree(r / A.SEEDS_DIR / "R0_beam", arm)
    with pytest.raises(SystemExit, match="hashes its inputs first"):
        A.stage_faudit_screen(argparse.Namespace(), root=r)
    files = [arm / "plans.jsonl"] + [arm / f"shard_{i}of4" / f for i in range(4) for f in ("plans.jsonl",
                                                                                         "fcache.jsonl")]
    (r / A.SCREEN_INPUTS_REL).write_text(json.dumps({A.rel(r, p): A.sha256_file(p) for p in files}))
    assert A.stage_faudit_screen(argparse.Namespace(), root=r) == 0
    rep = json.loads((r / A.FAUDIT_SCREEN_REL).read_text())
    assert rep["FREPLAY_R2"]["claim"] == A.CLAIMS["b4s_a_beam"] and len(rep["instances"]) == len(SIZES)
    with pytest.raises(SystemExit, match="written once"):
        A.stage_faudit_screen(argparse.Namespace(), root=r)
    (r / A.FAUDIT_SCREEN_REL).unlink()
    p = arm / "shard_0of4/fcache.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(A.PinError):
        A.stage_faudit_screen(argparse.Namespace(), root=r)


def test_registered_pins_are_the_real_files():
    assert A.check_pins(ROOT) == A.PINS                                   # read-only hashing of the real inputs
    FR = A.load_freplay()
    assert FR.PREREG_SHA == A.PINS["research/PREREG_FREPLAY.md"] and "torch" not in sys.modules
    if (A.TITAN_DIR / "tokenizer.json").exists():
        assert A.sha256_file(A.TITAN_DIR / "tokenizer.json") == A.TITAN_TOKENIZER_SHA256
    FL = A.flops_module()
    assert callable(FL.shared_prefix_breakdown) and "torch" not in sys.modules
