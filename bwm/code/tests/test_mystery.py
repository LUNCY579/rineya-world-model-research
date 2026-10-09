"""Unit tests for the Mystery Blocksworld generator fixes (bwmn/mystery.py, scripts/mystery_data.py). CPU, tokenizer
only, < 2 min. Scratch files go to runs/_tmp_tests_mystery/pytest/ and are removed.

  python -m pytest tests/test_mystery.py -q
"""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import mystery as M  # noqa: E402

SCRATCH = ROOT / "runs" / "_tmp_tests_mystery" / "pytest"
TOKENIZER = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                 "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze") / "tokenizer.json"
SEED = 4242


def _load_script():
    spec = importlib.util.spec_from_file_location("mystery_data", ROOT / "scripts" / "mystery_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = _load_script()


@pytest.fixture(scope="module")
def forb():
    return M.forbidden_words()  # no cache: tests never write outside the scratch dir


@pytest.fixture(scope="module")
def test_sigs():
    return M.test_signatures()


@pytest.fixture(scope="module")
def tok_fns():
    if not TOKENIZER.exists():
        pytest.skip("Titan tokenizer not available")
    return MD.token_fns(MD.load_tokenizer(TOKENIZER.parent))


@pytest.fixture(scope="module")
def a_slot(forb, test_sigs):
    """A Regime-A slot with a plan of >= 4 actions (word-named objects)."""
    for i in (0, 2, 4, 5, 7, 9, 10, 12):
        inst, rows, rej = M.make_instance("A", SEED, i, forb, test_sigs)
        if len(inst["plan"]) >= 4:
            return inst, rows
    raise AssertionError("no Regime-A slot with a plan of >= 4 actions")


@pytest.fixture(scope="module")
def b_slot(forb, test_sigs):
    return M.make_instance("B", SEED, 0, forb, test_sigs)[:2]


@pytest.fixture
def scratch(request):
    d = SCRATCH / request.node.name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def teardown_module(module):
    shutil.rmtree(SCRATCH, ignore_errors=True)


# ----------------------------------------------------------------------------- suffix-row signature guard (B-02)

def test_suffix_guard_drops_exactly_the_leaking_rows(a_slot, test_sigs):
    inst, _ = a_slot
    sigs = M.bw_suffix_signatures(inst)
    assert len(sigs) == len(inst["plan"]) and sigs[0] == inst["sig"]
    # the signatures are those of the renamed (state_t, goal) mapped back to Blocksworld space
    R = inst["renaming"].inverse()
    s = inst["init"]
    for t, a in enumerate(inst["plan"]):
        assert B.signature(inst["bw"]["objs"], R.facts(s), R.facts(inst["goal"])) == sigs[t]
        s, _ = M.step(inst["domain"], s, a)
    fake = set(test_sigs) | {sigs[2], sigs[0]}  # t = 0 is never dropped (the instance itself is excluded upstream)
    st = Counter()
    rows = M.instance_rows(inst, random.Random(0), test_sigs=fake, stats=st)
    ts = [r["t"] for r in rows["proposer"]]
    dropped = [t for t in range(1, len(sigs)) if sigs[t] in fake]
    assert 2 in dropped and 0 in ts and not set(dropped) & set(ts)
    assert rows["suffix_dropped"] == st["suffix_test_signature"] == len(dropped)
    assert all(r["sig"] not in fake for r in rows["proposer"] if r["t"] > 0)
    assert len(rows["proposer"]) == len(inst["plan"]) - len(dropped) + 1  # + noisy row


def test_suffix_guard_requires_test_sigs(a_slot):
    with pytest.raises(ValueError):
        M.instance_rows(a_slot[0], random.Random(0))


def test_suffix_guard_fires_on_real_test_set(forb, test_sigs):
    """Some generated slot has an intermediate state that is a renaming of a real PlanBench problem; exactly those
    rows are missing and every emitted row's signature (recomputed from the prompt text) is outside the test set."""
    for i in range(150):
        inst, rows, _ = M.make_instance("A", SEED, i, forb, test_sigs)
        if inst["suffix_rows_dropped"]:
            break
    else:
        pytest.fail("guard never fired in 150 slots")
    sigs = M.bw_suffix_signatures(inst)
    leaking = {t for t in range(1, len(sigs)) if sigs[t] in test_sigs}
    assert leaking and leaking.isdisjoint(r["t"] for r in rows["proposer"])
    R, dom, objs = inst["renaming"].inverse(), inst["domain"], set(inst["objs"])
    for r in rows["proposer"]:
        u = M.prompt_user(r["prompt"])
        sig = B.signature(inst["bw"]["objs"], R.facts(M.state_from_list(u["state"], dom, objs)),
                          R.facts(M.state_from_list(u["goal"], dom, objs)))
        assert sig == r["sig"] and sig not in test_sigs


# ----------------------------------------------------------------------------- NameGen filters

def test_banned_substrings_cover_deceptive_and_blocksworld_words():
    ban = set(M.banned_substrings())
    myst = M.mystery_domain()
    assert set(myst.preds) | set(myst.actions) <= ban
    assert {"pick", "down", "stack", "clear", "table", "hand", "empty", "hold", "holding", "block", "tower"} <= ban
    assert all(len(w) >= 4 for w in ban) and "on" not in ban and "up" not in ban


def test_namegen_rejects_substrings_pddl_words_and_forbidden(forb):
    ng = M.NameGen(random.Random(0), forb)
    for w in ("zhand", "spainor", "tabled", "kopicko", "bloblock", "feastu", "and", "either", "forall", "planet"):
        assert not ng.ok(w), w
    assert ng.ok("droshdoiv")
    ws = ng.sample(3000, 1, 3)
    assert not [w for w in ws if any(b in w for b in ng.banned) or w in forb or w in M.PDDL_WORDS]


def test_namegen_rejects_single_token_words(forb, tok_fns):
    count, single = tok_fns
    ng = M.NameGen(random.Random(1), forb, single_token=single)
    for w in ("spoon", "hound", "drip", "chop", "blush"):
        assert single(w) and not ng.ok(w), w
    ws = ng.sample(1500, 1, 3)
    assert not [w for w in ws if count(w) == 1 or count(" " + w) == 1]
    # without the filter such words do occur (the forbidden vocabulary alone misses them)
    raw = M.NameGen(random.Random(1), forb).sample(1500, 1, 3)
    assert any(single(w) for w in raw)


# ----------------------------------------------------------------------------- single-letter objects

def test_letter_slot_share_in_30_50_percent():
    for n in range(2, 1001):
        share = sum(M.letter_objects(i) for i in range(n)) / n
        assert 0.3 <= share <= 0.5, (n, share)


def test_letter_slots_use_single_letters(forb, test_sigs, b_slot):
    for i in range(5):
        inst, _, _ = M.make_instance("A", SEED, i, forb, test_sigs)
        letters = M.letter_objects(i)
        assert inst["letter_objects"] == letters
        assert all((len(o) == 1 and o in M.LETTERS) == letters for o in inst["objs"])
        assert all(len(w) >= 2 for w in list(inst["domain"].preds) + list(inst["domain"].actions))
    assert not b_slot[0]["letter_objects"] and all(len(o) >= 2 for o in b_slot[0]["objs"])
    b1 = M.make_instance("B", SEED, 1, forb, test_sigs)[0]
    assert b1["letter_objects"] and all(o in M.LETTERS for o in b1["objs"])


# ----------------------------------------------------------------------------- Blocksworld-fragment containment

def _contains_ctrl():
    """Blocksworld fragment {clear, holding, on} x {put-down, stack, unstack}, renamed with unstack's parameters
    permuted, hidden by an extra binary predicate (also an extra precondition of 'stack') and an extra action."""
    bw = M.bw_domain()
    frag = M._project(bw, {"clear", "holding", "on"}, ("put-down", "stack", "unstack"))
    fr = M.Renaming({"clear": "ka", "holding": "kb", "on": "kc"}, {"put-down": "ma", "stack": "mb", "unstack": "mc"},
                    {}, frozenset(), {"unstack": (1, 0)}).domain(frag)
    sc = fr.actions["mb"]
    acts = dict(fr.actions, mb=M.Schema(sc.params, tuple(sorted(sc.pre + (("kd", "X", "Y"),))), sc.add, sc.dele),
                md=M.Schema(("X", "Y"), (("kd", "X", "Y"),), (("kd", "Y", "X"),), (("kd", "X", "Y"),)))
    return M.Domain("ctrl", dict(fr.preds, kd=2), acts)


def test_contained_fragment_is_rejected():
    d = _contains_ctrl()
    assert not M.is_blocksworld_family(d)
    assert M.contains_blocksworld_fragment(d)
    assert M.domain_rejection(d) == "blocksworld_fragment"
    assert M.contains_blocksworld_fragment(M.mystery_domain()) and M.contains_blocksworld_fragment(M.bw_domain())


def test_non_blocksworld_domain_is_not_flagged():
    # unary 3-cycle p -> q -> r -> p: 3 predicates x 3 actions, but no Blocksworld mechanics
    cyc = {f"a{i}": M.Schema(("X",), ((p, "X"),), ((q, "X"),), ((p, "X"),))
           for i, (p, q) in enumerate((("pa", "pb"), ("pb", "pc"), ("pc", "pa")))}
    d = M.Domain("cycle", {"pa": 1, "pb": 1, "pc": 1}, cyc)
    assert not M.contains_blocksworld_fragment(d) and M.domain_rejection(d) is None


def test_fragments_need_three_predicates_and_three_actions():
    for inv, hs in M.bw_fragment_hashes().items():
        assert len(inv[0]) >= 3 and hs
    # a 2-predicate Blocksworld fragment (holding / clear via put-down + stack-like) is below the threshold
    bw = M.bw_domain()
    two = M._project(bw, {"holding", "clear"}, ("pick-up", "put-down", "stack"))
    assert not M.contains_blocksworld_fragment(two)


def test_accepted_regime_b_domains_contain_no_fragment(b_slot):
    assert not M.contains_blocksworld_fragment(b_slot[0]["domain"])
    assert M.domain_rejection(b_slot[0]["domain"]) is None


# ----------------------------------------------------------------------------- token budget + per-slot seeding

def test_token_budget_enforced(forb, test_sigs, tok_fns):
    count, single = tok_fns
    inst, rows, rej = M.make_instance("A", SEED, 3, forb, test_sigs, single_token=single, count_tokens=count,
                                      max_tokens=900)
    for r in rows["proposer"]:
        p = count(r["prompt"])
        assert r["n_tokens"] == {"actions": p + count(r["target_actions"]) + 1,
                                 "states": p + count(r["target_states"]) + 1}
        assert max(r["n_tokens"].values()) <= 900
    assert all(r["n_tokens"] == count(r["prompt"]) + count(r["target"]) + 1 <= 900 for r in rows["forward"])
    with pytest.raises(RuntimeError):
        M.make_instance("A", SEED, 3, forb, test_sigs, count_tokens=count, max_tokens=50, max_attempts=5)
    with pytest.raises(ValueError):
        M.make_instance("A", SEED, 3, forb, test_sigs, count_tokens=count)


def test_budget_redraw_is_counted(forb, test_sigs):
    count = lambda text: len(text) // 4
    big = M.make_instance("A", SEED, 6, forb, test_sigs, count_tokens=count, max_tokens=10 ** 6)
    need = M.row_token_counts(big[1], count)
    inst, rows, rej = M.make_instance("A", SEED, 6, forb, test_sigs, count_tokens=count, max_tokens=need - 1)
    assert rej["token_budget"] >= 1 and M.row_token_counts(rows, count) <= need - 1


def test_slot_is_a_pure_function_of_seed_and_index(forb, test_sigs, b_slot):
    rec = lambda x: (M.instance_record(x[0], "id"), x[1])
    a2 = rec(M.make_instance("A", SEED, 2, forb, test_sigs))
    M.make_instance("A", SEED, 3, forb, test_sigs)
    random.seed(0)  # global RNG state must not matter
    assert rec(M.make_instance("A", SEED, 2, forb, test_sigs)) == a2
    assert rec(M.make_instance("A", SEED + 1, 2, forb, test_sigs)) != a2
    assert rec(M.make_instance("A", SEED, 7, forb, test_sigs)) != a2
    assert rec(M.make_instance("B", SEED, 0, forb, test_sigs)) == rec(b_slot)
    # a signature already used by an earlier slot is redrawn
    a2s = M.make_instance("A", SEED, 2, forb, test_sigs)[0]["sig"]
    again, _, rej = M.make_instance("A", SEED, 2, forb, test_sigs, seen={a2s})
    assert again["sig"] != a2s and rej["duplicate_in_sample"] >= 1


# ----------------------------------------------------------------------------- noisy rows, records, renaming

def test_noisy_row_resampled_when_unchanged(a_slot, test_sigs):
    inst, _ = a_slot

    class Forced(random.Random):  # randrange -> 0 once; choice returns the original first action twice
        def __init__(self):
            super().__init__(0)
            self.queue, self.first = [inst["plan"][0], inst["plan"][0]], True

        def randrange(self, *a, **k):
            if self.first:
                self.first = False
                return 0
            return super().randrange(*a, **k)

        def choice(self, seq):
            if self.queue and isinstance(seq, list) and seq and isinstance(seq[0], tuple) and self.queue[0] in seq:
                return self.queue.pop(0)
            return super().choice(seq)

    rng = Forced()
    rows = M.instance_rows(inst, rng, test_sigs=test_sigs)
    assert not rng.queue  # both unchanged draws happened and were rejected
    noisy = rows["proposer"][-1]
    assert noisy["t"] == -1 and noisy["target_actions"] != M.target_actions(inst["plan"])
    keys = [(r["prompt"], r["target_actions"]) for r in rows["proposer"]]
    assert len(set(keys)) == len(keys)


def test_record_and_renaming_round_trip(a_slot, b_slot):
    for inst in (a_slot[0], b_slot[0]):
        back = M.instance_from_record(json.loads(json.dumps(M.instance_record(inst, "x"))))
        assert back["domain"].same_as(inst["domain"]) and back["plan"] == list(inst["plan"])
        assert (back["init"], back["goal"], back["objs"]) == (inst["init"], inst["goal"], inst["objs"])
    R = a_slot[0]["renaming"]
    assert M.Renaming.from_json(json.loads(json.dumps(R.to_json()))) == R
    Ri, bw = R.inverse(), a_slot[0]["bw"]
    assert Ri.facts(R.facts(bw["init"])) == bw["init"]
    assert [Ri.action(R.action(x)) for x in bw["plan"]] == list(bw["plan"])


# ----------------------------------------------------------------------------- test signatures (Blocksworld-Hard)

def test_test_signatures_include_blocksworld_hard(test_sigs):
    hard = M.hard_test_signatures()
    small = [p for p in M.HARD_DIR.glob("*.pddl") if len(M.parse_problem(p)[0]) <= 7]
    assert len(small) == len(hard) == 2 and hard <= test_sigs
    assert hard - B.test_signatures()  # blocks.test_signatures alone misses them
    assert B.test_signatures() | M.mystery_test_signatures() <= test_sigs


# ----------------------------------------------------------------------------- write_domain + VAL

def test_write_domain_round_trip(a_slot, b_slot, scratch):
    for i, dom in enumerate((M.bw_domain(), M.mystery_domain(), a_slot[0]["domain"], b_slot[0]["domain"])):
        p = scratch / f"d{i}.pddl"
        p.write_text(M.write_domain(dom))
        assert M.parse_domain(p).same_as(dom)


@pytest.mark.skipif(not B.VAL_EXE.exists(), reason="VAL not available")
def test_val_agrees_on_generated_domains(a_slot, b_slot, scratch):
    for inst, rows in (a_slot, b_slot):
        noisy = M.parse_plan(rows["proposer"][-1]["target_actions"], inst["domain"], set(inst["objs"]))[0]
        assert MD.val_check_generated(inst, [noisy], scratch / "val") >= 3


# ----------------------------------------------------------------------------- evaluation report

def test_eval_set_is_603_instances_262_problems():
    ev = M.mystery_eval_set()
    assert len(ev) == 603 and len({it["sig"] for it in ev}) == 262
    by = {}
    for it in ev:
        by.setdefault(it["n_blocks"], set()).add(it["sig"])
    assert {k: len(v) for k, v in by.items()} == {3: 22, 4: 185, 5: 55}
    rep = M.eval_report(ev, [1] * len(ev), n_boot=50)
    assert rep["all"]["per_instance"]["acc"] == rep["all"]["dedup_signature"]["acc"] == 1.0
    assert rep["all"]["dedup_signature"]["n_distinct"] == 262
    assert {k: v["per_instance"]["n"] for k, v in rep["by_blocks"].items()} == {3: 101, 4: 447, 5: 55}


def test_eval_report_dedup_math():
    items = [{"sig": "s1", "objs": "abc"}, {"sig": "s1", "objs": "abc"}, {"sig": "s1", "objs": "abc"},
             {"sig": "s2", "objs": "abc"}, {"sig": "s3", "objs": "abcd"}]
    rep = M.eval_report(items, [1, 1, 1, 0, 0], n_boot=200)
    a = rep["all"]
    assert a["per_instance"]["acc"] == pytest.approx(3 / 5)
    assert a["dedup_signature"]["acc"] == pytest.approx(1 / 3)  # s1 = 1, s2 = 0, s3 = 0
    assert a["dedup_signature"]["n_distinct"] == 3
    assert a["dedup_signature"]["solved_all_renamings"] == 1 == a["dedup_signature"]["solved_any_renaming"]
    lo, hi = a["dedup_signature"]["cluster_boot95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert rep["by_blocks"][3]["per_instance"]["acc"] == pytest.approx(3 / 4)
    assert rep["by_blocks"][3]["dedup_signature"]["acc"] == pytest.approx(1 / 2)
    assert rep["by_blocks"][4]["dedup_signature"]["acc"] == 0.0
    part = M.eval_report(items, [1, 0, 0, 1, 0], n_boot=0)["all"]["dedup_signature"]
    assert part["acc"] == pytest.approx((1 / 3 + 1 + 0) / 3) and part["solved_any_renaming"] == 2


# ----------------------------------------------------------------------------- streaming, resume, no overwrite

def _ctx(forb, test_sigs, **kw):
    from types import SimpleNamespace
    count = lambda text: len(text) // 4
    d = dict(seed=SEED, forb=forb, test_sigs=test_sigs, deceptive=sorted(set(M.mystery_domain().preds) |
                                                                          set(M.mystery_domain().actions)),
             banned=M.banned_substrings(), single_token=lambda w: False, count=count, max_tokens=4000,
             max_states=20000, val_checks=0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_no_silent_overwrite(scratch):
    (scratch / "instances_A.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit):
        MD.prepare_out(scratch, resume=False, overwrite=False)
    with pytest.raises(SystemExit):
        MD.prepare_out(scratch, resume=True, overwrite=True)
    assert MD.prepare_out(scratch, resume=True, overwrite=False) == ["instances_A.jsonl"]
    assert (scratch / "instances_A.jsonl").exists()
    MD.prepare_out(scratch, resume=False, overwrite=True)
    assert not (scratch / "instances_A.jsonl").exists()


def test_resume_refuses_other_config(scratch):
    MD.check_config(scratch, {"seed": 1}, resume=False)
    assert MD.check_config(scratch, {"seed": 1}, resume=True) == "resumed"
    with pytest.raises(SystemExit):
        MD.check_config(scratch, {"seed": 2}, resume=True)


def test_streaming_resume_after_crash_equals_fresh_run(forb, test_sigs, scratch):
    ctx = _ctx(forb, test_sigs)
    run, ref = scratch / "run", scratch / "ref"
    run.mkdir()
    ref.mkdir()
    MD.generate_regime(run, "A", 4, ctx, log=lambda m: None)
    # crash mid-slot: rows of the uncommitted slot A0004 plus a torn record line
    with open(run / "proposer_A.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"id": "A0004", "prompt": "partial"}) + "\n" + '{"id": "A0004", "pro')
    with open(run / "instances_A.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"id": "A0004", "regi')
    info = MD.generate_regime(run, "A", 7, ctx, log=lambda m: None)
    assert info["resumed_from"] == 4 and info["removed_uncommitted_lines"] == 3 and info["generated_now"] == 3
    MD.generate_regime(ref, "A", 7, ctx, log=lambda m: None)
    for f in ("instances_A.jsonl", "proposer_A.jsonl", "forward_A.jsonl"):
        assert (run / f).read_bytes() == (ref / f).read_bytes(), f
    summ = MD.verify_regime(run, "A", 7, ctx)
    assert summ["rows_signature_rechecked"] == summ["proposer_rows"] > 0
    with pytest.raises(SystemExit):  # never silently shrink a larger existing run
        MD.generate_regime(run, "A", 5, ctx, log=lambda m: None)


def test_verify_catches_a_leaking_row(forb, test_sigs, scratch):
    """The on-disk audit recomputes signatures from the prompt: a row whose state is a test problem fails."""
    ctx = _ctx(forb, test_sigs)
    MD.generate_regime(scratch, "A", 3, ctx, log=lambda m: None)
    rows = [json.loads(x) for x in (scratch / "proposer_A.jsonl").read_text(encoding="utf-8").splitlines()]
    with pytest.raises(AssertionError):  # pretend the last emitted row's problem were a test instance
        MD.verify_regime(scratch, "A", 3, _ctx(forb, set(test_sigs) | {rows[-1]["sig"]}))
    MD.verify_regime(scratch, "A", 3, ctx)  # the real data passes under the real test set
