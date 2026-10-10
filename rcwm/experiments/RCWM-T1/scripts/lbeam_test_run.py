"""CPU tests of scripts/lbeam_run.py (PREREG-LB runner, freeze tool, verifier) and scripts/lbeam_analysis.py, plus the
computed power / sizing record. No GPU, no lease, no real model (tiny random Qwen3 + LoRA stand-ins only;
AutoModelForCausalLM.from_pretrained is disabled), 2 threads, below-normal priority, outputs in the session scratchpad.
Each part runs in its own process (RAM < 1.5 GB):

  python -B scripts/lbeam_test_run.py --part plan       # --plan path without torch; GEN_PLAN_LB.json; turn rule
  python -B scripts/lbeam_test_run.py --part analysis   # analysis dry run (every wording branch, statistics, metrics)
                                                        # -> LB_DRYRUN.{json,md}; every registered sentence of the code
                                                        # appears verbatim in PREREG_LB_DRAFT.md
  python -B scripts/lbeam_test_run.py --part power      # SimulatedF sizing + paired outcomes on dev2_primary (model-free)
                                                        # -> power of the registered rules (resampled to N = 180) and an
                                                        # exact independence grid -> LB_POWER.json
  python -B scripts/lbeam_test_run.py --part tiny       # tiny-model end to end: real tiny generation; a rule-stub run on
                                                        # the dev set through the analysis; crash / resume; deferral
                                                        # (nothing fits; mid-run; the no-progress stall stop); anchor
                                                        # failure; batch-invariance failure; re-check failure (a
                                                        # verdict-eligible stop: the analysis renders 'not evaluable');
                                                        # an abandoned report-only stage; a core stage failing three
                                                        # times; tamper / lock / one-waiter / fingerprint / runtime
                                                        # stops; run_main / main() exit codes; analysis refusals
  python -B scripts/lbeam_test_run.py --part rehearsal  # the real full-size mint path (N = 180, the real freshness union)
                                                        # under a DIFFERENT seed tag; the draw is discarded unprinted
  python -B scripts/lbeam_test_run.py --part freeze     # fake root: freeze refusals (co-sign without the draft sha or
                                                        # the salt commitment, wrong / short salt, BWM draft not the
                                                        # cited one, records not bound to the draft, a union source
                                                        # that errs, data/ changing during the union build, a missing
                                                        # input) -> freeze (draw minted from the seed rule with BWM's
                                                        # revealed salt, never shown) -> verify -> hook identity ->
                                                        # registered run -> registered analysis (create-only) ->
                                                        # tamper (inputs, draw, co-sign record, BWM planner) / amendment
  python -B scripts/lbeam_test_run.py --part hook       # the queue hook draft under bash with a stub step(): every
                                                        # exit-code path, the analysis trigger, idempotency

Writes docs/ledger_beam/RUN_TEST_<PART>.json (with the code sha256s; refused once PREREG-LB is frozen). Exit 0 iff every
check passes.
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # never "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["BWM_DEVICE"] = "cpu"
os.environ["RCWM_DUTY_FORCE"] = "1"
os.environ["RCWM_DUTY_RATIO"] = "0"
os.environ["RCWM_BREATHE_S"] = "0"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOCS = ROOT / "docs" / "ledger_beam"
SCRATCH = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad")
RESULTS: list = []
NUM: dict = {}


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:500]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({str(detail)[:300]})" if detail != "" else ""), flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)


def nice():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def import_torch_cpu():
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.synchronize = lambda *a, **k: None
    torch.set_num_threads(2)
    torch.Tensor.cuda = lambda self, *a, **k: self      # generation.generate_texts moves inputs with .cuda()
    return torch


# ================================================================ part: plan (no torch)
def part_plan() -> dict:
    import lbeam_run as LR
    argv = sys.argv
    sys.argv = ["lbeam_run.py", "--plan"]
    try:
        rc = LR.main()
    finally:
        sys.argv = argv
    check("--plan exits 0 without importing torch and writes GEN_PLAN_LB.json (or leaves it, once frozen)",
          rc == 0 and "torch" not in sys.modules and (DOCS / "GEN_PLAN_LB.json").is_file(), rc)
    res = LR.plan(verbose=False)
    NUM["plan"] = {k: res[k] for k in res if k.startswith(("turns_", "gpu_h_"))}
    NUM["plan"]["stages"] = res["stages"]
    t_exp = res["turns_expected_planning_rate"]
    check("the expected plan schedules every GPU stage; later turns re-run A0 (anchor + batch invariance) first",
          isinstance(t_exp, list) and sorted(s for t in t_exp for s in t["stages"] if not s.startswith("A0 re-check"))
          == sorted(LR.GPU_STAGES) and all(t["stages"][0].startswith("A0 re-check") for t in t_exp[1:]), t_exp)
    check("G1 (the primary pair) fits the first turn at its CEILING at the observed rate, and at its expected size at "
          "the planning rate (at the ceiling AND the planning rate it would overrun the 1.25 x rule: disclosed)",
          isinstance(res["turns_ceiling_observed_rate"], list) and "G1" in res["turns_ceiling_observed_rate"][0]["stages"]
          and "G1" in t_exp[0]["stages"] and not isinstance(res["turns_ceiling_planning_rate"], list),
          res["turns_ceiling_observed_rate"])
    n = LR.counts(180)
    full = LR.simulate_turns(LR.OBSERVED_RATE, n, cap_h=4.5)
    short = LR.simulate_turns(LR.OBSERVED_RATE, n, cap_h=3.5)
    check("with a 3.5 h cap the D-22 rule moves stages to a later turn, and every later turn re-checks A0 first",
          len(short[0]["stages"]) < len(full[0]["stages"]) and all(t["stages"][0].startswith("A0 re-check") for t in short[1:]),
          short)
    check("estimates: A0 counts 2 x 832 anchor steps + 2 x 256 batched + 2 x 256 batch-1 generations",
          n["A0_batched"] == 2 * 832 + 512 and n["A0_batch1"] == 512, n)
    check("torch was never imported by the plan part", "torch" not in sys.modules)
    return {}


# ================================================================ part: analysis dry run (no torch)
def part_analysis() -> dict:
    import lbeam_analysis as LA
    import lbeam_run as LR
    frozen = (ROOT / LR.FREEZE_REL).exists()
    out_dir = DOCS if not frozen else Path(tempfile.mkdtemp(prefix="lbeam_dry_", dir=SCRATCH))
    res = LA.dry_run(out_dir)
    check("analysis dry run passes: statistics unit tests, metrics unit tests", res["stats_selftest"]["ok"] and res["metrics_selftest"]["ok"],
          {**res["stats_selftest"]["cases"], **res["metrics_selftest"]["cases"]})
    ws = res["wording_selftest"]
    for k, v in ws["cases"].items():
        check(f"wording branch: {k}", v["ok"], v.get("got", ""))
    check("every branch and every co-primary status is exercised by the wording self-test",
          set(ws["every_branch_exercised"]) == set(LA.BRANCHES) and set(ws["every_status_exercised"]) == set(LA.CP_STATUSES),
          ws["every_branch_exercised"])
    draft = (ROOT / LR.DRAFT_REL).read_text(encoding="utf-8") if (ROOT / LR.DRAFT_REL).is_file() else ""
    missing = LA.registered_strings_missing(draft) if draft else ["draft missing"]
    check(f"every registered sentence the analysis can render ({len(LA.registered_strings())}: headlines, readings, "
          "statuses, conclusions, qualifiers, not-evaluable and audit sentences, limits) appears verbatim in "
          "PREREG_LB_DRAFT.md", draft and not missing, missing[:4])
    consts = {"DELTA_MIN 0.20": "+0.20", "ALPHA 0.01": "p < 0.01", "RATIO 0.80 / 0.70": ">= 0.80",
              "BOOT 10,000 seed 17": "10,000", "REFUSED 1%": "1%", "N 180": "180", "BATCHINV 256 keys": "256",
              "batch-invariance 0.99": "99%", "audit margin 0.05": "-0.05", "seed tag": LR.SEED_TAG,
              "salt length": f"{LR.SALT_MIN_LEN} characters", "no-progress stop": "two consecutive turns",
              "report-only abandoned after two failures": "failed twice", "core stop after three failures": "three times"}
    check("the draft states the registered constants", all(v in draft for v in consts.values()),
          [k for k, v in consts.items() if v not in draft])
    check("the dry-run record binds the current draft and hook draft (sha256), as the freeze requires",
          res["draft_sha256"] == sha(ROOT / LR.DRAFT_REL) and res["hook_draft_sha256"] == sha(ROOT / LR.HOOK_DRAFT_REL)
          and res["ok"])
    check("torch was never imported by the analysis part", "torch" not in sys.modules)
    NUM["dry_run_dir"] = str(out_dir)
    return {}


# ================================================================ shared tiny-model machinery
def tiny_setup():
    torch = import_torch_cpu()
    import transformers
    from transformers import LogitsProcessorList

    def _no_real_models(*a, **k):
        raise AssertionError("a real model load was attempted inside the tiny test")
    transformers.AutoModelForCausalLM.from_pretrained = _no_real_models
    from rcwmt import generation, runtime
    runtime.gpu_temp_c = lambda: None
    generation.breathe = lambda: None
    generation.throttle_list = lambda: LogitsProcessorList([])
    generation.max_tokens = lambda fam: 12
    return torch


CFG_TINY = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e6,
                tie_word_embeddings=True)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_base(seed: int = 2002):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    with torch.no_grad():
        cfg = Qwen3Config(**CFG_TINY)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(seed)
        m = Qwen3ForCausalLM(cfg)
        g = torch.Generator().manual_seed(seed)
        for _, p in m.named_parameters():
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return m.eval()


def make_adapter(path: Path, seed: int = 31) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    with torch.no_grad():
        pm = get_peft_model(build_base(), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS,
                                                     task_type="CAUSAL_LM"))
        g = torch.Generator().manual_seed(seed)
        for n, p in pm.named_parameters():
            if "lora_B" in n or "lora_A" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        pm.save_pretrained(str(path))
    del pm
    gc.collect()


def load_tiny_peft(adapter_dir: Path):
    from peft import PeftModel
    d = Path(adapter_dir)
    src = d / "default" if (d / "default").exists() else d
    return PeftModel.from_pretrained(build_base(), str(src), adapter_name="t2", is_trainable=False).eval()


_TOK: list = []


def tok():
    if not _TOK:
        from rcwmt import runtime
        _TOK.append(runtime.tokenizer())
    return _TOK[0]


class FakeRuntime:
    """Stands in for rcwmt.runtime under the runner's RealBackend: tiny models, recorded calls, no real lock."""

    def __init__(self, tmp: Path):
        self.LOCK_PATH = tmp / "fake_gpu.lock"
        self.WAITING = {"rcwm": tmp / "fake_gpu.waiting.rcwm.json", "bwm": tmp / "fake_gpu.waiting.bwm.json"}
        self.calls: list = []
        self.fp = None
        self.runtime_override: list = []

    @contextlib.contextmanager
    def gpu_lease(self, label, job):
        self.calls.append(("lease", label))
        self.LOCK_PATH.write_text(json.dumps({"label": label, "pid": os.getpid(), "created": time.time()}), encoding="utf-8")
        try:
            yield
        finally:
            self.LOCK_PATH.unlink()
            self.calls.append(("release", label, time.time()))

    def wait_for_resources(self):
        self.calls.append(("wait",))

    def tokenizer(self):
        return tok()

    def unload(self, _):
        gc.collect()
        self.calls.append(("unload",))

    def runtime_info(self):
        import torch
        import transformers
        real = {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu (tiny test)"}
        return {**real, **self.runtime_override.pop(0)} if self.runtime_override else real


def make_backend(rt: FakeRuntime, device: str = "cpu-tiny"):
    import lbeam_run as LR
    import peft

    class TinyBackend(LR.RealBackend):
        def __init__(self, rt_):
            self.rt = rt_
            self.device = device

        def thermal_wait(self):
            pass

        def load(self, cfg):
            self.rt.calls.append(("load", "sparse+t2"))
            model = load_tiny_peft(cfg.adapter_dir)
            return model, tok(), {"fingerprint": self.rt.fp or (LR.FP["sparse"] + "-tiny-standin"),
                                  "versions": {"peft": peft.__version__}}
    return TinyBackend(rt)


def side_of(model) -> str:
    import lbeam_f as LF
    return LF.peft_side(model)


def make_stub(t2_level: str = "trigger", bad_batch1: bool = False):
    """Text generator standing in for generation.generate_texts: T2 side (PEFT adapter active) follows the frame's rule
    ('rule') or treats only 'void' forms as a no-op ('trigger', PREREG-3's finding); the OFF side (LoRA disabled, or the
    base alone) applies every event as posted (status-blind). bad_batch1: one-prompt calls perturb a third of the
    answers (a batch-composition dependence)."""
    import p3_common as P3C

    def gen(model, tok_, message_lists, max_new_tokens, **kw):
        side = side_of(model)
        out = []
        for msgs in message_lists:
            u = json.loads(msgs[1]["content"])
            b, e = u["before"], dict(u["event"])
            if side in ("off", "plain"):
                e["status"] = "posted"
            elif t2_level == "trigger" and e["status"] not in ("void", "Void", "VOID", "voided"):
                e["status"] = "posted"
            ans = P3C.rule_gold(b, e)
            if bad_batch1 and len(message_lists) == 1 and int(hashlib.sha256(msgs[1]["content"].encode()).hexdigest()[:6], 16) % 3 == 0:
                k = sorted(ans)[0]
                ans = {**ans, k: ans[k] + 1}
            out.append(json.dumps(ans))
        return out
    return gen


def single_adapter_refs(tmp: Path, chains: list, adapter_dir: Path, gen=None, tag: str = "real") -> dict:
    """References as T2's eval wrote them: OFF = the base alone; ON = a PeftModel with only adapter 't2'."""
    import lbeam_run as LR
    from rcwmt import generation
    real = generation.generate_texts
    if gen is not None:
        generation.generate_texts = gen
    refs = {}
    try:
        for s in ("off", "on"):
            m = build_base() if s == "off" else load_tiny_peft(adapter_dir)
            rr = generation.chain(m, tok(), chains, "chain_one")
            p = tmp / "refs" / f"ref_{tag}_{s}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("".join(json.dumps(x) + "\n" for x in LR.chain_rows(chains, rr)))
            refs[s] = p
            del m
            gc.collect()
    finally:
        generation.generate_texts = real
    return refs


def write_draw(path: Path, insts: list) -> Path:
    import lbeam_problems as P
    path.parent.mkdir(parents=True, exist_ok=True)
    text = P.dumps_set(insts)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    with open(path.with_name(path.name.replace(".jsonl", ".SOURCE.json")), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"file_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return path


def raises_stop(fn, needle: str = "") -> tuple:
    import lbeam_run as LR
    try:
        fn()
    except LR.Stop as e:
        return needle in str(e), str(e)[:260], e.persist
    return False, "no Stop raised", None


def tree_hashes(d: Path, rels) -> dict:
    return {r: sha(d / r) for r in rels if (d / r).is_file()}


class GenPatch:
    def __init__(self, gen):
        from rcwmt import generation
        self.g, self.gen, self.real = generation, gen, None

    def __enter__(self):
        self.real = self.g.generate_texts
        self.g.generate_texts = self.gen

    def __exit__(self, *a):
        self.g.generate_texts = self.real


# ================================================================ part: tiny
def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def complete(LR, c, rt=None, max_calls: int = 3, device: str = "cpu-tiny") -> list:
    """Call the job (run_main: the exit-code mapping) until it no longer defers (at most max_calls); the exit codes."""
    rcs = []
    Path(c.out).parent.mkdir(parents=True, exist_ok=True)
    for _ in range(max_calls):
        rcs.append(LR.run_main(c, make_backend(rt or FakeRuntime(Path(c.out).parent), device)))
        if rcs[-1] != 5:
            break
    return rcs


def part_tiny(keep: bool) -> dict:
    tiny_setup()
    import lbeam_run as LR
    import lbeam_problems as P
    import lbeam_analysis as LA
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="lbeam_run_tiny_", dir=SCRATCH if SCRATCH.exists() else None))
    print(f"workspace {tmp}", flush=True)
    try:
        adapter = tmp / "adapter"
        make_adapter(adapter)
        chains = LR.pilot_anchor_chains()[:8]
        refs_real = single_adapter_refs(tmp, chains, adapter)
        stub = make_stub("trigger")
        refs_stub = single_adapter_refs(tmp, chains, adapter, gen=stub, tag="stub")
        small = P.make_set(P.SetSpec(name="lbt", seed=777001, n_in_per_depth=2, n_oor_per_depth=0))[0]
        draw_small = write_draw(tmp / "draws" / "small.jsonl", small)
        dev = P.load_set(ROOT / LR.DEV_REL)
        draw_dev = write_draw(tmp / "draws" / "dev.jsonl", dev)
        (tmp / "runs").mkdir(parents=True, exist_ok=True)
        print(f"setup done ({time.time() - t0:.0f}s): 8 anchor chains, draws {len(small)} / {len(dev)}", flush=True)

        def cfg(name, draw, refs, **kw):
            return LR.Cfg(root=ROOT, out=tmp / "runs" / name, draw=draw, draw_source=draw.with_name(draw.name.replace(".jsonl", ".SOURCE.json")),
                          anchor_chains=chains, anchor_refs=refs, adapter_dir=adapter, require_freeze=False,
                          bi_n_inst=kw.pop("bi_n_inst", 2), identity={"tiny": "standin", "adapter_name": "t2"}, **kw)

        rels_small = [LR.rel_out(p, Path("x")) for st in LR.STAGE_ORDER for p in LR.stage_required(st, Path("x"))
                      if not p.name.endswith(("anchor.json", "batch_invariance.json", "accounting.json"))]
        planning_dirs = ("primary", "fixed", "w4e24", "audit", "cpu")

        print("1 real tiny generation, end to end (two calls: C1 never runs in the call that used the GPU)", flush=True)
        rt = FakeRuntime(tmp)
        ca = cfg("run_a", draw_small, refs_real)
        rc1 = LR.run_main(ca, make_backend(rt))
        p1 = read_json(ca.out / "provenance.json")
        n_calls = len(rt.calls)
        check("first call: every GPU stage recorded under the one lease; C1 pending (it never runs in a call that used "
              "the GPU): exit 5, no DONE.json", rc1 == 5 and set(p1["stages"]) == set(LR.GPU_STAGES)
              and not (ca.out / "DONE.json").exists(), (rc1, sorted(p1["stages"])))
        rc2 = LR.run_main(ca, make_backend(rt))
        prov = read_json(ca.out / "provenance.json")
        check("second call: C1 runs first with no lease, no model load and no CUDA use, then DONE.json (exit 0); every "
              "stage recorded; not marked registered", rc2 == 0 and (ca.out / "DONE.json").is_file()
              and set(prov["stages"]) == set(LR.STAGE_ORDER) and prov["registered"] is False and len(rt.calls) == n_calls
              and prov["cpu_runs"][0]["cuda_initialized_before"] is False and prov["cpu_runs"][0]["lease_held"] is False,
              rc2)
        leases = [c for c in rt.calls if c[0] == "lease"]
        check("exactly one lease, label 'rcwm-t1:lbeam'; resource wait before the load; unload at the end",
              leases == [("lease", LR.LEASE_LABEL)] and [c[0] for c in rt.calls][:3] == ["lease", "wait", "load"]
              and ("unload",) in rt.calls, [c[0] for c in rt.calls])
        anc = read_json(ca.out / "anchor" / "anchor.json")
        bi = read_json(ca.out / "anchor" / "batch_invariance.json")
        check("A0 anchor: OFF (disable_adapter) and ON reproduce single-adapter references (states AND texts 1.0)",
              anc["pass"] and anc["off"]["identical_states"] == anc["on"]["identical_states"] == 1.0
              and anc["off"]["identical_texts"] == anc["on"]["identical_texts"] == 1.0, {s: anc[s] for s in ("off", "on")})
        check("A0 batch invariance: one generation call batched vs one per key (batch 1), states identical, recorded",
              bi["pass"] and all(v["generation_calls_batched"] == 1 and v["generation_calls_batch1"] == bi["n_keys"]
                                 and v["max_prompts_per_model_generate_batch1"] == 1 for v in bi["arms"].values()), bi["arms"])
        rel_t = [c[2] for c in rt.calls if c[0] == "release"][0]
        check("C1 (ORACLE / IGNORE_STATUS) ran after A0 passed, after the lease release, flagged outside the lease and "
              "CUDA-free", prov["stages"]["C1"]["after_a0"] is True and prov["cpu_runs"][0]["start_unix"] >= rel_t
              and prov["cpu_runs"][0]["lease_held"] is False and prov["cpu_runs"][0]["cuda_initialized_after"] is False)
        t_ = prov["turns"][0]
        check("provenance records only planned generation counts for planning stages; the turn records the cap, the start "
              "check, the attempts and the hashes of every loaded BWM module (blocks_planner.py at its pin)",
              all("generations" not in prov["stages"][s] and prov["stages"][s]["generations_planned_expected"] >= 0
                  for s in ("G1", "G2", "G3", "G4")) and t_["cap_h"] == LR.CAP_H and "start_check" in t_
              and t_["bwm_modules_sha256"].get(LR.BWM_PLANNER_REL) == LR.BWM_PLANNER_SHA
              and prov["attempts"] == {"A0": [0], "G1": [0], "G2": [0], "G3": [0], "G4": [0], "C1": ["cpu"]},
              prov["attempts"])
        I = LA.Inputs(ca.out, draw_small, registered=False)
        res = LA.analyse(I)
        check("the analysis validates the tiny run in fixture mode and computes the decision (all answers unparseable: "
              "T2 = OFF = 0 successes)", res["decision"]["cp1"]["t2_success"] == 0 == res["decision"]["cp1"]["off_success"],
              res["wording"]["headline"])
        n_calls = len(rt.calls)
        h_a = tree_hashes(ca.out, rels_small)
        rc3 = LR.run_main(ca, make_backend(rt))
        check("a rerun after DONE is a no-op (exit 0, no lease, files unchanged)",
              rc3 == 0 and len(rt.calls) == n_calls and tree_hashes(ca.out, rels_small) == h_a)

        print("2 rule-stub run on the dev set (T2 trigger-level, OFF status-blind) through the analysis", flush=True)
        with GenPatch(stub):
            cd = cfg("run_dev", draw_dev, refs_stub)
            rcs = complete(LR, cd)
        ep = {n_: [json.loads(l) for l in (cd.out / r).read_text(encoding="utf-8").splitlines()]
              for st in LR.STAGE_ORDER for n_, r in LR.FILES[st].items() if n_.startswith(("eps:", "fixed:"))}
        strip = lambda rows: [{k: v for k, v in r.items()} for r in rows]  # noqa: E731
        check("stub run exits 5 then 0; canonical T2 (trigger-level) == ORACLE episodes at W2 E12 and W4 E24 and in "
              "select_fixed (statuses are posted / void only); OFF (status-blind) == IGNORE_STATUS",
              rcs == [5, 0] and strip(ep["eps:T2:W2E12"]) == strip(ep["eps:ORACLE:W2E12"])
              and strip(ep["eps:T2:W4E24"]) == strip(ep["eps:ORACLE:W4E24"])
              and [r["plan"] for r in ep["fixed:T2"]] == [r["plan"] for r in ep["fixed:ORACLE"]]
              and strip(ep["eps:OFF:W2E12"]) == strip(ep["eps:IGNORE_STATUS:W2E12"])
              and [r["plan"] for r in ep["fixed:OFF"]] == [r["plan"] for r in ep["fixed:IGNORE_STATUS"]], rcs)
        check("the D-35 audit arm (void rendered 'cancelled', keys reversed, word names) with the trigger-level stub "
              "reproduces IGNORE_STATUS's episodes, not ORACLE's",
              strip(ep["eps:T2@heldout_combined:W2E12"]) == strip(ep["eps:IGNORE_STATUS:W2E12"])
              and strip(ep["eps:T2@heldout_combined:W2E12"]) != strip(ep["eps:T2:W2E12"]))
        I = LA.Inputs(cd.out, draw_dev, registered=False)
        res = LA.analyse(I)
        D = res["decision"]
        NUM["stub_dev_analysis"] = {"branch": D["branch"], "cp1": {k: D["cp1"][k] for k in ("t2_success", "off_success", "delta", "p_holm")},
                                    "cp2": {k: D["cp2"][k] for k in ("n", "t2_success", "off_success", "delta", "p_holm")},
                                    "ratio": D["ratio"]["ratio"], "audit": res["audit"]["reading"],
                                    "ratio_needs_lookahead": res["ratio_needs_lookahead"].get("ratio"),
                                    "t2_vs_ignore_status": {k: res["t2_vs_ignore_status"][k] for k in ("b_t2_only", "c_off_only", "p_raw")}}
        check("analysis of the stub run: T2 == ORACLE (ratio 1.0, success 0.85 and needs-lookahead 0.58 as in the kit's "
              "ORACLE sizing), OFF == IGNORE_STATUS (0.283), both co-primaries PASS -> YES_NEAR",
              D["ratio"]["ratio"] == 1.0 and abs(D["ratio"]["oracle"] - 0.85) < 1e-9 and D["branch"] == "YES_NEAR"
              and res["arms"]["eps:OFF:W2E12"]["success"] == 0.283 and D["cp2"]["n"] == 43, NUM["stub_dev_analysis"])
        check("analysis report-only metrics: void-twin exposure ORACLE 0, OFF (status-blind) > 0 (applied as posted and "
              "not a no-op); the audit reads surface-dependent; refused keys 0 for every exact-equivalent arm",
              res["void_twin_exposure"]["eps:ORACLE:W2E12"]["exposure"] == 0.0
              and (res["void_twin_exposure"]["eps:OFF:W2E12"]["exposure"] or 0) > 0
              and (res["void_twin_exposure"]["eps:OFF:W2E12"]["exposure_not_noop"] or 0) > 0
              and res["void_twin_exposure"]["fixed:T2"]["includes_stored_distractors"] is True
              and res["audit"]["reading"] == LA.AUDIT_TEXT["dependent"]
              and res["cost"]["T2:W2E12"]["refused_unique"] == 0, {k: v["exposure"] for k, v in res["void_twin_exposure"].items()})
        dd = res["discordant_by_depth"]
        quals = res["wording"]["qualifiers"]
        check("report-only additions: T2/ORACLE on the needs-lookahead stratum (1.0 here), T2 vs IGNORE_STATUS, the "
              "co-primaries' discordant pairs by d* (summing to b and c), the instrument record",
              res["ratio_needs_lookahead"]["ratio"] == 1.0 and any(q.startswith("Report-only: T2/ORACLE on the 43") for q in quals)
              and any(q.startswith("Report-only: T2 vs IGNORE_STATUS") for q in quals)
              and sum(v["t2_only"] for v in dd["cp1_all"].values()) == D["cp1"]["b_t2_only"]
              and sum(v["off_only"] for v in dd["cp2_needs_lookahead"].values()) == D["cp2"]["c_off_only"]
              and res["instrument"]["terminal"] == "done" and res["instrument"]["anchor"]["pass"] is True
              and not res["not_evaluable"], {"la": res["ratio_needs_lookahead"].get("ratio")})
        md = LA.render(res)
        check("the mechanical render contains the headline, the limits, every arm row and the instrument block",
              res["wording"]["headline"] in md and all(x in md for x in LA.LIMITS) and all(k in md for k in res["arms"])
              and "## Instrument and deviations (report-only)" in md)

        print("3 crash in G2 + stray .partial -> resume", flush=True)
        with GenPatch(stub):
            cb = cfg("run_b", draw_small, refs_stub)
            check("the uninterrupted reference run completes in two calls (5, then 0)", complete(LR, cb) == [5, 0])
            h_b = tree_hashes(cb.out, rels_small)
            res_b = LA.analyse(LA.Inputs(cb.out, draw_small, registered=False))
            cc = cfg("run_c", draw_small, refs_stub)
            real_stage = LR.run_planning_stage

            def crashing(st, *a, **k):
                if st == "G2":
                    (cc.out / "fixed").mkdir(parents=True, exist_ok=True)
                    (cc.out / "fixed" / "fixed_T2.jsonl.partial").write_text("half", encoding="utf-8")
                    raise RuntimeError("simulated crash in G2")
                return real_stage(st, *a, **k)
            LR.run_planning_stage = crashing
            try:
                try:
                    LR.run_main(cc, make_backend(FakeRuntime(tmp)))
                    crashed = False
                except RuntimeError:
                    crashed = True
            finally:
                LR.run_planning_stage = real_stage
            pc = read_json(cc.out / "provenance.json")
            check("a crash propagates out of run_main (the interpreter's exit 1: the hook's one retry), leaves A0 and G1 "
                  "recorded, G2 partial, no C1, no STOPPED.json, the lock released",
                  crashed and set(pc["stages"]) == {"A0", "G1"} and not (cc.out / ".run.lock").exists()
                  and LR.stage_states(pc, cc.out)["G2"] == "partial" and not (cc.out / "STOPPED.json").exists())
            rc = LR.run_main(cc, make_backend(FakeRuntime(tmp)))
            pc = read_json(cc.out / "provenance.json")
        orph = list((cc.out / "_orphans").glob("G2_*"))
        last = pc["turns"][-1]
        check("the resume orphans the partial G2 (never deleted, recorded), runs C1 BEFORE its lease, re-runs A0 first "
              "(anchor AND batch invariance, both recorded and passing), runs G2-G4, completes; G2's failed attempt is "
              "counted", rc == 0 and orph and (orph[0] / "fixed_T2.jsonl.partial").is_file() and last["anchor_recheck"]["pass"]
              and last["batchinv_recheck"]["pass"] and last["stages_run"] == ["G2", "G3", "G4"] and "C1" in pc["stages"]
              and pc["cpu_runs"][-1]["end_unix"] <= last["acquired_unix"] and pc["attempts"]["G2"] == [0, 1]
              and [o["stage"] for o in pc["orphaned"]] == ["G2"], last.get("stages_run"))
        check("the resumed run's outputs equal the uninterrupted run byte for byte (episodes, caches, anchor rows)",
              tree_hashes(cc.out, rels_small) == h_b, [r for r in rels_small if tree_hashes(cc.out, [r]) != tree_hashes(cb.out, [r])])
        I = LA.Inputs(cc.out, draw_small, registered=False)
        r_c = LA.analyse(I)
        check("the analysis accepts the resumed run (re-check recorded for the resumed turn; caches extend per arm) and "
              "lists the orphaned stage in the instrument record", r_c["n"] == len(small)
              and r_c["instrument"]["orphaned_stages"][0]["stage"] == "G2" and len(r_c["instrument"]["orphan_dirs"]) == 1)

        print("4 deferral (nothing fits; mid-run; the no-progress stall)", flush=True)
        with GenPatch(stub):
            rt_g = FakeRuntime(tmp)
            cg = cfg("run_g", draw_small, refs_stub, cap_h=0.6, rate=0.01)
            rc = LR.run_main(cg, make_backend(rt_g))
            pg = read_json(cg.out / "provenance.json")
            check("nothing fits the cap: every GPU stage deferred, exit 5, no model load, nothing generated, no C1; the "
                  "turn is counted as no progress (1)",
                  rc == 5 and pg["turns"][-1]["deferred"] == list(LR.GPU_STAGES) and not pg["stages"]
                  and not [c for c in rt_g.calls if c[0] == "load"] and not (cg.out / "anchor").exists()
                  and pg["turns"][-1]["no_progress"] is True and pg["no_progress_turns"] == 1)
            cg.cap_h, cg.rate = LR.CAP_H, LR.RATE_GENS_PER_S
            rcs = complete(LR, cg, rt_g)
            pg = read_json(cg.out / "provenance.json")
            check("the next turns complete (5: C1 pending, then 0), reset the no-progress count and equal the "
                  "uninterrupted run", rcs == [5, 0] and pg["no_progress_turns"] == 0 and tree_hashes(cg.out, rels_small) == h_b, rcs)
            real_est = LR.stage_est_s
            LR.stage_est_s = lambda st, rate, n, mode="expected": {"A0": 60.0, "G1": 60.0}.get(st, 10 * 3600.0)
            try:
                ch = cfg("run_h", draw_small, refs_stub)
                rc = LR.run_main(ch, make_backend(FakeRuntime(tmp)))
            finally:
                LR.stage_est_s = real_est
            ph = read_json(ch.out / "provenance.json")
            check("mid-run deferral at a stage boundary: A0 and G1 run, G2-G4 deferred (exit 5); C1 waits for the next call",
                  rc == 5 and ph["turns"][-1]["deferred"] == ["G2", "G3", "G4"] and set(ph["stages"]) == {"A0", "G1"},
                  ph["turns"][-1].get("deferred"))
            rc = LR.run_main(ch, make_backend(FakeRuntime(tmp)))
            ph = read_json(ch.out / "provenance.json")
            check("the resumed call runs C1 before its lease, re-runs A0 (anchor + batch invariance) before G2, completes, "
                  "equals the uninterrupted run", rc == 0 and ph["turns"][-1]["anchor_recheck"]["pass"]
                  and ph["turns"][-1]["batchinv_recheck"]["pass"] and tree_hashes(ch.out, rels_small) == h_b
                  and ph["cpu_runs"][-1]["end_unix"] <= ph["turns"][-1]["acquired_unix"])
            # the reviewer's stall: an A0 that takes longer than its estimate (fake clock: +1.2 h in batch invariance)
            off = [0.0]
            real_time, real_bi = time.time, LR.run_batch_invariance
            n180 = LR.counts(180)

            def slow_bi(*a, **k):
                r = real_bi(*a, **k)
                off[0] += 1.2 * 3600
                return r
            time.time = lambda: real_time() + off[0]
            LR.run_batch_invariance = slow_bi
            LR.stage_est_s = lambda st, rate, n, mode="expected": real_est(st, rate, n180, mode)
            try:
                rt_s = FakeRuntime(tmp)
                cs = cfg("run_stall", draw_small, refs_stub)
                rcs = [LR.run_main(cs, make_backend(rt_s)) for _ in range(3)]
            finally:
                time.time, LR.run_batch_invariance, LR.stage_est_s = real_time, real_bi, real_est
            ps = read_json(cs.out / "provenance.json")
            stp = read_json(cs.out / "STOPPED.json") if (cs.out / "STOPPED.json").exists() else {}
            check("stall (an A0 measured at 1.2 h, registered 180-instance estimates): turn 1 runs A0 and defers G1 (5); the "
                  "next turns defer on the MEASURED A0 without loading the model; the second turn without a recorded stage "
                  "stops (3, persisted, no verdict). One model load in all",
                  rcs == [5, 5, 3] and len([c for c in rt_s.calls if c[0] == "load"]) == 1
                  and "no stage was recorded in 2 consecutive turns" in stp.get("reason", "")
                  and stp.get("verdict_eligible") is False and ps["turns"][1]["start_check"]["a0_s"] > 3600
                  and "C1" in ps["stages"] and "G1" not in ps["stages"], (rcs, stp.get("reason", "")[:120]))

        print("5 anchor failure, batch-invariance failure, re-check failure (verdict-eligible stop)", flush=True)
        bad = tmp / "refs" / "ref_bad_off.jsonl"
        rows = [json.loads(l) for l in refs_stub["off"].read_text(encoding="utf-8").splitlines()]
        for x in rows:
            x["states"][-1] = '{"A":0,"B":0,"C":0,"D":0}'
        with open(bad, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in rows))
        with GenPatch(stub):
            cx = cfg("run_x", draw_small, {"off": bad, "on": refs_stub["on"]})
            rc = LR.run_main(cx, make_backend(FakeRuntime(tmp)))
            anc = read_json(cx.out / "anchor" / "anchor.json")
            stx = read_json(cx.out / "STOPPED.json")
            check("an anchor failure stops (exit 3, STOPPED.json persisted, no verdict) before ANY planning output (no "
                  "primary/ fixed/ w4e24/ audit/ cpu/ dirs, no batch-invariance run)", rc == 3 and anc["pass"] is False
                  and stx["verdict_eligible"] is False and "ANCHOR FAILED" in stx["reason"]
                  and not any((cx.out / d).exists() for d in planning_dirs)
                  and not (cx.out / "anchor" / "batch_invariance.json").exists(), stx["reason"][:120])
            b0 = (cx.out / "STOPPED.json").read_bytes()
            rc = LR.run_main(cx, make_backend(FakeRuntime(tmp)))
            check("after the stop, STOPPED.json refuses every rerun (exit 3, not persisted again: the record is unchanged)",
                  rc == 3 and (cx.out / "STOPPED.json").read_bytes() == b0)
        with GenPatch(make_stub("trigger", bad_batch1=True)):
            cy = cfg("run_y", draw_small, refs_stub)
            rc = LR.run_main(cy, make_backend(FakeRuntime(tmp)))
            biy = read_json(cy.out / "anchor" / "batch_invariance.json")
            check("a batch-invariance failure (answers depend on batch composition) stops (exit 3, persisted, no verdict) "
                  "before any planning output; batch_invariance.json records pass false", rc == 3 and biy["pass"] is False
                  and read_json(cy.out / "STOPPED.json")["verdict_eligible"] is False
                  and not any((cy.out / d).exists() for d in planning_dirs), biy["arms"])
            (cy.out / "STOPPED.json").unlink()                 # an operator moved it aside: the failed record still refuses
            ok3, msg3, _ = raises_stop(lambda: LR.run(cy, make_backend(FakeRuntime(tmp))), "batch_invariance.json check failed")
            check("with STOPPED.json moved aside, the failed batch_invariance.json still refuses the run", ok3, msg3)
        with GenPatch(stub):
            real_est = LR.stage_est_s
            LR.stage_est_s = lambda st, rate, n, mode="expected": {"A0": 60.0, "G1": 60.0}.get(st, 10 * 3600.0)
            try:
                cz = cfg("run_z", draw_small, refs_stub)
                rc = LR.run_main(cz, make_backend(FakeRuntime(tmp)))
            finally:
                LR.stage_est_s = real_est
            cz.anchor_refs = {"off": bad, "on": refs_stub["on"]}
            rc2 = LR.run_main(cz, make_backend(FakeRuntime(tmp)))
            pz = read_json(cz.out / "provenance.json")
            stz = read_json(cz.out / "STOPPED.json")
            check("a failed A0 re-check on a resumed turn (after C1 ran before the lease) stops (exit 3, persisted) before "
                  "any planning stage; RECHECK_FAILED_*.json written; G2-G4 never run; A0, G1 and C1 recorded and unchanged: "
                  "the stop is verdict-eligible", rc == 5 and rc2 == 3 and list((cz.out / "anchor").glob("RECHECK_FAILED_*.json"))
                  and set(pz["stages"]) == {"A0", "G1", "C1"} and pz["turns"][-1]["stages_run"] == []
                  and stz["verdict_eligible"] is True and "RE-CHECK FAILED" in stz["reason"], stz.get("reason", "")[:120])
            rz = LA.analyse(LA.Inputs(cz.out, draw_small, registered=False))
            sent_z = LA.not_evaluable_sentence(rz["not_evaluable"])
            check("the analysis renders the verdict of the verdict-eligible stop from A0, G1 and C1, with G2, G3 and G4 "
                  "'not evaluable' (not run: stopped), mechanically; the failed re-check is in the instrument record",
                  set(rz["not_evaluable"]) == {"G2", "G3", "G4"} and all(v["kind"] == "stopped" for v in rz["not_evaluable"].values())
                  and sent_z in rz["wording"]["qualifiers"] and "not run: the run ended by a deterministic stop" in sent_z
                  and rz["instrument"]["terminal"] == "stopped" and rz["instrument"]["recheck_failed_files"]
                  and rz["wording"]["branch"] == res_b["wording"]["branch"] and "| not evaluable |" in LA.render(rz), sent_z)

        print("6 an abandoned report-only stage; a core stage failing three times", flush=True)
        with GenPatch(stub):
            real_stage = LR.run_planning_stage

            def crash_in(stage):
                def f(st, *a, **k):
                    if st == stage:
                        raise RuntimeError(f"simulated crash in {stage}")
                    return real_stage(st, *a, **k)
                return f

            def calls(c, n_):
                outs = []
                for _ in range(n_):
                    try:
                        outs.append(LR.run_main(c, make_backend(FakeRuntime(tmp))))
                    except RuntimeError:
                        outs.append("crash")
                return outs
            cab = cfg("run_ab", draw_small, refs_stub)
            LR.run_planning_stage = crash_in("G3")
            try:
                outs = calls(cab, 3)
            finally:
                LR.run_planning_stage = real_stage
            pab = read_json(cab.out / "provenance.json")
            dab = read_json(cab.out / "DONE.json") if (cab.out / "DONE.json").exists() else {}
            check("G3 (report-only) crashes twice: the third call abandons it (not evaluable), runs G4 and writes DONE.json "
                  "listing G3 as not evaluable (calls: crash, crash, 0)",
                  outs == ["crash", "crash", 0] and pab["abandoned"]["G3"]["failed_attempts"] == 2
                  and set(dab.get("not_evaluable") or {}) == {"G3"} and set(pab["stages"]) == set(LR.STAGE_ORDER) - {"G3"}, outs)
            rab = LA.analyse(LA.Inputs(cab.out, draw_small, registered=False))
            check("the analysis renders G3 'not evaluable' (abandoned after 2 failed attempts), no W4 E24 line, and the same "
                  "branch as the complete run (the branch needs only A0, G1 and C1)",
                  rab["not_evaluable"] == {"G3": {"kind": "abandoned", "n": 2}} and rab["w4e24"] is None
                  and any("G3 (T2 at enum W4 E24), abandoned after 2 failed attempts" in q for q in rab["wording"]["qualifiers"])
                  and rab["wording"]["branch"] == res_b["wording"]["branch"] and "eps:T2:W4E24" not in rab["arms"])
            cco = cfg("run_core", draw_small, refs_stub)
            LR.run_planning_stage = crash_in("G1")
            try:
                outs = calls(cco, 4)
            finally:
                LR.run_planning_stage = real_stage
            sco = read_json(cco.out / "STOPPED.json") if (cco.out / "STOPPED.json").exists() else {}
            check("G1 (core) failing three times: the fourth call stops (exit 3, persisted, no verdict: G1 not recorded)",
                  outs == ["crash", "crash", "crash", 3] and "core stage G1 failed 3 attempts" in sco.get("reason", "")
                  and sco.get("verdict_eligible") is False, outs)
            check("the analysis refuses the core-stage stop", _refuses(lambda: LA.Inputs(cco.out, draw_small, False)))

        print("7 tamper, fingerprint, runtime, lock, one-waiter, main()", flush=True)
        import psutil
        with GenPatch(stub):
            ct = cfg("run_t", draw_small, refs_stub)
            shutil.copytree(cb.out, ct.out)
            (ct.out / "DONE.json").unlink()
            p = ct.out / LR.FILES["G1"]["eps:T2:W2E12"]
            p.write_text(p.read_text(encoding="utf-8").replace('"verified": true', '"verified": false', 1), encoding="utf-8")
            rc = LR.run_main(ct, make_backend(FakeRuntime(tmp)))
            stt = read_json(ct.out / "STOPPED.json") if (ct.out / "STOPPED.json").exists() else {}
            check("a recorded output whose hash changed stops the job (exit 3, persisted, not verdict-eligible: a recorded "
                  "output changed)", rc == 3 and "changed after it was recorded" in stt.get("reason", "")
                  and stt.get("verdict_eligible") is False and stt.get("recorded_outputs_unchanged") is False)
            rt_f = FakeRuntime(tmp)
            rt_f.fp = "deadbeef0000"
            cf = cfg("run_f", draw_small, refs_stub)
            rc = LR.run_main(cf, make_backend(rt_f))
            check("a Titan-Sparse fingerprint mismatch stops before any generation (exit 3, persisted)",
                  rc == 3 and "fingerprint" in read_json(cf.out / "STOPPED.json")["reason"] and not (cf.out / "anchor").exists())
            rt_r = FakeRuntime(tmp)
            rt_r.runtime_override = [{"transformers": "0.0.0-drift"}]
            cr = cfg("run_r", draw_small, refs_stub)
            rc = LR.run_main(cr, make_backend(rt_r))
            check("runtime drift after the load stops before any generation (exit 3, persisted); the lease is released",
                  rc == 3 and "differs from the registered" in read_json(cr.out / "STOPPED.json")["reason"]
                  and not (cr.out / "anchor").exists() and rt_r.calls[-1][0] == "release")
            cl = cfg("run_l", draw_small, refs_stub)
            cl.out.mkdir(parents=True)
            (cl.out / ".run.lock").write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()}), encoding="utf-8")
            rc_l = LR.run_main(cl, make_backend(FakeRuntime(tmp)))
            (cl.out / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8")
            rcs = complete(LR, cl)
            check("a live run lock refuses (exit 3, not persisted); a stale one is reclaimed and the run completes",
                  rc_l == 3 and rcs == [5, 0] and not (cl.out / ".run.lock").exists() and not (cl.out / "STOPPED.json").exists(), rcs)
            parent = psutil.Process().parent()
            rt_w = FakeRuntime(tmp)
            cw = cfg("run_w", draw_small, refs_stub)
            rt_w.WAITING["rcwm"].write_text(json.dumps({"pid": parent.pid, "label": "rcwm-t1:other", "since": time.time()}), encoding="utf-8")
            ok1, m1, p1_ = raises_stop(lambda: LR.run(cw, make_backend(rt_w)), "one-waiter rule")
            rt_w.WAITING["rcwm"].unlink()
            rt_w.LOCK_PATH.write_text(json.dumps({"pid": parent.pid, "label": "rcwm-t1:other", "created": time.time()}), encoding="utf-8")
            ok2, m2, p2_ = raises_stop(lambda: LR.run(cw, make_backend(rt_w)), "holds the GPU lease")
            check("one-waiter rule: a live RCWM waiter or holder refuses (not persisted; no lease, nothing written)",
                  ok1 and ok2 and p1_ is False and p2_ is False and not [c for c in rt_w.calls if c[0] == "lease"]
                  and not cw.out.exists(), f"{m1} | {m2}")
            rt_w.LOCK_PATH.write_text(json.dumps({"pid": parent.pid, "label": "bwm-001:job", "created": time.time()}), encoding="utf-8")
            rt_w.WAITING["rcwm"].write_text(json.dumps({"pid": 999999, "label": "rcwm-t1:old", "since": 1.0}), encoding="utf-8")
            okx = LR.one_waiter_problem(make_backend(rt_w)) is None
            rt_w.WAITING["rcwm"].unlink()
            rt_w.LOCK_PATH.unlink()
            check("one-waiter rule: a dead RCWM marker and a live BWM holder do not refuse", okx)
        real_rb, argv = LR.RealBackend, sys.argv
        existed = (ROOT / "runs" / "lbeam").exists()
        tb = make_backend(FakeRuntime(tmp))                # built before the patch (it subclasses RealBackend)
        LR.RealBackend = lambda: tb
        sys.argv = ["lbeam_run.py"]
        try:
            rc_m = LR.main()
        finally:
            LR.RealBackend, sys.argv = real_rb, argv
        check("main() (the registered job) on the real, unfrozen root exits 3 and persists nothing (no runs/lbeam)",
              rc_m == 3 and not existed and not (ROOT / "runs" / "lbeam").exists(), rc_m)

        print("8 analysis refusals on tampered copies of a complete run", flush=True)

        def copy_run(name, src=None):
            d = tmp / "runs" / name
            shutil.copytree(src or cb.out, d)
            return d
        d1 = copy_run("an_hash")
        p = d1 / LR.FILES["G1"]["eps:OFF:W2E12"]
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        d2 = copy_run("an_done")
        (d2 / "DONE.json").unlink()
        d3 = copy_run("an_prov")
        pv = read_json(d3 / "provenance.json")
        pv["batch_invariance"]["pass"] = False
        (d3 / "provenance.json").write_text(json.dumps(pv), encoding="utf-8")
        d4 = copy_run("an_stop")
        (d4 / "STOPPED.json").write_text("{}", encoding="utf-8")
        d6 = copy_run("an_stop_ineligible")
        (d6 / "DONE.json").unlink()
        (d6 / "STOPPED.json").write_text(json.dumps({"reason": "x", "verdict_eligible": False,
                                                     "provenance_sha256": sha(d6 / "provenance.json")}), encoding="utf-8")
        d7 = copy_run("an_stop_prov", cz.out)
        pv7 = read_json(d7 / "provenance.json")
        pv7["note"] = "edited after the stop"
        (d7 / "provenance.json").write_text(json.dumps(pv7), encoding="utf-8")
        d8 = copy_run("an_cpu_cuda")
        pv8 = read_json(d8 / "provenance.json")
        pv8["cpu_runs"][0]["cuda_initialized_before"] = True
        (d8 / "provenance.json").write_text(json.dumps(pv8), encoding="utf-8")
        dn8 = read_json(d8 / "DONE.json")
        dn8["provenance_sha256"] = sha(d8 / "provenance.json")
        (d8 / "DONE.json").write_text(json.dumps(dn8), encoding="utf-8")
        check("the analysis refuses: a changed output (hash), a missing DONE.json, a provenance edited after DONE "
              "(batch invariance), DONE.json and STOPPED.json together, a STOPPED.json that is not verdict-eligible, a "
              "verdict-eligible stop whose provenance changed afterwards, a C1 run not recorded CUDA-free; and refuses a "
              "registered analysis of an unregistered run",
              all(_refuses(lambda d_=d_: LA.Inputs(d_, draw_small, False)) for d_ in (d1, d2, d3, d4, d6, d7, d8))
              and _refuses(lambda: LA.Inputs(cb.out, draw_small, True)))
        d5 = copy_run("an_cache")
        cp_ = d5 / LR.FILES["G2"]["cache:T2"]
        lines = cp_.read_text(encoding="utf-8").splitlines()
        if len(lines) > 2:
            lines[1], lines[2] = lines[2], lines[1]
        cp_.write_text("\n".join(lines) + "\n", encoding="utf-8")
        errs = []
        caches = {(st, k[6:]): LA.cache_rows(d5 / r, k[6:], errs, r) for st in LR.STAGE_ORDER for k, r in LR.FILES[st].items()
                  if k.startswith("cache:")}
        check("a later cache that does not extend the earlier stage's entries is detected (F must be the first computed "
              "answer)", bool(LA.cache_chain_errors(caches, small)))
    finally:
        NUM["peak_rss_mb"] = peak_mb()
        if keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    NUM["seconds"] = round(time.time() - t0, 1)
    return {}


# ================================================================ part: freeze (fake root)
TEST_SALT = "freeze-test-salt-0123456789abcdef"     # stands in for BWM's salt (the real one is BWM's, revealed at the freeze)


def _refuse_msg(fn) -> str | None:
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


def part_freeze(keep: bool) -> dict:
    tiny_setup()
    import lbeam_run as LR
    import lbeam_problems as P
    import lbeam_analysis as LA
    from collections import OrderedDict
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="lbeam_freeze_", dir=SCRATCH if SCRATCH.exists() else None))
    froot = tmp / "fakeroot"
    saved = {k: getattr(LR, k) for k in ("PINS", "N_PER_DEPTH", "build_exclusions", "BWM_ROOT", "BWM_DRAFT_SHA")}
    sys_path0 = list(sys.path)
    try:
        (froot / "rcwmt").mkdir(parents=True)
        for p in (ROOT / "rcwmt").glob("*.py"):
            shutil.copy2(p, froot / "rcwmt" / p.name)
        for rel in LR.REQUIRED_INPUTS:
            src = ROOT / rel
            if rel.startswith("scripts/") or rel in (LR.DRAFT_REL, LR.HOOK_DRAFT_REL, "docs/ledger_beam/INTERFACE.md",
                                                     "docs/ledger_beam/dev2_primary.SOURCE.json", LR.DEV_REL,
                                                     "docs/ledger_beam/TEST_ORACLE.json", "docs/ledger_beam/TEST_TINY.json",
                                                     "docs/ledger_beam/TEST_FRESH.json"):
                (froot / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, froot / rel)
        for rel in LR.SPENT_DEV_SETS:
            shutil.copy2(ROOT / rel, froot / rel)
        adapter = froot / LR.ADAPTER_REL
        make_adapter(adapter)
        if (adapter / "default").exists():
            for f in (adapter / "default").iterdir():
                shutil.move(str(f), str(adapter / f.name))
        chains = LR.pilot_anchor_chains()[:8]
        stub = make_stub("trigger")
        refs = single_adapter_refs(tmp, chains, adapter, gen=stub, tag="stub")
        for s, rel in LR.ANCHOR_REF_RELS.items():
            shutil.copy2(refs[s], froot / rel)
        (froot / "data").mkdir(parents=True, exist_ok=True)
        (froot / "data" / "bank_pilot.json").write_text("{}\n", encoding="utf-8")      # anchor chains come from the Cfg
        d = froot / "docs" / "ledger_beam"

        def write_records():
            code = {r: sha(froot / r) for r in LR.RUN_CODE}
            bind = {"draft_sha256": sha(froot / LR.DRAFT_REL), "hook_draft_sha256": sha(froot / LR.HOOK_DRAFT_REL)}
            for part in LR.RUN_TEST_PARTS:
                (d / f"RUN_TEST_{part.upper()}.json").write_text(json.dumps({"ok": True, "code_sha256": code, **bind}), encoding="utf-8")
            (d / "LB_DRYRUN.json").write_text(json.dumps({"ok": True, "code_sha256": {r: sha(froot / r) for r in LR.DRYRUN_CODE},
                                                          **bind}), encoding="utf-8")
            (d / "LB_DRYRUN.md").write_text("dry run (freeze test)\n", encoding="utf-8")
            (d / "LB_POWER.json").write_text(json.dumps({"ok": True, "code_sha256": code}), encoding="utf-8")
            (d / "GEN_PLAN_LB.json").write_text(json.dumps({"stages": [1]}), encoding="utf-8")
        write_records()
        LR.PINS = {r: sha(froot / r) for r in saved["PINS"]}
        LR.N_PER_DEPTH = 1                                  # 5 instances (the registered 36 per d* is not needed here)
        union_src = P.load_set(froot / LR.DEV_REL)[:20]

        def small_union(root, log=print):
            L1 = P._L1()
            pay, ops = set(), set()
            for inst in union_src:
                for b, e in P.binding_transitions(inst):
                    pay.add(L1.payload(b, e))
                    ops.add(L1.opkey(P.LEDGER, b, e))
            return {"pay": pay, "ops": ops, "sources": {"test": {"payloads": len(pay)}}, "rule": "freeze-test union",
                    "historical": False}
        LR.build_exclusions = small_union
        missing = sorted(r for r in LR.required_inputs(froot) if not (froot / r).is_file() and r not in (LR.DRAW_REL, LR.DRAW_SOURCE_REL))
        check("the fake root holds every required input except the draw (the real code, draft, hook draft, kit records; "
              "tiny adapter and references at their registered paths)", not missing, missing)
        ident = {"tiny": "standin", "adapter_name": "t2"}

        def cfg_r(**kw):
            return LR.Cfg(root=froot, out=froot / "runs" / "lbeam", draw=froot / LR.DRAW_REL, draw_source=froot / LR.DRAW_SOURCE_REL,
                          dev=froot / LR.DEV_REL, anchor_chains=chains, anchor_refs={s: froot / r for s, r in LR.ANCHOR_REF_RELS.items()},
                          adapter_dir=adapter, require_freeze=True, bi_n_inst=2, identity=ident, **kw)

        rc = LR.run_main(cfg_r(), make_backend(FakeRuntime(tmp), "cuda"))
        check("the job before any freeze refuses WITHOUT persisting (exit 3, no STOPPED.json, no output dir)",
              rc == 3 and not (froot / "runs" / "lbeam").exists(), rc)
        appr = "freeze test approval 2026-10-09 (no real freeze; DECISIONS D-test)"
        dsha = sha(froot / LR.DRAFT_REL)
        commit = LR.salt_commit(TEST_SALT)
        cos = f"BWM co-sign (freeze test only) of PREREG_LB_DRAFT.md sha256 {dsha} commit:{commit}"

        def fz_(a=appr, c=cos, s=TEST_SALT):
            return lambda: LR.freeze(a, c, s, froot)
        refusals = OrderedDict()
        refusals["no approval"] = raises_stop(fz_(a=""), "--approval")
        refusals["an approval without a DECISIONS entry"] = raises_stop(fz_(a="approved by the operator today"),
                                                                        "must cite the DECISIONS entry")
        refusals["no co-sign"] = raises_stop(fz_(c=""), "--bwm-cosign")
        refusals["a co-sign without the draft's sha256"] = raises_stop(fz_(c=f"BWM co-sign commit:{commit}"),
                                                                       "does not quote this draft's sha256")
        refusals["a co-sign without a salt commitment"] = raises_stop(fz_(c=f"BWM co-sign {dsha}"), "no salt commitment")
        refusals["a salt that does not match the commitment"] = raises_stop(fz_(s=TEST_SALT + "x"), "does not match the commitment")
        short = "short-salt"
        refusals["a salt shorter than 16 characters"] = raises_stop(
            fz_(c=f"BWM co-sign {dsha} commit:{LR.salt_commit(short)}", s=short), "shorter than")
        LR.BWM_DRAFT_SHA = "0" * 64
        refusals["BWM's draft is not the cited version"] = raises_stop(fz_(), "not the cited version")
        LR.BWM_DRAFT_SHA = saved["BWM_DRAFT_SHA"]
        (froot / "runs" / "lbeam").mkdir(parents=True)
        refusals["an lbeam output exists"] = raises_stop(fz_(), "outputs exist")
        shutil.rmtree(froot / "runs" / "lbeam")
        (d / "RUN_TEST_TINY.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
        refusals["a pre-freeze record failed"] = raises_stop(fz_(), "pre-freeze checks failed")
        write_records()
        dp = froot / LR.DRAFT_REL
        draft0 = dp.read_bytes()
        dp.write_bytes(draft0 + b"\nedited after the records\n")
        refusals["the draft changed after the pre-freeze records (a co-sign quoting the edited draft)"] = raises_stop(
            fz_(c=f"BWM co-sign {sha(dp)} commit:{commit}"), "produced with the current draft")
        dp.write_bytes(draft0)
        mv = froot / "docs" / "ledger_beam" / "INTERFACE.md"
        shutil.move(str(mv), str(tmp / "INTERFACE.md.aside"))
        refusals["a frozen input is missing"] = raises_stop(fz_(), "frozen inputs missing (nothing was minted)")
        shutil.move(str(tmp / "INTERFACE.md.aside"), str(mv))

        def union_err(root, log=print):
            u = small_union(root)
            u["sources"]["data/broken.jsonl"] = {"error": "ValueError: freeze test"}
            return u

        def union_writes(root, log=print):
            (froot / "data" / "concurrent.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
            return small_union(root)
        LR.build_exclusions = union_err
        refusals["a source of the freshness union errs"] = raises_stop(fz_(), "could not be read")
        LR.build_exclusions = union_writes
        refusals["data/ changes while the union is built"] = raises_stop(fz_(), "data/ changed while")
        (froot / "data" / "concurrent.jsonl").unlink()
        LR.build_exclusions = small_union
        (froot / (LR.PREREG_REL + ".partial")).write_text("half\n", encoding="utf-8")
        refusals["an interrupted freeze"] = raises_stop(fz_(), "interrupted before its renames")
        (froot / (LR.PREREG_REL + ".partial")).unlink()
        (froot / LR.DRAW_REL).write_text("x\n", encoding="utf-8")
        refusals["a draw without a freeze"] = raises_stop(fz_(), "draw exists without a freeze")
        (froot / LR.DRAW_REL).unlink()
        nothing = not any((froot / r).exists() or LR._partial(froot / r).exists()
                          for r in (LR.DRAW_REL, LR.DRAW_SOURCE_REL, LR.PREREG_REL, LR.FREEZE_REL))
        check(f"the freeze refuses ({len(refusals)} cases: " + "; ".join(refusals) + "); never persisted, nothing minted "
              "or renamed", all(v[0] and v[2] is False for v in refusals.values()) and nothing,
              {k: v[1][:90] for k, v in refusals.items() if not (v[0] and v[2] is False)})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rec = LR.freeze(appr, cos, TEST_SALT, froot)
        printed = buf.getvalue()
        ok_v, errs, fz = LR.verify_freeze(froot)
        text = (froot / LR.PREREG_REL).read_text(encoding="utf-8")
        src = json.loads((froot / LR.DRAW_SOURCE_REL).read_text(encoding="utf-8"))
        insts = P.load_set(froot / LR.DRAW_REL)
        check("the freeze fills the REAL draft (status line with approval and co-sign, section 10 with the draw seed, the "
              "salt commitment, the data/ snapshot and the seal), writes PREREG_LB_FREEZE.json, and verifies",
              ok_v and text.count("**Status:** FROZEN") == 1 and cos in text and "## 10. Freeze record" in text
              and rec["draw"]["file_sha256"] in text and f"`{LR.DRAW_REL}`" in text and commit in text
              and rec["bwm"]["draft_sha256_at_freeze"] == LR.BWM_DRAFT_SHA and fz["draft_sha256"] == dsha, errs)
        sr = src["seed_rule"]
        re_ex = small_union(froot)
        dig = LR.union_digest(re_ex)
        check("the draw's seed follows the recorded rule (tag, draft sha256, union digest, BWM's revealed salt); another "
              "salt gives another seed (RCWM cannot compute the draw before BWM reveals the salt); not a spent seed; the "
              "draw re-mints byte for byte from the rule; 1 per d* (the patched size); fresh; the data/ snapshot recorded",
              sr["seed"] == LR.seed_from_rule(dsha, dig, TEST_SALT) and sr["seed"] != LR.seed_from_rule(dsha, dig, TEST_SALT + "x")
              and sr["seed"] not in LR.SPENT_SEEDS and sr["bwm_commit"] == commit
              and LR.mint_draw(dsha, re_ex, TEST_SALT)[2] == (froot / LR.DRAW_REL).read_text(encoding="utf-8")
              and sorted(i["opt_depth"] for i in insts) == list(LR.DEPTHS) and src["overlap"]["payload_overlap"] == 0
              and src["data_snapshot"]["files"] == 1, sr["seed"])
        check("the freeze never prints an instance (ids, init, goal or vocabulary): only the seal and the count",
              not any(i["id"] in printed or json.dumps(list(i["init"])) in printed for i in insts) and rec["draw"]["file_sha256"] in printed,
              printed.strip()[-160:])
        check("a second freeze refuses", raises_stop(fz_(), "already frozen")[0])
        half = tmp / "halfroot"
        half.mkdir()
        shutil.copy2(froot / LR.PREREG_REL, half / LR.PREREG_REL)
        shutil.copy2(froot / LR.FREEZE_REL, half / (LR.FREEZE_REL + ".partial"))
        ok_h, errs_h, _ = LR.verify_freeze(half)
        check("an interrupted last rename is diagnosed by --verify with the recovery step", not ok_h and "last rename" in errs_h[0], errs_h[:1])
        fzp = froot / LR.FREEZE_REL
        fz_bytes = fzp.read_bytes()
        fzj = json.loads(fz_bytes)
        fzj["bwm_cosign"] = fzj["bwm_cosign"].replace(commit, LR.salt_commit("another salt 0123456789"))
        fzp.write_text(json.dumps(fzj, indent=1) + "\n", encoding="utf-8")
        ok_t, errs_t, _ = LR.verify_freeze(froot)
        fzp.write_bytes(fz_bytes)
        check("a freeze record whose co-sign carries another salt commitment fails verification",
              not ok_t and any("salt does not match" in e for e in errs_t), errs_t[:2])
        hook = froot / LR.HOOK_INSTALLED_REL
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_bytes((froot / LR.HOOK_DRAFT_REL).read_bytes() + b"# edited after the freeze\n")
        rc = LR.run_main(cfg_r(), make_backend(FakeRuntime(tmp), "cuda"))
        check("an installed 70_lbeam.sh that is not the frozen draft byte for byte refuses the run (exit 3, not persisted, "
              "nothing written)", rc == 3 and not (froot / "runs" / "lbeam").exists(), rc)
        shutil.copy2(froot / LR.HOOK_DRAFT_REL, hook)
        with GenPatch(stub):
            rcs = complete(LR, cfg_r(), device="cuda")
        pr = json.loads((froot / "runs" / "lbeam" / "provenance.json").read_text(encoding="utf-8"))
        check("a run under the (fake) freeze completes (5: C1 pending, then 0); provenance marks it registered under that "
              "freeze and draw", rcs == [5, 0] and pr["registered"] is True and pr["freeze_prereg_sha256"] == rec["prereg_sha256"]
              and pr["draw_sha256"] == rec["draw"]["file_sha256"], rcs)
        res = LA.run_registered(root=froot)
        outd = froot / "runs" / "lbeam" / "analysis"
        check("the registered analysis accepts it (freeze verified, every registered provenance check) and writes "
              "PREREG_LB_RESULTS.{json,md} as one create-only set with a registered headline and the instrument record",
              all((outd / n_).is_file() for n_ in LA.OUT_NAMES) and res["wording"]["headline"] in LA.HEADLINE.values()
              and not list(outd.glob("*.partial")) and res["instrument"]["bwm_modules_drift"] == []
              and "## Instrument and deviations" in (outd / LA.OUT_NAMES[1]).read_text(encoding="utf-8"), res["wording"]["headline"])
        check("a second registered analysis refuses (create-only)", _refuses(lambda: LA.run_registered(root=froot)))
        check("the registered analysis refuses a run made by a non-GPU backend (device label)", _registered_refuses_cpu(froot, tmp, LR, LA))
        tgt = froot / "scripts" / "lbeam_analysis.py"
        old = sha(tgt)
        tgt.write_bytes(tgt.read_bytes() + b"\n# changed\n")
        ok_c, errs_c, _ = LR.verify_freeze(froot)
        shutil.move(str(outd), str(tmp / "analysis_moved"))
        check("a changed frozen input fails verification, and the registered analysis then refuses",
              not ok_c and any("lbeam_analysis.py" in e for e in errs_c) and _refuses(lambda: LA.run_registered(root=froot)), errs_c[:2])
        am = {"file": "scripts/lbeam_analysis.py", "old_sha256": old, "new_sha256": sha(tgt), "reason": "test"}
        (froot / "PREREG_LB_AMENDMENT_1.json").write_text(json.dumps(am), encoding="utf-8")
        ok_a0, errs_a0, _ = LR.verify_freeze(froot)
        (froot / "PREREG_LB_AMENDMENT_1.json").write_text(json.dumps({**am, "approved_by": "operator (test)"}), encoding="utf-8")
        ok_a, errs_a, _ = LR.verify_freeze(froot)
        check("an amendment without 'approved_by' does not verify; a logged amendment (old -> new hash chain, reason, "
              "approved_by) restores verification", not ok_a0 and any("approved_by" in e for e in errs_a0) and ok_a, errs_a[:2])
        (froot / "PREREG_LB_AMENDMENT_1.json").unlink()
        tgt.write_bytes(tgt.read_bytes()[: -len(b"\n# changed\n")])
        dr = froot / LR.DRAW_REL
        keep_bytes = dr.read_bytes()
        dr.write_bytes(keep_bytes.replace(b'"attempt":', b'"attempt" :', 1))
        ok_d, errs_d, _ = LR.verify_freeze(froot)
        cz = cfg_r()
        cz.out = tmp / "runs" / "z"
        ok_z, msg_z, _ = raises_stop(lambda: LR.run(cz, make_backend(FakeRuntime(tmp), "cuda")), "refusing to run")
        dr.write_bytes(keep_bytes)
        check("a tampered draw fails verification, and run() then refuses", not ok_d and any("LB_CONFIRM_DRAW" in e for e in errs_d) and ok_z, errs_d[:2])
        ok_f, _, _ = LR.verify_freeze(froot)
        check("restoring the draw restores verification", ok_f)
        bwm_copy = tmp / "bwm_copy"
        (bwm_copy / "bwmn").mkdir(parents=True)
        (bwm_copy / "research").mkdir(parents=True)
        (bwm_copy / LR.BWM_PLANNER_REL).write_bytes((saved["BWM_ROOT"] / LR.BWM_PLANNER_REL).read_bytes()
                                                    + b"\n# tampered (freeze test copy)\n")
        shutil.copy2(saved["BWM_ROOT"] / LR.BWM_DRAFT_REL, bwm_copy / LR.BWM_DRAFT_REL)
        LR.BWM_ROOT = bwm_copy
        try:
            ok_b, errs_b, _ = LR.verify_freeze(froot)
            ok_i, msg_i, _ = raises_stop(LR.import_plan_beam, "differs from the pinned sha256")
            cb_ = cfg_r()
            cb_.out = tmp / "runs" / "bwm"
            ok_r, msg_r, _ = raises_stop(lambda: LR.run(cb_, make_backend(FakeRuntime(tmp), "cuda")), "refusing to run")
            m_a = _refuse_msg(lambda: LA.run_registered(root=froot))
        finally:
            LR.BWM_ROOT = saved["BWM_ROOT"]
            sys.path[:] = sys_path0
        check("a changed BWM planner (a patched BWM_ROOT copy) fails verification; import_plan_beam, the job and the "
              "registered analysis refuse", not ok_b and any("blocks_planner.py differs" in e for e in errs_b) and ok_i
              and ok_r and m_a is not None and "does not verify" in m_a, (errs_b[:1], msg_i[:80], (m_a or "")[:80]))
        check("restoring BWM_ROOT restores verification", LR.verify_freeze(froot)[0])
    finally:
        for k, v in saved.items():
            setattr(LR, k, v)
        sys.path[:] = sys_path0
        NUM["peak_rss_mb"] = peak_mb()
        if keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    NUM["seconds"] = round(time.time() - t0, 1)
    return {}


def _refuses(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def _registered_refuses_cpu(froot: Path, tmp: Path, LR, LA) -> bool:
    d = tmp / "cpu_copy"
    shutil.copytree(froot / "runs" / "lbeam", d)
    pv = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    for t in pv["turns"]:
        t["device"] = "cpu-tiny"
    (d / "provenance.json").write_text(json.dumps(pv), encoding="utf-8")
    done = json.loads((d / "DONE.json").read_text(encoding="utf-8"))
    done["provenance_sha256"] = sha(d / "provenance.json")
    (d / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    shutil.rmtree(d / "analysis", ignore_errors=True)
    try:
        LA.Inputs(d, froot / LR.DRAW_REL, registered=True, root=froot)
    except SystemExit as e:
        return "not generated by the registered GPU backend" in str(e)
    return False


# ================================================================ part: power (model-free sizing; torch only for the import)
SIM_SCENARIOS = {
    "optimistic: T2 = SIM_T2 (PREREG-2 four-step exact 0.698 -> 0.914 per step), OFF = SIM_OFF": ("SIM_T2", None),
    "central: T2 = SIM_T2_P3 (PREREG-3 posted accuracy 0.84), OFF = SIM_OFF": ("SIM_T2_P3", None),
    "pessimistic: T2 posted errors 0.25 and void applied 0.05, OFF = SIM_OFF": (None, {"p_void_applied": 0.05, "p_wrong": 0.25}),
}
SIM_SEEDS = (0, 1, 2, 3, 4)


def part_power() -> dict:
    import_torch_cpu()
    import numpy as np
    import lbeam_run as LR
    import lbeam_problems as P
    import lbeam_proposer as PR
    import lbeam_f as LF
    torch_mod = sys.modules.pop("torch")
    try:
        import lbeam_analysis as LA
    finally:
        sys.modules["torch"] = torch_mod
    BP = LR.import_plan_beam()
    t0 = time.time()
    dev = P.load_set(ROOT / LR.DEV_REL)
    PI, own = P.planner_instances(dev), P.owner_of_action(dev)
    greedy = np.array([P.greedy_solvable(i) for i in dev], bool)
    depth = np.array([i["opt_depth"] for i in dev])
    prop = PR.EnumerateProposer()

    def beam(F, cname="W2E12"):
        W, E = LR.CONFIGS[cname]
        return BP.plan_beam(PI, prop, F, eval_seed=LR.EVAL_SEED, **PR.beam_kwargs(prop, W, E))

    def succ(eps):
        by = {i["id"]: i for i in dev}
        return np.array([P.execute(by[e["id"]], e["plan"])["success"] for e in eps], bool)

    orc_inner = LF.LedgerF("ORACLE")
    orc = succ(beam(LF.guarded(orc_inner, dev)))
    orc4 = succ(beam(LF.guarded(LF.LedgerF("ORACLE"), dev), "W4E24"))
    check("ORACLE enum W2 E12 on dev2_primary reproduces the kit's sizing (0.850; needs-lookahead 0.581)",
          abs(orc.mean() - 0.85) < 1e-9 and abs(orc[~greedy].mean() - 25 / 43) < 1e-9, (orc.mean(), orc[~greedy].mean()))
    sizing = {"G1_T2": [], "G1_OFF": [], "G2_T2": [], "G2_OFF": [], "G3_T2": [], "G4_proxy": []}
    refused = {"T2_P3": [], "OFF": [], "T2": []}
    outcomes = {}
    for s in SIM_SEEDS:
        # T2-like (central preset) with the stage order and the shared per-arm memo of the runner
        inner = LF.SimulatedF("SIM_T2_P3", seed=s)
        F = LF.guarded(inner, dev)
        t2c = succ(beam(F))
        q1 = inner.model_queries
        refused["T2_P3"].append(F.oor_key_share() or 0.0)
        inner.reset_accounting()
        PR.select_fixed(dev, LF.guarded(inner, dev), eval_seed=LR.EVAL_SEED)
        q2 = inner.model_queries - q1
        inner.reset_accounting()
        beam(LF.guarded(inner, dev), "W4E24")
        q3 = inner.model_queries - q1 - q2
        sizing["G1_T2"].append(q1 / len(dev))
        sizing["G2_T2"].append(q2 / len(dev))
        sizing["G3_T2"].append(q3 / len(dev))
        inner_o = LF.SimulatedF("SIM_OFF", seed=s)
        Fo = LF.guarded(inner_o, dev)
        offs = succ(beam(Fo))
        o1 = inner_o.model_queries
        refused["OFF"].append(Fo.oor_key_share() or 0.0)
        inner_o.reset_accounting()
        PR.select_fixed(dev, LF.guarded(inner_o, dev), eval_seed=LR.EVAL_SEED)
        sizing["G1_OFF"].append(o1 / len(dev))
        sizing["G2_OFF"].append((inner_o.model_queries - o1) / len(dev))
        sizing["G4_proxy"].append(o1 / len(dev))
        inner_t = LF.SimulatedF("SIM_T2", seed=s)
        Ft = LF.guarded(inner_t, dev)
        t2o = succ(beam(Ft))
        refused["T2"].append(Ft.oor_key_share() or 0.0)
        inner_p = LF.SimulatedF(None, seed=s, **SIM_SCENARIOS["pessimistic: T2 posted errors 0.25 and void applied 0.05, OFF = SIM_OFF"][1])
        t2p = succ(beam(LF.guarded(inner_p, dev)))
        # SimulatedF draws its errors from sha256(seed, key) uniforms shared by every preset, so at one seed T2's errors
        # are nested in OFF's: a simulation artifact. OFF from seed + 100 decorrelates the pair.
        offd = succ(beam(LF.guarded(LF.SimulatedF("SIM_OFF", seed=s + 100), dev)))
        outcomes[s] = {"SIM_T2": t2o, "SIM_T2_P3": t2c, "PESS": t2p, "OFF": offs, "OFF_DEC": offd}
        print(f"  seed {s}: T2 {t2o.mean():.3f} / T2_P3 {t2c.mean():.3f} / pess {t2p.mean():.3f}; OFF {offs.mean():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    size = {k: round(float(np.mean(v)), 2) for k, v in sizing.items()}
    NUM["sizing_per_instance_new_keys"] = size
    NUM["refused_share"] = {k: {"mean": round(float(np.mean(v)), 5), "max": round(float(np.max(v)), 5)} for k, v in refused.items()}
    reg = {"G1_T2": LR.SIZING["G1"]["T2"], "G1_OFF": LR.SIZING["G1"]["OFF"], "G2_T2": LR.SIZING["G2"]["T2"],
           "G2_OFF": LR.SIZING["G2"]["OFF"], "G3_T2": LR.SIZING["G3"]["T2"], "G4_proxy": LR.SIZING["G4"]["T2@heldout_combined"]}
    check("the runner's registered SIZING constants equal the measured sizing (to 0.01 keys per instance)",
          all(abs(size[k] - reg[k]) <= 0.011 for k in size), {"measured": size, "registered": reg})
    dev_rates = {}
    for key, col in (("SIM_T2", "SIM_T2"), ("SIM_T2_P3", "SIM_T2_P3"), ("PESS", "PESS"), ("OFF", "OFF")):
        a = np.array([outcomes[s][col] for s in SIM_SEEDS])
        dev_rates[key] = {"all": round(float(a.mean()), 4), "needs_lookahead": round(float(a[:, ~greedy].mean()), 4),
                          "greedy": round(float(a[:, greedy].mean()), 4)}
    dev_rates["ORACLE"] = {"all": round(float(orc.mean()), 4), "needs_lookahead": round(float(orc[~greedy].mean()), 4),
                           "greedy": round(float(orc[greedy].mean()), 4), "W4E24": round(float(orc4.mean()), 4)}
    for key in ("SIM_T2", "SIM_T2_P3", "PESS"):
        a = np.array([outcomes[s][key] for s in SIM_SEEDS])
        dev_rates[key]["ratio_all"] = round(float(a.mean() / orc.mean()), 4)
        dev_rates[key]["ratio_needs_lookahead"] = round(float(a[:, ~greedy].mean() / orc[~greedy].mean()), 4)
        dev_rates[key]["needs_lookahead_by_seed"] = [round(float(outcomes[s][key][~greedy].mean()), 4) for s in SIM_SEEDS]
    a = np.array([outcomes[s]["OFF_DEC"] for s in SIM_SEEDS])
    dev_rates["OFF_DEC"] = {"all": round(float(a.mean()), 4), "needs_lookahead": round(float(a[:, ~greedy].mean()), 4),
                            "greedy": round(float(a[:, greedy].mean()), 4)}
    NUM["dev_success_rates"] = dev_rates
    power, power_dec = {}, {}
    R_PER_SEED, B = 400, 2000
    for off_col, store in (("OFF", power), ("OFF_DEC", power_dec)):
        for name, (preset, _) in SIM_SCENARIOS.items():
            col = preset or "PESS"
            parts = []
            for s in SIM_SEEDS:
                trip = np.stack([outcomes[s][col], outcomes[s][off_col], orc], 1).astype(np.int8)
                parts.append(LA.power_resample(trip, greedy, depth, R_PER_SEED, B, seed=1000 + s, n_per_depth=36))
            agg = {"R": R_PER_SEED * len(SIM_SEEDS), "B_bootstrap": B, "n_per_depth": 36}
            for k in ("P_cp1_pass", "P_cp2_pass", "P_near_oracle", "P_near_bar_met"):
                agg[k] = round(float(np.mean([p[k] for p in parts])), 4)
            agg["P_cp2_pass_by_seed"] = [p["P_cp2_pass"] for p in parts]
            brs = {}
            for p in parts:
                for b_, v in p["P_branch"].items():
                    brs[b_] = brs.get(b_, 0.0) + v / len(parts)
            agg["P_branch"] = {k: round(v, 4) for k, v in sorted(brs.items())}
            agg["n_needs_lookahead_mean"] = round(float(np.mean([p["n_needs_lookahead"]["mean"] for p in parts])), 1)
            agg["delta1_mean"] = round(float(np.mean([p["delta1_mean"] for p in parts])), 4)
            agg["delta2_mean"] = round(float(np.mean([p["delta2_mean"] for p in parts])), 4)
            store[name] = agg
            print(f"  power [{off_col} {name[:12]}]: CP1 {agg['P_cp1_pass']}, CP2 {agg['P_cp2_pass']} "
                  f"(by seed {agg['P_cp2_pass_by_seed']}), near-oracle {agg['P_near_oracle']}", flush=True)
    NUM["power_resampled"] = power
    NUM["power_resampled_off_decorrelated"] = power_dec
    grid = {"cp1_n180": {}, "cp2_n64": {}, "cp2_n55": {}, "cp2_n75": {}}
    for pt in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for po in (0.10, 0.16, 0.25, 0.35, 0.45):
            grid["cp1_n180"][f"T2 {pt:.2f} / OFF {po:.2f}"] = [round(LA.exact_pass_prob(180, pt, po, a), 4) for a in (0.005, 0.01)]
    for n_ in (64, 55, 75):
        for pt in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            for po in (0.02, 0.05, 0.10, 0.20):
                grid[f"cp2_n{n_}"][f"T2 {pt:.2f} / OFF {po:.2f}"] = [round(LA.exact_pass_prob(n_, pt, po, a), 4) for a in (0.005, 0.01)]
    NUM["power_exact_independence_grid"] = {"note": "P(delta >= +0.20 and exact McNemar p < alpha) for [alpha 0.005 (a lower "
                                                    "bound on the Holm pass), alpha 0.01 (an upper bound)]; T2 and OFF "
                                                    "outcomes independent within an instance", **grid}
    check("power computed: resampled scenarios (as simulated, and with OFF decorrelated) and the exact grid are complete",
          len(power) == 3 and len(power_dec) == 3 and all(len(v) > 0 for v in grid.values()))
    cells = [v[1] for k, v in grid["cp1_n180"].items()
             if round(float(k.split()[1]) - float(k.split()[4]), 2) >= 0.30]
    NUM["cp1_min_power_at_true_gain_0.30_or_more"] = round(min(cells), 4)
    rec = {"ok": all(r["ok"] for r in RESULTS), "record": "LB_POWER", "sizing": {"per_instance_new_keys": size,
                                                                                 "refused_share": NUM["refused_share"],
                                                                                 "method": "dev2_primary (120), SimulatedF seeds 0-4, the runner's stage order and shared per-arm memo"},
           "dev_success_rates": dev_rates, "power_resampled": power, "power_resampled_off_decorrelated": power_dec,
           "cp1_min_power_at_true_gain_0.30_or_more": NUM["cp1_min_power_at_true_gain_0.30_or_more"],
           "power_exact_independence_grid": NUM["power_exact_independence_grid"],
           "method": ("Each scenario pairs per-instance success of a T2-like and an OFF-like SimulatedF with ORACLE on the "
                      "120 dev instances (enum W2 E12, guarded, registered max_depth); a confirmation-sized set (36 per d*) "
                      "is resampled with replacement within d* and the registered decide() is applied (Holm across the two "
                      "co-primaries; ratio bootstrap with B = 2,000, computed only when the point ratio is >= 0.80). 400 "
                      "replicates per seed x 5 seeds. 'power_resampled' pairs T2 and OFF at the SAME SimulatedF seed: "
                      "SimulatedF draws errors from sha256(seed, key) uniforms shared across presets, so T2's errors are "
                      "nested in OFF's (a simulation artifact, not an estimate of the real correlation); "
                      "'power_resampled_off_decorrelated' takes OFF from seed + 100."),
           "code_sha256": {r: sha(ROOT / r) for r in LR.RUN_CODE}, "seconds": round(time.time() - t0, 1)}
    if not (ROOT / LR.FREEZE_REL).exists():
        with open(DOCS / "LB_POWER.json", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {DOCS / 'LB_POWER.json'}")
    return {}


# ================================================================ part: rehearsal (full-size mint path, a different seed)
def part_rehearsal() -> dict:
    """The REAL freeze-time mint path at full size (N = 180, the real freshness union with the name-skipping harvest
    and the spent dev sets), under a DIFFERENT seed tag, so the rehearsal draw can never be the confirmation draw. The
    rehearsal draw is held in memory only: never written, never printed; only counts are recorded."""
    import lbeam_run as LR
    import lbeam_problems as P
    t0 = time.time()
    real_tag = LR.SEED_TAG
    LR.SEED_TAG = "PREREG-LB REHEARSAL ONLY - never the confirmation draw"
    salt = "rehearsal-salt-NOT-BWM-0000"          # BWM's real salt is unknown to RCWM until the freeze
    try:
        snap0 = LR.data_snapshot(ROOT)
        excl = LR.build_exclusions(ROOT, log=lambda *a: None)
        snap1 = LR.data_snapshot(ROOT)
        t_union = time.time() - t0
        draft_sha = sha(ROOT / LR.DRAFT_REL)
        dig = LR.union_digest(excl)
        hist = excl["sources"].get("historical", {})
        check("the rebuilt union covers the historical harvest (HSKILL_CONFIRM_V2* skipped by name, never opened), the "
              "pinned sources, PREREG-3's bank, the other data files and the three spent dev sets",
              hist.get("payloads", 0) > 0 and all(r in excl["sources"] for r in LR.SPENT_DEV_SETS)
              and "bank_p3_oor.jsonl" in excl["sources"] and len(excl["sources"]) >= 30
              and any(k.startswith("data/t2br_superseded/") for k in excl["sources"]),
              {"sources": len(excl["sources"]), "payloads": len(excl["pay"]), "operative": len(excl["ops"]),
               "skipped_by_name": hist.get("skipped_by_name_never_opened")})
        check("no source of the rebuilt union errs (the freeze refuses one that does), and the data/ snapshot (every "
              "data/**/*.json[l]) is recorded before and after the build (the freeze refuses a change)",
              not LR.union_errors(excl) and len(snap0) > 0,
              {"errors": LR.union_errors(excl), "data_files": len(snap0), "changed_during_build": snap0 != snap1})
        t1 = time.time()
        insts, rec, text = LR.mint_draw(draft_sha, excl, salt)
        t_mint = time.time() - t1
        seed_reh = rec["seed_rule"]["seed"]
        seed_real_tag = int(hashlib.sha256(f"{real_tag}|{draft_sha}|{dig}|{salt}".encode()).hexdigest()[:12], 16)
        check("the rehearsal seed differs from what the real tag would give with the same stand-in salt, and the real "
              "seed needs BWM's salt (unknown before the freeze): the rehearsal draw is not the confirmation draw; not a "
              "spent seed", seed_reh != seed_real_tag and seed_reh not in LR.SPENT_SEEDS)
        check("full-size mint: 180 instances, 36 per d*, 0 payload / 0 operative overlaps with the union, verified, "
              "regenerated byte for byte from (seed, spec, recorded rejections)",
              len(insts) == 180 and rec["per_stratum"] == {f"in-d{d}": 36 for d in LR.DEPTHS}
              and rec["overlap"]["payload_overlap"] == 0 and rec["overlap"]["operative_overlap"] == 0
              and rec["verify_ok"] and rec["regenerated_equal"], {"per_stratum": rec["per_stratum"], "overlap": rec["overlap"]})
        check("the SOURCE record carries no instance content (no init / goal / vocabulary / plan fields)",
              not any(k in json.dumps(rec) for k in ('"init"', '"goal"', '"vocab"', '"opt_plan"')))
        NUM["rehearsal"] = {"union_seconds": round(t_union, 1), "mint_seconds": round(t_mint, 1),
                            "union_payloads": len(excl["pay"]), "union_operative": len(excl["ops"]),
                            "union_sources": len(excl["sources"]), "fresh_rejections": len(rec["fresh_rejections"]),
                            "data_files_snapshot": len(snap0), "data_changed_during_build": snap0 != snap1,
                            "closure_and_other_rejections": {k: sum(v.values()) for k, v in rec["rejected"].items()},
                            "note": "rehearsal tag; the draw itself was discarded unprinted"}
        del insts, text, excl
    finally:
        LR.SEED_TAG = real_tag
    return {}


# ================================================================ part: hook (the queue hook draft under bash, stub step)
HOOK_SIM = r'''set -u
HOOK="$1"
run_case() {
  local name="$1" frozen="$2" stopped="$3" done="$4" results="$5" rcs="$6" stopkind="$7" passes="$8"
  rm -rf w; mkdir -p w/runs/lbeam/analysis; cd w
  [ "$frozen" = 1 ] && touch PREREG_LB_FREEZE.json
  [ "$stopped" = eligible ] && printf '{\n "reason": "x",\n "verdict_eligible": true\n}\n' > runs/lbeam/STOPPED.json
  [ "$stopped" = ineligible ] && printf '{\n "reason": "x",\n "verdict_eligible": false\n}\n' > runs/lbeam/STOPPED.json
  [ "$done" = 1 ] && touch runs/lbeam/DONE.json
  [ "$results" = 1 ] && touch runs/lbeam/analysis/PREREG_LB_RESULTS.json
  CALLS=""; RCS=($rcs); N=0
  step() {
    if [ "$1" = scripts/lbeam_run.py ]; then
      local rc=${RCS[$N]:-0}; N=$((N+1)); CALLS="$CALLS run:$rc"
      [ "$rc" = 0 ] && touch runs/lbeam/DONE.json
      if [ "$rc" = 3 ]; then printf '{\n "verdict_eligible": %s\n}\n' "$stopkind" > runs/lbeam/STOPPED.json; fi
      return $rc
    fi
    CALLS="$CALLS analysis"; touch runs/lbeam/analysis/PREREG_LB_RESULTS.json; return 0
  }
  local i
  for i in $(seq 1 "$passes"); do source "$HOOK"; CALLS="$CALLS |"; done
  echo "CASE $name =>$CALLS"
  cd ..
}
'''
# (name, frozen, STOPPED.json before, DONE.json before, results before, exit codes of the run calls, kind of a stop the
#  run writes on exit 3, chain passes) -> the calls made, pass by pass ('|' ends a pass)
HOOK_CASES = [
    ("not frozen", 1 - 1, "none", 0, 0, "0", "false", 1, "|"),
    ("STOPPED.json (not verdict-eligible) exists", 1, "ineligible", 0, 0, "0", "false", 1, "|"),
    ("STOPPED.json (verdict-eligible) exists: the analysis once", 1, "eligible", 0, 0, "0", "false", 2, "analysis | |"),
    ("STOPPED.json (verdict-eligible) and the results exist", 1, "eligible", 0, 1, "0", "false", 1, "|"),
    ("exit 0: the analysis follows; a second pass makes no call (idempotent)", 1, "none", 0, 0, "0", "false", 2,
     "run:0 analysis | |"),
    ("exit 3 (not verdict-eligible): never retried, no analysis", 1, "none", 0, 0, "3", "false", 2, "run:3 | |"),
    ("exit 3 (verdict-eligible): never retried; the analysis once", 1, "none", 0, 0, "3", "true", 2, "run:3 analysis | |"),
    ("exit 5 then 5: exactly one resume call per pass; the next pass resumes again", 1, "none", 0, 0, "5 5 0", "false", 2,
     "run:5 run:5 | run:0 analysis |"),
    ("exit 5 then 0: one resume call, then the analysis", 1, "none", 0, 0, "5 0", "false", 1, "run:5 run:0 analysis |"),
    ("crash then crash: exactly one retry", 1, "none", 0, 0, "1 1", "false", 1, "run:1 run:1 |"),
    ("crash then 3: one retry, no analysis", 1, "none", 0, 0, "1 3", "false", 1, "run:1 run:3 |"),
    ("crash then 0: one retry, then the analysis", 1, "none", 0, 0, "1 0", "false", 1, "run:1 run:0 analysis |"),
    ("exit 5 then a verdict-eligible 3: the analysis once", 1, "none", 0, 0, "5 3", "true", 1, "run:5 run:3 analysis |"),
    ("DONE.json exists, no results: the analysis only", 1, "none", 1, 0, "0", "false", 1, "analysis |"),
    ("DONE.json and the results exist: nothing", 1, "none", 1, 1, "0", "false", 1, "|"),
]


def part_hook() -> dict:
    import subprocess
    import lbeam_run as LR
    hook = ROOT / LR.HOOK_DRAFT_REL
    txt = hook.read_bytes()
    bash = Path(r"C:/Program Files/Git/bin/bash.exe")
    if not bash.is_file():
        bash = Path(shutil.which("bash") or "bash")
    tmp = Path(tempfile.mkdtemp(prefix="lbeam_hook_", dir=SCRATCH if SCRATCH.exists() else None))
    try:
        script = tmp / "sim.sh"
        lines = [HOOK_SIM]
        for name, fr, st, dn, rs, rcs, sk, passes, _ in HOOK_CASES:
            lines.append(f"run_case '{name}' {fr} {st} {dn} {rs} '{rcs}' {sk} {passes}\n")
        with open(script, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(lines))
        r = subprocess.run([str(bash), str(script).replace("\\", "/"), str(hook).replace("\\", "/")], cwd=str(tmp),
                           capture_output=True, text=True, timeout=120)
        got = {}
        for ln in r.stdout.splitlines():
            if ln.startswith("CASE ") and " =>" in ln:
                k, v = ln[5:].split(" =>", 1)
                got[k] = " ".join(v.split())
        check("the simulation ran under bash (exit 0, every case reported)", r.returncode == 0 and len(got) == len(HOOK_CASES),
              (r.returncode, r.stderr[-300:]))
        for name, *_rest, want in HOOK_CASES:
            check(f"hook: {name}: calls '{want}'", got.get(name) == " ".join(want.split()), got.get(name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("the hook draft is LF-only, names the installed path 70_lbeam.sh, and is NOT installed",
          b"\r" not in txt and b"scripts/queue_hooks/70_lbeam.sh" in txt and not (ROOT / LR.HOOK_INSTALLED_REL).exists())
    NUM["hook_draft_sha256"] = sha(hook)
    return {}


# ================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("plan", "analysis", "power", "tiny", "freeze", "rehearsal", "hook"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    nice()
    t0 = time.time()
    try:
        {"plan": part_plan, "analysis": part_analysis, "power": part_power, "rehearsal": part_rehearsal,
         "tiny": lambda: part_tiny(a.keep), "freeze": lambda: part_freeze(a.keep), "hook": part_hook}[a.part]()
    except Exception as e:                                  # a crash is a failed check, recorded
        import traceback
        traceback.print_exc()
        check(f"part {a.part} ran to completion", False, f"{type(e).__name__}: {e}")
    import lbeam_run as LR
    rec = {"part": a.part, "ok": bool(RESULTS) and all(r["ok"] for r in RESULTS), "n_checks": len(RESULTS),
           "n_failed": sum(not r["ok"] for r in RESULTS), "checks": RESULTS, "numbers": NUM,
           "code_sha256": {r: sha(ROOT / r) for r in LR.RUN_CODE}, "bwm_blocks_planner_sha256": sha(LR.BWM_ROOT / LR.BWM_PLANNER_REL),
           "draft_sha256": sha(ROOT / LR.DRAFT_REL), "hook_draft_sha256": sha(ROOT / LR.HOOK_DRAFT_REL),
           "torch_imported": "torch" in sys.modules, "peak_rss_mb": peak_mb(), "seconds": round(time.time() - t0, 1),
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "CPU only; tiny random Qwen3 + LoRA stand-ins; no real model, no GPU, no lease"}
    print(f"part {a.part}: {rec['n_checks'] - rec['n_failed']}/{rec['n_checks']} checks pass in {rec['seconds']}s; "
          f"peak RSS {rec['peak_rss_mb']} MB", flush=True)
    if not a.no_record:
        if (ROOT / LR.FREEZE_REL).exists():
            print("PREREG-LB is frozen: RUN_TEST records are frozen inputs and are not rewritten")
        else:
            with open(DOCS / f"RUN_TEST_{a.part.upper()}.json", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rec, indent=1, default=str) + "\n")
            print(f"wrote {DOCS / f'RUN_TEST_{a.part.upper()}.json'}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
