"""CPU tests of the T2b prerequisite screen kit (data audit, scorer, analysis rule, runner) with tiny random Qwen3
stand-ins. No real model is loaded (AutoModelForCausalLM.from_pretrained is disabled; the vendored trainer's load_core
returns the tiny model), no GPU is touched (CUDA_VISIBLE_DEVICES="" before torch, is_available/device_count patched),
no lease is taken (a fake runtime stands in for rcwmt.runtime under the runner's own RealBackend code). Threads 2. One
process. Runner outputs go to a temporary directory only.

Stand-ins: Qwen3 with 2 blocks, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary and the real
tokenizer (read-only). Training runs the REAL vendored trainer (train_generic.main -> lm.attach_lora r16 ->
lm.train_sft) for 2 steps at lr 0.05 (so the adapter visibly changes greedy outputs). Pacing-only deviations:
generation.breathe / throttle_list no-ops, Tensor.cuda the identity, a 12-token decode cap, BWM_DEVICE=cpu.

Checks
  1 frames: D-int messages == env.multi_step_messages; render_chat == tok.apply_chat_template for every arm (8-, 10-,
    12-bit, with and without surface variants); t2bp_chain('dint') == generation.chain('chain_m1') OFF and ON
  2 data audit: the written files pass; perturbed copies (composition, a broken minimal pair, a wrong gold, a filler off
    by one token, a reordered trace target) are refused
  3 surface diversity: rendered SD files pass the same audit, keep the canonical rows and targets, use only training
    variants, and actually vary the system text and field names
  4 scorer self-test (golds 100%, heuristics and shortcut policies at their floors, negatives rejected and classified
    format vs value, D-int parser == env.parse_state) and filler token equality under the HF tokenizer == the tokenizers
    library count on every training and dev row
  5 analysis dry run: every synthetic scenario, rule unit, close-call unit, R5 addendum unit and refusal gives the
    pre-stated result
  6 full tiny run (3 arms): exit 0, DONE.json, one lease 'rcwm-t1:t2b-prereq', resource wait before every load, train
    before eval per arm, anchor pass recorded, OFF != ON, analysis validators accept the outputs, a rerun is a no-op;
    an oracle-generator run on the whole dev bank; frozen-rule stops (a changed frozen input or a missing record refuses
    before the lease; code changed between turns refuses the resume; a completed run refuses other arms, e.g.
    --with-filler after DONE; the analysis' frozen check); a surface-diversity run with stage E3 (held-out surface)
  7 crash + resume: a crash inside training (partial train dir) and a crash inside E2 (stray .partial) are orphaned and
    redone; the lease is released by the crash; every output equals the uninterrupted run byte for byte
  8 deferral: nothing fits -> exit 5, no load, no training; an arm-level deferral (pair does not fit) and a stage-level
    deferral (E2 does not fit) resume on the next call (re-anchored) and finish with outputs identical to run 6
  9 stops: one-waiter rule (live RCWM marker / live RCWM holder refuse, not persisted; dead marker and live BWM holder
    pass); live run lock refuses, stale lock reclaimed; anchor failure (persisted; no evaluation output); fingerprint
    mismatch; runtime drift; a recorded output changed; a data file changed after recording
 10 --plan logic: the screen fits one turn at the planning rate, the filler arm needs a second, a short cap splits turns

  python -B scripts/t2bp_test.py [--keep] [--no-record]       -> docs/t2b/prereq/TINY_TEST.json
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # NOT "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["RCWM_DUTY_FORCE"] = "1"
os.environ["RCWM_DUTY_RATIO"] = "0"
os.environ["RCWM_BREATHE_S"] = "0"
os.environ["BWM_DEVICE"] = "cpu"
os.environ["BWM_DUTY_RATIO"] = "0"
os.environ["BWM_YIELD_S"] = "0"
sys.dont_write_bytecode = True

import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.synchronize = lambda *a, **k: None
torch.set_num_threads(2)
torch.Tensor.cuda = lambda self, *a, **k: self      # generation.generate_texts moves inputs with .cuda()

import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor_bwm"))
sys.path.insert(0, str(ROOT / "vendor_bwm" / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import transformers  # noqa: E402
from transformers import AutoTokenizer, LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from peft import PeftModel  # noqa: E402

from rcwmt import env, generation, runtime  # noqa: E402
from bwmn import lm  # noqa: E402
import t2bp_common as C  # noqa: E402
import t2bp_make_data as MD  # noqa: E402
import t2bp_audit as AU  # noqa: E402
import t2bp_run as R  # noqa: E402
import t2bp_analysis as A  # noqa: E402


def _no_real_models(*a, **k):
    raise AssertionError("a real model load was attempted inside the tiny test")


transformers.AutoModelForCausalLM.from_pretrained = _no_real_models
runtime.gpu_temp_c = lambda: None
generation.breathe = lambda: None
generation.throttle_list = lambda: LogitsProcessorList([])
MAXNEW = 12
generation.max_tokens = lambda fam: MAXNEW
lm.YIELD_S = 0.0
REAL_CHAIN = R.t2bp_chain
REAL_GENERATE = generation.generate_texts

KEEP = "--keep" in sys.argv
RECORD = "--no-record" not in sys.argv
CFG = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e6,
           tie_word_embeddings=True)
SEEDS = {"sparse": 2002, "supa": 1001}
STEPS, LR_TINY = 2, 0.05
RESULTS: list = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({str(detail)[:200]})" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def build(arm: str):
    cfg = Qwen3Config(**CFG)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(SEEDS[arm])
    m = Qwen3ForCausalLM(cfg)
    g = torch.Generator().manual_seed(SEEDS[arm])
    for _, p in m.named_parameters():
        if p.dim() == 2:
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return m.eval()


_TOK = []


def tok():
    if not _TOK:
        t = AutoTokenizer.from_pretrained(str(C.TOKENIZER_DIR), local_files_only=True)
        if t.pad_token_id is None:
            t.pad_token = "<|endoftext|>"
        _TOK.append(t)
    return _TOK[0]


def fake_load_core(name: str):
    assert name == "titan-sparse", name
    return build("sparse"), tok()


lm.load_core = fake_load_core


class FakeRuntime:
    """Stands in for rcwmt.runtime under RealBackend: tiny models, recorded calls, no real lock."""

    def __init__(self, tmp: Path):
        self.LOCK_PATH = tmp / "fake_gpu.lock"
        self.WAITING = {"rcwm": tmp / "fake_gpu.waiting.rcwm.json", "bwm": tmp / "fake_gpu.waiting.bwm.json"}
        self.calls: list = []
        self.fp_override: dict = {}
        self.runtime_override: list = []

    @contextmanager
    def gpu_lease(self, label, job):
        self.calls.append(("lease", label))
        self.LOCK_PATH.write_text(json.dumps({"label": label, "pid": os.getpid(), "created": time.time()}), encoding="utf-8")
        try:
            yield
        finally:
            self.LOCK_PATH.unlink()
            self.calls.append(("release", label))

    def wait_for_resources(self):
        self.calls.append(("wait",))

    def load_arm(self, arm):
        self.calls.append(("load", arm))
        return build(arm), {"param_fingerprint": self.fp_override.get(arm, R.FP[arm] + "-tiny-standin")}

    def tokenizer(self):
        return tok()

    def unload(self, _):
        gc.collect()
        self.calls.append(("unload",))

    def runtime_info(self):
        import peft  # noqa: F401
        real = {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu (tiny test)"}
        return {**real, **self.runtime_override.pop(0)} if self.runtime_override else real


class TinyBackend(R.RealBackend):
    device = "cpu-tiny"

    def __init__(self, rt):
        self.rt = rt
        self.train_calls: list = []

    def thermal_wait(self):
        pass

    def vram_allocated_mib(self):
        return 0.0

    def train(self, rows_path, out_dir, steps, seed, lr):
        self.train_calls.append(Path(out_dir).parent.name)
        self.rt.calls.append(("train", Path(out_dir).parent.name))
        return super().train(rows_path, out_dir, steps, seed, lr)


def raises_stop(fn, needle: str = "") -> tuple[bool, str, bool | None]:
    try:
        fn()
    except R.Stop as e:
        return needle in str(e), str(e)[:240], e.persist
    return False, "no Stop raised", None


def stage_files(out: Path, arms) -> list:
    return [R.rel_out(p, out) for a in arms for st in R.STAGES for p in R.stage_required(a, st, out)]


def tree_hashes(out: Path, rels) -> dict:
    return {r: sha(out / r) for r in rels}


def make_anchor_ref(tmp: Path, chains: list, name: str = "anchor_ref.jsonl") -> Path:
    m = build("sparse")
    rr = generation.chain(m, tok(), chains, "chain_m1")
    p = tmp / name
    p.write_text("".join(json.dumps(x) + "\n" for x in R.chain_rows(chains, rr)), encoding="utf-8")
    del m
    gc.collect()
    return p


def side_of(model) -> str:
    """The active side from PEFT's own LoRA layer state (oracle generator only); a bare base model is OFF."""
    from peft.tuners.lora import LoraLayer
    for mod in model.modules():
        if isinstance(mod, LoraLayer):
            return "off" if mod.disable_adapters else "on"
    return "off"


_CNT = []


def oracle_generate(model, tok_, message_lists, max_new_tokens, **kw):
    """ON: the stated rule, in the arm's own format (arm and width read from the prompt); OFF: copy the input state."""
    import re
    if not _CNT:
        _CNT.append(MD.counter())
    side = side_of(model)
    out = []
    for msgs in message_lists:
        sys_t = msgs[0]["content"]
        arm = ("bbits" if "binary digits" in sys_t else "strace" if "keys cmp, calc and after" in sys_t
               else "ffill" if "keys fill" in sys_t else "dint")
        w = int(re.search(r"(\d+)-bit", sys_t).group(1))
        u = json.loads(msgs[1]["content"])
        b, ev = u["before"], dict(u["events"][0])
        if arm == "bbits":
            b = int(b, 2)
            for k in ("add", "delete", "pre_on", "pre_off"):
                ev[k] = int(ev[k], 2)
        out.append(C.target_text(arm, b, ev, w, _CNT[0]) if side == "on" else C.render_prediction(arm, b, b, ev, w))
    return out


def _peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)


def main() -> int:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="t2bp_tiny_"))
    print(f"tiny test workspace: {tmp}", flush=True)
    dev_rows, parents = C.load_dev()
    dchains = C.dev_chains(parents)
    tr = {a: C.load_jsonl(C.TRAIN_FILES[a]) for a in C.ARMS}
    try:
        # ------------------------------------------------------------ 1 frames
        print("1 frames", flush=True)
        w8 = [r for r in dev_rows + tr["dint"] if r["width"] == C.WIDTH_ID]
        check(f"D-int messages == env.multi_step_messages on every 8-bit dev and training row ({len(w8)})",
              all(C.messages("dint", r["before"], r["event"], r["width"]) == env.multi_step_messages(C.FAM, r["before"], [r["event"]])
                  for r in w8))
        ood = [r for r in dev_rows if r["width"] != C.WIDTH_ID]
        check("OOD D-int prompts differ from T2's frame only in the width word ('a 10-bit' / 'a 12-bit')",
              all(C.messages("dint", r["before"], r["event"], r["width"])[0]["content"] ==
                  env.SYSTEM_MULTI[C.FAM].replace("an 8-bit", f"a {r['width']}-bit") and
                  C.messages("dint", r["before"], r["event"], r["width"])[1] == env.multi_step_messages(C.FAM, r["before"], [r["event"]])[1]
                  for r in ood))
        bad = 0
        sample = dev_rows[::7] + tr["dint"][::25]
        for a in C.ARMS:
            for r in sample:
                for sf in (None, {"para": 2, "names": 3}, {"para": 4, "names": 4}):
                    msgs = C.messages(a, r["before"], r["event"], r["width"], sf)
                    bad += int(C.render_chat(msgs) != tok().apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                                               enable_thinking=False))
        check(f"render_chat == tok.apply_chat_template ({len(sample) * 3 * len(C.ARMS)} prompts, 8/10/12-bit, surface variants)", bad == 0, bad)
        ad = tmp / "adapter_probe"
        _make_probe_adapter(ad)
        pm = PeftModel.from_pretrained(build("sparse"), str(ad), adapter_name=R.ADAPTER_NAME, is_trainable=False).eval()
        ch16 = dchains[:16]
        for side in ("off", "on"):
            with R.side_ctx(pm, side):
                a_ = REAL_CHAIN(pm, tok(), "dint", ch16)
                b_ = generation.chain(pm, tok(), [{"family": C.FAM, "s0": c["s0"], "events": c["events"]} for c in ch16], "chain_m1")
            check(f"t2bp_chain('dint') == generation.chain('chain_m1') on 16 dev chains ({side}): texts and states",
                  [x["texts"] for x in a_] == [x["texts"] for x in b_] and
                  [[None if s is None else env.dumps(s) for s in x["states"]] for x in a_] ==
                  [[None if s is None else env.dumps(s) for s in x["states"]] for x in b_])
        generation.generate_texts = oracle_generate
        try:
            for side in ("off", "on"):
                with R.side_ctx(pm, side):
                    a_ = REAL_CHAIN(pm, tok(), "dint", dchains)
                    b_ = generation.chain(pm, tok(), [{"family": C.FAM, "s0": c["s0"], "events": c["events"]} for c in dchains], "chain_m1")
                live4 = sum(x["states"][4] is not None for x in a_)
                check(f"under the oracle generator ({side}): t2bp_chain('dint') == generation.chain('chain_m1') on all 240 dev "
                      f"chains, {live4} alive at k=4", live4 == len(dchains) and [x["texts"] for x in a_] == [x["texts"] for x in b_]
                      and [[None if s is None else env.dumps(s) for s in x["states"]] for x in a_] ==
                      [[None if s is None else env.dumps(s) for s in x["states"]] for x in b_])
        finally:
            generation.generate_texts = REAL_GENERATE
        del pm
        gc.collect()

        # ------------------------------------------------------------ 2 data audit
        print("2 data audit", flush=True)
        au = AU.audit("")
        hs = au["core"]["heuristic_accuracy"]
        check("the written training files pass the audit (balance, quotas, pairs, renderings, filler, SOURCE hashes)",
              au["ok"], {k: round(v, 4) for k, v in hs.items()})
        cnt = MD.counter()

        def audit_variant(mutate) -> bool:
            rows = {a: copy.deepcopy(v) for a, v in tr.items()}
            mutate(rows)
            return MD.audit_files(rows, cnt)["ok"]

        def drop_inapp(rows):
            for a in rows:
                rows[a] = [r for r in rows[a] if r["type"] != "Zi"][:]
        check("refused: composition without the add=0 inapplicable rows (int_add balance breaks)", not audit_variant(drop_inapp))

        def break_pair(rows):
            pid = next(r["pair_id"] for r in rows["dint"] if r.get("pair_id"))
            for a in rows:
                for r in rows[a]:
                    if r.get("pair_id") == pid and r["pair_role"] == "inapp":
                        r["event"] = {**r["event"], "cost": r["event"]["cost"] + 1}
        check("refused: a minimal pair whose members differ outside one precondition bit", not audit_variant(break_pair))

        def wrong_gold(rows):
            for a in rows:
                rows[a][0]["gold"] = rows[a][0]["gold"] ^ 1
        check("refused: a wrong gold", not audit_variant(wrong_gold))

        def short_fill(rows):
            r = rows["ffill"][3]
            r["target"] = r["target"].replace('"fill":" .', '"fill":"', 1)
        check("refused: a filler target one token short of the trace", not audit_variant(short_fill))

        def reorder_trace(rows):
            r = rows["strace"][5]
            d = json.loads(r["target"])
            r["target"] = json.dumps({"calc": d["calc"], "cmp": d["cmp"], "after": d["after"]}, separators=(",", ":"))
        check("refused: a trace target with keys out of the literal order", not audit_variant(reorder_trace))

        # ------------------------------------------------------------ 3 surface diversity
        print("3 surface diversity", flush=True)
        canon = MD.canonical_from_arm_rows(tr["dint"])
        surf = MD.assign_surfaces(canon, True)
        sd_rows = {a: MD.render_arm(canon, a, cnt, surf) for a in C.ARMS}
        au_sd = MD.audit_files(sd_rows, cnt)
        check("surface-diversity renderings pass the same audit", au_sd["ok"], au_sd["checks"])
        check("surface diversity keeps the canonical rows and every target",
              all(MD.canonical_from_arm_rows(sd_rows[a]) == canon and [r["target"] for r in sd_rows[a]] == [r["target"] for r in tr[a]]
                  for a in C.ARMS))
        used = {(s["para"], s["names"]) for s in surf.values()}
        check("only training variants are used (held-out paraphrase/field set 4 never), all 16 combinations appear",
              all(p in C.TRAIN_SURFACE_IDX and n in C.TRAIN_SURFACE_IDX for p, n in used) and len(used) == 16, sorted(used))
        pair_same = all(surf[x["id"]] == surf[y["id"]] for x in canon for y in canon[:0])
        mp = {}
        for r in canon:
            if r["pair_id"]:
                mp.setdefault(r["pair_id"], []).append(surf[r["id"]])
        check("pair members share one surface; system texts and user keys vary",
              all(v[0] == v[1] for v in mp.values()) and len({r["prompt"].split("<|im_end|>")[0] for r in sd_rows["dint"]}) >= 16
              and any('"state":' in r["prompt"] for r in sd_rows["dint"]) and pair_same is True)
        check("surface diversity is OFF by default: canonical files carry surface null",
              all(r["surface"] is None for a in C.ARMS for r in tr[a]))

        # ------------------------------------------------------------ 4 scorer
        print("4 scorer self-test", flush=True)
        st = A.scorer_selftest(dev_rows, dchains)
        check("scorer self-test: golds 100% in every format, heuristics at their exact floors, negatives rejected, "
              "D-int parser == env.parse_state", st["ok"],
              {k: f"{v['correct']}/{v['n']}" for k, v in st["golds"].items()})
        hf = lambda s: len(tok()(s, add_special_tokens=False).input_ids)   # noqa: E731
        bad_fill = sum(hf(r["target"]) != hf(C.trace_text(r["before"], r["event"])) for r in tr["ffill"])
        bad_cnt = sum(hf(r["target"]) != cnt(r["target"]) for a in C.ARMS for r in tr[a][::10])
        dev_fill = sum(hf(C.target_text("ffill", r["before"], r["event"], r["width"], hf)) != hf(C.trace_text(r["before"], r["event"]))
                       for r in dev_rows)
        check(f"filler token count == trace token count under the HF tokenizer on all {len(tr['ffill']):,} training and "
              f"{len(dev_rows):,} dev rows; HF counts == tokenizers-library counts", bad_fill == 0 and dev_fill == 0 and bad_cnt == 0,
              (bad_fill, dev_fill, bad_cnt))

        # ------------------------------------------------------------ 5 analysis dry run
        print("5 analysis dry run", flush=True)
        dr = A.dry_run(dev_rows, dchains)
        check(f"analysis dry run: {dr['counts']['scenarios']} synthetic scenarios, {dr['counts']['rule_units']} rule units, "
              f"{dr['counts']['close_call_units']} close-call units, {dr['counts']['filler_units']} R5 addendum units and "
              f"{dr['counts']['refusals']} refusals give the pre-stated results", dr["ok"], dr["counts"])

        # ------------------------------------------------------------ 6 full tiny run
        print("6 full tiny run", flush=True)
        anchor_chains = R.pilot_anchor_chains()[:8]
        anchor_ref = make_anchor_ref(tmp, anchor_chains)
        rule_dir = tmp / "rule"
        rule_dir.mkdir()
        (rule_dir / C.SCREEN_RULE_MD).write_text(A.rule_md(), encoding="utf-8", newline="\n")
        rule_rec = {"record": "T2BP_SCREEN_RULE", "revision": 0, "inputs": C.frozen_inputs(),
                    "rule_md_sha256": sha(rule_dir / C.SCREEN_RULE_MD),
                    "rule_text_sha256": hashlib.sha256(A.rule_text().encode("utf-8")).hexdigest()}
        rule_path = rule_dir / C.SCREEN_RULE_JSON
        rule_path.write_text(json.dumps(rule_rec), encoding="utf-8")

        def cfg_for(name, **kw):
            base = dict(tag=name, out=tmp / "runs" / name, steps=STEPS, lr=LR_TINY, anchor_chains=anchor_chains,
                        anchor_ref=anchor_ref, anchor_ref_sha=None, dev_subset=(2, 2), rule_record=rule_path)
            base.update(kw)
            return R.Cfg(**base)

        rt = FakeRuntime(tmp)
        be = TinyBackend(rt)
        cfg = cfg_for("run_a")
        rc = R.run(cfg, be)
        out = cfg.out
        check("tiny run exits 0 and writes DONE.json", rc == 0 and (out / "DONE.json").is_file(), rc)
        prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        check("every arm's stages T/E1/E2 recorded with file hashes", all(set(prov["stages"][a]) == set(R.STAGES) for a in cfg.arms))
        leases = [c for c in rt.calls if c[0] == "lease"]
        check("one lease, label 'rcwm-t1:t2b-prereq'", leases == [("lease", R.LEASE_LABEL)], leases)
        seq = [c[0] if c[0] not in ("load", "train") else f"{c[0]}:{c[1]}" for c in rt.calls]
        order_ok = all(seq.index(f"train:{a}") < seq.index("load:sparse", seq.index(f"train:{a}")) for a in cfg.arms)
        waits_ok = all(seq[i - 1] == "wait" for i, s in enumerate(seq) if s.startswith("load") or s.startswith("train"))
        check("per arm: train, then evaluation load; a resource wait before every training and load; one anchor per turn",
              order_ok and waits_ok and len(prov["turns"]) == 1 and prov["turns"][0]["anchor"]["pass"], seq)
        check("anchor reproduces the base-model chain_m1 reference (states and texts 1.0)",
              prov["turns"][0]["anchor"]["identical_states"] == 1.0 and prov["turns"][0]["anchor"]["identical_texts"] == 1.0)
        sub_rows, sub_ch = R.load_dev(cfg)
        outputs = A.validate_run(out, sub_rows, sub_ch, list(cfg.arms))
        diff = {a: sum(x != y for x, y in zip(outputs[a]["off"]["onestep"], outputs[a]["on"]["onestep"])) for a in cfg.arms}
        check("OFF != ON one-step texts in every arm (the trained adapter is active)", all(v > 0 for v in diff.values()), diff)
        res = A.analyse(outputs, sub_rows, sub_ch)
        check("the analysis validators accept the run and the rule produces a verdict", bool(res["rule"]["verdict"]), res["rule"]["verdict"])
        prov_a = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        check("provenance records the frozen code hashes and the screen-rule record at the first turn; anchors_ok",
              prov_a["code_sha256"] == R.code_hashes() and prov_a["screen_rule_sha256"] == sha(rule_path) and A.anchors_ok(prov_a))
        ac = json.loads((out / "dint" / "train" / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
        check("adapters are the T2 recipe's LoRA (r16, alpha 32, the 7 target modules)",
              ac["r"] == 16 and ac["lora_alpha"] == 32 and sorted(ac["target_modules"]) ==
              sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        rels = stage_files(out, cfg.arms)
        h_a = tree_hashes(out, rels)
        n = len(rt.calls)
        rc2 = R.run(cfg, be)
        check("a rerun after DONE is a no-op (exit 0, no lease, files unchanged)", rc2 == 0 and len(rt.calls) == n and tree_hashes(out, rels) == h_a)

        # ------------------------------------------------------------ 6b oracle run on the FULL dev bank
        print("6b oracle-generator run on the full dev bank", flush=True)
        generation.generate_texts = oracle_generate
        try:
            ref_o = make_anchor_ref(tmp, anchor_chains, name="anchor_ref_oracle.jsonl")
            rt_o = FakeRuntime(tmp)
            cfg_o = cfg_for("run_oracle", anchor_ref=ref_o, dev_subset=None)
            rc_o = R.run(cfg_o, TinyBackend(rt_o))
        finally:
            generation.generate_texts = REAL_GENERATE
        outputs_o = A.validate_run(cfg_o.out, dev_rows, dchains, list(cfg_o.arms))
        res_o = A.analyse(outputs_o, dev_rows, dchains, st)
        mo = res_o["rule"]["metrics"]
        off_fc = {a: res_o["arms"][a]["off"]["onestep"]["FC"] for a in cfg_o.arms}
        off_fa = {a: res_o["arms"][a]["off"]["onestep"]["FA"] for a in cfg_o.arms}
        check(f"oracle run (ON = the rule in each arm's format, OFF = copy) on all {len(dev_rows):,} rows and {len(dchains)} "
              "chains: exit 0; the analysis validates every file and reads ON IAW/OOR/K4/ID/WF 1.0, FC 0, FA 0; OFF FC 1.0, "
              "FA 0; verdict WINNER=dint (tie rule)",
              rc_o == 0 and res_o["rule"]["verdict"] == "WINNER=dint" and all(
                  mo[a]["IAW"] == mo[a]["OOR"] == mo[a]["K4"] == mo[a]["ID"] == mo[a]["WF"] == 1.0 and mo[a]["FC"] == 0.0
                  and mo[a]["FA"] == 0.0 for a in cfg_o.arms)
              and all(v == 1.0 for v in off_fc.values()) and all(v == 0.0 for v in off_fa.values()), (res_o["rule"]["verdict"], mo, off_fc))
        md = A.render_md(res_o)
        check("the markdown rendering carries the verdict and every stratum", "WINNER=dint" in md and all(s_ in md for s_ in C.DEV_STRATA))
        one_arm = A.analyse({"bbits": outputs_o["bbits"]}, dev_rows, dchains, st)
        check("a single-arm (seed 1/2) run is analysed descriptively with NO_VERDICT", one_arm["rule"]["verdict"] == "NO_VERDICT")
        mech = {a: {k: res_o["arms"][a]["on"]["onestep"]["strata"]["D_int_add_wrong"].get(k) for k in ("cmp_ok", "calc_ok", "flag_ok")}
                for a in cfg_o.arms}
        check("oracle run: mechanism scores 1.0 (S-trace cmp_ok/calc_ok, B-bits flag_ok)",
              mech["strace"]["cmp_ok"] == 1.0 and mech["strace"]["calc_ok"] == 1.0 and mech["bbits"]["flag_ok"] == 1.0, mech)

        # ------------------------------------------------------------ 6c frozen rule, resume code check, DONE arms
        print("6c frozen rule", flush=True)
        bad_rec = dict(rule_rec, inputs=dict(rule_rec["inputs"], **{"scripts/t2bp_run.py": "0" * 64}))
        bad_path = rule_dir / "SCREEN_RULE_bad.json"
        bad_path.write_text(json.dumps(bad_rec), encoding="utf-8")
        rt_z = FakeRuntime(tmp)
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_z", rule_record=bad_path), TinyBackend(rt_z)), "frozen screen rule")
        check("a frozen input that differs from SCREEN_RULE.json refuses before the lease (not persisted, no lease, no output)",
              ok and pers is False and not rt_z.calls and not (tmp / "runs" / "run_z" / "provenance.json").exists(), msg)
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_z", rule_record=rule_dir / "missing.json"), TinyBackend(rt_z)),
                                    "not a screen-rule record")
        check("a missing screen-rule record refuses before the lease (not persisted)", ok and pers is False and not rt_z.calls, msg)
        check("the analysis' frozen check: [] on the record, problems on a changed input hash and on a changed SCREEN_RULE.md",
              A.check_frozen(rule_path) == [] and A.check_frozen(bad_path) != [] and
              (lambda: ((rule_dir / C.SCREEN_RULE_MD).write_text("x", encoding="utf-8"), A.check_frozen(rule_path))[1])() != [])
        (rule_dir / C.SCREEN_RULE_MD).write_text(A.rule_md(), encoding="utf-8", newline="\n")
        check("restored SCREEN_RULE.md verifies again", A.check_frozen(rule_path) == [])
        rt_h = FakeRuntime(tmp)
        cfg_h = cfg_for("run_h", cap_h=0.5)
        rc_h = R.run(cfg_h, TinyBackend(rt_h))
        ph = cfg_h.out / "provenance.json"
        prov_h = json.loads(ph.read_text(encoding="utf-8"))
        prov_h["code_sha256"]["rcwmt/generation.py"] = "f" * 64
        ph.write_text(json.dumps(prov_h), encoding="utf-8")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_h"), TinyBackend(rt_h)), "code changed since the run's first turn")
        check("a resume under code that differs from the run's first turn refuses (not persisted, no second lease)",
              rc_h == 5 and ok and pers is False and [c[0] for c in rt_h.calls].count("lease") == 1, msg)
        n = len(rt.calls)
        cfg_x = cfg_for("run_a", arms=C.SCREEN_ARMS + ("ffill",))
        ok, msg, pers = raises_stop(lambda: R.run(cfg_x, be), "never extended")
        check("--with-filler after a completed 3-arm run is an explicit stop (not persisted, no lease, files unchanged)",
              ok and pers is False and len(rt.calls) == n and tree_hashes(out, rels) == h_a
              and not (out / "STOPPED.json").exists(), msg)

        # ------------------------------------------------------------ 6d surface diversity run (stage E3)
        print("6d surface-diversity run", flush=True)
        sd_dir = tmp / "sd_data"
        sd_dir.mkdir()
        sd_files = {}
        for a in C.SCREEN_ARMS:
            sd_files[a] = sd_dir / C.TRAIN_FILES_SD[a].name
            sd_files[a].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sd_rows[a]), encoding="utf-8")
        sd_src = sd_dir / C.TRAIN_SOURCE_SD.name
        sd_src.write_text(json.dumps({"files": {p_.name: sha(p_) for p_ in sd_files.values()}}), encoding="utf-8")
        seen = []
        real_gen = R.gen_texts

        def spy_gen(model, tok_, message_lists):
            seen.extend(m_[1]["content"] for m_ in message_lists)
            return real_gen(model, tok_, message_lists)
        R.gen_texts = spy_gen
        try:
            rt_s = FakeRuntime(tmp)
            cfg_s = cfg_for("sd_run", data_variant="sd", train_files=sd_files, train_source=sd_src)
            rc_s = R.run(cfg_s, TinyBackend(rt_s))
        finally:
            R.gen_texts = real_gen
        prov_s = json.loads((cfg_s.out / "provenance.json").read_text(encoding="utf-8"))
        held = [x for x in seen if '"mask":' in x and '"moves":' in x]
        check("surface-diversity run: exit 0, stages T/E1/E2/E3 per arm, E3 outputs written, held-out field names used only "
              "in E3 (OFF+ON: 2 x rows x arms prompts)",
              rc_s == 0 and prov_s["stages_per_arm"] == list(R.STAGES_SD) and
              all(set(prov_s["stages"][a]) == set(R.STAGES_SD) for a in C.SCREEN_ARMS) and
              all((cfg_s.out / a / "onestep_heldout_on.jsonl").is_file() for a in C.SCREEN_ARMS) and
              len(held) == 2 * len(R.load_dev(cfg_s)[0]) * len(C.SCREEN_ARMS), (rc_s, len(held)))
        s_rows, s_ch = R.load_dev(cfg_s)
        out_s = A.validate_run(cfg_s.out, s_rows, s_ch, list(C.SCREEN_ARMS))
        res_s = A.analyse(out_s, s_rows, s_ch, anchors=A.anchors_ok(prov_s))
        check("the analysis validates E3 and reports the held-out surface descriptively (the verdict ignores it)",
              all("heldout_surface_descriptive" in res_s["arms"][a]["on"] for a in C.SCREEN_ARMS)
              and res_s["arms"]["dint"]["on"]["heldout_surface_descriptive"]["surface"] == C.HELDOUT_SURFACE)

        # ------------------------------------------------------------ 7 crash + resume
        print("7 crash + resume", flush=True)
        rt_b = FakeRuntime(tmp)
        be_b = TinyBackend(rt_b)
        cfg_b = cfg_for("run_b")
        orig_train = TinyBackend.train
        state = {"train": 0, "chain": 0}

        def crashing_train(self, rows_path, out_dir, steps, seed, lr):
            state["train"] += 1
            if state["train"] == 2:                                # inside the second arm's training
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / "log.txt").write_text("half a training\n", encoding="utf-8")
                raise RuntimeError("simulated crash in training")
            return orig_train(self, rows_path, out_dir, steps, seed, lr)
        TinyBackend.train = crashing_train
        try:
            R.run(cfg_b, be_b)
            crashed = False
        except RuntimeError:
            crashed = True
        finally:
            TinyBackend.train = orig_train
        check("crash in training propagates; the lease is released", crashed and not rt_b.LOCK_PATH.exists()
              and rt_b.calls[-1] == ("release", R.LEASE_LABEL))

        def crashing_chain(model, tok_, arm, chains):
            state["chain"] += 1
            if arm == "strace" and state["chain"] == 2:            # inside strace E2, ON side
                (cfg_b.out / "strace" / "chains_off.jsonl.partial").write_text("half", encoding="utf-8")
                raise RuntimeError("simulated crash in E2")
            return REAL_CHAIN(model, tok_, arm, chains)
        R.t2bp_chain = crashing_chain
        try:
            R.run(cfg_b, be_b)
            crashed2 = False
        except RuntimeError:
            crashed2 = True
        finally:
            R.t2bp_chain = REAL_CHAIN
        orph = sorted(p.name for p in (cfg_b.out / "_orphans").iterdir())
        check("the resume orphaned the partial training dir; a crash inside E2 propagates", crashed2 and
              any(x.startswith("strace_T_") for x in orph), orph)
        rc_b = R.run(cfg_b, be_b)
        orph = sorted(p.name for p in (cfg_b.out / "_orphans").iterdir())
        prov_b = json.loads((cfg_b.out / "provenance.json").read_text(encoding="utf-8"))
        check("the second resume orphaned the partial E2, re-anchored and finished (exit 0)",
              rc_b == 0 and any(x.startswith("strace_E2_") for x in orph) and all("anchor" in t for t in prov_b["turns"] if t["stages_run"]),
              [t.get("stages_run") for t in prov_b["turns"]])
        check("every output of the crashed-and-resumed run equals the uninterrupted run byte for byte (adapters included)",
              tree_hashes(cfg_b.out, rels) == h_a)

        # ------------------------------------------------------------ 8 deferral
        print("8 deferral", flush=True)
        rt_c = FakeRuntime(tmp)
        be_c = TinyBackend(rt_c)
        cfg_c = cfg_for("run_c", cap_h=0.5)
        rc_c = R.run(cfg_c, be_c)
        prov_c = json.loads((cfg_c.out / "provenance.json").read_text(encoding="utf-8"))
        check("nothing fits (cap 0.5 h = the stage margin): exit 5, every stage deferred, no training, no load",
              rc_c == 5 and not [c for c in rt_c.calls if c[0] in ("load", "train")] and len(prov_c["turns"][0]["deferred"]) == 9,
              prov_c["turns"][0]["deferred"])
        cfg_c2 = cfg_for("run_c")
        orig_est = R.estimates

        def est_big_strace(*a, **k):
            e = orig_est(*a, **k)
            e["arms"]["strace"]["T"] = 99 * 3600.0
            return e
        R.estimates = est_big_strace
        try:
            rc_c2 = R.run(cfg_c2, be_c)
        finally:
            R.estimates = orig_est
        prov_c = json.loads((cfg_c.out / "provenance.json").read_text(encoding="utf-8"))
        check("arm-level deferral: D-int pair runs; S-trace (pair does not fit) and B-bits deferred; exit 5",
              rc_c2 == 5 and prov_c["turns"][-1]["stages_run"] == ["dint:T", "dint:E1", "dint:E2"]
              and prov_c["turns"][-1]["deferred"][:1] == ["strace:T"], prov_c["turns"][-1])
        orig_cs = R.G2.can_start
        calls = {"n": 0}

        def cs_fail_e2(elapsed, est, reserve, cap):
            calls["n"] += 1
            return False if calls["n"] == 3 else orig_cs(elapsed, est, reserve, cap)   # pair, E1, E2 -> E2 refused
        R.G2.can_start = cs_fail_e2
        try:
            rc_c3 = R.run(cfg_c2, be_c)
        finally:
            R.G2.can_start = orig_cs
        prov_c = json.loads((cfg_c.out / "provenance.json").read_text(encoding="utf-8"))
        check("stage-level deferral: S-trace trained and E1 done, E2 deferred with B-bits; exit 5",
              rc_c3 == 5 and prov_c["turns"][-1]["stages_run"] == ["strace:T", "strace:E1"]
              and prov_c["turns"][-1]["deferred"][:1] == ["strace:E2"], prov_c["turns"][-1])
        rc_c4 = R.run(cfg_c2, be_c)
        prov_c = json.loads((cfg_c.out / "provenance.json").read_text(encoding="utf-8"))
        check("the next call resumes (re-anchors, S-trace E2 then B-bits) and completes; outputs == run 6 byte for byte",
              rc_c4 == 0 and prov_c["turns"][-1]["stages_run"][0] == "strace:E2" and "anchor" in prov_c["turns"][-1]
              and tree_hashes(cfg_c.out, rels) == h_a, prov_c["turns"][-1]["stages_run"])

        # ------------------------------------------------------------ 9 stops
        print("9 stops", flush=True)
        rt_d = FakeRuntime(tmp)
        be_d = TinyBackend(rt_d)
        cfg_d = cfg_for("run_d")
        rt_d.WAITING["rcwm"].write_text(json.dumps({"pid": os.getppid(), "label": "rcwm-t1:other", "since": time.time()}), encoding="utf-8")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "one-waiter")
        check("a live RCWM waiting marker refuses (not persisted, no lease)", ok and pers is False and not rt_d.calls, msg)
        rt_d.WAITING["rcwm"].write_text(json.dumps({"pid": 999999, "label": "rcwm-t1:dead", "since": time.time()}), encoding="utf-8")
        check("a dead RCWM marker does not refuse", R.one_waiter_problem(be_d) is None)
        rt_d.WAITING["rcwm"].unlink()
        rt_d.LOCK_PATH.write_text(json.dumps({"label": "bwm:bundle", "pid": os.getppid(), "created": time.time()}), encoding="utf-8")
        check("a live BWM lease holder does not refuse", R.one_waiter_problem(be_d) is None)
        rt_d.LOCK_PATH.write_text(json.dumps({"label": "rcwm-t1:prereg3", "pid": os.getppid(), "created": time.time()}), encoding="utf-8")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "holds the GPU lease")
        check("a live RCWM lease holder refuses (not persisted)", ok and pers is False, msg)
        rt_d.LOCK_PATH.unlink()
        cfg_d.out.mkdir(parents=True, exist_ok=True)
        import psutil
        (cfg_d.out / ".run.lock").write_text(json.dumps({"pid": os.getppid(), "create_time": psutil.Process(os.getppid()).create_time()}), encoding="utf-8")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "run lock")
        check("a live run lock refuses (not persisted)", ok and pers is False, msg)
        (cfg_d.out / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8")
        bad_ref = tmp / "anchor_bad.jsonl"
        rows_ = C.load_jsonl(anchor_ref)
        for x in rows_:
            x["states"] = ["-1"] + x["states"][1:]           # a different start state: the states can never match
        bad_ref.write_text("".join(json.dumps(x) + "\n" for x in rows_), encoding="utf-8")
        cfg_d = cfg_for("run_d", anchor_ref=bad_ref)
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "ANCHOR FAILED")
        check("stale run lock reclaimed; anchor failure stops (persisted) before any evaluation output",
              ok and pers is True and not (cfg_d.out / "dint" / "onestep_on.jsonl").exists() and not (cfg_d.out / ".run.lock").exists(), msg)
        rt_e = FakeRuntime(tmp)
        rt_e.fp_override["sparse"] = "0000deadbeef"
        cfg_e = cfg_for("run_e")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_e, TinyBackend(rt_e)), "fingerprint")
        check("a base fingerprint mismatch stops (persisted)", ok and pers is True, msg)
        rt_f = FakeRuntime(tmp)
        rt_f.runtime_override = [{"transformers": "0.0.0"}]
        cfg_f = cfg_for("run_f")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_f, TinyBackend(rt_f)), "runtime")
        check("runtime drift at the evaluation load stops (persisted; nothing evaluated)",
              ok and pers is True and not (cfg_f.out / "dint" / "onestep_off.jsonl").exists(), msg)
        f = cfg.out / "bbits" / "chains_on.jsonl"
        orig_bytes = f.read_bytes()
        (cfg.out / "DONE.json").rename(cfg.out / "DONE.json.bak")
        f.write_bytes(orig_bytes + b"\n")
        ok, msg, pers = raises_stop(lambda: R.run(cfg, be), "changed after it was recorded")
        f.write_bytes(orig_bytes)
        (cfg.out / "DONE.json.bak").rename(cfg.out / "DONE.json")
        check("a recorded output whose sha256 changed stops (persisted)", ok and pers is True, msg)
        tmpd = tmp / "data_copy"
        tmpd.mkdir()
        for a in C.SCREEN_ARMS:
            shutil.copy(C.TRAIN_FILES[a], tmpd / C.TRAIN_FILES[a].name)
        with (tmpd / C.TRAIN_FILES["bbits"].name).open("a", encoding="utf-8") as fh:
            fh.write("\n")
        cfg_g = cfg_for("run_g", train_files={a: tmpd / C.TRAIN_FILES[a].name for a in C.SCREEN_ARMS})
        ok, msg, pers = raises_stop(lambda: R.run(cfg_g, TinyBackend(FakeRuntime(tmp))), "differs from its SOURCE record")
        check("a training file that differs from its SOURCE record stops before the lease", ok, msg)

        # ------------------------------------------------------------ 10 plan
        print("10 plan", flush=True)
        pl = R.plan(verbose=False)
        tp, to = pl["planning"]["turns_screen"], pl["observed"]["turns_screen"]
        check("plan: the 3-arm screen takes 2 turns at the planning rate (OFF of non-native frames at the 160-token cap) "
              "and 1 at the observed rates; every turn within the cap; each arm's T/E1/E2 in one turn; the filler control "
              "run is one turn",
              len(tp) == 2 and len(to) == 1 and all(t["projected_h"] <= R.CAP_H for t in tp + to) and
              all(sum(f"{a}:T" in t["stages"] for t in tp) == 1 and
                  [t_ for t_ in tp if f"{a}:T" in t_["stages"]][0]["stages"].count(f"{a}:E2") == 1 for a in C.SCREEN_ARMS) and
              len(pl["planning"]["turns_filler_control_run_if_strace_wins"]) == 1,
              (pl["planning"]["gpu_h_screen"], pl["observed"]["gpu_h_screen"]))
        short = R.simulate_turns(R.Cfg(cap_h=3.0), C.SCREEN_ARMS)
        check("plan: a 3.0 h cap splits the screen into several turns, each arm's pair in one turn",
              len(short) >= 2 and all(any(s.endswith(":T") for s in t["stages"]) for t in short), short)
        gp = json.loads((C.DOCS / "GEN_PLAN.json").read_text(encoding="utf-8"))
        check("docs/t2b/prereq/GEN_PLAN.json was written by --plan without importing torch", gp.get("torch_imported") is False)
    finally:
        if not KEEP:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = all(r["ok"] for r in RESULTS)
    rec = {"ok": ok, "n_checks": len(RESULTS), "passed": sum(r["ok"] for r in RESULTS), "checks": RESULTS,
           "seconds": round(time.time() - t0, 1), "peak_rss_mb": _peak_mb(), "threads": torch.get_num_threads(),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "code_sha256": {r: sha(ROOT / r) for r in ("scripts/t2bp_run.py", "scripts/t2bp_common.py", "scripts/t2bp_analysis.py",
                                                     "scripts/t2bp_make_data.py", "scripts/t2bp_audit.py", "scripts/t2bp_test.py")},
           "data_sha256": {p.name: sha(p) for p in list(C.TRAIN_FILES.values()) + [C.DEV]}}
    print(f"{rec['passed']}/{rec['n_checks']} checks passed in {rec['seconds']} s; peak RSS {rec['peak_rss_mb']} MB", flush=True)
    if RECORD:
        C.write_doc("TINY_TEST.json", json.dumps(rec, indent=1) + "\n")
    return 0 if ok else 1


@torch.no_grad()
def _make_probe_adapter(path: Path) -> None:
    from peft import LoraConfig, get_peft_model
    m = build("sparse")
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    g = torch.Generator().manual_seed(7)
    for n_, p in pm.named_parameters():
        if "lora_" in n_:
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    pm.save_pretrained(str(path))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAILED: {e}", flush=True)
        if RECORD:
            C.write_doc("TINY_TEST.json", json.dumps({"ok": False, "checks": RESULTS, "error": str(e)[:500]}, indent=1) + "\n")
        sys.exit(1)
