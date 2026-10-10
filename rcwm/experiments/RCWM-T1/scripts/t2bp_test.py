"""CPU tests of the T2b prerequisite screen kit, revision 4 (data audit, surface proofs, amendment-4b filler, scorer,
analysis rule, runner, the report-only held-out-surface chains E4) with tiny random Qwen3 stand-ins. No real model is loaded (AutoModelForCausalLM.from_pretrained is
disabled; the vendored trainer's load_core returns the tiny model), no GPU is touched (CUDA_VISIBLE_DEVICES="" before
torch, is_available/device_count patched), no lease is taken (a fake runtime stands in for rcwmt.runtime under the
runner's own RealBackend code). Threads 2. One process. Runner outputs go to a temporary directory only.

Stand-ins: Qwen3 with 2 blocks, hidden 64, 4 heads, 2 KV heads, head_dim 16, the real vocabulary and the real
tokenizer (read-only). Training runs the REAL vendored trainer (train_generic.main -> lm.attach_lora r16 ->
lm.train_sft) for 2 steps at lr 0.05 (so the adapter visibly changes greedy outputs) on the REAL surface-diversity
training files. Pacing-only deviations: generation.breathe / throttle_list no-ops, Tensor.cuda the identity, a 12-token
decode cap, BWM_DEVICE=cpu.

Checks
  1 frames: D-int canonical messages == env.multi_step_messages; OOR prompts differ from 8-bit prompts only in the
    width word, in EVERY surface (incl. the held-out paraphrase, whose state sentence words the width its own way);
    render_chat == tok.apply_chat_template for every arm (8-, 10-, 12-bit; canonical, training sd and held-out
    surfaces with permuted key orders); t2bp_chain('dint') == generation.chain('chain_m1') OFF and ON; no kit file has
    CRLF line endings
  2 data audit: the sd (primary) and canonical files pass; perturbed copies are refused (composition, a broken minimal
    pair, a wrong gold, a filler of another length, a reordered trace target, and a training row rendered with a
    held-out paraphrase / field-name variant / top-level order / event order, or with a prompt that differs from its
    recorded surface)
  3 surface proofs (C2): the held-out key orders regenerate from ORDER_SEED (reversed + 2 random event orders; reversed
    only for the 2-key top object); every one of the 6,000 training prompts read back FROM ITS TEXT has no held-out
    factor and no held-out text fragment; the sd files keep the canonical rows and targets; diversity used; the E3
    rendering rotates the 3 held-out event orders within every stratum and reads back as the held-out surface; the E4
    chain rendering rotates them by chain index (80/80/80) and every step reads back as the held-out surface
  4 scorer self-test and amendment 4b: golds, floors, negatives; FILL_K == the formula under the HF tokenizer; one fill
    string on every training and dev target; no filler shorter than its trace; the fill length carries zero
    information about the answer (lookup == majority share, mutual information 0), while the revision-2 row-matched
    FILL length (trace-head tokens - 3; it reproduces every row of the kept revision-2 file) would have revealed
    applicability (0.749 train, 0.837 dev 8-bit, 0.813 dev OOR; the figures quoted in the rule and DESIGN.md)
  5 analysis dry run: every synthetic scenario (incl. the held-out-surface ones), report-only unit (revision 4: the
    verdict never depends on E4), rule unit, close-call unit, R5 addendum unit and refusal gives the pre-stated result
  6 full tiny run (3 arms, sd files): exit 0, DONE.json, stages T/E1/E2/E3/E4, E3 and E4 ON only and in the held-out
    surface only, one lease 'rcwm-t1:t2b-prereq', resource wait before every load, train before eval per arm, anchor
    pass recorded, OFF != ON, analysis validators accept the outputs (HS and K4_HS present), a rerun is a no-op; an
    oracle-generator run on the whole dev bank (HS 1.0, K4_HS 1.0); the R5 addendum path; frozen-rule stops; the
    canonical variant is refused by the runner and by the analysis; a held-out chain file whose states differ from
    parse(texts) is refused
  7 crash + resume: a crash inside training (partial train dir) and a crash inside E2 (stray .partial) are orphaned and
    redone; the lease is released by the crash; every output equals the uninterrupted run byte for byte
  8 deferral: nothing fits -> exit 5, no load, no training; an arm-level deferral (pair does not fit) and a stage-level
    deferral (E2 does not fit) resume on the next call (re-anchored) and finish with outputs identical to run 6
  9 stops: one-waiter rule (live RCWM marker / live RCWM holder refuse, not persisted; dead marker and live BWM holder
    pass); live run lock refuses, stale lock reclaimed; anchor failure (persisted; no evaluation output); fingerprint
    mismatch; runtime drift; a recorded output changed; a data file changed after recording
 10 --plan logic (E4 included): the sd screen takes 2 turns at the planning rate and 1 at the observed rate; per-factor
    held-out renderings would need a third planning turn (omitted); the filler control run is one turn; every arm's
    whole pair x 1.25 fits an empty 4.5 h turn; a 3.5 h cap splits the screen, a 3.0 h cap is refused (S-trace with
    E3/E4 needs more)

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
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
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
LF_EXEMPT = {"TINY_TEST.json", "SCREEN_RULE_rev1.json"}


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
        self.LOCK_PATH.write_text(json.dumps({"label": label, "pid": os.getpid(), "created": time.time()}), encoding="utf-8", newline="\n")
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
    p.write_text("".join(json.dumps(x) + "\n" for x in R.chain_rows(chains, rr)), encoding="utf-8", newline="\n")
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


def oracle_generate(model, tok_, message_lists, max_new_tokens, **kw):
    """ON: the stated rule, in the arm's own format; OFF: copy the input state. The arm, the width, the state and the
    event are read back from the prompt itself (any surface: t2bp_common.read_prompt)."""
    side = side_of(model)
    out = []
    for msgs in message_lists:
        x = C.read_prompt(msgs)
        a, w, b, ev = x["arm"], x["width"], x["before"], x["event"]
        out.append(C.target_text(a, b, ev, w) if side == "on" else C.render_prediction(a, b, b, ev, w))
    return out


def _peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)


def _norm_width(t: str, w: int) -> str:
    """The text with its width word replaced by a placeholder (and the article before a number normalised)."""
    t = re.sub(r"\ban? (?=\d)", "<A> ", t)
    return t.replace(str(w), "<W>")


def main() -> int:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="t2bp_tiny_"))
    print(f"tiny test workspace: {tmp}", flush=True)
    dev_rows, parents = C.load_dev()
    dchains = C.dev_chains(parents)
    tr = {a: C.load_jsonl(C.TRAIN_FILES[a]) for a in C.ARMS}
    trs = {a: C.load_jsonl(C.TRAIN_FILES_SD[a]) for a in C.ARMS}
    try:
        # ------------------------------------------------------------ 1 frames
        print("1 frames", flush=True)
        w8 = [r for r in dev_rows + tr["dint"] if r["width"] == C.WIDTH_ID]
        check(f"D-int canonical messages == env.multi_step_messages on every 8-bit dev and training row ({len(w8)})",
              all(C.messages("dint", r["before"], r["event"], r["width"]) == env.multi_step_messages(C.FAM, r["before"], [r["event"]])
                  for r in w8))
        ood = [r for r in dev_rows if r["width"] != C.WIDTH_ID]
        check("OOD D-int canonical prompts differ from T2's frame only in the width word ('a 10-bit' / 'a 12-bit')",
              all(C.messages("dint", r["before"], r["event"], r["width"])[0]["content"] ==
                  env.SYSTEM_MULTI[C.FAM].replace("an 8-bit", f"a {r['width']}-bit") and
                  C.messages("dint", r["before"], r["event"], r["width"])[1] == env.multi_step_messages(C.FAM, r["before"], [r["event"]])[1]
                  for r in ood))
        bad_w, n_w = [], 0
        for a in C.ARMS:
            for i in range(len(C.PARAPHRASES)):
                for j in range(len(C.FIELD_VARIANTS)):
                    sf = {"para": i, "names": j}
                    t8 = C.system_text(a, 8, sf)
                    for w in C.OOD_WIDTHS:
                        n_w += 1
                        tw = C.system_text(a, w, sf)
                        if _norm_width(tw, w) != _norm_width(t8, 8) or str(w) not in tw or "8" in tw:
                            bad_w.append((a, i, j, w))
        ho = C.system_text("dint", 12, C.heldout_surface(0))
        check(f"OOR system texts differ from the 8-bit texts only in the width word in every surface ({n_w} texts: 4 arms x "
              "5 paraphrases x 5 field variants x 10/12 bits); the held-out paraphrase words the width its own way",
              not bad_w and ho.startswith("The state is a flag mask 12 bits wide."), (bad_w[:3], ho[:60]))
        bad = 0
        sample = dev_rows[::7] + tr["dint"][::25]
        sd_sample = {r["id"]: r["surface"] for r in trs["dint"][::25]}
        n_pr = 0
        for a in C.ARMS:
            for r in sample:
                sfs = [None, {"para": 2, "names": 3}, sd_sample.get(r["id"]) or {"para": 1, "names": 2, "order": {
                    "top": list(C.OBJ_SLOTS["top"]), "event": list(C.train_orders("event")[777])}}]
                if r.get("split") == "dev":
                    sfs.append(C.heldout_surface_of(r))
                for sf in sfs:
                    msgs = C.messages(a, r["before"], r["event"], r["width"], sf)
                    n_pr += 1
                    bad += int(C.render_chat(msgs) != tok().apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                                               enable_thinking=False))
        check(f"render_chat == tok.apply_chat_template ({n_pr} prompts: 8/10/12-bit; canonical, training sd surfaces with "
              "permuted key orders, held-out surfaces)", bad == 0, bad)
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

        crlf = []
        for p in sorted(list((ROOT / "scripts").glob("t2bp_*")) + list((ROOT / "data").glob("t2bp_*")) + list(C.DOCS.glob("*"))):
            if p.is_file() and p.name not in LF_EXEMPT and b"\r\n" in p.read_bytes():
                crlf.append(p.name)
        check("line endings: no kit script, data file or record has CRLF line endings (exempt: TINY_TEST.json, rewritten "
              "with LF at the end of this test, and the superseded SCREEN_RULE_rev1.json, kept byte for byte because the "
              "revision-2 record pins its sha256)", not crlf, crlf)

        # ------------------------------------------------------------ 2 data audit
        print("2 data audit", flush=True)
        au = AU.audit("sd")
        hs_ = au["core"]["heuristic_accuracy"]
        check("the PRIMARY (surface-diversity) training files pass the audit (balance, quotas, pairs, renderings, 4b filler, "
              "surface proof, rows/targets == canonical, SOURCE hashes)", au["ok"], {k: round(v, 4) for k, v in hs_.items()})
        auc = AU.audit("canonical")
        check("the canonical files (the row source) pass the audit", auc["ok"], [k for k, v in auc["checks"].items() if not v])
        cnt = MD.counter()

        def audit_variant(mutate, base=None) -> bool:
            rows = {a: copy.deepcopy(v) for a, v in (base or trs).items()}
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

        def longer_fill(rows):
            r = rows["ffill"][3]
            r["target"] = r["target"].replace('"fill":"', '"fill":" .', 1)
        check("refused: one filler target with a different fill length (amendment 4b: one constant per kind)",
              not audit_variant(longer_fill))

        def reorder_trace(rows):
            r = rows["strace"][5]
            d = json.loads(r["target"])
            r["target"] = json.dumps({"calc": d["calc"], "cmp": d["cmp"], "after": d["after"]}, separators=(",", ":"))
        check("refused: a trace target with keys out of the literal order", not audit_variant(reorder_trace))

        def heldout_mut(factor):
            def mut(rows):
                k = 7
                for a in rows:
                    r = rows[a][k]
                    sf = copy.deepcopy(r["surface"])
                    if factor == "para":
                        sf["para"] = C.HELDOUT_SURFACE_IDX
                    elif factor == "names":
                        sf["names"] = C.HELDOUT_SURFACE_IDX
                    elif factor == "top":
                        sf["order"]["top"] = list(C.HELDOUT_ORDERS["top"][0])
                    else:
                        sf["order"]["event"] = list(C.HELDOUT_ORDERS["event"][int(factor[-1])])
                    r["surface"] = sf
                    r["prompt"] = C.render_chat(C.messages(a, r["before"], r["event"], r["width"], sf))
            return mut
        ref = {f: not audit_variant(heldout_mut(f)) for f in ("para", "names", "top", "event0", "event1", "event2")}
        check("refused: one training row rendered (consistently) with the held-out paraphrase, field names, top-level order "
              "or any of the 3 held-out event orders", all(ref.values()), ref)

        def silent_heldout(rows):                      # the prompt uses a held-out order, the record claims a training one
            r = rows["bbits"][11]
            sf = copy.deepcopy(r["surface"])
            sf["order"]["event"] = list(C.HELDOUT_ORDERS["event"][1])
            r["prompt"] = C.render_chat(C.messages("bbits", r["before"], r["event"], r["width"], sf))
        check("refused: a prompt in a held-out order whose recorded surface claims a training order (read back from the text)",
              not audit_variant(silent_heldout))

        # ------------------------------------------------------------ 3 surface proofs
        print("3 surface proofs", flush=True)
        ho_ = C.derive_heldout_orders()
        check("held-out key orders regenerate from ORDER_SEED: event = reversed + 2 fixed random orders, top = reversed only "
              "(a 2-key object has no third order); training subsets exclude them (event 5,037 orders, top 1)",
              ho_ == C.HELDOUT_ORDERS and ho_["event"][0] == tuple(reversed(C.OBJ_SLOTS["event"])) and
              len(set(ho_["event"])) == 3 and ho_["top"] == (tuple(reversed(C.OBJ_SLOTS["top"])),) and
              len(C.train_orders("event")) == 5037 and C.train_orders("top") == [C.OBJ_SLOTS["top"]] and
              not set(C.train_orders("event")) & set(ho_["event"]), ho_)
        frags = C.heldout_fragments()
        n_pr, held_hits, frag_hits = 0, 0, 0
        for a in C.ARMS:
            for r in trs[a]:
                n_pr += 1
                got = C.recover_surface(C.split_chat(r["prompt"]))
                held_hits += int(bool(C.heldout_factors(got)))
                frag_hits += sum(f in r["prompt"] for f in frags.values())
        check(f"PROOF: none of the {n_pr:,} training prompts (4 sd files) uses a held-out paraphrase, field-name variant or "
              "key order (each surface read back from the prompt TEXT), and none contains a held-out text fragment "
              f"({len(frags)} fragments scanned)", n_pr == 6000 and held_hits == 0 and frag_hits == 0, (held_hits, frag_hits))
        canon = MD.canonical_from_arm_rows(tr["dint"])
        check("the sd files keep the canonical rows and every target (literal order); the canonical files carry surface null",
              all(MD.canonical_from_arm_rows(trs[a]) == canon and [r["target"] for r in trs[a]] == [r["target"] for r in tr[a]]
                  for a in C.ARMS) and all(r["surface"] is None for a in C.ARMS for r in tr[a]))
        st_ = au["surface"]["stats"]
        mp = {}
        for r in trs["dint"]:
            if r["pair_id"]:
                mp.setdefault(r["pair_id"], []).append(r["surface"])
        check("diversity used: all 16 paraphrase x field-name combinations, the canonical event order on ~1/4 of the rows, "
              ">= 500 distinct event orders, the top-level order always canonical; pair members share one surface; every "
              "arm uses the same per-row surface",
              st_["paraphrase_x_names_combinations"] == 16 and 0.2 <= st_["event_order_canonical_share"] <= 0.32 and
              st_["distinct_event_orders"] >= 500 and st_["top_orders"] == {"before,events": 1500} and
              all(v[0] == v[1] for v in mp.values()) and
              all([r["surface"] for r in trs[a]] == [r["surface"] for r in trs["dint"]] for a in C.ARMS),
              {k: st_[k] for k in ("event_order_canonical_share", "distinct_event_orders")})
        pos = Counter()
        idx_ok = True
        for r in dev_rows:
            idx_ok &= C.stratum_index(r) == pos[r["stratum"]]
            pos[r["stratum"]] += 1
        rot = {s: Counter(C.stratum_index(r) % 3 for r in dev_rows if r["stratum"] == s) for s in C.DEV_STRATA}
        hs_ok = True
        for a in C.ARMS:
            for r in dev_rows:
                msgs = C.messages(a, r["before"], r["event"], r["width"], C.heldout_surface_of(r))
                x = C.read_prompt(msgs)
                hs_ok &= (x["surface"]["para"] == 4 and x["surface"]["names"] == 4 and
                          tuple(x["surface"]["order"]["top"]) == C.HELDOUT_ORDERS["top"][0] and
                          tuple(x["surface"]["order"]["event"]) == C.HELDOUT_ORDERS["event"][C.stratum_index(r) % 3] and
                          x["before"] == r["before"] and x["event"] == r["event"] and x["width"] == r["width"] and x["arm"] == a)
        check("E3 rendering: dev id suffix == index within the stratum on all 1,050 rows; the 3 held-out event orders rotate "
              "evenly within every stratum; every E3 prompt (4 arms) reads back as paraphrase 4 + names 4 + reversed top + "
              "its held-out event order, encoding the row's own state and event",
              idx_ok and hs_ok and all(max(c.values()) - min(c.values()) <= 1 and len(c) == 3 for c in rot.values()),
              {s: dict(c) for s, c in list(rot.items())[:2]})
        rot4 = Counter(j % 3 for j in range(len(dchains)))
        hc_ok = True
        for a in C.ARMS:
            for j, c in enumerate(dchains):
                sf = C.heldout_surface_of_chain(j)
                hc_ok &= sf == C.heldout_surface(j)
                for t, (stt, e) in enumerate(zip(C.chain_gold_states(c), c["events"])):
                    x = C.read_prompt(C.messages(a, stt, e, c["width"], sf))
                    hc_ok &= (x["surface"]["para"] == 4 and x["surface"]["names"] == 4 and
                              tuple(x["surface"]["order"]["top"]) == C.HELDOUT_ORDERS["top"][0] and
                              tuple(x["surface"]["order"]["event"]) == C.HELDOUT_ORDERS["event"][j % 3] and
                              x["before"] == stt and x["event"] == e and x["arm"] == a)
        check("E4 rendering (report-only held-out-surface chains): the 3 held-out event orders rotate by chain index over "
              f"the {len(dchains)} dev chains ({dict(sorted(rot4.items()))}); every step of every chain (4 arms, gold "
              "states) reads back as paraphrase 4 + names 4 + reversed top + its chain's held-out event order",
              hc_ok and len(dchains) == 240 and sorted(rot4.values()) == [80, 80, 80], dict(rot4))

        # ------------------------------------------------------------ 4 scorer + amendment 4b
        print("4 scorer self-test and amendment 4b", flush=True)
        st = A.scorer_selftest(dev_rows, dchains)
        check("scorer self-test: golds 100% in every format, heuristics at their exact floors, negatives rejected, "
              "D-int parser == env.parse_state, one constant fill on every dev gold, filler >= trace tokens", st["ok"],
              {k: f"{v['correct']}/{v['n']}" for k, v in st["golds"].items()})
        hf = lambda s: len(tok()(s, add_special_tokens=False).input_ids)   # noqa: E731
        items = [(r["before"], r["event"]) for r in canon + dev_rows]
        for c in dchains:
            stt = C.chain_gold_states(c)
            items += [(stt[t], e) for t, e in enumerate(c["events"])]
        k_hf = C.fill_k_required(items, hf)
        short = sum(hf(C.target_text("ffill", b, e, 12)) < hf(C.trace_text(b, e)) for b, e in items)
        same_cnt = sum(hf(r["target"]) != cnt(r["target"]) for a in C.ARMS for r in trs[a][::10])
        check(f"amendment 4b under the HF tokenizer: FILL_K {C.FILL_K} == the formula over {len(items):,} train+dev items "
              f"(longest trace head - base); no filler target shorter than its trace; HF counts == tokenizers-library counts",
              k_hf == C.FILL_K and short == 0 and same_cnt == 0, (k_hf, short, same_cnt))
        fills = {r["target"][:r["target"].index('","after":')] for rr in (tr["ffill"], trs["ffill"]) for r in rr}
        fills |= {C.target_text("ffill", r["before"], r["event"], r["width"])[:-len(',"after":' + str(r["gold"]) + "}") - 1]
                  for r in dev_rows}
        info = MD.filler_information(trs["ffill"], cnt)
        check("FILL LENGTH CARRIES ZERO INFORMATION: one fill string on all 3,000 training targets (canonical + sd) and all "
              "1,050 dev golds; predicting applicability / change / after / type from the fill length == the majority "
              "share; mutual information 0 bits",
              len(fills) == 1 and info["zero_information"] and
              all(v == 0 for v in info["mutual_information_bits"].values()),
              (len(fills), info["fill_length_lookup_accuracy"], info["mutual_information_bits"]))
        rev2_rows = C.load_jsonl(MD.REV2_FILLER)
        rev2_ok = len(rev2_rows) == 1500 and all(
            MD.fill_units_of(r["target"]) == MD.rev2_fill_units(r["before"], r["event"], cnt) == cnt(C.trace_head(r["before"], r["event"])) - 3
            and cnt(r["target"]) == cnt(C.trace_text(r["before"], r["event"])) for r in rev2_rows)
        check("revision-2 fill length formula: rev2_fill_units (trace-head tokens - 3) reproduces the fill of every row of "
              "the kept revision-2 row-matched file (1,500 rows), whose whole target has the trace target's token count",
              rev2_ok and cnt(C.FILL_BASE_TEXT) == 3)
        rv = info["rev2_row_matched_length_would_reveal_descriptive"]
        lk = {"train": rv, "dev_8bit": MD.rev2_length_leak([r for r in dev_rows if r["width"] == C.WIDTH_ID], cnt),
              "dev_oor": MD.rev2_length_leak([r for r in dev_rows if r["width"] != C.WIDTH_ID], cnt)}
        quoted = {"train": (0.749, 0.500), "dev_8bit": (0.837, 0.600), "dev_oor": (0.813, 0.667)}
        quoted_ok = all(round(lk[k]["applicable_lookup_accuracy"], 3) == q[0] and round(lk[k]["applicable_majority_share"], 3) == q[1]
                        for k, q in quoted.items())
        rule_quotes = all(s_ in A.rule_text() for s_ in ("0.749", "0.500", "0.837", "0.813", "0.600", "0.667")) and \
            all(s_ in A.RULE_PREFACE for s_ in ("0.749", "0.837", "0.813")) and "perfectly" not in A.rule_text().split(
                "Revision 4 (")[0]
        check("descriptive contrast: the revision-2 row-matched FILL length would have revealed applicability well above "
              "the majority share (0.749 vs 0.500 on the training rows, 0.837 vs 0.600 on the 8-bit and 0.813 vs 0.667 on "
              "the out-of-range dev rows), but not perfectly; the figures quoted in the rule text and its preface equal "
              "the recomputed ones; the 4b residual is >= 0 on every row",
              quoted_ok and rule_quotes and all(0.6 < lk[k]["applicable_lookup_accuracy"] < 0.95 for k in lk) and
              info["residual_tokens_descriptive"]["min"] >= 0,
              ({k: (v["applicable_lookup_accuracy"], v["applicable_majority_share"]) for k, v in lk.items()},
               info["residual_tokens_descriptive"]))

        # ------------------------------------------------------------ 5 analysis dry run
        print("5 analysis dry run", flush=True)
        dr = A.dry_run(dev_rows, dchains)
        check(f"analysis dry run: {dr['counts']['scenarios']} synthetic scenarios, {dr['counts']['report_only_units']} "
              f"report-only units (E4 never changes the rule outcome), {dr['counts']['rule_units']} rule units, "
              f"{dr['counts']['close_call_units']} close-call units, {dr['counts']['filler_units']} R5 addendum units and "
              f"{dr['counts']['refusals']} refusals give the pre-stated results",
              dr["ok"] and dr["counts"]["passed"] == dr["counts"]["total"], dr["counts"])
        rev3 = {k: v for k, v in dr["scenarios"].items() if k.startswith("rev3: surface-locked")}
        check("the held-out-surface gate changes the outcome: in both surface-locked scenarios S-trace has the larger R "
              "under revision 2's formula (no HS) but B-bits wins under revision 3",
              len(rev3) == 2 and all(v["R_without_HS_revision2_formula"]["strace"] > v["R_without_HS_revision2_formula"]["bbits"]
                                     and v["got"] == "WINNER=bbits" for v in rev3.values()),
              {k[:40]: (v["R_without_HS_revision2_formula"], v["R"]) for k, v in rev3.items()})

        # ------------------------------------------------------------ 6 full tiny run
        print("6 full tiny run (surface-diversity files, stages T/E1/E2/E3)", flush=True)
        anchor_chains = R.pilot_anchor_chains()[:8]
        anchor_ref = make_anchor_ref(tmp, anchor_chains)
        rule_dir = tmp / "rule"
        rule_dir.mkdir()
        (rule_dir / C.SCREEN_RULE_MD).write_text(A.rule_md(), encoding="utf-8", newline="\n")
        rule_rec = {"record": "T2BP_SCREEN_RULE", "revision": 0, "inputs": C.frozen_inputs(),
                    "rule_md_sha256": sha(rule_dir / C.SCREEN_RULE_MD),
                    "rule_text_sha256": hashlib.sha256(A.rule_text().encode("utf-8")).hexdigest()}
        rule_path = rule_dir / C.SCREEN_RULE_JSON
        rule_path.write_text(json.dumps(rule_rec), encoding="utf-8", newline="\n")

        def cfg_for(name, **kw):
            base = dict(tag=name, out=tmp / "runs" / name, steps=STEPS, lr=LR_TINY, anchor_chains=anchor_chains,
                        anchor_ref=anchor_ref, anchor_ref_sha=None, dev_subset=(2, 2), rule_record=rule_path)
            base.update(kw)
            return R.Cfg(**base)

        seen = []
        real_gen = R.gen_texts

        def spy_gen(model, tok_, message_lists):
            seen.extend((side_of(model), m_) for m_ in message_lists)
            return real_gen(model, tok_, message_lists)
        R.gen_texts = spy_gen
        rt = FakeRuntime(tmp)
        be = TinyBackend(rt)
        cfg = cfg_for("run_a")
        try:
            rc = R.run(cfg, be)
        finally:
            R.gen_texts = real_gen
        out = cfg.out
        check("tiny run on the sd files exits 0 and writes DONE.json", rc == 0 and (out / "DONE.json").is_file()
              and cfg.train_files == C.TRAIN_FILES_SD and cfg.data_variant == "sd", rc)
        prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        check("every arm's stages T/E1/E2/E3/E4 recorded with file hashes; E3 and E4 wrote the ON file only; E4 is "
              "recorded as report-only with its chain surface",
              R.STAGES == ("T", "E1", "E2", "E3", "E4") and
              all(set(prov["stages"][a]) == set(R.STAGES) for a in cfg.arms) and prov["data_variant"] == "sd" and
              all(list(prov["stages"][a]["E3"]["files"]) == [f"{a}/onestep_heldout_on.jsonl"] for a in cfg.arms) and
              all(list(prov["stages"][a]["E4"]["files"]) == [f"{a}/chains_heldout_on.jsonl"] for a in cfg.arms) and
              all(prov["stages"][a]["E4"].get("report_only") is True and
                  prov["stages"][a]["E4"].get("surface") == C.HELDOUT_SURFACE_CHAINS for a in cfg.arms) and
              not any((out / a / "onestep_heldout_off.jsonl").exists() or (out / a / "chains_heldout_off.jsonl").exists()
                      for a in cfg.arms))
        sub_rows, sub_ch = R.load_dev(cfg)
        key = lambda m_: json.dumps(m_, sort_keys=True)   # noqa: E731
        held = [(s_, m_) for s_, m_ in seen if C.recover_surface(m_)["names"] == C.HELDOUT_SURFACE_IDX]
        want_held = Counter()
        n_e4 = 0
        for a in cfg.arms:
            for r in sub_rows:                                              # E3: one prompt per row
                want_held[key(C.messages(a, r["before"], r["event"], r["width"], C.heldout_surface_of(r)))] += 1
            e4 = C.load_jsonl(out / a / "chains_heldout_on.jsonl")          # E4: one prompt per generated step
            for j, (c, x) in enumerate(zip(sub_ch, e4)):
                for t in range(len(x["texts"])):
                    want_held[key(C.messages(a, json.loads(x["states"][t]), c["events"][t], c["width"],
                                             C.heldout_surface_of_chain(j)))] += 1
                    n_e4 += 1
        nonheld = [m_ for _, m_ in seen if C.recover_surface(m_)["names"] != C.HELDOUT_SURFACE_IDX]
        check("held-out-surface prompts appear only in E3 and E4, adapter ON only: exactly one per row (E3) and one per "
              "generated chain step in its chain's held-out surface (E4), as a multiset; every other evaluation prompt is "
              "the canonical surface",
              all(s_ == "on" for s_, _ in held) and Counter(key(m_) for _, m_ in held) == want_held and n_e4 > 0 and
              len(held) == len(sub_rows) * len(cfg.arms) + n_e4 and
              all(C.recover_surface(m_)["para"] == 0 and C.recover_surface(m_)["names"] == 0 and
                  C.recover_surface(m_)["order"] == {o: list(C.OBJ_SLOTS[o]) for o in C.OBJ_SLOTS} for m_ in nonheld),
              (len(held), n_e4, len(nonheld)))
        leases = [c for c in rt.calls if c[0] == "lease"]
        check("one lease, label 'rcwm-t1:t2b-prereq'", leases == [("lease", R.LEASE_LABEL)], leases)
        seq = [c[0] if c[0] not in ("load", "train") else f"{c[0]}:{c[1]}" for c in rt.calls]
        order_ok = all(seq.index(f"train:{a}") < seq.index("load:sparse", seq.index(f"train:{a}")) for a in cfg.arms)
        waits_ok = all(seq[i - 1] == "wait" for i, s in enumerate(seq) if s.startswith("load") or s.startswith("train"))
        check("per arm: train, then evaluation load; a resource wait before every training and load; one anchor per turn",
              order_ok and waits_ok and len(prov["turns"]) == 1 and prov["turns"][0]["anchor"]["pass"], seq)
        check("anchor reproduces the base-model chain_m1 reference (states and texts 1.0)",
              prov["turns"][0]["anchor"]["identical_states"] == 1.0 and prov["turns"][0]["anchor"]["identical_texts"] == 1.0)
        outputs = A.validate_run(out, sub_rows, sub_ch, list(cfg.arms))
        diff = {a: sum(x != y for x, y in zip(outputs[a]["off"]["onestep"], outputs[a]["on"]["onestep"])) for a in cfg.arms}
        check("OFF != ON one-step texts in every arm (the trained adapter is active)", all(v > 0 for v in diff.values()), diff)
        res = A.analyse(outputs, sub_rows, sub_ch)
        check("the analysis validators accept the run (E3 and E4 included) and the rule produces a verdict with HS for "
              "every arm; K4_HS is reported for every arm outside the rule's metrics (report-only)",
              bool(res["rule"]["verdict"]) and res["rule"]["verdict"] != "NO_VERDICT" and
              all(res["rule"]["metrics"][a]["HS"] is not None for a in cfg.arms) and
              all(res["heldout_chains_report_only"]["arms"][a]["K4_HS"] is not None and
                  "K4_HS" not in res["rule"]["metrics"][a] for a in cfg.arms), res["rule"]["verdict"])
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
        check(f"oracle run (ON = the rule in each arm's format in any surface, OFF = copy) on all {len(dev_rows):,} rows "
              f"and {len(dchains)} chains: exit 0; the analysis validates every file and reads ON IAW/OOR/K4/HS/ID/WF 1.0, "
              "FC 0, FA 0; OFF FC 1.0, FA 0; verdict WINNER=dint (tie rule)",
              rc_o == 0 and res_o["rule"]["verdict"] == "WINNER=dint" and all(
                  mo[a]["IAW"] == mo[a]["OOR"] == mo[a]["K4"] == mo[a]["HS"] == mo[a]["ID"] == mo[a]["WF"] == 1.0
                  and mo[a]["FC"] == 0.0 and mo[a]["FA"] == 0.0 for a in cfg_o.arms)
              and all(v == 1.0 for v in off_fc.values()) and all(v == 0.0 for v in off_fa.values()), (res_o["rule"]["verdict"], mo, off_fc))
        hso = res_o["arms"]["strace"]["on"]["heldout_surface"]
        check("held-out-surface descriptives: per-stratum cells, OOR in the held-out surface, accuracy by held-out event "
              "order (3 orders x 8-bit/OOR), paired held-out minus in-surface contrasts (0 under the oracle)",
              hso["OOR"] == 1.0 and len(hso["by_heldout_event_order"]) == 6 and
              all(v["acc"] == 1.0 for v in hso["by_heldout_event_order"].values()) and
              all(v["diff"] == 0.0 for v in hso["paired_heldout_minus_in_surface"].values()))
        hk = res_o["heldout_chains_report_only"]["arms"]
        hco = {a: res_o["arms"][a]["on"]["heldout_surface_chains"] for a in cfg_o.arms}
        check("E4 under the oracle (all 240 chains in the held-out surface, every arm): K4_HS 1.0, K4 - K4_HS 0, no broken "
              "chain, 80 chains per held-out event order at 1.0; reported outside the rule's metrics",
              all(hk[a]["K4_HS"] == 1.0 and hk[a]["K4"] == 1.0 and hk[a]["K4_minus_K4_HS"] == 0.0 for a in cfg_o.arms) and
              all(v["broken_by_k4"] == 0.0 and [x["n"] for x in v["k4_by_heldout_event_order"].values()] == [80, 80, 80]
                  and all(x["acc"] == 1.0 for x in v["k4_by_heldout_event_order"].values()) for v in hco.values()) and
              all("K4_HS" not in mo[a] for a in cfg_o.arms), {a: hk[a] for a in cfg_o.arms})
        md = A.render_md(res_o)
        check("the markdown rendering carries the verdict, HS, every stratum (in-surface and held-out) and the report-only "
              "E4 table (K4 vs K4_HS per arm)",
              "WINNER=dint" in md and "| HS |" in md and md.count("| D_int_add_wrong |") == 2 and all(s_ in md for s_ in C.DEV_STRATA)
              and "| Arm | K4 (in-surface) | K4_HS (held-out surface) |" in md and "REPORT-ONLY" in md)
        one_arm = A.analyse({"bbits": outputs_o["bbits"]}, dev_rows, dchains, st)
        check("a single-arm (seed 1/2) run is analysed descriptively with NO_VERDICT", one_arm["rule"]["verdict"] == "NO_VERDICT")
        mech = {a: {k: res_o["arms"][a]["on"]["onestep"]["strata"]["D_int_add_wrong"].get(k) for k in ("cmp_ok", "calc_ok", "flag_ok")}
                for a in cfg_o.arms}
        check("oracle run: mechanism scores 1.0 (S-trace cmp_ok/calc_ok, B-bits flag_ok)",
              mech["strace"]["cmp_ok"] == 1.0 and mech["strace"]["calc_ok"] == 1.0 and mech["bbits"]["flag_ok"] == 1.0, mech)
        rc_sm = A.screen_main(cfg_o.out, dev_rows, dchains, st)
        scr = json.loads((cfg_o.out / "analysis" / "T2BP_SCREEN.json").read_text(encoding="utf-8"))
        check("analysis CLI path on the oracle run: T2BP_SCREEN.json/.md written (verdict WINNER=dint, rule revision 4), a "
              "second call refuses (create-only, exit 3)", rc_sm == 0 and scr["verdict"] == "WINNER=dint" and
              scr["rule_revision"] == 4 and (cfg_o.out / "analysis" / "T2BP_SCREEN.md").is_file() and
              A.screen_main(cfg_o.out, dev_rows, dchains, st) == 3)
        generation.generate_texts = oracle_generate
        try:
            cfg_f1 = cfg_for("run_ffill", anchor_ref=ref_o, dev_subset=None, arms=("ffill",))
            rc_f1 = R.run(cfg_f1, TinyBackend(FakeRuntime(tmp)))
        finally:
            generation.generate_texts = REAL_GENERATE
        fake_screen = tmp / "runs" / "screen_strace"
        (fake_screen / "analysis").mkdir(parents=True)
        shutil.copy(cfg_o.out / "provenance.json", fake_screen / "provenance.json")
        (fake_screen / "analysis" / "T2BP_SCREEN.json").write_text(json.dumps(
            {"verdict": "WINNER=strace", "rule": {"verdict": "WINNER=strace", "R": {"dint": .5, "strace": .9, "bbits": .5},
                                                  "metrics": {a: {} for a in C.SCREEN_ARMS}}}), encoding="utf-8", newline="\n")
        rc_ref = A.filler_main(cfg_f1.out, cfg_o.out, dev_rows, dchains, st)       # the screen named D-int: refuse
        written_on_refusal = (cfg_f1.out / "analysis" / "T2BP_FILLER.json").exists()
        rc_fm = A.filler_main(cfg_f1.out, fake_screen, dev_rows, dchains, st)
        fil = json.loads((cfg_f1.out / "analysis" / "T2BP_FILLER.json").read_text(encoding="utf-8"))
        fres = fil["descriptive"].get("filler_residual_covariate_descriptive") or {}
        check("R5 addendum path: a separate oracle filler run (sd, with E3; R 1.0 >= 0.9 - 0.05) after an S-trace screen "
              "flags TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE; after the D-int screen it refuses (exit 3); the filler's "
              "residual length is reported as a covariate",
              rc_f1 == 0 and rc_fm == 0 and fil["addendum"]["flags"] == [A.FLAG_FILLER] and
              fil["filler_metrics_on"]["HS"] == 1.0 and rc_ref == 3 and not written_on_refusal and
              fres.get("available") is True and fres["residual_tokens"]["min"] >= 0,
              (rc_f1, rc_ref, rc_fm, fil["addendum"]["filler_control"], fres.get("residual_tokens")))

        # ------------------------------------------------------------ 6c frozen rule, C1, resume code check, DONE arms
        print("6c frozen rule and C1", flush=True)
        bad_rec = dict(rule_rec, inputs=dict(rule_rec["inputs"], **{"scripts/t2bp_run.py": "0" * 64}))
        bad_path = rule_dir / "SCREEN_RULE_bad.json"
        bad_path.write_text(json.dumps(bad_rec), encoding="utf-8", newline="\n")
        rt_z = FakeRuntime(tmp)
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_z", rule_record=bad_path), TinyBackend(rt_z)), "frozen screen rule")
        check("a frozen input that differs from SCREEN_RULE.json refuses before the lease (not persisted, no lease, no output)",
              ok and pers is False and not rt_z.calls and not (tmp / "runs" / "run_z" / "provenance.json").exists(), msg)
        bad_rec2 = dict(rule_rec, inputs=dict(rule_rec["inputs"], **{"data/t2bp_train_sd_strace.jsonl": "0" * 64}))
        bad_path2 = rule_dir / "SCREEN_RULE_bad2.json"
        bad_path2.write_text(json.dumps(bad_rec2), encoding="utf-8", newline="\n")
        ok2, msg2, pers2 = raises_stop(lambda: R.run(cfg_for("run_z", rule_record=bad_path2), TinyBackend(rt_z)), "frozen screen rule")
        check("a surface-diversity training file that differs from SCREEN_RULE.json refuses before the lease",
              ok2 and pers2 is False and not rt_z.calls, msg2)
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_z", rule_record=rule_dir / "missing.json"), TinyBackend(rt_z)),
                                    "not a screen-rule record")
        check("a missing screen-rule record refuses before the lease (not persisted)", ok and pers is False and not rt_z.calls, msg)
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_canon", data_variant="", train_files=dict(C.TRAIN_FILES),
                                                          train_source=C.TRAIN_SOURCE), TinyBackend(rt_z)), "canonical-only")
        check("C1: a canonical-variant run is refused before the lease (not persisted, no lease, no output)",
              ok and pers is False and not rt_z.calls and not (tmp / "runs" / "run_canon").exists(), msg)
        fake_canon = tmp / "runs" / "canon_copy"
        fake_canon.mkdir(parents=True)
        pc = json.loads((cfg_o.out / "provenance.json").read_text(encoding="utf-8"))
        pc["data_variant"] = ""
        (fake_canon / "provenance.json").write_text(json.dumps(pc), encoding="utf-8", newline="\n")
        (fake_canon / "DONE.json").write_text("{}", encoding="utf-8", newline="\n")
        try:
            A.validate_run(fake_canon, dev_rows, dchains, list(C.SCREEN_ARMS))
            refused = False
        except A.Refuse:
            refused = True
        check("C1: the analysis refuses to score a run that is not the surface-diversity build", refused)
        tam = tmp / "runs" / "e4_tamper"
        shutil.copytree(cfg_o.out, tam, ignore=shutil.ignore_patterns("train", "analysis"))
        f4 = tam / "dint" / "chains_heldout_on.jsonl"
        rows4 = C.load_jsonl(f4)
        rows4[0]["states"][1] = env.dumps(json.loads(rows4[0]["states"][1]) ^ 1)    # a state that its text does not give
        f4.write_text("".join(json.dumps(x) + "\n" for x in rows4), encoding="utf-8", newline="\n")
        pt = json.loads((tam / "provenance.json").read_text(encoding="utf-8"))
        pt["stages"]["dint"]["E4"]["files"]["dint/chains_heldout_on.jsonl"] = sha(f4)
        (tam / "provenance.json").write_text(json.dumps(pt), encoding="utf-8", newline="\n")
        try:
            A.validate_run(tam, dev_rows, dchains, list(cfg_o.arms))
            refused4, why4 = False, ""
        except A.Refuse as e_:
            refused4, why4 = True, str(e_)
        os.replace(f4, tam / "dint" / "moved_aside.jsonl")
        try:
            A.validate_run(tam, dev_rows, dchains, list(cfg_o.arms))
            refused4b = False
        except A.Refuse:
            refused4b = True
        check("the analysis validates E4 like E2: a held-out chain file whose recorded states differ from parse(texts) is "
              "refused (hash re-recorded), and a complete run without its E4 file is refused",
              refused4 and "held-out-surface chains (E4)" in why4 and "states != parse(texts)" in why4 and refused4b, why4)
        check("the analysis' frozen check: [] on the record, problems on a changed input hash and on a changed SCREEN_RULE.md",
              A.check_frozen(rule_path) == [] and A.check_frozen(bad_path) != [] and
              (lambda: ((rule_dir / C.SCREEN_RULE_MD).write_text("x", encoding="utf-8", newline="\n"), A.check_frozen(rule_path))[1])() != [])
        (rule_dir / C.SCREEN_RULE_MD).write_text(A.rule_md(), encoding="utf-8", newline="\n")
        check("restored SCREEN_RULE.md verifies again", A.check_frozen(rule_path) == [])
        rt_h = FakeRuntime(tmp)
        cfg_h = cfg_for("run_h", cap_h=0.5)
        rc_h = R.run(cfg_h, TinyBackend(rt_h))
        ph = cfg_h.out / "provenance.json"
        prov_h = json.loads(ph.read_text(encoding="utf-8"))
        prov_h["code_sha256"]["rcwmt/generation.py"] = "f" * 64
        ph.write_text(json.dumps(prov_h), encoding="utf-8", newline="\n")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_for("run_h"), TinyBackend(rt_h)), "code changed since the run's first turn")
        check("a resume under code that differs from the run's first turn refuses (not persisted, no second lease)",
              rc_h == 5 and ok and pers is False and [c[0] for c in rt_h.calls].count("lease") == 1, msg)
        n = len(rt.calls)
        cfg_x = cfg_for("run_a", arms=C.SCREEN_ARMS + ("ffill",))
        ok, msg, pers = raises_stop(lambda: R.run(cfg_x, be), "never extended")
        check("--with-filler after a completed 3-arm run is an explicit stop (not persisted, no lease, files unchanged)",
              ok and pers is False and len(rt.calls) == n and tree_hashes(out, rels) == h_a
              and not (out / "STOPPED.json").exists(), msg)

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
                (Path(out_dir) / "log.txt").write_text("half a training\n", encoding="utf-8", newline="\n")
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

        def crashing_chain(model, tok_, arm, chains, **kw):
            state["chain"] += 1
            if arm == "strace" and state["chain"] == 2:            # inside strace E2, ON side
                (cfg_b.out / "strace" / "chains_off.jsonl.partial").write_text("half", encoding="utf-8", newline="\n")
                raise RuntimeError("simulated crash in E2")
            return REAL_CHAIN(model, tok_, arm, chains, **kw)
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
        check("every output of the crashed-and-resumed run equals the uninterrupted run byte for byte (adapters, E3 and E4 "
              "included)", tree_hashes(cfg_b.out, rels) == h_a and any(r_.endswith("chains_heldout_on.jsonl") for r_ in rels))

        # ------------------------------------------------------------ 8 deferral
        print("8 deferral", flush=True)
        rt_c = FakeRuntime(tmp)
        be_c = TinyBackend(rt_c)
        cfg_c = cfg_for("run_c", cap_h=0.5)
        rc_c = R.run(cfg_c, be_c)
        prov_c = json.loads((cfg_c.out / "provenance.json").read_text(encoding="utf-8"))
        check("nothing fits (cap 0.5 h = the stage margin): exit 5, every stage deferred (3 arms x T/E1/E2/E3/E4), no "
              "training, no load", rc_c == 5 and not [c for c in rt_c.calls if c[0] in ("load", "train")] and
              len(prov_c["turns"][0]["deferred"]) == 15, prov_c["turns"][0]["deferred"])
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
              rc_c2 == 5 and prov_c["turns"][-1]["stages_run"] == ["dint:T", "dint:E1", "dint:E2", "dint:E3", "dint:E4"]
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
        check("stage-level deferral: S-trace trained and E1 done, E2 deferred with E3, E4 and B-bits; exit 5",
              rc_c3 == 5 and prov_c["turns"][-1]["stages_run"] == ["strace:T", "strace:E1"]
              and prov_c["turns"][-1]["deferred"][:3] == ["strace:E2", "strace:E3", "strace:E4"], prov_c["turns"][-1])
        rc_c4 = R.run(cfg_c2, be_c)
        prov_c = json.loads((cfg_c.out / "provenance.json").read_text(encoding="utf-8"))
        check("the next call resumes (re-anchors, S-trace E2/E3/E4 then B-bits) and completes; outputs == run 6 byte for byte",
              rc_c4 == 0 and prov_c["turns"][-1]["stages_run"][:3] == ["strace:E2", "strace:E3", "strace:E4"]
              and "anchor" in prov_c["turns"][-1] and tree_hashes(cfg_c.out, rels) == h_a, prov_c["turns"][-1]["stages_run"])

        # ------------------------------------------------------------ 9 stops
        print("9 stops", flush=True)
        rt_d = FakeRuntime(tmp)
        be_d = TinyBackend(rt_d)
        cfg_d = cfg_for("run_d")
        rt_d.WAITING["rcwm"].write_text(json.dumps({"pid": os.getppid(), "label": "rcwm-t1:other", "since": time.time()}), encoding="utf-8", newline="\n")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "one-waiter")
        check("a live RCWM waiting marker refuses (not persisted, no lease)", ok and pers is False and not rt_d.calls, msg)
        rt_d.WAITING["rcwm"].write_text(json.dumps({"pid": 999999, "label": "rcwm-t1:dead", "since": time.time()}), encoding="utf-8", newline="\n")
        check("a dead RCWM marker does not refuse", R.one_waiter_problem(be_d) is None)
        rt_d.WAITING["rcwm"].unlink()
        rt_d.LOCK_PATH.write_text(json.dumps({"label": "bwm:bundle", "pid": os.getppid(), "created": time.time()}), encoding="utf-8", newline="\n")
        check("a live BWM lease holder does not refuse", R.one_waiter_problem(be_d) is None)
        rt_d.LOCK_PATH.write_text(json.dumps({"label": "rcwm-t1:prereg3", "pid": os.getppid(), "created": time.time()}), encoding="utf-8", newline="\n")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "holds the GPU lease")
        check("a live RCWM lease holder refuses (not persisted)", ok and pers is False, msg)
        rt_d.LOCK_PATH.unlink()
        cfg_d.out.mkdir(parents=True, exist_ok=True)
        import psutil
        (cfg_d.out / ".run.lock").write_text(json.dumps({"pid": os.getppid(), "create_time": psutil.Process(os.getppid()).create_time()}), encoding="utf-8", newline="\n")
        ok, msg, pers = raises_stop(lambda: R.run(cfg_d, be_d), "run lock")
        check("a live run lock refuses (not persisted)", ok and pers is False, msg)
        (cfg_d.out / ".run.lock").write_text(json.dumps({"pid": 999999, "create_time": 1.0}), encoding="utf-8", newline="\n")
        bad_ref = tmp / "anchor_bad.jsonl"
        rows_ = C.load_jsonl(anchor_ref)
        for x in rows_:
            x["states"] = ["-1"] + x["states"][1:]           # a different start state: the states can never match
        bad_ref.write_text("".join(json.dumps(x) + "\n" for x in rows_), encoding="utf-8", newline="\n")
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
        f = cfg.out / "bbits" / "onestep_heldout_on.jsonl"
        orig_bytes = f.read_bytes()
        (cfg.out / "DONE.json").rename(cfg.out / "DONE.json.bak")
        f.write_bytes(orig_bytes + b"\n")
        ok, msg, pers = raises_stop(lambda: R.run(cfg, be), "changed after it was recorded")
        f.write_bytes(orig_bytes)
        (cfg.out / "DONE.json.bak").rename(cfg.out / "DONE.json")
        check("a recorded output whose sha256 changed stops (persisted; here the E3 held-out-surface file)", ok and pers is True, msg)
        tmpd = tmp / "data_copy"
        tmpd.mkdir()
        for a in C.SCREEN_ARMS:
            shutil.copy(C.TRAIN_FILES_SD[a], tmpd / C.TRAIN_FILES_SD[a].name)
        with (tmpd / C.TRAIN_FILES_SD["bbits"].name).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n")
        cfg_g = cfg_for("run_g", train_files={a: tmpd / C.TRAIN_FILES_SD[a].name for a in C.SCREEN_ARMS})
        ok, msg, pers = raises_stop(lambda: R.run(cfg_g, TinyBackend(FakeRuntime(tmp))), "differs from its SOURCE record")
        check("a training file that differs from its SOURCE record stops before the lease", ok, msg)

        # ------------------------------------------------------------ 10 plan
        print("10 plan", flush=True)
        pl = R.plan(verbose=False)
        tp, to = pl["planning"]["turns_screen"], pl["observed"]["turns_screen"]
        check("plan (sd, E3 and E4 ON): the 3-arm screen takes 2 turns at the planning rate (OFF of non-native frames at "
              "the 160-token cap) and 1 at the observed rates; every turn within the cap; each arm's T/E1/E2/E3/E4 in one "
              "turn; the filler control run is one turn; per-factor held-out renderings would need a third planning turn",
              len(tp) == 2 and len(to) == 1 and all(t["projected_h"] <= R.CAP_H for t in tp + to) and
              all(sum(f"{a}:T" in t["stages"] for t in tp) == 1 and
                  all([t_ for t_ in tp if f"{a}:T" in t_["stages"]][0]["stages"].count(f"{a}:{s}") == 1
                      for s in ("E1", "E2", "E3", "E4"))
                  for a in C.SCREEN_ARMS) and
              all(pl[m_]["arms"][a]["E4_s"] > 0 and pl[m_]["arms"][a]["generations"]["E4"] == 960
                  for m_ in ("planning", "observed") for a in C.ARMS) and
              len(pl["planning"]["turns_filler_control_run_if_strace_wins"]) == 1 and
              len(pl["planning"]["turns_screen_if_per_factor_heldout_added"]) == 3 and
              pl["per_factor_heldout_renderings"]["run"] is False,
              (pl["planning"]["gpu_h_screen"], pl["observed"]["gpu_h_screen"]))
        est_p = R.estimates(R.Cfg(planning=True))
        need_h = {a: round(R.G2.STAGE_SAFETY * (sum(est_p["arms"][a][s] for s in R.STAGES) + R.LOAD_S + est_p["anchor"]) / 3600, 2)
                  for a in C.ARMS}
        check("plan: every arm's whole pair (T + E1 + E2 + E3 + E4 + load + anchor) x 1.25 fits an empty 4.5 h turn at the "
              "planning rate, so no arm can be deferred for ever (the largest needs a turn of more than 3.0 h)",
              all(v <= R.CAP_H for v in need_h.values()) and max(need_h.values()) > 3.0, need_h)
        short = R.simulate_turns(R.Cfg(cap_h=3.5), C.SCREEN_ARMS)
        check("plan: a 3.5 h cap splits the screen into several turns, each arm's pair in one turn",
              len(short) >= 2 and all(any(s.endswith(":T") for s in t["stages"]) for t in short), short)
        try:
            R.simulate_turns(R.Cfg(cap_h=3.0), C.SCREEN_ARMS)
            too_small = False
        except RuntimeError:
            too_small = True
        check("plan: a 3.0 h cap is too small for the S-trace pair with E3 and E4 (the simulator refuses instead of looping)",
              too_small)
        gp = json.loads((C.DOCS / "GEN_PLAN.json").read_text(encoding="utf-8"))
        check("docs/t2b/prereq/GEN_PLAN.json was written by --plan (sd, stages T/E1/E2/E3/E4) without importing torch",
              gp.get("torch_imported") is False and gp.get("data_variant") == "sd" and gp.get("stages_per_arm") == list(R.STAGES)
              and gp.get("code_sha256", {}).get("scripts/t2bp_run.py") == sha(ROOT / "scripts" / "t2bp_run.py"))
    finally:
        if not KEEP:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = all(r["ok"] for r in RESULTS)
    rec = {"ok": ok, "n_checks": len(RESULTS), "passed": sum(r["ok"] for r in RESULTS), "checks": RESULTS,
           "seconds": round(time.time() - t0, 1), "peak_rss_mb": _peak_mb(), "threads": torch.get_num_threads(),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
           "code_sha256": {r: sha(ROOT / r) for r in ("scripts/t2bp_run.py", "scripts/t2bp_common.py", "scripts/t2bp_analysis.py",
                                                     "scripts/t2bp_make_data.py", "scripts/t2bp_audit.py", "scripts/t2bp_test.py")},
           "data_sha256": {p.name: sha(p) for p in list(C.TRAIN_FILES.values()) + list(C.TRAIN_FILES_SD.values())
                           + [C.DEV, C.TRAIN_SOURCE, C.TRAIN_SOURCE_SD]}}
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
