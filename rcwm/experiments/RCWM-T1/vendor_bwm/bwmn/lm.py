# RCWM-T1 local patch (2026-10-06): training made device-agnostic for a CPU smoke test (3 edits, marked).
# Vendored 2026-10-06T17:51:17-05:00 from C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/bwmn/lm.py (BWM-001 session, with permission; see RCWM-T1 DECISIONS D-15).
"""Native language-model utilities: load a Titan core, attach LoRA, masked SFT, batched generation.

Conventions
- FP32 weights end to end (Titan-Sparse/Transition are FP32-only; BF16, including autocast, rounds their update away).
- TF32 disabled (cuDNN defaults it ON on this card).
- Prompt and target are tokenized separately and concatenated; prompt and padding get -100; target + EOS supervised.
  Overlength rows are rejected, never truncated.
- Loss is computed only at supervised positions (no full [B, L, V] logits tensor).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
MODELS = {
    "titan-sparse": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
    "titan-transition": CORES / "Rineya-Titan-Transition-1.0.0-devfreeze",
    "titan-exp": CORES / "Rineya-Titan-Experimental-1.0.0-devfreeze",
    "supa": CORES / "Rineya-Supa-1.7.0-devfreeze",
    "qwen3-0.6b": Path(r"C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"),
}
EOS_IDS = [151645, 151643]


# The RTX 4070 Laptop also drives the display. Driver resets (nvlddmkm TDR, event 153) and two unexpected shutdowns
# coincided with heavy FP32 jobs and VRAM spilling into shared memory. Safeguards (D-20):
#  - cap PyTorch allocations (fail fast with OOM instead of paging into shared memory),
#  - synchronize + sleep briefly between batches/steps so the compositor gets GPU time,
#  - small generation batches.
MEM_FRACTION = float(__import__("os").environ.get("BWM_MEM_FRACTION", "0.70"))
YIELD_S = float(__import__("os").environ.get("BWM_YIELD_S", "0.05"))


TEMP_HIGH, TEMP_RESUME, TEMP_WARM = 82, 75, 78
PEER_FLAG = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/"
                 r"experiments/RCWM-T1/runs/.thermal_pause")  # shared watchdog flag (RCWM-T1 thermal_watchdog.py)
_thermal = {"last_check": 0.0, "last_temp": None, "last_report": 0.0}


def gpu_temp() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["C:/Windows/System32/nvidia-smi.exe", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception as exc:
        print(f"  [thermal] WARNING cannot read GPU temperature: {exc!r}", flush=True)
        return None


def thermal_guard() -> None:
    """Pause the job (GPU idle) while (a) the shared watchdog flag exists or (b) the GPU reads >= 82 C, until the
    flag is gone and the GPU reads <= 75 C (max 10 min per pause). Temperature is sampled at most every 10 s; an
    unreadable temperature is treated as hot (fail safe)."""
    if __import__("os").environ.get("BWM_DEVICE", "cuda") == "cpu":
        return  # CPU-lane jobs put no load on the GPU; pausing them on GPU heat only wastes CPU time (34 pauses observed)
    now = time.time()
    hot = PEER_FLAG.exists()
    if not hot and now - _thermal["last_check"] >= 10:
        _thermal["last_check"] = now
        t = gpu_temp()
        _thermal["last_temp"] = t
        hot = t is None or t >= TEMP_HIGH
        if now - _thermal["last_report"] >= 300:
            _thermal["last_report"] = now
            print(f"  [thermal] {time.strftime('%H:%M:%S')} GPU {t} C", flush=True)
    if not hot:
        return
    t0 = time.time()
    while time.time() - t0 < 600:
        time.sleep(5)
        t = gpu_temp()
        if not PEER_FLAG.exists() and t is not None and t <= TEMP_RESUME:
            break
    _thermal["last_temp"] = gpu_temp()
    print(f"  [thermal] {time.strftime('%H:%M:%S')} paused {time.time() - t0:.0f}s, GPU now {_thermal['last_temp']} C",
          flush=True)


DUTY_RATIO = float(__import__("os").environ.get("BWM_DUTY_RATIO", "3.0"))  # idle time = 3 x compute time (~25% duty)
_duty = {"since": None}


def cuda_ok() -> bool:
    """True only for GPU-lane processes. CPU-lane processes (BWM_DEVICE=cpu) never call into CUDA: on WDDM even
    torch.cuda.is_available() registers the process with the driver, and the shared lease then mistakes it for a
    GPU holder."""
    return __import__("os").environ.get("BWM_DEVICE", "cuda") != "cpu" and torch.cuda.is_available()


def gpu_yield(seconds: float | None = None) -> None:
    """Synchronize, then idle for max(minimum, DUTY_RATIO x the GPU time since the previous yield). This proportional
    duty cap keeps the laptop GPU's boost bursts below the temperatures at which driver resets occurred (agreed with
    the RCWM-T1 session), then applies the thermal guard."""
    if cuda_ok():
        torch.cuda.synchronize()
    now = time.time()
    busy = 0.0 if _duty["since"] is None else now - _duty["since"]
    base = YIELD_S if seconds is None else seconds
    time.sleep(max(base, 0.12 if seconds is None else base, DUTY_RATIO * busy))
    thermal_guard()
    _duty["since"] = time.time()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_ok():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_core(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = MODELS[name]
    if cuda_ok():
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, low_cpu_mem_usage=True,
                                                 device_map={"": __import__("os").environ.get("BWM_DEVICE", "cuda")})
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    assert tok.eos_token_id == 151645 and tok.pad_token_id == 151643
    return model, tok


def attach_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                adapter_name: str = "default", seed: int = 0):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets),
                     bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg, adapter_name=adapter_name)


def load_lora(model, adapter_dir: Path, adapter_name: str = "default"):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=adapter_name, is_trainable=False)


def count_trainable(model) -> tuple[int, int]:
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a


# ----------------------------------------------------------------------------- tokenization

@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_target: int


def encode_pair(tok, prompt: str, target: str, max_len: int, eos: bool = True) -> Encoded | None:
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(target, add_special_tokens=False).input_ids
    if eos:
        t = t + [tok.eos_token_id]
    ids = p + t
    if len(ids) > max_len:
        return None
    return Encoded(ids, [-100] * len(p) + t, len(t))


def collate(batch: list[Encoded], pad_id: int):
    L = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        lab[i, :n] = torch.tensor(e.labels)
        att[i, :n] = 1
    return ids, lab, att


def _decoder_and_head(model):
    """Return (decoder producing hidden states, lm_head) for plain or PEFT-wrapped Qwen3."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model  # PeftModel -> LoraModel -> Qwen3ForCausalLM
    return m.model, m.lm_head


def supervised_logits(model, ids, att, lab):
    """Hidden states for all positions, logits only where the next token is supervised."""
    dec, head = _decoder_and_head(model)
    h = dec(input_ids=ids, attention_mask=att).last_hidden_state  # [B, L, D]
    tgt = lab[:, 1:]
    mask = tgt != -100
    hs = h[:, :-1][mask]  # [N, D]
    return head(hs).float(), tgt[mask], mask


def _ce_sum_chunk(head, hs, tgt):
    return F.cross_entropy(head(hs).float(), tgt, reduction="sum")


def chunked_ce(model, ids, att, lab, chunk: int = 256):
    """Token-mean CE over supervised positions. Each chunk's [chunk, V] logits are recomputed in backward
    (checkpointed), so at most one chunk of full-vocabulary logits is alive. Mathematically identical to
    F.cross_entropy(head(h[sup]), tgt)."""
    from torch.utils.checkpoint import checkpoint

    dec, head = _decoder_and_head(model)
    h = dec(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    mask = tgt != -100
    hs = h[:, :-1][mask]
    t = tgt[mask]
    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        total = total + checkpoint(_ce_sum_chunk, head, hs[i:i + chunk], t[i:i + chunk], use_reentrant=False)
    return total / t.numel(), t.numel()


# ----------------------------------------------------------------------------- training

def wsd_lr(step: int, total: int, peak: float, warm: float = 0.1, decay: float = 0.2, floor: float = 0.1) -> float:
    w = max(1, int(total * warm))
    d0 = int(total * (1 - decay))
    if step < w:
        return peak * (step + 1) / w
    if step < d0:
        return peak
    prog = (step - d0) / max(1, total - d0)
    return peak * (1 - (1 - floor) * prog)


def train_sft(model, tok, rows: list[dict], *, steps: int, batch_size: int = 16, lr: float = 1e-4,
              max_len: int = 320, seed: int = 0, log_every: int = 50, logger=print,
              grad_clip: float = 1.0, grad_ckpt: bool = True, row_order: list[int] | None = None,
              on_step=None, micro: int = 4) -> dict:
    """Masked next-token SFT, token-mean loss over supervised positions, WSD schedule.

    row_order (optional) fixes the exact sequence of row indices so that matched arms see identical rows."""
    set_determinism(seed)
    rng = random.Random(seed)
    enc, keep = [], []
    rejected = 0
    for i, r in enumerate(rows):
        e = encode_pair(tok, r["prompt"], r["target"], max_len)
        if e is None:
            rejected += 1
        else:
            enc.append(e)
            keep.append(i)
    if rejected:
        raise ValueError(f"{rejected} overlength rows; refuse to train with silent drops")
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if row_order is None:
        row_order = []
        while len(row_order) < steps * batch_size:
            idx = list(range(len(enc)))
            rng.shuffle(idx)
            row_order.extend(idx)
    assert len(row_order) >= steps * batch_size
    model.train()
    hist, gns = [], []
    t0 = time.time()
    tokens_seen = sup_tokens = 0
    if cuda_ok():
        torch.cuda.reset_peak_memory_stats()
    for s in range(steps):
        b = [enc[row_order[s * batch_size + j]] for j in range(batch_size)]
        cur = wsd_lr(s, steps, lr)
        for g in opt.param_groups:
            g["lr"] = cur
        # Gradient accumulation over microbatches (short GPU bursts for the laptop's thermals). Each microbatch's
        # token-mean loss is weighted by its share of the batch's supervised tokens, so the gradient equals the
        # full-batch token-mean gradient up to floating-point order.
        n_total = sum(e.n_target for e in b)
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        n_sup = 0
        for m0 in range(0, batch_size, micro):
            mb = b[m0:m0 + micro]
            ids, lab, att = collate(mb, tok.pad_token_id)
            _dev = next(model.parameters()).device; ids, lab, att = ids.to(_dev), lab.to(_dev), att.to(_dev)  # RCWM-T1 local patch: device-agnostic
            mloss, mn = chunked_ce(model, ids, att, lab)
            if not torch.isfinite(mloss):
                raise FloatingPointError(f"non-finite loss at step {s}")
            (mloss * (mn / n_total)).backward()
            loss_val += float(mloss) * mn / n_total
            n_sup += mn
            tokens_seen += int(att.sum())
            if m0 + micro < batch_size:
                gpu_yield()
        loss = torch.tensor(loss_val)
        gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite grad at step {s}")
        opt.step()
        gpu_yield()
        sup_tokens += int(n_sup)
        hist.append(float(loss))
        gns.append(float(gn))
        if on_step is not None:
            on_step(s + 1, model)
        if (s + 1) % log_every == 0 or s == 0:
            k = min(log_every, len(hist))
            logger(f"step {s+1}/{steps} loss {sum(hist[-k:])/k:.4f} lr {cur:.2e} gn {float(gn):.2f} "
                   f"{time.time()-t0:.0f}s peak {(torch.cuda.max_memory_allocated() if cuda_ok() else 0)/2**30:.2f}GiB")
    model.eval()
    if grad_ckpt:
        model.gradient_checkpointing_disable()
    return {"steps": steps, "batch_size": batch_size, "rows_available": len(enc), "lr_peak": lr,
            "final_loss_avg50": sum(hist[-50:]) / len(hist[-50:]),
            "loss_curve_every50": [round(sum(hist[i:i+50]) / len(hist[i:i+50]), 5) for i in range(0, len(hist), 50)],
            "max_grad_norm": max(gns), "tokens_processed": tokens_seen, "supervised_tokens": sup_tokens,
            "wall_s": round(time.time() - t0, 1), "peak_mem_gib": (torch.cuda.max_memory_allocated() if cuda_ok() else 0) / 2**30}


# ----------------------------------------------------------------------------- generation

@torch.no_grad()
def generate(model, tok, prompts: list[str], *, max_new_tokens: int = 64, max_batch_tokens: int = 2000,
             do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, n: int = 1,
             seed: int | None = None) -> list[list[dict]]:
    """Batched left-padded generation. Returns, per prompt, a list of n completions
    {text, n_tokens, hit_eos}. Batch size adapts to keep (prompt+new)*batch*n under max_batch_tokens."""
    tok.padding_side = "left"
    out: list[list[dict]] = []
    if seed is not None:
        torch.manual_seed(seed)
        if cuda_ok():
            torch.cuda.manual_seed_all(seed)
    i = 0
    while i < len(prompts):
        plen = len(tok(prompts[i], add_special_tokens=False).input_ids)
        bs = max(1, max_batch_tokens // ((plen + max_new_tokens) * n))
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=EOS_IDS,
                  num_return_sequences=n)
        if do_sample:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        g = model.generate(**enc, **kw)
        gpu_yield()
        new = g[:, enc.input_ids.shape[1]:].tolist()
        for j in range(len(chunk)):
            comps = []
            for r in range(n):
                ids = new[j * n + r]
                cut = len(ids)
                for k, t in enumerate(ids):
                    if t in EOS_IDS:
                        cut = k + 1
                        break
                hit = cut <= len(ids) and any(t in EOS_IDS for t in ids[:cut])
                comps.append({"text": tok.decode(ids[:cut], skip_special_tokens=True), "n_tokens": cut, "hit_eos": hit})
            out.append(comps)
        i += len(chunk)
    tok.padding_side = "right"
    return out


@torch.no_grad()
def sample_crn(model, tok, prompts: list[str], seeds: list[int | None], *, max_new_tokens: int = 64,
               temperature: float = 1.0, top_p: float = 0.95, max_batch_tokens: int = 2000) -> list[dict]:
    """Batched decoding with per-sequence randomness (common random numbers).

    seeds[i] is None -> greedy; otherwise sequence i draws its own Gumbel noise from torch.Generator(seed), so its
    sample depends only on (prompt, seed), not on what else shares the batch. Top-p filtering is applied before the
    Gumbel-max draw, which is equivalent to sampling from the renormalised nucleus. Only last-position logits are
    materialised (logits_to_keep=1). max_batch_tokens budgets (prompt+new)*batch; DynamicCache grows by torch.cat, so
    the KV cache transiently exists twice (FP32: ~2 x 224 KiB per token) -- 6000 tokens keeps it near 2.7 GB."""
    from transformers import DynamicCache

    if model.device.type == "cpu":  # CPU decoding is bandwidth-bound: wide batches; no VRAM/thermal constraint
        max_batch_tokens = max(max_batch_tokens, int(__import__("os").environ.get("BWM_CPU_BATCH_TOKENS", "8000")))
    tok.padding_side = "left"
    res: list[dict | None] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    i = 0
    while i < len(order):
        plen = len(tok(prompts[order[i]], add_special_tokens=False).input_ids)
        bs = max(1, max_batch_tokens // (plen + 16 + max_new_tokens))
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        att = enc.attention_mask
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        cache = DynamicCache()
        out = model(input_ids=enc.input_ids, attention_mask=att, position_ids=pos, past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :].float()
        B = len(idx)
        gens = [None if seeds[j] is None else torch.Generator(device=model.device).manual_seed(int(seeds[j]) % (2**63 - 1))
                for j in idx]
        greedy_mask = torch.tensor([g is None for g in gens], device=model.device)
        finished = torch.zeros(B, dtype=torch.bool, device=model.device)
        toks = []
        eos = torch.tensor(EOS_IDS, device=model.device)
        last_pos = pos[:, -1:]
        for step in range(max_new_tokens):
            nxt = logits.argmax(-1)
            if not bool(greedy_mask.all()):
                lp = torch.log_softmax(logits / max(temperature, 1e-6), -1)
                if top_p < 1.0:
                    sp, si = lp.sort(-1, descending=True)
                    cum = sp.exp().cumsum(-1)
                    drop = cum - sp.exp() > top_p  # keep the smallest prefix with mass >= top_p
                    sp = sp.masked_fill(drop, float("-inf"))
                    lp = torch.full_like(lp, float("-inf")).scatter(-1, si, sp)
                noise = torch.zeros_like(lp)
                for b, g in enumerate(gens):
                    if g is not None:
                        u = torch.rand(lp.shape[-1], generator=g, device=model.device).clamp_(1e-12, 1 - 1e-12)
                        noise[b] = -torch.log(-torch.log(u))
                samp = (lp + noise).argmax(-1)
                nxt = torch.where(greedy_mask, nxt, samp)
            nxt = torch.where(finished, torch.full_like(nxt, tok.pad_token_id), nxt)
            toks.append(nxt)
            if step % 8 == 7:
                gpu_yield(0.01)
            finished = finished | torch.isin(nxt, eos)
            if bool(finished.all()):
                break
            att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=att.device)], -1)
            last_pos = last_pos + 1
            out = model(input_ids=nxt[:, None], attention_mask=att, position_ids=last_pos, past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            logits = out.logits[:, -1, :].float()
        T = torch.stack(toks, 1).tolist()
        for b, j in enumerate(idx):
            ids = T[b]
            cut = len(ids)
            hit = False
            for k, t in enumerate(ids):
                if t in EOS_IDS:
                    cut, hit = k + 1, True
                    break
            res[j] = {"text": tok.decode(ids[:cut], skip_special_tokens=True), "n_tokens": cut, "hit_eos": hit}
        del cache, out
        gpu_yield()
        i += len(idx)
    tok.padding_side = "right"
    return res


@torch.no_grad()
def score_targets(model, tok, pairs: list[tuple[str, str]], batch_size: int = 8, max_len: int = 512) -> list[float]:
    """Sum log-prob of target (incl. EOS) given prompt, computed at supervised positions only."""
    res = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = [encode_pair(tok, p, t, max_len) for p, t in chunk]
        ids, lab, att = collate(enc, tok.pad_token_id)
        ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
        logits, tgt, mask = supervised_logits(model, ids, att, lab)
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        per = torch.zeros(mask.shape, device=lp.device)
        per[mask] = lp
        res.extend(per.sum(-1).tolist())
        gpu_yield()
    return res


def save_adapter(model, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    (out_dir / "RUN_META.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
