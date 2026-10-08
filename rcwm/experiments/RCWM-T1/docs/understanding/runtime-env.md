# Runtime report: loading Rineya Titan checkpoints on this machine (read-only; no model weights loaded)

## 1. Recommendation

**Primary venv:** `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`
- Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, CUDA available, BF16 supported.
- This is the runtime recorded in the Titan programme's own receipts:
  - 59 result JSONs in `ChatBubble-titan-clare-sparse-v1/experiments` record `"torch": "2.6.0+cu124"`; 55 record `"transformers": "5.14.1"`.
  - 11 identical `runtime` blocks read `{"python":"3.11.15","torch":"2.6.0+cu124","transformers":"5.14.1","device":"NVIDIA GeForce RTX 4070 Laptop GPU","dtype":"float32","tf32":false,"use_cache":false}`.
  - `experiments/interference_mechanism_v1/RUNBOOK_GPU.md` names it explicitly: "This is the development environment. Do NOT use `.venv-rab001-cu126`: it has no `accelerate`."
- Its one gap: **no scikit-learn**.

**Secondary, fully equipped venv:** `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`
- Python 3.11.15, torch 2.12.1+cu126, transformers 5.14.1. Has sklearn 1.9.0, scipy 1.17.1, peft 0.19.1 and accelerate 1.14.0.
- It is the most-referenced interpreter in the harness code: 1465 path hits, against 219 for hermes.
- Use it when an analysis needs sklearn (linear probes, etc.).
- It has the same transformers version and gives a byte-identical chat-template render. The torch version differs (2.12 vs 2.6), so FP32 kernels may differ by ULPs from the recorded receipts.

**Rule:** if a new number will be compared head-to-head with a recorded Titan receipt (e.g. CONFIRM-002 430/864, 183→250/336), use hermes. Otherwise either venv works; pick one per experiment and record it.

## 2. Candidate venvs (probed with `import` only, no model load)

| venv (python.exe under `Scripts/`) | py | torch (CUDA build) | cuda_available | transformers | safetensors | accelerate | peft | numpy | sklearn | scipy | tokenizers | hf_hub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv` (on PATH) | 3.11.15 | 2.6.0+cu124 (12.4) | True, bf16 True | 5.14.1 | 0.8.0 | 1.14.0 | 0.20.0 | 2.4.6 | **MISSING** | 1.17.1 | 0.22.2 | 1.27.0 |
| `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv` | 3.11.15 | 2.12.1+cu126 (12.6) | True, bf16 True | 5.14.1 | 0.8.0 | 1.14.0 | 0.19.1 | 2.4.6 | 1.9.0 | 1.17.1 | 0.22.2 | 1.24.0 |
| `C:/Users/clclc/BusinessProject/.venv-mq001` | 3.11.15 | 2.10.0+cu126 | True | 4.57.1 | 0.7.0 | MISSING | MISSING | 2.4.6 | MISSING | MISSING | 0.22.2 | 0.36.2 |
| `C:/Users/clclc/BusinessProject/rineya-quality-pilot-v1/native-runtime-cu126-v1` | 3.11.15 | 2.10.0+cu126 | True | 4.57.1 | 0.7.0 | MISSING | MISSING | 2.4.6 | MISSING | 1.17.1 | 0.22.2 | 0.36.2 |
| `C:/Users/clclc/BusinessProject/SupaExperiment/.venv-cuda` | 3.11.15 | 2.11.0+cu128 | True | 5.16.1 | 0.8.0 | MISSING | MISSING | 2.4.6 | MISSING | MISSING | 0.23.1 | 1.29.0 |
| `C:/Users/clclc/BusinessProject/SupaExperiment-nonrecurrent-adaptation-v1/.venv` | 3.11.15 | 2.14.0+cu130 | **False** (cudaErrorNotSupported; driver 566.07 = CUDA 12.7 is too old for cu130) | 5.14.1 | 0.8.0 | MISSING | MISSING | 2.4.6 | MISSING | MISSING | 0.22.2 | 1.30.0 |

Two other venvs are referenced but unusable:
- `C:/Users/clclc/BusinessProject/.venv-rab001-cu126` is referenced in `ChatBubble-titan-tangent-task-arithmetic-v1/experiments/delta_multi_reasoning_v1/build/run_battery.sh:10` and `inference/run_matrix.sh:22` (as `PY_RENDER`; `PY_GEN` there is Chat-Bubble/.venv). It **no longer exists on disk**, and the runbook says it lacked accelerate anyway.
- `titan_tangent_arithmetic_v1/VALIDATION.json` records python 3.13.5 / torch 2.10.0+cpu, a CPU-only validation host that is not present.

Not probed: `LiveSearching/.venv` and `WayVoice/.venv`, which are not referenced by the Titan harness.

None of the venvs has transformers 4.51.3, the version behind the published long-context numbers (`matched_conditions.transformers = 4.51.3`, digest `a329eb7bb0`). That suite is SPENT anyway, so you cannot reproduce those numbers bit-exactly here.

## 3. Machine resources (measured 2026-10-05 ~22:37–22:45)

**GPU (`nvidia-smi`):** RTX 4070 Laptop, driver 566.07, CUDA 12.7, WDDM. 8188 MiB total.
- Used 1393 MiB, free ~6795 MiB on the first read; used 1630, free 6319 MiB on the second.
- No compute processes hold the card: everything listed is C+G (Unity Editor ×2, Chrome, Edge, ChatGPT/Codex, Claude).

**Host RAM (WMI):** 31.22 GiB total. Free physical memory swung from 8.26 GiB to **3.82 GiB** within minutes. Free commit is 9.9 GiB. RAM is the binding resource.

**Disk:** C: has 33.1 GiB free.

**GPU lease:** `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock` is **absent**, so no session holds the lease. A concurrent Rineya session is alive, though: hermes-venv python running `ChatBubble-rineya-synth-centralized-reasoning-v1/experiments/rineya_synth_centralized_reasoning_v1/observer_…`. Take the lease before any GPU job:
- Implementation: `ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/gpu_lease.py`, `LOCK = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")`, `ORPHAN_AFTER_SECONDS = 900`. It also scans for foreign python using more than 512 MiB of RSS.

**Programme resource floors** (`experiments/clare_sparse_composition_v1/clare/resources.py`):

| Floor | Value |
|---|---|
| `HOST_FLOOR_PROBE_GIB` | 3.0 |
| `HOST_FLOOR_TRAIN_GIB` | 3.6 |
| `HOST_GUARD_GIB` (between updates) | 1.5 |
| `VRAM_FLOOR_GIB` (free before scoring) | 3.5 |
| `DISK_FLOOR_GIB` | 4.0 |

Current free RAM (3.82) clears the probe floor but barely clears the train floor.

## 4. Checkpoints (safetensors headers parsed directly, no tensors read)

All are `Qwen3ForCausalLM` with: 28 layers, hidden 1024, 16 Q heads / 8 KV heads, head_dim 128, intermediate 3072, vocab 151936, rope_theta 1e6, max_pos 40960, `tie_word_embeddings: true`. None has an `auto_map`.

| Dir under `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/` | file bytes | tensors / dtype | params | config `torch_dtype` | weight SHA256 (from CORE.json) |
|---|---|---|---|---|---|
| `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze` | 2,384,234,992 | 310 F32 (meta `sparse001_keep=0.25`) | 596,049,920 | float32 | a28629c9…1284 |
| `Rineya-Titan-Experimental-1.0.0-devfreeze` | 1,192,135,064 | 310 BF16 | 596,049,920 | bfloat16 | 537201676…31b7 |
| `Rineya-Titan-Transition-1.0.0-devfreeze` (= Sparse + 0.25·P16(δ_H), CONFIRM-002 TH25) | 2,384,235,008 | 310 F32 (meta `experiment=CONFIRM-002, arm=TH25`) | 596,049,920 | float32 | a4d73353…9567 |
| `Rineya-Supa-1.7.0-devfreeze` (base for all deltas) | 1,192,135,096 | 310 BF16 | 596,049,920 | — | 962abc5b…d24b |

- None of the files contains `lm_head.weight`; it is tied to `model.embed_tokens.weight` (151936×1024 = 155.6M params, 26% of the total).
- Parameters per decoder layer = 15,730,944 (attention 6,291,456; MLP 9,437,184).
- Titan-Transition's `config.json` is 756 bytes against 726 for the others. The content is identical; the difference is line endings only.
- tokenizer_config.json and tokenizer.json are byte-identical across all three Titans (md5 b06e103a… and 6423133b…).

**FP32 must not be downcast.** The Sparse README says rounding it to BF16 zeroes 28.49% of kept delta elements, because the update is only 0.056% of the weight norm. Titan-Experimental is stored in BF16, but every published number was measured after upcasting to FP32 ("Native BF16 serving is not qualified"). In transformers 5.14.1, `from_pretrained` defaults `dtype` to `"auto"` (`modeling_utils.py` around line 4247: `if dtype is None: dtype = "auto"`). Omitting `dtype` therefore loads Experimental in BF16 and Sparse/Transition in FP32. **Always pass `dtype=torch.float32`.** `torch_dtype=` still works but warns "`torch_dtype` is deprecated! Use `dtype` instead!" (line 1616).

## 5. How existing code loads models

`_family_system_runtime.py` **loads no model**. It only replays and routes saved prediction JSONLs: `route_predictions` with ops `select_if_arm_unknown` and `consensus_replace`, and `normalize()` strips "answer:", articles and punctuation. It verifies `canonical_prediction_sha256`. It is irrelevant to runtime.

Harness loaders (paths under `ChatBubble-titan-clare-sparse-v1/experiments/` unless noted):

| File:line | Call |
|---|---|
| `composition_confirmation_v2/score_boundary.py:103` (CONFIRM-002 scorer) | `AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()` |
| `score_boundary.py` preamble | `torch.backends.cuda.matmul.allow_tf32=False`; `cudnn.allow_tf32=False`; `torch.cuda.set_per_process_memory_fraction(0.78, 0)` (comment: "past VRAM, Windows pages into host RAM instead of failing") |
| `score_boundary.py` generate | `do_sample=False, eos_token_id=eos` (list from generation_config), `pad_token_id=eos[-1]`, trailing EOS stripped, `decode(skip_special_tokens=True).strip()` |
| `fresh_dual_skill_instrument_v1/score_arms.py:137` | same FP32, `device_map={"": "cuda"}`, `low_cpu_mem_usage=True`, inside the `lease()` context |
| `clare_sparse_composition_v1/clare/native.py:216` (training host) | `dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True, device_map={"": device}`, then FP32 compute wrappers. Asserts tied embeddings and sets `use_cache=False`. Comment: "device_map loads tensors straight onto the device: host RAM … never holds the full model." |
| `titan_model_soup_mini_v1/collect_native.py:204` | `torch_dtype=torch.float32, device_map="auto"`; asserts all floating parameters are FP32; prompt plus generation ≤ 4096 tokens; `max_new_tokens=64` |
| `ChatBubble-titan-tangent-task-arithmetic-v1/experiments/titan_tangent_arithmetic_v1/host.py:64` (CPU) | `trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True`; `torch.use_deterministic_algorithms(True)`; `set_float32_matmul_precision("highest")`; `set_num_threads(1)`; seed 17; psutil RAM guard |

- **trust_remote_code:** never needed for Titan (stock Qwen3, no `auto_map`). `modeling_supa_a1.py` in the cores root is a separate Supa-A1 abstention-gate model ("Qwen3 with the abstention decision folded into the output logits") and does not apply to Titan.
- **attn_implementation:** default sdpa for generation and scoring; `"eager"` in training and tangent (JVP) code. Qwen3 in transformers 5.14.1 supports sdpa, flash and flex.

## 6. Generation config and chat template

`generation_config.json` is identical for all three Titans:
- `bos 151643`, `eos_token_id [151645, 151643]`, `pad 151643`.
- `do_sample: true, temperature 0.6, top_k 20, top_p 0.95` (Qwen3 thinking-mode defaults).
- **Every programme scorer overrides this with greedy `do_sample=False`.** Pass it explicitly, or the generation config will sample.

Tokenizer (`Qwen2Tokenizer`):
- `eos_token=<|im_end|>` (151645), `pad_token=<|endoftext|>` (151643), `bos_token=None`, `add_bos_token=False`. `len(tok)=151669`; the model vocab is padded to 151936.
- Added tokens: `<think>`=151667 and `</think>`=151668 (non-special, so `skip_special_tokens` does not strip them). `<tool_call>` 151657/151658 and `<tool_response>` 151665/151666.

Chat template: this is the stock Qwen3 template.
- It **includes `<think>` handling**:
  - Assistant turns after the last user query are rendered `'<|im_start|>assistant\n<think>\n' + reasoning + '\n</think>\n\n' + content`.
  - Earlier assistant turns have their `<think>…</think>` stripped, and `reasoning_content` is supported.
  - Tools are injected via `<tools>` in the system block.
- **`enable_thinking`:** only `enable_thinking is false` changes the output. It appends `'<think>\n\n</think>\n\n'` after the generation prompt. Omitting it and `enable_thinking=True` render identically, with no think block, so the model may open `<think>` itself.
- **All Titan scorers use `enable_thinking=False`**: `score_boundary.py:46`, `clare/native.py:247`, `clare/recipe.py:114`, and `titan_opallo_skill_vector_v1/run_native_probe.py:40`. `score_arms.py:20` hard-codes the equivalent string with a system turn. Training also used `assistant_response_plus_eos` under this template.

CPU-only tokenizer check, run in Chat-Bubble/.venv with CUDA_VISIBLE_DEVICES="" (the hermes venv gives identical output):

```
default / enable_thinking=True:
'<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n'
  15 tok [151644, 872, 198, 3838, 374, 220, 17, 10, 18, 30, 151645, 198, 151644, 77091, 198]
enable_thinking=False:
'<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  19 tok [..., 151644, 77091, 198, 151667, 271, 151668, 271]
```

The template inserts no default system prompt.

## 7. Memory budget (analytic)

**Weights:**
- FP32: 596,049,920 × 4 = 2.384 GB (2.22 GiB) VRAM.
- BF16 storage: 1.19 GB.
- After loading FP32 weights plus a ~0.3–0.5 GiB CUDA context, about 3.4–3.6 GiB of the current 6.17 GiB free VRAM remains. That sits right at the programme's 3.5 GiB scoring floor.

**KV cache:** 2 × 28 layers × 8 KV heads × 128 × 4 B = 224 KiB/token in FP32, or 0.875 GiB at 4096 tokens (half that in BF16). Batch-1 greedy generation up to 4K context fits.

**Logits:** 151936 × 4 B = 593.5 KiB per position.
- A teacher-forced forward on a full sequence materializes T × 0.58 MiB: 1024 tokens is 0.58 GiB; 4096 tokens is 2.32 GiB, plus softmax temporaries.
- For NLL scoring, pass `logits_to_keep`, slice hidden states to the answer span and apply `lm_head` yourself, or chunk.
- `generate()` keeps only the last position.

**Host RAM:** with `device_map={"": "cuda"}` and `low_cpu_mem_usage=True`, safetensors streams through mmap and RSS stays well under the model size. Peak host use is roughly torch/CUDA import (~0.5–1 GiB) plus transient page cache. Keep the harness guard: refuse to start below 3.0 GiB of available psutil RAM. At the current 3.82 GiB free, run **one model process at a time**, and close the Unity Editor instances if possible.

**Training feasibility** (relevant to the research direction):
- Full FP32 AdamW on the tail 20 layers is 314,618,880 trainable params × 16 B (weights, gradients, two moments) = 5.03 GB, before frozen weights and activations. **This does not fit in 8 GB VRAM.** The programme used CPU AdamW for that reason.
- Delta arithmetic (Sparse, Transition, task vectors) is cheap: FP32 delta = loaded Titan − Supa-1.7 upcast. Do it in float64 on CPU per tensor to avoid holding two 2.4 GB copies.
- LoRA via peft is feasible on GPU.

## 8. Exact loading code (recommended)

```python
# Run with: C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe
# (or C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe when sklearn is needed)
import json, gc, sys, psutil, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

CORES = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
MODEL = CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze"   # or -Transition- / -Experimental-
# Programme numerics: FP32, TF32 off, highest matmul precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
assert psutil.virtual_memory().available / 2**30 >= 3.0, "host RAM below probe floor"
torch.cuda.set_per_process_memory_fraction(0.78, 0)   # avoid WDDM paging VRAM into host RAM

# GPU lease (shared card; lock at %TEMP%/rineya-gpu-0.lock)
sys.path.insert(0, "C:/Users/clclc/BusinessProject/ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1")
from gpu_lease import lease

eos = json.loads((MODEL / "generation_config.json").read_text())["eos_token_id"]  # [151645, 151643]
tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
with lease(label="titan-probe", job="describe job"):
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        dtype=torch.float32,              # NEVER omit: v5 default "auto" keeps Titan-Experimental in BF16
        local_files_only=True,
        trust_remote_code=False,          # stock Qwen3; no auto_map
        low_cpu_mem_usage=True,
        device_map={"": "cuda"},          # stream straight to GPU; host RAM never holds full model
        attn_implementation="sdpa",       # "eager" if you need output_attentions / JVP code paths
    ).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight

    msgs = [{"role": "user", "content": "What is 2+3?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)      # programme convention
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=False, max_new_tokens=64,   # override sampling defaults
                             eos_token_id=eos, pad_token_id=eos[-1])
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    while new and new[-1] in eos: new = new[:-1]
    reply = tok.decode(new, skip_special_tokens=True).strip()
    del model; gc.collect(); torch.cuda.empty_cache()
```

For hidden-state or probe work, call `model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1)`. That returns 29 hidden-state tensors (embeddings + 28 layers), each [1, T, 1024], with only last-position logits. Move the hidden states to CPU as float32 and run sklearn in Chat-Bubble/.venv, or fit probes in torch inside hermes.

## 9. Caveats for the lead researcher

1. The README numbers are not reproducible in these venvs. The long-context 161/162/164 figures used transformers 4.51.3 / DenseReference, and that suite is SPENT. CONFIRM-002 numbers (Sparse 430/864 transition, 183/336 boundary; Transition 434 and 250) were produced in hermes with torch 2.6.0 and transformers 5.14.1, greedy decoding, FP32, thinking disabled.
2. Free RAM fluctuates by about 4.5 GiB because of other apps (Unity ×2, browsers, other agent sessions). Re-check right before launch.
3. `SupaExperiment-nonrecurrent-adaptation-v1/.venv` (cu130) cannot use the GPU with driver 566.07.
4. Scratch probe scripts are in `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad/` (`probe_env.py`, `probe_tok.py`). No project files were modified.