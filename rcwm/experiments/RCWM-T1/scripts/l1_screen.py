"""PREREG-L1: promotion screen for L8-15 (RCWM-T1, DECISIONS D-26). DRAFT runner; writes NOTHING to the vault.

L8-15 = Supa 1.7 + Titan-Sparse's delta restricted to transformer blocks 8..15, stored FP32. Titan-Sparse - Supa is
exactly zero in the embeddings, the final norm and blocks 0..7 (D-21), so byte for byte every tensor of blocks 8..15
is Titan-Sparse's stored FP32 tensor and every other tensor is Supa's stored BF16 tensor widened to FP32 (exact).
It is the arm R12 called HlowT@16 and R14 called loc8_15; its per-group FP32 hashes must equal R14's record.

Design, provenance, endpoint roles and the decision rule: docs/l1/PREREG_L1_DRAFT.md. Constants marked REGISTERED
below are the preregistration's; the freeze (docs/l1/PREREG_L1_FREEZE.json) records this file's sha256, and `run` and
`analyze` refuse to start if anything frozen changed.

SUBCOMMANDS. Every one only READS the vault, the records and the ChatBubble trees.
  plan           GPU-time and disk estimate. Imports no torch, loads no model, writes nothing.
  integrity      CPU, no model: pinned input hashes, the vendored 5b4c22d code, SELECT-VIEWS-001's role and
                 floors, the long-context suite's SPENT bar, payload disjointness of SELECT-VIEWS-001 from the
                 RCWM banks that chose the cut, the fresh prompts, R14 identity facts, and the A-OOD bank (C6):
                 an independent re-check by scripts/l1_make_ood.py's verifier, including a fresh harvest of the
                 historical payload set (the reserved HSKILL_CONFIRM_V2* files are skipped by name, never opened).
  parity-cpu     CPU, no model: our scorers re-score the PUBLISHED raw predictions (Titan-Sparse s25d, Supa) and
                 must reproduce every published per-row verdict and count exactly; the frozen B28 source answers
                 re-score to 47/41; this environment's tokenizer/decoder return every published anchor text unchanged.
                 Checks on the SPENT long-context bank (its scorer, its recorded input-token counts) are report-only.
  fresh-prompts  Build docs/l1/offtask_fresh_prompts.jsonl (create-only; no model): 8 prompts per reply class
                 from the preservation slice, disjoint from the 248 R14 prompts in id AND template family, and from
                 every id and user text of the HSKILL-002 source pools (so from HSKILL-VIEWS-002 and the reserved,
                 unread HSKILL-CONFIRM-002 by construction; the CONFIRM file is never opened).
  build          Materialise <out>/arm-l8_15 (CPU, numpy byte transform, no model): pass 1 writes, pass 2 re-derives
                 the bytes into a hasher only, then the written file is re-read; gates B1-B7 below.
                 --dry-hash computes the identity without writing; --record stores it in docs/l1/L8_15_EXPECTED.json.
  freeze         Write docs/l1/PREREG_L1_FREEZE.json. Only after the adversarial review and the operator's approval.
                 Refuses if any L1 output exists anywhere under runs/ (or the given --out-dir).
  run            GPU under ONE lease (rcwm-t1:l1screen): supa, then sparse, on the pre-gate stages acq, ret, lc,
                 drift_fresh, drift_bridge; then the ANCHOR PARITY GATE (CPU, C0); then, only if it passed, the
                 post-gate work: sparse's and supa's ood stage (the 600 A-OOD rows, C6; sparse stays loaded across the
                 gate, supa is reloaded) and every l8_15 stage. Raw outputs only; resumable per stage. A failed gate
                 writes PARITY_FAIL.json: L8-15 is never loaded and no model reads the OOD bank.
  analyze        CPU: binds the run to the freeze, re-derives the parity gate, every endpoint, the REGISTERED decision
                 rule; L1_RESULTS.json / L1_RESULTS.md (create-only). After PARITY_FAIL it issues NON-COMPARABLE and
                 computes no L8-15 endpoint.

BUILD GATES (fatal; SanityError, exit 3)
  B1 input weight files hash to the published identities (Supa 962abc5b..., Titan-Sparse a28629c9...)
  B2 D-21 premise, element exact: Titan-Sparse's embeddings, final norm and blocks 0..7 equal Supa's widened bytes
  B3 pass 2 (independent re-derivation from the sources, hasher only) == pass 1; the file re-read from disk == both
  B4 per-group FP32 hashes of the file == R14 loc8_15's recorded group hashes == the composition of R14's recorded
     Supa / Titan-Sparse hashes (blocks 8..15 Titan, everything else Supa)
  B5 identity == docs/l1/L8_15_EXPECTED.json when that record exists (the frozen bytes)
  B6 the six sidecars are byte copies of Supa's (the programme's hf_checkpoint convention for a measured arm)
  B7 create-only: an existing arm directory is accepted only if it re-verifies, never overwritten

RUN GATES (fatal)
  G0 the environment equals the one recorded in the freeze, and on resume equals the first invocation's (python,
     torch, transformers, tokenizers, safetensors, numpy, torch CUDA build, NVIDIA driver, GPU name); the runner is
     the frozen file; a --no-freeze run refuses real weights and a real GPU; one run process per output directory
  G1 each arm's weight file hashes (sha256 + per-group) to its pinned identity before the lease is requested
  G2 the loaded model's per-group FP32 hashes == the file's (the measured weights ARE the file)
  G3 FP32 everywhere, TF32 off, float32_matmul_precision highest; no prompt truncated (retention prompts must fit
     the registered 512-token context; a long-context row that does not fit is recorded as skipped, which makes the
     report-only long-context numbers non-comparable; it never enters C0)
  PG the anchor parity gate (C0) on supa and sparse pre-gate outputs, before l8_15 is loaded and before any model
     reads the OOD bank (every stage that must follow it records the gate it followed; analyze checks that)

FORWARD PATH. Default `--forward dense`: the programme's own DenseReference (experiments/architecture_benchmark_001/
model.py at pilot commit 5b4c22d, vendored byte-exact in docs/l1/vendor_5b4c22d and hash-checked before import), the
same greedy loop as process_probe / retention_probe / long_context_probe (batch 1, KV cache, argmax, the official EOS
list, decode skip_special_tokens + strip). `--forward hf` (Qwen3ForCausalLM) exists for the tiny-model cross-check and
as a fallback; it would need its own amendment before any decision rests on it.

GPU SAFETY (DECISIONS I-10..I-27): one lease for the whole job (rcwmt.runtime.gpu_lease: alternation, the thermal
flag, the dedicated-memory holder check); runtime.breathe() after every row (duty cycle + watchdog flag); a 60 ms
pause and a flag check every 8 decode steps. No process is spawned while waiting or running (I-27): the lease's
tasklist / PowerShell helpers are replaced in this process by psutil and an in-process PDH read of the same Windows
counter, and runtime.gpu_temp_c's nvidia-smi spawn by the in-process NVML read of scripts/sensors_noproc.py.
Before the lease (nothing held): host RAM >= 6 GiB and a live watchdog (runs/thermal_log.txt written within 3 min).
Inside the lease, before each load: the pause flag clear, RAM >= 5 GiB within 5 min, a live watchdog; while running:
a watchdog silent for 3 min, or a pause flag with RAM < 5 GiB for 15 min, ends the process with exit code 5
(resumable; the lease is released on the way out).

CPU path (--device cpu): the tiny-model test only (scripts/l1_test_tiny.py). CUDA is hidden with the empty string
before torch is imported and torch.cuda.is_available / device_count are patched (never "-1", DECISIONS I-21).

  python -B scripts/l1_screen.py plan
  python -B scripts/l1_screen.py integrity
  python -B scripts/l1_screen.py parity-cpu
  python -B scripts/l1_screen.py build --dry-hash [--record]
  python -B scripts/l1_screen.py build
  python -B scripts/l1_screen.py run --device cuda [--wait-for PATH]
  python -B scripts/l1_screen.py analyze

EXIT CODES: 0 done or nothing pending (including a written PARITY_FAIL.json); 3 a sanity / identity / freeze /
environment failure (deterministic: do not retry); 4 the --wait-for gate timed out; 5 resumable (host RAM, watchdog
or a RAM-caused pause; the lease was released; rerunning resumes from the saved stages).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # importing vendored / foreign code must never write __pycache__


def _argv_value(flag: str, default: str | None = None) -> str | None:
    for i, x in enumerate(sys.argv):
        if x == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return default


def _argv_values(flag: str) -> list[str]:
    out = []
    for i, x in enumerate(sys.argv):
        if x == flag and i + 1 < len(sys.argv):
            out.append(sys.argv[i + 1])
        elif x.startswith(flag + "="):
            out.append(x.split("=", 1)[1])
    return out


def wait_for_files(paths, poll_s: float = 60.0, max_h: float = 0.0) -> bool:
    """Queue gate: block until every path exists (relative to RCWM-T1). Standard library only, so a waiting process
    holds a few MB and never touches the driver. False (caller exits 4, nothing loaded or written) after max_h."""
    import time as _t
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    ps = [p if p.is_absolute() else root / p for p in (_P(x) for x in paths)]
    t0 = _t.time()
    missing = [p for p in ps if not p.exists()]
    if missing:
        print(f"[{_t.strftime('%H:%M:%S')}] gate: waiting for {[str(p) for p in missing]}", flush=True)
    while missing:
        if max_h > 0 and _t.time() - t0 >= max_h * 3600:
            print(f"[{_t.strftime('%H:%M:%S')}] gate: still missing after {max_h:g} h; nothing loaded", flush=True)
            return False
        _t.sleep(poll_s)
        missing = [p for p in ps if not p.exists()]
    return True


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "run" and _argv_values("--wait-for"):
    if not wait_for_files(_argv_values("--wait-for"), float(_argv_value("--wait-poll", "60")),
                          float(_argv_value("--wait-max-h", "0"))):
        sys.exit(4)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "run" and _argv_value("--device") == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # as the programme's supervised wrapper set it
    else:                                                # every other subcommand is CPU-only: never touch the driver
        os.environ["CUDA_VISIBLE_DEVICES"] = ""           # empty string, NEVER "-1" (I-21)
        for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(_k, str(_argv_value("--threads", "2")))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "l1"
VENDOR = DOCS / "vendor_5b4c22d"
AB = VENDOR / "experiments" / "architecture_benchmark_001"
OUT_DEFAULT = ROOT / "runs" / "l1_screen"
BP = Path(r"C:/Users/clclc/BusinessProject")
CORES = BP / "Chat-Bubble" / "Rineya_Published_Cores"
REC = BP / "rineya-dataset-retention-records"
CED = BP / "ChatBubble-ced-native-loader-qualification"

# Inputs. A tuple means "the first copy that exists" (same pinned sha). Tests repoint entries at tiny stand-ins.
PATHS: dict = {
    "supa_dir": CORES / "Rineya-Supa-1.7.0-devfreeze",
    "sparse_dir": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
    "views": REC / "select-views-001" / "SELECT_VIEWS.jsonl",
    "floors": REC / "select-views-001" / "FLOORS.json",
    "sv_build": REC / "select-views-001" / "BUILD.json",
    "lc_prompts": BP / "Chat-Bubble" / "Rineya_Long_Context_Demo_v1" / "data" / "all_prompts.jsonl",
    "lc_key": BP / "Chat-Bubble" / "Rineya_Long_Context_Demo_v1" / "data" / "all_answer_key.jsonl",
    "probe": AB / "data" / "monitor_probe.jsonl",
    "audit": AB / "evidence" / "retention-design-v1" / "PROBE_LABEL_AUDIT.json",
    "protocol": AB / "protocol.json",
    "b28": (REC / "rescued-compose-lane-outputs-20260922" / "retention_B28_a329eb7b.json",
            CED / "experiments" / "architecture_benchmark_001" / "outputs" / "retention-a329eb7b" / "B28" / "retention.json"),
    "anchor_process_sparse": REC / "sparse001-20260917" / "evidence" / "PROCESS_int001_s25d.json",
    "anchor_ret_sparse": REC / "sparse001-20260917" / "evidence" / "RETENTION_PROBE_int001_s25d.json",
    "anchor_lc_sparse": REC / "sparse001-20260917" / "evidence" / "LONGCTX_int001_s25d.json",
    "anchor_process_supa": (REC / "rescued-ced-checkout-20260922" / "docs" / "model-quality" / "audit-2026-09-17"
                            / "rrp015-evidence" / "PROCESS_rrp015_supa.json",
                            CED / "docs" / "model-quality" / "audit-2026-09-17" / "rrp015-evidence" / "PROCESS_rrp015_supa.json"),
    "anchor_ret_supa": (REC / "rescued-compose-lane-outputs-20260922" / "retention_B28_a329eb7b.json",
                        CED / "experiments" / "architecture_benchmark_001" / "outputs" / "retention-a329eb7b" / "B28" / "retention.json"),
    "anchor_lc_supa": AB / "evidence" / "retention-design-v1" / "LONGCTX_supa.json",
    "frontier_det": REC / "sparse001-20260917" / "FRONTIER_DET.json",
    "quarantine": CED / "research" / "reasoning_posttraining_v1" / "QUARANTINE.json",
    # HSKILL-002: the build record names the source pools of BOTH sibling banks (identical in both build records);
    # the fresh prompts and the bridge exclude every id and user text of those pools. HSKILL_CONFIRM_V2.jsonl
    # (operator-reserved, UNSPENT_AND_UNREAD) is never opened by this file.
    "hskill_views_build": REC / "hskill-views-002" / "HSKILL_VIEWS_V2_BUILD.json",
    "hskill_quarantine": REC / "hskill-views-002" / "QUARANTINE.json",
    "hskill_views": REC / "hskill-views-002" / "HSKILL_VIEWS_V2.jsonl",
    "rci001_seal": REC / "rci001-20260918" / "RCI_SEAL.json",
    "rci002_seal": REC / "rci002-20260918" / "RCI_SEAL.json",
    "gate_v2_prereg": REC / "forge-v2-prereg-001" / "GATE_V2_PREREG.md",
    "offtask_used": ROOT / "data" / "offtask_prompts.jsonl",
    "r14_meta": ROOT / "runs" / "bank_pilot" / "localized" / "supa" / "offtask_meta.json",
    "r14_lp_supa": ROOT / "runs" / "bank_pilot" / "localized" / "supa" / "offtask_logprobs.npy",
    "r14_lp_sparse": ROOT / "runs" / "bank_pilot" / "localized" / "sparse" / "offtask_logprobs.npy",
    "r14_lp_l8_15": ROOT / "runs" / "bank_pilot" / "localized" / "loc8_15" / "offtask_logprobs.npy",
    "vendor_refonly": DOCS / "vendor_5b4c22d" / "VENDOR_REFONLY.json",
    "offtask_fresh": DOCS / "offtask_fresh_prompts.jsonl",
    "offtask_fresh_source": DOCS / "offtask_fresh_prompts.SOURCE.json",
    "slice": BP / "Chat-Bubble" / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0" / "data" / "sft"
             / "rineya_v1_v2_preservation_slice_14000.jsonl",
    "r14_run": ROOT / "runs" / "bank_pilot" / "localized" / "RUN.json",
    "r14_loc": ROOT / "runs" / "bank_pilot" / "localized" / "loc8_15" / "ARM.json",
    "pilot_bank": ROOT / "data" / "bank_pilot.json",
    # amendment A-OOD (C6): the rule-extrapolation bank, its build record and builder; rcwmt/env.py supplies the
    # registered parser (parse_state, lenient) and is pinned because C6 is scored with it
    "ood_bank": DOCS / "ood_bank.jsonl",
    "ood_bank_source": DOCS / "ood_bank.SOURCE.json",
    "ood_builder": ROOT / "scripts" / "l1_make_ood.py",
    "rcwmt_env": ROOT / "rcwmt" / "env.py",
    "expected": DOCS / "L8_15_EXPECTED.json",
    "freeze": DOCS / "PREREG_L1_FREEZE.json",
    "prereg": DOCS / "PREREG_L1_DRAFT.md",
    "test_script": ROOT / "scripts" / "l1_test_tiny.py",
    "queue_hook": DOCS / "queue_hook_l1.sh.draft",
    "runs_root": ROOT / "runs",
}

SHA: dict = {   # pinned sha256 of every input (verified at use)
    "supa_weights": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    "sparse_weights": "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284",
    "views": "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3",
    "floors": "96b4ed55dddc68398974cd9b4cb4fe6ccf2ca0e0503f02f1d62f9253690ed59e",
    "sv_build": "c34b560dfbda9737e0d60722886db38c12c091f25a62e41898c29fc141d4091c",
    "lc_prompts": "2619bd2c4a503e8d65e1d22d5090738b91229365b400332ade29c110dece0af5",
    "lc_key": "b95cf52c29a44b3f4eb18d2979c18e3b941d93709a06d908c30295ac6ed66398",
    "probe": "5f6df1c14aa082eb2f61024a2652809077dbc91a6d7a4e19933d55f7890ae266",
    "audit": "4857cf868202763e76bc610f6e787aebee20b9d18bfa7451f42c603ed871e6d7",
    "protocol": "ead4ddf07f8098b01cb4dda1db40df451ee4495adff80e33cfe1603807830ef7",
    "b28": "ec3fca1e8b1eb418a6b848dadfea06235cf7c4317ddf6930061c84147afa9a8f",
    "anchor_process_sparse": "a101c0c69f99f722a4c527c72c8b91dd61ea0c12c8fcbddc3fbf7d353a19ab40",
    "anchor_ret_sparse": "eb85c169dc02e96f1b248c854bd9bbe804947445cce98d2df0b828bd8f8f6939",
    "anchor_lc_sparse": "8f53d99de6bf9c947a8ccb3c5f0e39a17cc137c1f6211efb762c5c75f8cecba5",
    "anchor_process_supa": "ecc80042ab44da19a565c02c6a9dbd7f0158210601d594117641d66e764e1f6c",
    "anchor_ret_supa": "ec3fca1e8b1eb418a6b848dadfea06235cf7c4317ddf6930061c84147afa9a8f",
    "anchor_lc_supa": "1305bdac5d7781712d3304e65acefebe8f1be53fc3b9624a690280336b8e6563",
    "frontier_det": "d1e761919a2ae982ab83a6ba26bafbfd7122db0762dd52b5753c63480e73d19a",
    "quarantine": "a20fc72bb85871b64900c1e47ef7c6b832f1d9de86f92051d26fde5e2dc2ec66",
    "offtask_used": "ae92870bef9ab18e99229af71d764d12b0544d40590f2faf2c7d1e9356f3d6bc",
    "slice": "5ee0c283b290cf89f6fcfbcf52bd2bcd03f6c63bb06793b9fa4b957d91973d50",
    "r14_run": "d94f6a0784a84843f425345c09b47af3a08209d15eef3fb0377b99c8acaf4556",
    "r14_loc": "939a98b57b517725dab4ed1bd316173245cc9ccb64f868491cc598d72c83277f",
    "pilot_bank": "7c9b432c30e1abf142e10ec7413ee8383c6b51c5410a4fe7246906f7af659e7a",
    "hskill_views_build": "45103800078c99b3442b6011dbdf779c40a9c344bd20e5784057239fe5b89396",
    "hskill_quarantine": "8679bc88c0203daca68ee5945f5e1962100707da4733d63eec45fb2752908f2e",
    "hskill_views": "ba1f2982fd5d08aac861d4547b1dadb33d20e6022ff9aa06934e1ad2e7154723",
    "rci001_seal": "9381238711e8fa3c8beea246ea426eff45925d3ebe04d23bca9d6edc3bddd4c9",
    "rci002_seal": "fbcdae68ad007fbc72b2d4a1953160756e2549320ac1c98b11f24e2e7a693db3",
    "gate_v2_prereg": "3e33554fff42cc6ef564f5f4488ed47f9241eb8c3769cf24abee4464608fbabb",
    "r14_meta": "291ea1c90af07d17787141f2643d3387ea904c3a70ae2126083160b153acd479",
    "r14_lp_supa": "9c5d4be302a624811fb304c41b0fd9061f9315409d311b6a659b146c27d0dba2",
    "r14_lp_sparse": "6735ec7d14eb3a67450e265c40c8cc41b65cb40e5f9470f34f23e3593b21627c",
    "r14_lp_l8_15": "d20d0108ae57c4e3aa6b9545920fb32ab19bb57c3ea9198ccfd72858a72b2362",
    "ood_bank": "ac665c115b8c098e6bf58f6182a4a5a73b6fa1d87401afb1b3a55eeb165e940c",
    "ood_bank_source": "a373686250caeb503061e267faec7b261cf089c8e6832f9789591f26886b08dd",
    "rcwmt_env": "92abb6787588bfb0a38d2d73618f0aeb9e70b9dc4a2d86f428712cd4466db1e2",
}
L8_15_REGISTERED_SHA = "6282b45676fab8b2446d4363373b94931aea16c346ba769c284b81fcb0f97fe3"   # L8_15_EXPECTED.json
# the real weight identities: a --no-freeze run (tests only) refuses to touch any of them
REAL_WEIGHT_SHA = frozenset({SHA["supa_weights"], SHA["sparse_weights"], L8_15_REGISTERED_SHA})
VENDOR_SHA = {   # byte copies of 5b4c22d (docs/l1/vendor_5b4c22d/VENDOR.json records the git blob ids)
    "__init__.py": "d97acbe1959e8b2b7dee93e89ef729be14956dabd6c7320c943ff4ba3957f621",
    "budget.py": "cdeae5d9b422595310e608eb4d234dbb5740e147ba5eb43626c1315b5d340dce",
    "model.py": "f4c1f29710f14291baa89be875ba9b3f74fb02601532c1b482eb049d6c8ec3ac",
    "transfer.py": "ddf8e1e88db63c27d59aa3e0db9d1521692a3f6af82dcf8273e6e862433763ee",
    "source.py": "5e98a759c9c546b26129b66e8c2916f5dbcff74781f6eb34a54d9990ce6dc22a",
    "retention.py": "c52b6a48cb811e3c272793a66765e82152fe170e8307eea153c3b1b6c5c58322",
}
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")

# ------------------------------------------------------------------------------------------------ REGISTERED
ARMS = ("supa", "sparse", "l8_15")
N_LAYERS = 28
L8_15_BLOCKS = tuple(range(8, 16))
DELTA_BLOCKS = tuple(range(8, 28))                  # D-21: Titan-Sparse differs from Supa in exactly these blocks
FAMILIES = ("ledger_replay", "prerequisite_plan", "register_machine")
PUBLISHED_FLOORS = {   # select-views-001/FLOORS.json == process_probe.INSTRUMENTS['select-views-001']['floors']
    "ledger_replay": {"copy_before": 30, "constant_majority": 1, "n": 128},
    "prerequisite_plan": {"copy_before": 0, "constant_majority": 10, "n": 128},
    "register_machine": {"copy_before": 9, "constant_majority": 1, "n": 128},
}
ACQ_MAX_NEW, LC_MAX_NEW, LC_MAX_CONTEXT = 96, 24, 4096      # process_probe / long_context_probe defaults
# retention: protocol.json decoding.max_new_tokens (24) and max_context (512), read from the vendored protocol
ARITH_BAR = 62                     # GATE v2 term 1 (DATASET_PROGRAMME_PREREG.md:140-141 via ADJUDICATION_001 D3)
RETAINED_BAR = 37                  # GATE v2 term 2 (same provenance)
# C2 and C4 share ONE margin logic (PREREG-L1 section 5): a within-run paired contrast against the parent, which may
# neither be a loss the paired exact test detects nor exceed half the parent recipe's own cross-draw span on that
# endpoint (REP-001: three independent FP32 controls of the RRP-015 recipe, seeds 17 / 23 / 29). They are value
# judgments, not noise floors: "the cut may cost less than half of what retraining the parent moves the endpoint".
REP001_CONTROL = {"ledger_replay": (54, 71, 61), "prerequisite_plan": (95, 85, 93), "register_machine": (85, 73, 74),
                  "retained": (40, 38, 36)}        # PROCESS_rrp015_control / PROCESS_int001_rep_s23_ctrl / _s29_ctrl
ACQ_FAMILY_TOL = {f: (max(v) - min(v)) // 2 for f, v in REP001_CONTROL.items() if f in FAMILIES}   # 8 / 5 / 6
RETAINED_MARGIN = (max(REP001_CONTROL["retained"]) - min(REP001_CONTROL["retained"])) // 2          # 2
DETECT_P = 0.05                    # C2: a family fails if lost > gained and the two-sided exact McNemar p < 0.05
DRIFT_UPPER = 0.80                 # C5: D-23's primary criterion, replicated on fresh prompts
PARITY_FRACTION, PARITY_MIN_ROWS = 0.03, 3          # per-row text allowance max(3, floor(0.03 n))
PARITY_TOL = {"acq_total": 3, "arith_permissive": 2, "arith_strict": 2, "retained": 1}      # decisive (C0)
PARITY_TOL_REPORT = {"lc_total": 3, "unknown_substring": 1, "unknown_literal": 1}           # report-only (SPENT bank)
RET_ROW_TOL, RET_CLASS_TOL = 1, 1  # Titan-Sparse retention: source-correct rows / per-class counts (C0)
BRIDGE_TOP1_MIN, BRIDGE_KL_MAX = 0.99, 1e-3        # anchors vs R14's stored log-probs on the bridge prompts (C0)
BRIDGE_R14_RATIO = 0.6772          # R14's KL(loc8_15||Supa)/KL(sparse||Supa) on the same 228 bridge prompts (P6)
BOOT, BOOT_SEED = 10000, 20261009
FRESH_SEED, FRESH_PER_CLASS = 20261009, 8
FRESH_N = 456                      # 57 reply classes x 8: the 5 SRB HSKILL classes have no row outside the pools
BRIDGE_N = 228                     # the 248 R14 prompts minus the 20 that are rows of the HSKILL-002 pools
# C6 (amendment A-OOD, D-28(a)): out-of-distribution / rule-extrapolation rows, paired L8-15 vs Titan-Sparse per
# stratum. Bank: docs/l1/ood_bank.jsonl (scripts/l1_make_ood.py, seed 20261028, no model call).
OOD_STRATA = ("L_big", "L_void", "R_wrap", "R_boundary", "P_inapp", "P_appdel")
OOD_N_PER_STRATUM = 100
OOD_N = len(OOD_STRATA) * OOD_N_PER_STRATUM          # 600
OOD_MAX_NEW = ACQ_MAX_NEW                            # 96: the programme's cap for these families in this frame
OOD_TOL_MIN, OOD_TOL_FRAC = 3, 0.05                  # per stratum: net >= -max(3, ceil(0.05 n)) = -5 at n = 100
OOD_PARSE = "rcwmt.env.parse_state(family, prediction, lenient=True) == json.loads(gold)"
LEASE_LABEL = "rcwm-t1:l1screen"
LEASE_JOB = ("RCWM-T1 PREREG-L1 promotion screen (supa, sparse, l8_15): SELECT-VIEWS-001, arithmetic panel, "
             "long context, off-task drift, OOD rule extrapolation")
PROMOTE = "PROMOTE (eligible; the operator's explicit go-ahead is still required, D-26)"
DO_NOT_PROMOTE = "DO NOT PROMOTE"
NON_COMPARABLE = "NON-COMPARABLE (no verdict issued)"
DECISION_RULE = (
    "L8-15 is eligible for promotion as an experimental core - promotion itself still requires the operator's "
    "explicit go-ahead (D-26) - if and only if, in the L1 run, ALL of the following hold. "
    "C0 identity and comparability, settled before any L8-15 output exists: the measured L8-15 weights are the frozen "
    "bytes (sha256 in L8_15_EXPECTED.json; per-group FP32 hashes equal to R14's loc8_15 arm); every build and load "
    "gate passed; the forward path is the programme's DenseReference (vendored from pilot commit 5b4c22d); the run "
    "used the frozen runner and the environment recorded in the freeze, in every invocation; the full decisive "
    "instruments were run (no row subset) and no decisive prompt was truncated or skipped; the L1 scorers reproduce "
    "every published per-row verdict and count on the anchors' published SELECT-VIEWS-001 and arithmetic-panel "
    "predictions, the L1 tokenizer and decoder return every one of those published texts unchanged (encode, decode, "
    "strip), and every arithmetic prompt fits the registered 512-token context; and the anchor parity gate, evaluated after "
    "both anchors and before L8-15 is loaded, passes: Supa 1.7 reproduces its published SELECT-VIEWS-001 texts on all "
    "but max(3, floor(3% of n)) rows, its canonical total within +-3 and its floor verdicts identically; "
    "Titan-Sparse does the same and also reproduces arithmetic_permissive and arithmetic_strict each within +-2 and "
    "references_retained within +-1 of its published 72, 63 and 41, its correct set on the 47 permissive and 41 "
    "strict source-correct rows up to 1 row per policy, its per-class arm_correct and terminated counts within +-1, "
    "and its 24 published arithmetic samples in text and termination on all but 3; and both anchors agree with R14's "
    "stored first-token log-probabilities on the 228 bridge prompts (top-1 agreement >= 0.99, mean KL(L1||R14) < "
    "0.001 nats). If the anchor parity gate fails, L8-15 is never loaded and no arm generates on the OOD bank. "
    "C1 acquisition floors (GATE v2 term 3): on SELECT-VIEWS-001 every family's two-sided exact binomial 95% lower "
    "bound on canonical_correct exceeds its published binding floor (ledger_replay 0.2344, prerequisite_plan 0.0781, "
    "register_machine 0.0703). "
    "C2 acquisition non-inferiority, per family, paired within the run against Titan-Sparse on canonical "
    "correctness: in every family the net change (gained - lost) is >= -8 (ledger_replay), >= -5 "
    "(prerequisite_plan), >= -6 (register_machine), and no family shows a detected loss (lost > gained with a "
    "two-sided exact McNemar p < 0.05). "
    "C3 arithmetic (GATE v2 term 1): arithmetic_permissive(L8-15) >= 62 of 150. "
    "C4 source answers retained (GATE v2 term 2 plus non-inferiority): references_retained(L8-15) >= 37 of 47 AND "
    ">= references_retained(Titan-Sparse, same run) - 2. "
    "C5 off-task drift on the 456 fresh prompts: the paired percentile-bootstrap 95% upper bound (10,000 resamples "
    "over prompts, seed 20261009) of mean first-token KL(L8-15||Supa) / mean KL(Titan-Sparse||Supa) is < 0.80. "
    "C6 out-of-distribution rule extrapolation (amendment A-OOD, D-28(a)), paired within the run against "
    "Titan-Sparse on the 600-row OOD bank (docs/l1/ood_bank.jsonl, sha256 recorded in ood_bank.SOURCE.json and "
    "pinned in the runner; built with no model call from data that played no part in choosing the cut: six strata "
    "of 100 one-step transitions in Titan's trained one-step frame, L_big, L_void, R_wrap, R_boundary, P_inapp and "
    "P_appdel), every arm run on it only after the anchor parity gate passed, greedy with at most 96 new tokens, and "
    "a row correct iff rcwmt.env.parse_state(family, prediction, lenient=True) equals the oracle state: in every "
    "stratum the net change (gained - lost) is >= -max(3, ceil(5% of n)), which is -5 at n = 100, and no stratum "
    "shows a detected loss (lost > gained with a two-sided exact McNemar p < 0.05). Supa's OOD scores are "
    "report-only. "
    "If C0 fails the run is NON-COMPARABLE: no verdict, nothing promoted, no L8-15 endpoint computed, and this "
    "registration ends (a rerun is a new registration with a fresh drift draw that may not reuse these prompts, and "
    "with a fresh OOD bank if any arm has generated on this one). "
    "If C0 holds and any of C1-C6 fails: DO NOT PROMOTE. Long context and literal UNKNOWN (including every parity "
    "and scorer check on that SPENT bank), Supa's arithmetic against its CPU-measured source answers, the legacy "
    "per-class monitor and the bridge drift ratio are report-only and never enter C0-C6. No threshold may change "
    "after any L1 output exists.")
REGISTERED = {"arms": ARMS, "l8_15_blocks": [L8_15_BLOCKS[0], L8_15_BLOCKS[-1]], "families": FAMILIES,
              "published_floors": PUBLISHED_FLOORS, "acq_max_new": ACQ_MAX_NEW, "lc_max_new": LC_MAX_NEW,
              "lc_max_context": LC_MAX_CONTEXT, "arith_bar": ARITH_BAR, "retained_bar": RETAINED_BAR,
              "rep001_control": REP001_CONTROL, "acq_family_tol": ACQ_FAMILY_TOL, "retained_margin": RETAINED_MARGIN,
              "detect_p": DETECT_P, "drift_upper": DRIFT_UPPER,
              "parity_fraction": PARITY_FRACTION, "parity_min_rows": PARITY_MIN_ROWS, "parity_tol": PARITY_TOL,
              "parity_tol_report": PARITY_TOL_REPORT, "ret_row_tol": RET_ROW_TOL, "ret_class_tol": RET_CLASS_TOL,
              "bridge_top1_min": BRIDGE_TOP1_MIN, "bridge_kl_max": BRIDGE_KL_MAX, "bridge_r14_ratio": BRIDGE_R14_RATIO,
              "boot": BOOT, "boot_seed": BOOT_SEED, "fresh_seed": FRESH_SEED, "fresh_per_class": FRESH_PER_CLASS,
              "fresh_n": FRESH_N, "bridge_n": BRIDGE_N, "l8_15_sha256": L8_15_REGISTERED_SHA,
              "ood_strata": OOD_STRATA, "ood_n_per_stratum": OOD_N_PER_STRATUM, "ood_n": OOD_N,
              "ood_max_new": OOD_MAX_NEW, "ood_tol_min": OOD_TOL_MIN, "ood_tol_frac": OOD_TOL_FRAC,
              "ood_parse": OOD_PARSE, "ood_bank_sha256": SHA["ood_bank"],
              "decision_rule": DECISION_RULE}
# Registered predictions: scored and reported, never decisive (PREREG-L1 section 7). P3, P7 and P8 are NO-LOSS
# predictions with a tolerance; equality is their expected value and none of them can be read as an improvement.
PREDICTIONS = {
    "P1": "L8-15 clears all three SELECT-VIEWS-001 family floors",
    "P2": "per-family net (L8-15 - Titan-Sparse): ledger_replay in [-12, 0], prerequisite_plan in [-4, +4], "
          "register_machine in [-6, +2]; aggregate in [-16, +2] (pilot projection -6 / 0 / -2.5, about -8)",
    "P3": "no retention loss: references_retained(L8-15) >= references_retained(Titan-Sparse) - 1",
    "P4": "arithmetic_permissive(L8-15) >= 62",
    "P5": "fresh-prompt drift ratio point estimate lies in [0.60, 0.76] (R14 on its 248: 0.680 [0.617, 0.735])",
    "P6": "bridge drift ratio (228 R14 prompts, this harness) is within +-0.05 of R14's 0.677 on the same 228",
    "P7": "no long-context loss (report-only): long_context(L8-15) >= long_context(Titan-Sparse) - 3",
    "P8": "no abstention loss (report-only): UNKNOWN substring(L8-15) >= UNKNOWN substring(Titan-Sparse) - 1",
    "P9": "both anchors reproduce their published per-row predictions on >= 99% of rows of every instrument "
          "(decisive and report-only alike)",
    "P10": "both anchors' bridge log-probabilities agree with R14's within mean KL 1e-4 nats (10x inside the bound)",
}

# Programme wall times on the RTX 4070 Laptop, FP32 DenseReference, no duty cycle (probe_terminal.json, SPARSE-001
# and REP-001 batteries): process 270-325 s / 384 rows, retention 46-59 s / 150, long context 151-192 s / 256.
SEC_PER_ROW = {"acq": (270 / 384, 325 / 384), "ret": (46 / 150, 59 / 150), "lc": (151 / 256, 192 / 256),
               "drift": (0.03, 0.08)}
# the OOD stage is the acquisition workload (same families, same one-step frame, same decoder and cap); `plan` scales
# the acquisition rate by the ratio of mean answer lengths (OOD gold vs SELECT-VIEWS-001 references, >= 1)
SEC_PER_ROW["ood"] = SEC_PER_ROW["acq"]
DUTY = {"low": 0.5, "expected": 1.5, "high": 3.0}       # rcwmt.runtime._duty_ratio bands (<70, 70-74, >=75 C)
BREATHE_MIN_S, TOKEN_PAUSE_EVERY, TOKEN_PAUSE_S = 0.12, 8, 0.06
PRE_GATE_STAGES = ("acq", "ret", "lc", "drift_fresh", "drift_bridge")    # the anchors run these BEFORE the gate
POST_GATE_STAGES = ("ood",)          # A-OOD: no model reads the OOD bank before the anchor parity gate has passed
STAGES = PRE_GATE_STAGES + POST_GATE_STAGES
# streamed loads in an uninterrupted run: supa (pre-gate), sparse (pre-gate, kept loaded through the CPU gate for its
# OOD stage), supa again (OOD only), l8_15 (every stage)
LOADS_PER_RUN = 4
STAGE_FILES = {"acq": ("acq.jsonl",), "ret": ("ret.jsonl",), "lc": ("lc.jsonl",),
               "drift_fresh": ("drift_fresh.npy", "drift_fresh_ids.json"),
               "drift_bridge": ("drift_bridge.npy", "drift_bridge_ids.json"), "ood": ("ood.jsonl",)}
HASH_CHUNK = 1 << 24
BUILD_CHUNK = 1 << 23
PROGRESS_EVERY = 50                    # a progress line every 50 rows of a stage
# host safety (ops review): thresholds relative to the watchdog's own (pause < 3 GB, RAM-caused resume >= 5 GB)
PRE_LEASE_RAM_GIB, PRE_LEASE_WAIT_S = 6.0, 2 * 3600       # wait with NOTHING held; then exit 5
LOAD_RAM_GIB, LOAD_RAM_WAIT_S = 5.0, 300                   # inside the lease, before each load; then exit 5
PAUSE_WAIT_BEFORE_LOAD_S = 1800                            # inside the lease, a pause flag before a load; then exit 5
RAM_PAUSE_EXIT_S = 900                                     # a pause flag with RAM < 5 GiB this long -> exit 5
WATCHDOG_STALE_S, WATCHDOG_WAIT_S = 180, 3600              # runs/thermal_log.txt is written at least once a minute
EXIT_SANITY, EXIT_WAIT, EXIT_RESUMABLE = 3, 4, 5

# test hooks
_SUBSET: dict | None = None            # rows per stage ({'acq': per family, 'ret': n, 'lc': n, 'drift': n,
                                       #                  'ood': per stratum})
_DENSE_CONFIG_HOOK = None              # callable(raw config.json dict) -> vendored budget.Config (tiny models)
_TENSOR_DEVICE: str | None = None      # run the cuda wiring with CPU tensors
_on_stage_done = None                  # callable(arm, stage)
_PARITY_HOOK = None                    # callable(gate dict) -> gate dict (tests force a pass or a failure)
_GUARD_FACTORY = None                  # callable(real: bool) -> host guard (tests record the guard calls)


class SanityError(RuntimeError):
    pass


class ResumableExit(RuntimeError):
    """Host RAM, the watchdog or a RAM-caused pause: stop now, release the lease, exit 5; a rerun resumes."""
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================================ files and hashes
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def resolve(key: str) -> Path:
    v = PATHS[key]
    if isinstance(v, (tuple, list)):
        for p in v:
            if Path(p).exists():
                return Path(p)
        return Path(v[0])
    return Path(v)


def pinned(key: str) -> Path:
    """The input's path after checking its sha256 against the pin (SanityError otherwise)."""
    p = resolve(key)
    if not p.exists():
        raise SanityError(f"input {key} missing: {p}")
    got = sha256_file(p)
    if key in SHA and got != SHA[key]:
        raise SanityError(f"input {key} ({p}) has sha256 {got[:12]}..., pinned {SHA[key][:12]}...")
    return p


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path, encoding: str = "utf-8") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding=encoding).splitlines() if l.strip()]


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes((json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))   # LF on every OS
    os.replace(tmp, path)


def write_create(path: Path, text: str) -> None:
    if Path(path).exists():
        raise SanityError(f"create-only output already exists: {path}")
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))   # bytes: a text-mode write turns LF into CRLF on Windows and breaks every
    os.replace(tmp, path)                   # hash recorded from the text (the fresh prompts' SOURCE record did)


def group_key(name: str) -> str:
    """run_patch._group_key."""
    return "block" + name.split(".")[2] if name.startswith("model.layers.") else name.rsplit(".", 1)[0]


def block_of(name: str) -> int | None:
    return int(name.split(".")[2]) if name.startswith("model.layers.") else None


def group_sort(groups) -> list[str]:
    bi = lambda g: int(g[5:]) if g.startswith("block") else None  # noqa: E731
    return sorted(groups, key=lambda g: (bi(g) is None, bi(g) or 0, g))


_ST_BYTES = {"F32": 4, "BF16": 2, "F16": 2}


def fp32_bytes(buf: bytes, dtype: str) -> bytes:
    """Stored little-endian values -> the FP32 bytes an FP32 load holds (BF16 -> FP32 is a 16-bit shift, exact)."""
    if dtype == "F32":
        return buf
    if dtype == "BF16":
        return (np.frombuffer(buf, dtype="<u2").astype("<u4") << 16).tobytes()
    if dtype == "F16":
        return np.frombuffer(buf, dtype="<f2").astype("<f4").tobytes()
    raise SanityError(f"dtype {dtype} is not hashed")


def read_header(path: Path) -> tuple[dict, dict, int, int]:
    """(tensor table without __metadata__, metadata, data base offset, file size)."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        raw = json.loads(fh.read(n))
    meta = raw.pop("__metadata__", {}) or {}
    return raw, meta, 8 + n, Path(path).stat().st_size


def file_identity(path: Path) -> dict:
    """sha256 of the file and sha1 per parameter group of its tensors widened to FP32, in ONE sequential read.
    The group digest is byte-for-byte run_localized.file_group_hashes / run_patch.group_hashes (sorted names within a
    group, name bytes then FP32 bytes, first 16 hex chars). Keys outside 'model.' are not parameters of the tied model."""
    hdr, meta, base, size = read_header(path)
    order = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])
    groups: dict[str, hashlib._Hash] = {}
    per_group_names: dict[str, list[str]] = defaultdict(list)
    for k in hdr:
        if k.startswith("model."):
            per_group_names[group_key(k)].append(k)
    # group digests need names sorted within the group; collect each group's FP32 bytes in sorted-name order by
    # buffering per-tensor digests is impossible (sha1 is sequential), so hash groups in a second pass if the file's
    # offset order is not sorted-name order within groups (it is for safetensors written by save_file and by build).
    sorted_ok = all(sorted(v) == [x for x in order if x in set(v)] for v in per_group_names.values())
    h_all = hashlib.sha256()
    with open(path, "rb") as fh:
        h_all.update(fh.read(base))
        pos = 0
        for name in order:
            a, b = hdr[name]["data_offsets"]
            if a != pos:
                raise SanityError(f"{path}: tensor data not contiguous at {name}")
            dt = hdr[name]["dtype"]
            if dt not in _ST_BYTES:
                raise SanityError(f"{path}: {name} has dtype {dt}")
            size_el = _ST_BYTES[dt]
            n_el = (b - a) // size_el
            if n_el * size_el != b - a or n_el != int(np.prod(hdr[name]["shape"], dtype=np.int64)):
                raise SanityError(f"{path}: {name} byte range does not match its shape")
            g = None
            if name.startswith("model.") and sorted_ok:
                key = group_key(name)
                g = groups.setdefault(key, hashlib.sha1())
                g.update(name.encode())
            for s in range(0, n_el, HASH_CHUNK):
                k = min(HASH_CHUNK, n_el - s)
                buf = fh.read(k * size_el)
                if len(buf) != k * size_el:
                    raise SanityError(f"{path}: short read in {name}")
                h_all.update(buf)
                if g is not None:
                    g.update(fp32_bytes(buf, dt))
            pos = b
        rest = fh.read()
        h_all.update(rest)
        if base + pos != size or rest:
            raise SanityError(f"{path}: {size - base - pos} trailing bytes after the tensor data")
    if not sorted_ok:   # fallback: second pass in sorted-name order per group
        groups = {}
        with open(path, "rb") as fh:
            for key in group_sort(per_group_names):
                g = groups.setdefault(key, hashlib.sha1())
                for name in sorted(per_group_names[key]):
                    a, b = hdr[name]["data_offsets"]
                    size_el = _ST_BYTES[hdr[name]["dtype"]]
                    g.update(name.encode())
                    fh.seek(base + a)
                    left = b - a
                    while left:
                        buf = fh.read(min(left, HASH_CHUNK * size_el))
                        g.update(fp32_bytes(buf, hdr[name]["dtype"]))
                        left -= len(buf)
    return {"sha256": h_all.hexdigest(), "bytes": size, "metadata": meta,
            "group_hashes": {k: groups[k].hexdigest()[:16] for k in group_sort(groups)},
            "dtypes": dict(Counter(v["dtype"] for v in hdr.values())), "tensors": len(hdr)}


# ============================================================================================ R14 identity
def r14_hashes() -> dict:
    """R14's recorded per-group hashes: Supa, Titan-Sparse (genuine loads, via run_patch's recorded hashes) and the
    loc8_15 arm built with run_patch.set_block_sources (G3/G4 there)."""
    run = read_json(pinned("r14_run"))
    loc = read_json(pinned("r14_loc"))
    src = run["source_hashes"]
    return {"supa": src["supa"], "sparse": src["sparse"], "l8_15": loc["group_hashes"],
            "loc8_15_fingerprint": loc.get("fingerprint"), "loc8_15_titan_blocks": loc.get("titan_blocks")}


def composed_l8_15(supa: dict, sparse: dict) -> dict:
    return {g: (sparse[g] if g.startswith("block") and int(g[5:]) in L8_15_BLOCKS else supa[g]) for g in supa}


# ============================================================================================ build (no model)
class _NullSink:
    def write(self, b: bytes) -> None:
        pass

    def commit(self) -> None:
        pass

    def abort(self) -> None:
        pass


class _FileSink:
    def __init__(self, path: Path):
        self.path, self.tmp = Path(path), Path(str(path) + ".tmp")
        self.fh = open(self.tmp, "wb")

    def write(self, b: bytes) -> None:
        self.fh.write(b)

    def commit(self) -> None:
        self.fh.close()
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self.fh.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def build_stream(supa_file: Path, sparse_file: Path, sink, chunk_el: int = BUILD_CHUNK) -> dict:
    """One pass over both cores, sequential reads only, host RAM about four chunks. Writes the L8-15 safetensors bytes
    to `sink` and returns its identity plus the B1/B2 checks and the delta accounting."""
    hs, ms, bs, zs = read_header(supa_file)
    ht, mt, bt, zt = read_header(sparse_file)
    if set(hs) != set(ht):
        raise SanityError("the two cores hold different tensor names")
    names = sorted(hs)
    for name in names:
        if hs[name]["shape"] != ht[name]["shape"]:
            raise SanityError(f"shape mismatch {name}")
        if hs[name]["dtype"] not in ("BF16", "F32") or ht[name]["dtype"] != "F32":
            raise SanityError(f"unexpected dtypes for {name}: supa {hs[name]['dtype']}, sparse {ht[name]['dtype']}")
        if not name.startswith("model."):
            raise SanityError(f"unexpected tensor outside the tied model: {name}")
    for hdr, z, base, label in ((hs, zs, bs, "supa"), (ht, zt, bt, "sparse")):
        pos = 0
        for name in names:                      # sorted-name order must be the file's offset order (sequential read)
            a, b = hdr[name]["data_offsets"]
            if a != pos:
                raise SanityError(f"{label}: tensors are not stored contiguously in sorted-name order ({name})")
            pos = b
        if base + pos != z:
            raise SanityError(f"{label}: file size does not match its header")
    # output header: what safetensors' save_file writes for these tensors (sorted names, __metadata__ first,
    # header padded with spaces to a multiple of 8)
    table, off = {"__metadata__": {"format": "pt"}}, 0
    for name in names:
        n_el = int(np.prod(hs[name]["shape"], dtype=np.int64))
        table[name] = {"dtype": "F32", "shape": hs[name]["shape"], "data_offsets": [off, off + 4 * n_el]}
        off += 4 * n_el
    hb = json.dumps(table, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    head = len(hb).to_bytes(8, "little") + hb
    h_out, h_s, h_t = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    groups: dict[str, hashlib._Hash] = {}
    delta = {i: {"elements": 0, "nonzero": 0, "energy": 0.0} for i in DELTA_BLOCKS}
    h_out.update(head)
    sink.write(head)
    fs, ft = open(supa_file, "rb"), open(sparse_file, "rb")
    try:
        h_s.update(fs.read(bs))
        h_t.update(ft.read(bt))
        for name in names:
            blk = block_of(name)
            n_el = int(np.prod(hs[name]["shape"], dtype=np.int64))
            ds = _ST_BYTES[hs[name]["dtype"]]
            g = groups.setdefault(group_key(name), hashlib.sha1())
            g.update(name.encode())
            for s in range(0, n_el, chunk_el):
                k = min(chunk_el, n_el - s)
                rs, rt = fs.read(k * ds), ft.read(k * 4)
                if len(rs) != k * ds or len(rt) != k * 4:
                    raise SanityError(f"short read in {name}")
                h_s.update(rs)
                h_t.update(rt)
                s32 = fp32_bytes(rs, hs[name]["dtype"])
                if blk is None or blk not in DELTA_BLOCKS:
                    if rt != s32:                                   # B2, element exact
                        raise SanityError(f"B2: Titan-Sparse differs from Supa outside blocks 8..27 ({name}); "
                                          f"the D-21 premise fails")
                else:
                    d = np.frombuffer(rt, "<f4").astype(np.float64) - np.frombuffer(s32, "<f4").astype(np.float64)
                    rec = delta[blk]
                    rec["elements"] += k
                    rec["nonzero"] += int(np.count_nonzero(d))
                    rec["energy"] += float(np.dot(d, d))
                    del d
                out = rt if (blk is not None and blk in L8_15_BLOCKS) else s32
                h_out.update(out)
                g.update(out)
                sink.write(out)
        if fs.read(1) or ft.read(1):
            raise SanityError("trailing bytes after the last tensor")
    finally:
        fs.close()
        ft.close()
    e_all = sum(r["energy"] for r in delta.values())
    e_815 = sum(delta[i]["energy"] for i in L8_15_BLOCKS)
    return {"out_weight_sha256": h_out.hexdigest(), "out_bytes": len(head) + off, "header_bytes": len(head),
            "tensors": len(names),
            "group_hashes": {k: groups[k].hexdigest()[:16] for k in group_sort(groups)},
            "input_sha256": {"supa": h_s.hexdigest(), "sparse": h_t.hexdigest()},
            "input_metadata": {"supa": ms, "sparse": mt},
            "b2_titan_equals_supa_outside_blocks_8_27": True,
            "delta_by_block": {str(i): {"nonzero_fraction": (r["nonzero"] / r["elements"]) if r["elements"] else 0.0,
                                        "energy_share": (r["energy"] / e_all) if e_all else 0.0}
                               for i, r in delta.items()},
            "delta_energy_share_blocks_8_15": (e_815 / e_all) if e_all else 0.0,
            "delta_norm_fraction_kept": math.sqrt(e_815 / e_all) if e_all else 0.0}


def _check_build_identity(rec: dict, check_expected: bool = True) -> dict:
    """B1, B4, B5 on a build_stream record."""
    out = {}
    for k, key in (("supa", "supa_weights"), ("sparse", "sparse_weights")):
        if rec["input_sha256"][k] != SHA[key]:
            raise SanityError(f"B1: {k} weights hash {rec['input_sha256'][k][:12]}..., pinned {SHA[key][:12]}...")
    out["B1"] = "input weights hash to the pinned identities"
    r = r14_hashes()
    comp = composed_l8_15(r["supa"], r["sparse"])
    bad = [g for g in set(rec["group_hashes"]) | set(r["l8_15"]) if rec["group_hashes"].get(g) != r["l8_15"].get(g)]
    if bad:
        raise SanityError(f"B4: group hashes differ from R14 loc8_15 in {group_sort(bad)[:6]}")
    if comp != r["l8_15"]:
        raise SanityError("B4: R14 loc8_15 is not the composition of R14's Supa/Titan-Sparse hashes")
    diff = [g for g in r["supa"] if r["supa"][g] != r["sparse"][g]]
    if group_sort(diff) != [f"block{i}" for i in DELTA_BLOCKS]:
        raise SanityError(f"B4: R14's cores differ in {group_sort(diff)}, not exactly blocks 8..27")
    out["B4"] = "per-group FP32 hashes == R14 loc8_15 == composition of R14's Supa / Titan-Sparse hashes"
    exp = resolve("expected")
    if check_expected and exp.exists():
        e = read_json(exp)
        if e["out_weight_sha256"] != rec["out_weight_sha256"] or e["group_hashes"] != rec["group_hashes"]:
            raise SanityError(f"B5: identity differs from the frozen expectation {exp}")
        out["B5"] = f"identity == {exp.name}"
    else:
        out["B5"] = "no expectation recorded yet" if check_expected else "skipped"
    return out


def _lower_priority() -> None:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def cmd_build(a) -> int:
    _lower_priority()
    supa_file = resolve("supa_dir") / "model.safetensors"
    sparse_file = resolve("sparse_dir") / "model.safetensors"
    if a.dry_hash:
        t0 = time.time()
        rec = build_stream(supa_file, sparse_file, _NullSink())
        rec["checks"] = _check_build_identity(rec, check_expected=not a.record)
        rec["seconds"] = round(time.time() - t0, 1)
        print(json.dumps({k: v for k, v in rec.items() if k != "delta_by_block"}, indent=1), flush=True)
        if a.record:
            body = {"record": "L1_L8_15_EXPECTED_IDENTITY", "schema": 1,
                    "what": ("The identity the L8-15 artifact must have: computed by `l1_screen.py build --dry-hash "
                             "--record` from the two published cores (no file written, no model loaded). "
                             "C0/B5 compare every build and every measured arm against it."),
                    "construction": "Supa 1.7 with Titan-Sparse's stored FP32 tensors in blocks 8..15; every other "
                                    "tensor Supa's BF16 value widened to FP32; safetensors metadata {'format': 'pt'}",
                    "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **{k: rec[k] for k in ("out_weight_sha256", "out_bytes", "header_bytes", "tensors", "group_hashes",
                                           "input_sha256", "delta_energy_share_blocks_8_15",
                                           "delta_norm_fraction_kept", "delta_by_block")},
                    "checks": rec["checks"]}
            write_create(resolve("expected"), json.dumps(body, indent=1) + "\n")
            log(f"recorded {resolve('expected')}")
        return 0
    out_dir = Path(a.out_dir) if a.out_dir else OUT_DEFAULT
    arm = out_dir / "arm-l8_15"
    if arm.exists():                                                         # B7
        rec = verify_arm_dir(arm)
        log(f"{arm} exists and re-verifies ({rec['out_weight_sha256'][:12]}...); nothing written")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "arm-l8_15.building"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    t0 = time.time()
    sink = _FileSink(tmp_dir / "model.safetensors")
    try:
        rec1 = build_stream(supa_file, sparse_file, sink)
        checks = _check_build_identity(rec1)
        sink.commit()
    except BaseException:
        sink.abort()
        raise
    log(f"pass 1 wrote {rec1['out_bytes']:,} bytes ({rec1['out_weight_sha256'][:12]}...) in {time.time() - t0:.0f}s")
    rec2 = build_stream(supa_file, sparse_file, _NullSink())                 # B3: independent re-derivation
    if (rec2["out_weight_sha256"], rec2["group_hashes"]) != (rec1["out_weight_sha256"], rec1["group_hashes"]):
        raise SanityError("B3: the second derivation differs from the first")
    disk = file_identity(tmp_dir / "model.safetensors")                     # B3: bytes on disk
    if disk["sha256"] != rec1["out_weight_sha256"] or disk["group_hashes"] != rec1["group_hashes"]:
        raise SanityError("B3: the file read back from disk differs from the derived bytes")
    checks["B3"] = "pass 2 (hasher only) == pass 1 == the file re-read from disk (sha256 and per-group hashes)"
    src = resolve("supa_dir")
    for f in SIDECARS:                                                       # B6
        shutil.copyfile(src / f, tmp_dir / f)
        if sha256_file(tmp_dir / f) != sha256_file(src / f):
            raise SanityError(f"B6: sidecar copy differs from source: {f}")
    checks["B6"] = "six sidecars are byte copies of Supa 1.7.0-devfreeze's"
    receipt = {"schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
               "what_this_is": ("PREREG-L1 arm L8-15: Supa 1.7 plus Titan-Sparse-Experimental's delta in blocks 8..15 "
                                "only (its stored FP32 tensors), every other tensor Supa's BF16 value widened to FP32. "
                                "FP32, never routed through BF16. No model was run to produce these bytes."),
               "experiment": "RCWM-T1 PREREG-L1 (DECISIONS D-26)",
               "source": str(src), "source_weight_sha256": SHA["supa_weights"],
               "delta_donor": str(resolve("sparse_dir")), "delta_donor_weight_sha256": SHA["sparse_weights"],
               "titan_sparse_blocks_kept": [L8_15_BLOCKS[0], L8_15_BLOCKS[-1]],
               "out_weight_sha256": rec1["out_weight_sha256"], "stored_dtype": "float32",
               "group_hashes": rec1["group_hashes"],
               "delta_energy_share_blocks_8_15": rec1["delta_energy_share_blocks_8_15"],
               "delta_norm_fraction_kept": rec1["delta_norm_fraction_kept"],
               "delta_by_block": rec1["delta_by_block"],
               "sidecars_copied": list(SIDECARS), "builder": "scripts/l1_screen.py build",
               "builder_sha256": sha256_file(Path(__file__)), "gates": checks,
               "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_seconds": round(time.time() - t0, 1),
               "quality_claim": False, "release_permitted": False}
    write_json(tmp_dir / "MATERIALIZATION.json", receipt)
    write_json(tmp_dir / "ARM.json", receipt)
    os.replace(tmp_dir, arm)
    log(f"built {arm} ({rec1['out_weight_sha256'][:12]}...); gates {sorted(checks)}")
    return 0


def verify_arm_dir(arm: Path) -> dict:
    """B7 / G1 for the L8-15 arm directory: receipt present, weights == receipt == expectation, sidecars == Supa's."""
    rec = read_json(arm / "MATERIALIZATION.json")
    ident = file_identity(arm / "model.safetensors")
    if ident["sha256"] != rec["out_weight_sha256"] or ident["group_hashes"] != rec["group_hashes"]:
        raise SanityError(f"{arm}: weights differ from their receipt")
    exp = resolve("expected")
    if exp.exists():
        e = read_json(exp)
        if e["out_weight_sha256"] != ident["sha256"]:
            raise SanityError(f"{arm}: weights {ident['sha256'][:12]}... are not the frozen bytes {e['out_weight_sha256'][:12]}...")
    r = r14_hashes()
    if ident["group_hashes"] != r["l8_15"]:
        raise SanityError(f"{arm}: group hashes differ from R14 loc8_15")
    for f in SIDECARS:
        if sha256_file(arm / f) != sha256_file(resolve("supa_dir") / f):
            raise SanityError(f"{arm}: sidecar {f} differs from Supa's")
    return {**rec, "verified_identity": ident}


# ============================================================================================ vendored programme code
_VMODS: dict = {}


def _vendor_check(names) -> None:
    man = {Path(f["path"]).name: f["sha256"] for f in read_json(VENDOR / "VENDOR.json")["files"]}
    for n in names:
        got = sha256_file(AB / n)
        if got != VENDOR_SHA[n] or man.get(n) != VENDOR_SHA[n]:
            raise SanityError(f"vendored {n}: sha256 {got[:12]}... is not the pinned 5b4c22d copy")


def _vendor_import(mod: str):
    import importlib
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    pkg = sys.modules.get("experiments")
    if pkg is not None and not all(str(p).startswith(str(VENDOR)) for p in getattr(pkg, "__path__", [])):
        raise SanityError("an 'experiments' package from elsewhere is already imported")
    m = importlib.import_module(f"experiments.architecture_benchmark_001.{mod}")
    if not str(Path(m.__file__).resolve()).startswith(str(VENDOR.resolve())):
        raise SanityError(f"{mod} imported from {m.__file__}, not from the vendored copy")
    return m


def vendor_retention():
    """The programme's typed scorer (pure python: math, re, fractions)."""
    if "retention" not in _VMODS:
        _vendor_check(["__init__.py", "retention.py"])
        _VMODS["retention"] = _vendor_import("retention")
    return _VMODS["retention"]


def vendor_forward() -> dict:
    """DenseReference and its key map / official-shape check (imports torch)."""
    if "model" not in _VMODS:
        _vendor_check(["__init__.py", "budget.py", "model.py", "transfer.py", "source.py"])
        for m in ("budget", "model", "transfer", "source"):
            _VMODS[m] = _vendor_import(m)
    return _VMODS


def protocol_policy() -> tuple[dict, dict]:
    p = read_json(pinned("protocol"))
    return p["G1_RETENTION"], p["decoding"]


# ============================================================================================ instruments: rows
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def load_views() -> list[dict]:
    """process_probe.load_views for select-views-001 (sha bound to FLOORS.json, role guard, structural leak checks)."""
    path = pinned("views")
    declared = read_json(pinned("floors"))["views_sha256"]
    if sha256_file(path) != declared:
        raise SanityError("views sha256 differs from FLOORS.json")
    rows = read_jsonl(path)
    for row in rows:
        if row["view_role"] != "ADMITTED_FOR_SELECTION":
            raise SanityError(f"a view claims a role this instrument does not grant: {row['view_role']}")
        blob = json.dumps(row["prompt"])
        if row["source_id"] in blob:
            raise SanityError(f"label-bearing source id leaked into the prompt: {row['id']}")
        if "proposed_after" in blob or '"valid"' in blob:
            raise SanityError(f"answer or label key leaked into the prompt: {row['id']}")
        if sorted(json.loads(row["prompt"][1]["content"]).keys()) != ["before", "event"]:
            raise SanityError(f"user content is not exactly before+event: {row['id']}")
    return rows


def floors_from(rows: list[dict]) -> dict:
    out = {}
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family]
        copy_before = sum(1 for r in fam if canonical(json.loads(r["prompt"][1]["content"])["before"]) == r["reference"])
        counts = Counter(r["reference"] for r in fam)
        out[family] = {"copy_before": copy_before, "constant_majority": counts.most_common(1)[0][1], "n": len(fam)}
    return out


def check_floors(rows: list[dict]) -> dict:
    got = floors_from(rows)
    pub = {f: {k: v[k] for k in ("copy_before", "constant_majority", "n")}
           for f, v in read_json(pinned("floors"))["per_family"].items()}
    if got != PUBLISHED_FLOORS or pub != PUBLISHED_FLOORS:
        raise SanityError(f"floors drifted from the published pre-registration: recomputed {got}, FLOORS.json {pub}")
    return got


def load_probe_rows(classes) -> list[dict]:
    """run.load_probe_rows."""
    rows = read_jsonl(pinned("probe"), encoding="utf-8-sig")
    seen = set()
    for row in rows:
        if (row.get("split") != "task" or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen
                or not isinstance(row.get("group_id"), str) or not row["group_id"]
                or row.get("reply_class") not in set(classes)
                or not isinstance(row.get("answer"), str) or not row["answer"].strip()
                or not isinstance(row.get("messages"), list) or not row["messages"]):
            raise SanityError("probe row identity, class, answer or messages invalid")
        seen.add(row["id"])
    if not rows:
        raise SanityError("empty probe")
    return rows


def load_lc() -> tuple[list[dict], dict]:
    rows = read_jsonl(pinned("lc_prompts"))
    key = {r["id"]: r for r in read_jsonl(pinned("lc_key"))}
    if set(key) != {r["id"] for r in rows}:
        raise SanityError("answer key does not cover exactly the prompt rows")
    return rows, key


def lc_messages(row: dict) -> list[dict]:
    """long_context_probe.render: system + user turns only, contents str()."""
    msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in row["messages"]
            if isinstance(m, dict) and m.get("role") in {"system", "user"} and "content" in m]
    if not msgs:
        raise SanityError(f"row {row.get('id')} has no system or user turn")
    return msgs


def norm_text(s) -> str:
    return " ".join(str(s).split()).lower()


def user_texts(messages) -> set[str]:
    return {norm_text(m.get("content", "")) for m in messages or [] if isinstance(m, dict) and m.get("role") == "user"}


_POOLS: dict = {}


def hskill_pools() -> dict:
    """Every id and every (whitespace-normalised, lower-cased) user text of the HSKILL-002 source pools: per class, the
    rows of that class in the source file the build record names for it. Both HSKILL-VIEWS-002 and the reserved
    HSKILL-CONFIRM-002 were drawn from exactly these pools (their build records carry identical `sources` and
    `source_sha256`), so a prompt outside the pools is outside both banks; the CONFIRM file itself is never opened."""
    if "ids" in _POOLS:
        return _POOLS
    b = read_json(pinned("hskill_views_build"))
    ids, txt, per_class = set(), set(), {}
    for cls, f in sorted(b["sources"].items()):
        p = Path(f)
        if sha256_file(p) != b["source_sha256"].get(p.name):
            raise SanityError(f"HSKILL pool source {p} does not hash to its build record")
        n = 0
        for r in read_jsonl(p):
            if (r.get("metadata") or {}).get("reply_class") == cls:
                ids.add(r["id"])
                txt |= user_texts(r.get("messages"))
                n += 1
        per_class[cls] = n
    _POOLS.update(ids=ids, text=txt, rows_by_class=per_class, sources=b["sources"], source_sha256=b["source_sha256"])
    return _POOLS


def in_pools(row: dict, pools: dict) -> bool:
    return row["id"] in pools["ids"] or bool(user_texts(row.get("messages")) & pools["text"])


def load_prompts(which: str) -> list[dict]:
    """fresh: the C5 draw (checked against its SOURCE record); bridge: the 248 R14 prompts minus every row of the
    HSKILL-002 pools (the bridge of P6 and of the C0 bridge-parity leg); used: all 248 (integrity only)."""
    if which == "fresh":
        p = resolve("offtask_fresh")
        src = read_json(resolve("offtask_fresh_source"))
        if sha256_file(p) != src["file_sha256"]:
            raise SanityError("fresh prompts differ from their SOURCE record")
        return read_jsonl(p)
    used = read_jsonl(pinned("offtask_used"))
    if which == "bridge":
        pools = hskill_pools()
        return [u for u in used if not in_pools(u, pools)]
    return used


def load_ood_bank() -> list[dict]:
    """C6's 600 rows (amendment A-OOD): pinned file, equal to its build record, 6 strata x 100 in stratum order."""
    p = pinned("ood_bank")
    src = read_json(pinned("ood_bank_source"))
    if src.get("file_sha256") != SHA["ood_bank"] or src.get("model_calls") != 0:
        raise SanityError("the OOD bank differs from its SOURCE record")
    rows = read_jsonl(p)
    per = Counter(r["stratum"] for r in rows)
    if (len(rows) != OOD_N or per != Counter({s: OOD_N_PER_STRATUM for s in OOD_STRATA})
            or len({r["id"] for r in rows}) != OOD_N):
        raise SanityError(f"OOD bank is not {len(OOD_STRATA)} strata x {OOD_N_PER_STRATUM}: {dict(per)}")
    return rows


_ENV: list = []


def ood_env():
    """rcwmt.env (the registered parser parse_state and the oracle step), hash-checked before import."""
    if not _ENV:
        pinned("rcwmt_env")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from rcwmt import env as _env
        if Path(_env.__file__).resolve() != resolve("rcwmt_env").resolve():
            raise SanityError(f"rcwmt.env imported from {_env.__file__}, not the pinned file")
        _ENV.append(_env)
    return _ENV[0]


def subset_rows(stage: str, rows: list[dict]) -> list[dict]:
    if not _SUBSET:
        return rows
    if stage == "ood":
        per = _SUBSET.get("ood", 1)
        return [r for s in OOD_STRATA for r in [x for x in rows if x["stratum"] == s][:per]]
    if stage == "acq":
        per = _SUBSET.get("acq", 2)
        return [r for f in FAMILIES for r in [x for x in rows if x["family"] == f][:per]]
    if stage == "ret":
        per = _SUBSET.get("ret", 2)
        cl = sorted({r["reply_class"] for r in rows})
        return [r for c in cl for r in [x for x in rows if x["reply_class"] == c][:per]]
    return rows[: _SUBSET.get("drift" if stage.startswith("drift") else stage, len(rows))]


# ============================================================================================ scorers
def binom_at_least(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """process_probe.clopper_pearson, verbatim arithmetic."""
    lower = 0.0
    if k > 0:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if binom_at_least(k, n, mid) > alpha / 2:
                hi = mid
            else:
                lo = mid
        lower = lo
    upper = 1.0
    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if 1 - binom_at_least(k + 1, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = hi
    return lower, upper


def score_acq(rows: list[dict], preds: dict) -> dict:
    """process_probe's scoring block: strict / canonical per row, per-family CP bounds and the floor rule."""
    floors = PUBLISHED_FLOORS
    detail = []
    for row in rows:
        got = preds[row["id"]]
        text = got["prediction"]
        strict = text == row["reference"]
        canonical_ok, parsed_ok = False, False
        try:
            parsed = json.loads(text)
            parsed_ok = True
            canonical_ok = canonical(parsed) == row["reference"]
        except (ValueError, TypeError):
            parsed_ok = False
        detail.append({"id": row["id"], "family": row["family"], "reference": row["reference"], "prediction": text,
                       "strict": strict, "canonical": canonical_ok, "parsed": parsed_ok,
                       "terminated": got["terminated"], "generated_tokens": got["generated_tokens"]})
    per_family = {}
    for family in sorted({r["family"] for r in rows}):
        fam = [d for d in detail if d["family"] == family]
        n = len(fam)
        strict = sum(d["strict"] for d in fam)
        canon = sum(d["canonical"] for d in fam)
        copy_k, const_k = floors[family]["copy_before"], floors[family]["constant_majority"]
        floor_k = max(copy_k, const_k)
        floor_p = floor_k / floors[family]["n"]        # the published floor rate (n = 128)
        lo, hi = clopper_pearson(canon, n)
        per_family[family] = {
            "n": n, "strict_correct": strict, "canonical_correct": canon,
            "parsed": sum(d["parsed"] for d in fam), "terminated": sum(d["terminated"] for d in fam),
            "copy_before_count": copy_k, "constant_majority_count": const_k,
            "binding_floor_predictor": "copy_before" if copy_k >= const_k else "constant_majority",
            "binding_floor_count": floor_k, "binding_floor_rate": round(floor_p, 4),
            "canonical_rate": round(canon / n, 4) if n else 0.0, "presentation_only_gain": canon - strict,
            "exact_binomial_p_vs_floor_one_sided": binom_at_least(canon, n, floor_p),
            "clopper_pearson_95": [round(lo, 4), round(hi, 4)], "clopper_pearson_lower_exact": lo,
            "clears_floor": lo > floor_p}
    total_canon = sum(d["canonical"] for d in detail)
    return {"detail": detail, "per_family": per_family,
            "aggregate": {"n": len(rows), "strict_correct": sum(d["strict"] for d in detail),
                          "canonical_correct": total_canon,
                          "canonical_rate": round(total_canon / len(rows), 4) if rows else 0.0,
                          "presentation_only_gain": total_canon - sum(d["strict"] for d in detail)},
            "floors_cleared": sum(per_family[f]["clears_floor"] for f in per_family)}


def ret_context():
    """Everything the retention scorer needs, with retention_probe's control: the frozen B28 source answers must
    re-score to exactly the frozen reference sets under both bare policies."""
    R = vendor_retention()
    policy, decoding = protocol_policy()
    rows = load_probe_rows(policy["probe_classes"])
    audit = read_json(pinned("audit"))
    overlays = {name: R.build_overlay(audit, name) for name in R.BARE_POLICIES}
    frozen = read_json(pinned("b28"))
    references = {k: set(v) for k, v in frozen["reference_ids_by_policy"].items()}
    source_raw = {r["id"]: r["source"] for r in frozen["rows"]}
    if set(source_raw) != {r["id"] for r in rows}:
        raise SanityError("frozen receipt does not cover exactly the probe rows")
    rc = {r["id"]: r["reply_class"] for r in rows}
    scored_by_policy = {}
    for name, ov in overlays.items():
        scored = {i: {**R.score_prediction(rec["raw"], rec["terminated"], ov["targets"][i]), "reply_class": rc[i]}
                  for i, rec in source_raw.items()}
        got = {i for i, r in scored.items() if r["correct"]}
        if got != references[name]:
            raise SanityError(f"re-scored source under {name} does not reproduce the frozen reference: "
                              f"{len(got)} vs {len(references[name])}")
        scored_by_policy[name] = scored
    return {"R": R, "policy": policy, "decoding": decoding, "rows": rows, "overlays": overlays,
            "references": references, "source_raw": source_raw, "source_scored": scored_by_policy}


def retention_classes(rows, source: dict, arm: dict, reference: set, classes) -> dict:
    """run._retention_classes."""
    counts = {}
    for cls in classes:
        ids = [r["id"] for r in rows if r["reply_class"] == cls]
        ref = [i for i in ids if i in reference]
        lost = sorted(i for i in ref if not arm[i]["correct"])
        counts[cls] = {"rows": len(ids), "reference_n": len(ref), "lost": len(lost), "lost_ids": lost,
                       "retained": len(ref) - len(lost),
                       "gained": sum(1 for i in ids if arm[i]["correct"] and i not in reference),
                       "terminated": sum(1 for i in ids if arm[i]["terminated"]),
                       "source_correct_terminated": sum(1 for i in ids if source[i]["correct"]),
                       "arm_correct": sum(1 for i in ids if arm[i]["correct"])}
    return counts


def score_ret(ctx: dict, outputs: list[dict]) -> dict:
    """retention_probe's per-policy block on arm outputs [{id, reply_class, text, terminated, new_tokens}]."""
    R, policy = ctx["R"], ctx["policy"]
    ids = {o["id"] for o in outputs}
    rows = [r for r in ctx["rows"] if r["id"] in ids]
    if len(ids) != len(outputs):
        raise SanityError("duplicate probe row id in predictions")
    by = {}
    for name, ov in ctx["overlays"].items():
        arm = {}
        for o in outputs:
            t = ov["targets"].get(o["id"])
            if t is None:
                raise SanityError(f"no reviewed target for probe row {o['id']}")
            arm[o["id"]] = {**R.score_prediction(o["text"], o["terminated"], t), "reply_class": o["reply_class"],
                            "new_tokens": o["new_tokens"]}
        src = {i: v for i, v in ctx["source_scored"][name].items() if i in ids}
        source_correct = {i for i, r in src.items() if r["correct"]}
        arm_correct = {i for i, r in arm.items() if r["correct"]}
        reference = source_correct | arm_correct
        counts = retention_classes(rows, src, arm, reference, policy["probe_classes"])
        verdict = R.monitor_verdict(counts, policy, policy["probe_classes"])
        by[name] = {"status": verdict["status"], "failed_classes": verdict["failed_classes"],
                    "uninformative_classes": verdict["uninformative_classes"],
                    "source_correct_total": len(source_correct), "arm_correct_total": len(arm_correct),
                    "reference_size": len(reference), "lost_total": sum(c["lost"] for c in counts.values()),
                    "arm_correct_outside_source": len(reference) - len(source_correct), "per_class": counts,
                    "retained": len(source_correct) - sum(c["lost"] for c in counts.values()),
                    "lost_ids": sorted(i for c in counts.values() for i in c["lost_ids"]),
                    "correct_ids": sorted(arm_correct)}
    return {"by_bare_policy": by, "arithmetic_permissive": by["permissive"]["arm_correct_total"],
            "arithmetic_strict": by["strict"]["arm_correct_total"],
            "retained": by["permissive"]["retained"], "retained_strict": by["strict"]["retained"],
            "registered_bare_policy": policy["bare_policy"]}


def lc_normalize(value: str) -> str:
    return " ".join(str(value).strip().lower().replace(".", " ").split())


def score_lc(key: dict, preds: dict) -> dict:
    """long_context_probe's acceptance rule plus both literal-UNKNOWN counters (gap G4)."""
    scored, by = {}, {}
    for i, record in preds.items():
        target = key[i]
        accepted = target.get("accepted_answers") or [target["answer"]]
        first = record["prediction"].splitlines()[0] if record["prediction"] else ""
        hit = lc_normalize(first) in {lc_normalize(x) for x in accepted}
        scored[i] = hit
        for stratum in ("split", "variant", "task_type", "needle_profile", "difficulty"):
            b = by.setdefault(stratum, {}).setdefault(target[stratum], {"items": 0, "hits": 0})
            b["items"] += 1
            b["hits"] += int(hit)
    unans = [i for i in preds if key[i]["variant"] == "unanswerable"]
    answ = [i for i in preds if key[i]["variant"] != "unanswerable"]
    return {"all_hits": sum(scored.values()), "rows": len(preds),
            "showcase_hits": sum(1 for i, v in scored.items() if v and key[i]["split"] == "showcase"),
            "stability_hits": sum(1 for i, v in scored.items() if v and key[i]["split"] == "stability_test"),
            "rows_skipped_context_overflow": sum(1 for r in preds.values() if r.get("status") == "skipped_context_overflow"),
            "unanswerable_rows": len(unans), "unanswerable_correct": sum(scored[i] for i in unans),
            "says_unknown_substring": sum(1 for i in unans if "unknown" in lc_normalize(preds[i]["prediction"])),
            "says_unknown_literal": sum(1 for i in unans if preds[i]["prediction"].strip() == "UNKNOWN"),
            "false_abstention_answerable": sum(1 for i in answ if "unknown" in lc_normalize(preds[i]["prediction"])),
            "per_row_correct": scored, "by_stratum": by}


def score_ood(preds: dict, bank: dict) -> dict:
    """C6's scorer: a row is correct iff rcwmt.env.parse_state(family, prediction, lenient=True) equals the oracle
    state (the bank's gold, rcwmt.env.step). preds: {id: {"prediction", "terminated", ...}}; bank: {id: row}."""
    E = ood_env()
    per_row, detail = {}, []
    for i, rec in preds.items():
        row = bank[i]
        got = E.parse_state(row["family"], rec["prediction"], lenient=True)
        ok = got is not None and got == json.loads(row["gold"])
        per_row[i] = ok
        detail.append({"id": i, "stratum": row["stratum"], "correct": ok, "parsed": got is not None,
                       "terminated": bool(rec.get("terminated"))})
    per_stratum = {}
    for s in OOD_STRATA:
        d = [x for x in detail if x["stratum"] == s]
        if d:
            per_stratum[s] = {"n": len(d), "correct": sum(x["correct"] for x in d), "parsed": sum(x["parsed"] for x in d),
                              "terminated": sum(x["terminated"] for x in d)}
    return {"per_row_correct": per_row, "per_stratum": per_stratum, "rows": len(detail),
            "correct": sum(per_row.values())}


def mcnemar_exact(base: dict, arm: dict) -> dict:
    """tabulate_sparse.mcnemar_exact: two-sided exact McNemar on discordant pairs (base -> arm)."""
    keys = set(base) & set(arm)
    if len(keys) != len(base) or len(keys) != len(arm):
        return {"error": "ROW ID MISMATCH"}
    lost = sum(1 for k in keys if base[k] and not arm[k])
    gained = sum(1 for k in keys if not base[k] and arm[k])
    n = lost + gained
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(lost, gained) + 1)) / 2 ** n)
    return {"lost": lost, "gained": gained, "net": gained - lost, "discordant": n, "p": p}


# ---------------------------------------------------------------------------------------------- drift statistics
def _lsm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def lp_compare(P: np.ndarray, Q: np.ndarray, chunk: int = 8):
    """t2_analysis.lp_compare: per-row KL(P||Q) of renormalised stored log-probs, and the argmax of each."""
    n = P.shape[0]
    kl, ap, aq = np.zeros(n), np.zeros(n, np.int64), np.zeros(n, np.int64)
    for s in range(0, n, chunk):
        a, b = _lsm(np.asarray(P[s:s + chunk])), _lsm(np.asarray(Q[s:s + chunk]))
        kl[s:s + chunk] = (np.exp(a) * (a - b)).sum(1)
        ap[s:s + chunk], aq[s:s + chunk] = a.argmax(1), b.argmax(1)
    return np.maximum(kl, 0.0), ap, aq


def boot_idx(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def pct(x) -> list:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [float("nan"), float("nan")]


def ratio(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict:
    """localized_analysis.ratio: paired percentile bootstrap of a ratio of means over prompts."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    nb, db = num[idx].mean(1), den[idx].mean(1)
    ok = db > 0
    return {"ratio": float(num.mean() / den.mean()) if den.mean() > 0 else float("nan"),
            "ci95": pct(nb[ok] / db[ok]), "dropped_zero_denominator": int((~ok).sum()),
            "diff": float(num.mean() - den.mean()), "diff_ci95": pct(nb - db),
            "num_mean": float(num.mean()), "den_mean": float(den.mean())}


def drift_block(lp: dict, n_boot: int = BOOT, seed: int = BOOT_SEED) -> dict:
    n = lp["supa"].shape[0]
    idx = boot_idx(n, n_boot, seed)
    per = {"kl_arm_supa": {}, "kl_supa_arm": {}, "top1_flip": {}}
    out = {"n_prompts": n, "arms": {}}
    for arm in ARMS:
        k_as, a_arm, a_supa = lp_compare(lp[arm], lp["supa"])
        k_sa, _, _ = lp_compare(lp["supa"], lp[arm])
        per["kl_arm_supa"][arm], per["kl_supa_arm"][arm] = k_as, k_sa
        per["top1_flip"][arm] = (a_arm != a_supa).astype(float)
        out["arms"][arm] = {"kl_arm_supa_mean": float(k_as.mean()), "kl_arm_supa_median": float(np.median(k_as)),
                            "kl_supa_arm_mean": float(k_sa.mean()), "top1_flips": int((a_arm != a_supa).sum())}
    out["ratio_l8_15_over_sparse"] = {
        "kl_arm_supa": ratio(per["kl_arm_supa"]["l8_15"], per["kl_arm_supa"]["sparse"], idx),
        "kl_supa_arm": ratio(per["kl_supa_arm"]["l8_15"], per["kl_supa_arm"]["sparse"], idx),
        "top1_flip": ratio(per["top1_flip"]["l8_15"], per["top1_flip"]["sparse"], idx)}
    ks = per["kl_arm_supa"]["sparse"]
    top = np.sort(ks)[::-1][: max(1, int(round(0.10 * n)))]
    out["flags"] = [f for f, bad in (
        ("CAUTION: Titan-Sparse flips Supa's first token on < 10 prompts", per["top1_flip"]["sparse"].sum() < 10),
        ("CAUTION: the top 10% of prompts carry > 50% of Titan-Sparse's KL", ks.sum() > 0 and top.sum() / ks.sum() > 0.5))
        if bad]
    return out


# ============================================================================================ decision rule
def c2_family(m: dict, family: str) -> dict:
    """C2 for one family from its paired exact McNemar record {lost, gained, net, p} (L8-15 vs Titan-Sparse)."""
    tol_ok = m["net"] >= -ACQ_FAMILY_TOL[family]
    detected_loss = m["lost"] > m["gained"] and m["p"] < DETECT_P
    return {"net": m["net"], "lost": m["lost"], "gained": m["gained"], "p": m["p"], "tolerance": -ACQ_FAMILY_TOL[family],
            "within_tolerance": tol_ok, "detected_loss": detected_loss, "pass": tol_ok and not detected_loss}


def ood_tol(n: int) -> int:
    """C6's per-stratum margin: max(3, ceil(5% of n)); 5 at the registered n = 100."""
    return max(OOD_TOL_MIN, math.ceil(OOD_TOL_FRAC * n - 1e-9))


def c6_stratum(m: dict, stratum: str) -> dict:
    """C6 for one stratum from its paired exact McNemar record {lost, gained, net, p, n} (L8-15 vs Titan-Sparse)."""
    if "error" in m or not m.get("n"):
        return {"stratum": stratum, "error": m.get("error", "no rows"), "pass": False}
    tol = ood_tol(m["n"])
    tol_ok = m["net"] >= -tol
    detected_loss = m["lost"] > m["gained"] and m["p"] < DETECT_P
    return {"n": m["n"], "net": m["net"], "lost": m["lost"], "gained": m["gained"], "p": m["p"], "tolerance": -tol,
            "within_tolerance": tol_ok, "detected_loss": detected_loss, "pass": tol_ok and not detected_loss}


def c6_min_lost_to_fail(n: int) -> int:
    """The fewest rows L8-15 must lose (gaining none) for a stratum of n rows to fail C6, by the margin (net <
    -max(3, ceil(5% n))) or by the detection clause (lost > gained, exact McNemar p < 0.05), whichever binds first.
    6 at n = 100 (the margin and the detection clause coincide there), 4 at n <= 60."""
    tol, k = ood_tol(n), 1
    while not (k > tol or 2.0 ** (1 - k) < DETECT_P):      # p of k lost, 0 gained is 2 / 2^k
        k += 1
    return k


def c6_binding(sparse_correct: int, n: int) -> dict:
    """REPORT-ONLY. L8-15 can lose only rows Titan-Sparse got right, so a stratum can fail C6 only if Titan-Sparse
    is correct on at least c6_min_lost_to_fail(n) of its rows. Otherwise C6 cannot fail there, whatever L8-15 does,
    and that stratum's pass carries no information."""
    need = c6_min_lost_to_fail(n)
    return {"n": n, "sparse_correct": sparse_correct, "min_lost_to_fail": need, "can_fail": sparse_correct >= need}


CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")


def decide(E: dict) -> dict:
    """The REGISTERED rule as a pure function. E: comparable (bool), comparable_reasons (list), floors_cleared,
    acq_family {family: {lost, gained, net, p}}, arith_perm_l, retained_l, retained_t, drift_ci_upper,
    ood_stratum {stratum: {lost, gained, net, p, n}}."""
    up = E.get("drift_ci_upper")
    fam = E.get("acq_family") or {}
    c2 = {f: c2_family(fam[f], f) for f in FAMILIES if f in fam}
    oo = E.get("ood_stratum") or {}
    c6 = {s: c6_stratum(oo[s], s) for s in OOD_STRATA if s in oo}
    c = {"C0": bool(E["comparable"]),
         "C1": E["floors_cleared"] == 3,
         "C2": len(c2) == len(FAMILIES) and all(v["pass"] for v in c2.values()),
         "C3": E["arith_perm_l"] >= ARITH_BAR,
         "C4": E["retained_l"] >= RETAINED_BAR and E["retained_l"] >= E["retained_t"] - RETAINED_MARGIN,
         "C5": up is not None and isinstance(up, (int, float)) and math.isfinite(up) and up < DRIFT_UPPER,
         "C6": len(c6) == len(OOD_STRATA) and all(v["pass"] for v in c6.values())}
    if not c["C0"]:
        verdict = NON_COMPARABLE
    elif all(c[k] for k in CONDITIONS[1:]):
        verdict = PROMOTE
    else:
        verdict = DO_NOT_PROMOTE
    return {"verdict": verdict, "conditions": c, "c2_by_family": c2, "c6_by_stratum": c6,
            "failed": [k for k in CONDITIONS if not c[k]],
            "inputs": {k: E.get(k) for k in ("comparable", "comparable_reasons", "floors_cleared", "acq_family",
                                             "arith_perm_l", "retained_l", "retained_t", "drift_ci_upper",
                                             "ood_stratum")},
            "rule": DECISION_RULE}


def fragile_flags(E: dict, floors_lo: dict) -> list[str]:
    """Report-only: a C1-C4 outcome within the anchor-parity tolerance of its threshold is FRAGILE - the harness change
    (transformers 4.51.3 -> this environment) is tolerated up to that size on the anchors, so the outcome could
    plausibly flip under the programme's original harness. floors_lo: family -> min canonical count that clears."""
    out = []
    for f, k in E.get("acq_canonical_l", {}).items():
        if abs(k - floors_lo[f]) <= PARITY_TOL["acq_total"]:
            out.append(f"C1 {f}: {k} correct, {floors_lo[f]} needed")
    for f, m in (E.get("acq_family") or {}).items():
        if abs(m["net"] - (-ACQ_FAMILY_TOL[f])) <= PARITY_TOL["acq_total"]:
            out.append(f"C2 {f}: net {m['net']:+d}, tolerance {-ACQ_FAMILY_TOL[f]}")
    if abs(E["arith_perm_l"] - ARITH_BAR) <= PARITY_TOL["arith_permissive"]:
        out.append(f"C3: arithmetic_permissive {E['arith_perm_l']}, bar {ARITH_BAR}")
    if (abs(E["retained_l"] - RETAINED_BAR) <= PARITY_TOL["retained"]
            or abs(E["retained_l"] - (E["retained_t"] - RETAINED_MARGIN)) <= PARITY_TOL["retained"]):
        out.append(f"C4: retained {E['retained_l']}, bars {RETAINED_BAR} and {E['retained_t'] - RETAINED_MARGIN}")
    for s, m in (E.get("ood_stratum") or {}).items():          # C6: paired nets, flagged like C2's
        if "net" in m and m.get("n") and abs(m["net"] - (-ood_tol(m["n"]))) <= PARITY_TOL["acq_total"]:
            out.append(f"C6 {s}: net {m['net']:+d}, tolerance {-ood_tol(m['n'])}")
    return out


def min_count_clearing(family: str, n: int = 128) -> int:
    floor_p = max(PUBLISHED_FLOORS[family]["copy_before"], PUBLISHED_FLOORS[family]["constant_majority"]) / \
        PUBLISHED_FLOORS[family]["n"]
    return next(k for k in range(n + 1) if clopper_pearson(k, n)[0] > floor_p)


def allowed_diff(n: int) -> int:
    return max(PARITY_MIN_ROWS, int(PARITY_FRACTION * n))


def text_parity(ours: dict, theirs: dict) -> dict:
    ids = sorted(ours)
    missing = [i for i in ids if i not in theirs]
    differ = [i for i in ids if i in theirs and ours[i] != theirs[i]]
    n = len(ids)
    ok = not missing and len(differ) <= allowed_diff(n)
    return {"n": n, "equal": n - len(differ) - len(missing), "differ": len(differ), "missing": len(missing),
            "allowed_differ": allowed_diff(n), "agreement": (n - len(differ) - len(missing)) / n if n else float("nan"),
            "differ_ids": differ[:20], "pass": ok}


# ============================================================================================ model backends
def _torch(device: str):
    if device == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    if device == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
    return torch


def _hash_params(named, torch) -> dict:
    """run_patch.group_hashes over (hf_name, tensor) pairs."""
    groups: dict[str, list] = {}
    for name, p in named:
        groups.setdefault(group_key(name), []).append((name, p))
    out = {}
    with torch.no_grad():
        for key in group_sort(groups):
            h = hashlib.sha1()
            for name, p in sorted(groups[key], key=lambda kv: kv[0]):
                h.update(name.encode())
                flat = p.detach().reshape(-1)
                for s in range(0, flat.numel(), HASH_CHUNK):
                    h.update(flat[s: s + HASH_CHUNK].float().cpu().numpy().tobytes())
            out[key] = h.hexdigest()[:16]
    return out


class DenseBackend:
    """The programme's DenseReference (vendored 5b4c22d), weights streamed tensor by tensor to the device as FP32.
    Equivalent to run._load (load_file -> import_qwen_state -> .to(device, float32)) without its CPU state-dict clone:
    same key map, same shape / finiteness checks, the same FP32 values."""
    name = "dense_reference@5b4c22d"

    def __init__(self, torch, dev):
        self.torch, self.dev = torch, dev
        self.V = vendor_forward()
        self.mapping = None

    def config(self, cfg_path: Path):
        raw = read_json(cfg_path)
        if _DENSE_CONFIG_HOOK is not None:
            return _DENSE_CONFIG_HOOK(raw)
        return self.V["source"].source_config(raw)

    def load(self, weight_file: Path, cfg_path: Path):
        torch = self.torch
        from safetensors import safe_open
        c = self.config(cfg_path)
        mapping = self.V["transfer"].native_key_map(c)
        with torch.device("meta"):
            m = self.V["model"].DenseReference(c)
        target = m.state_dict()
        state = {}
        with torch.no_grad(), safe_open(str(weight_file), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "lm_head.weight" in keys:
                if not torch.equal(f.get_tensor("lm_head.weight"), f.get_tensor("model.embed_tokens.weight")):
                    raise SanityError("untied or changed output head is forbidden")
                keys.discard("lm_head.weight")
            if keys != set(mapping):
                raise SanityError(f"native tensor inventory mismatch: missing {sorted(set(mapping) - keys)[:4]}, "
                                  f"unexpected {sorted(keys - set(mapping))[:4]}")
            for src, dst in mapping.items():
                t = f.get_tensor(src)
                if tuple(t.shape) != tuple(target[dst].shape) or not t.is_floating_point():
                    raise SanityError(f"invalid native tensor: {src}")
                t = t.to(device=self.dev, dtype=torch.float32)
                if not bool(torch.isfinite(t).all()):
                    raise SanityError(f"non-finite native tensor: {src}")
                state[dst] = t
        m.load_state_dict(state, strict=True, assign=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        if not all(p.dtype == torch.float32 and p.device.type == torch.device(self.dev).type for p in m.parameters()):
            raise SanityError("G3: a parameter is not FP32 on the device")
        self.mapping = mapping
        return m

    def group_hashes(self, m) -> dict:
        inv = {dst: src for src, dst in self.mapping.items()}
        return _hash_params(((inv[n], p) for n, p in m.named_parameters()), self.torch)

    def step(self, m, cur, past):
        out = m(cur, past=past, use_cache=True, last_position_only=True)
        return out.logits[0, -1], out.cache

    def first_logits(self, m, ids):
        return m(ids, last_position_only=True).logits[0, -1]


class HFBackend:
    """Qwen3ForCausalLM (transformers in this environment), FP32. Tiny-model cross-check and fallback only."""
    name = "hf_qwen3"

    def __init__(self, torch, dev):
        self.torch, self.dev = torch, dev

    def load(self, weight_file: Path, cfg_path: Path):
        torch = self.torch
        from safetensors import safe_open
        from transformers import Qwen3Config, Qwen3ForCausalLM
        raw = read_json(cfg_path)
        cfg = Qwen3Config(**{k: v for k, v in raw.items() if k not in ("torch_dtype", "transformers_version")})
        cfg._attn_implementation = "sdpa"
        with torch.device(self.dev):
            m = Qwen3ForCausalLM(cfg).to(torch.float32)
        params = dict(m.named_parameters())
        with torch.no_grad(), safe_open(str(weight_file), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            want = {k for k in params if not (k == "lm_head.weight" and cfg.tie_word_embeddings)}
            if keys - {"lm_head.weight"} != want:
                raise SanityError("HF tensor inventory mismatch")
            for k in sorted(want):
                params[k].data.copy_(f.get_tensor(k).to(torch.float32))
        m.eval()
        return m

    def group_hashes(self, m) -> dict:
        return _hash_params(m.named_parameters(), self.torch)

    def step(self, m, cur, past):
        out = m(input_ids=cur, past_key_values=past, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1], out.past_key_values

    def first_logits(self, m, ids):
        return m(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]


def _ram_gib() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


class HostGuard:
    """Host-safety checks of the ops review (DECISIONS I-23, I-26, I-27). real=False (CPU, tests) makes every check a
    no-op. Every wait here holds no model; pre_lease() also holds no lease."""

    def __init__(self, real: bool, root: Path | None = None, ram_fn=None, clock=None, sleep=None):
        self.real = real
        self.root = Path(root) if root is not None else ROOT
        self.ram_fn = ram_fn or _ram_gib
        self.clock, self.sleep = clock or time.time, sleep or time.sleep
        self.last_wd_check = -1e18

    def flag(self) -> Path:
        return self.root / "runs" / ".thermal_pause"

    def watchdog_age(self) -> float:
        try:
            return self.clock() - (self.root / "runs" / "thermal_log.txt").stat().st_mtime
        except OSError:
            return float("inf")

    def pre_lease(self) -> None:
        """Before the lease is requested: RAM >= 6 GiB and a live watchdog; wait (nothing held) up to 2 h, then exit 5."""
        if not self.real:
            return
        t0, said = self.clock(), False
        while True:
            ram, age = self.ram_fn(), self.watchdog_age()
            if ram >= PRE_LEASE_RAM_GIB and age <= WATCHDOG_STALE_S:
                return
            if not said:
                log(f"host gate (no lease held): available RAM {ram:.1f} GiB (need {PRE_LEASE_RAM_GIB}), watchdog log "
                    f"age {age:.0f} s (need <= {WATCHDOG_STALE_S}); waiting")
                said = True
            if self.clock() - t0 > PRE_LEASE_WAIT_S:
                raise ResumableExit(f"host gate not met within {PRE_LEASE_WAIT_S} s (RAM {ram:.1f} GiB, watchdog "
                                    f"age {age:.0f} s)")
            self.sleep(60)

    def before_load(self) -> None:
        """Inside the lease, before each model load: the pause flag clear, a live watchdog, RAM >= 5 GiB within 5 min."""
        if not self.real:
            return
        t0 = self.clock()
        while self.flag().exists():
            if self.clock() - t0 > PAUSE_WAIT_BEFORE_LOAD_S:
                raise ResumableExit("the watchdog's pause flag stayed set before a load")
            if self.watchdog_age() > WATCHDOG_STALE_S:
                raise ResumableExit("pause flag set and the watchdog is silent")
            self.sleep(5)
        if self.watchdog_age() > WATCHDOG_STALE_S:
            raise ResumableExit(f"the watchdog has not logged for {self.watchdog_age():.0f} s; not loading")
        t0 = self.clock()
        while self.ram_fn() < LOAD_RAM_GIB:
            if self.clock() - t0 > LOAD_RAM_WAIT_S:
                raise ResumableExit(f"host RAM stayed below {LOAD_RAM_GIB} GiB for {LOAD_RAM_WAIT_S} s; not loading")
            self.sleep(15)

    def during(self) -> None:
        """Between rows and inside the token pause: the watchdog alive (checked at most once a minute), and a pause flag
        waited out here - unless RAM stays < 5 GiB under it for 15 min or the watchdog dies, which ends the run (5)."""
        if not self.real:
            return
        now = self.clock()
        if now - self.last_wd_check >= 60:
            self.last_wd_check = now
            if self.watchdog_age() > WATCHDOG_STALE_S:
                raise ResumableExit(f"the watchdog has not logged for {self.watchdog_age():.0f} s")
        low_since = None
        while self.flag().exists():
            if self.ram_fn() < LOAD_RAM_GIB:
                low_since = low_since if low_since is not None else self.clock()
                if self.clock() - low_since > RAM_PAUSE_EXIT_S:
                    raise ResumableExit(f"a RAM-caused pause (< {LOAD_RAM_GIB} GiB) lasted {RAM_PAUSE_EXIT_S} s; "
                                        f"releasing the card")
            else:
                low_since = None
            if self.watchdog_age() > WATCHDOG_STALE_S:
                raise ResumableExit("pause flag set and the watchdog is silent")
            self.sleep(2)


class Pace:
    """GPU pacing: runtime.breathe() after every row; a short pause and a flag check every 8 decode steps; the host
    guard before each of them."""

    def __init__(self, cuda: bool, torch, runtime=None, guard: HostGuard | None = None):
        self.cuda, self.torch, self.runtime = cuda, torch, runtime
        self.guard = guard or HostGuard(False)
        self.rows = 0

    def row(self) -> None:
        self.rows += 1
        if self.cuda:
            self.guard.during()
            if self.runtime is not None:
                self.runtime.breathe()

    def token(self, i: int) -> None:
        if self.cuda and (i + 1) % TOKEN_PAUSE_EVERY == 0:
            if _TENSOR_DEVICE is None:
                self.torch.cuda.synchronize()
            time.sleep(TOKEN_PAUSE_S)
            if self.guard.real:
                self.guard.during()
            else:
                flag = ROOT / "runs" / ".thermal_pause"
                while flag.exists():
                    time.sleep(2)


# ---------------------------------------------------------------------------------- no process spawns (I-27)
def _psutil_python_rss() -> dict:
    """lease_vendored._heavy_python_pids without `tasklist`: python.exe processes and their RSS in bytes."""
    import psutil
    out = {}
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if (p.info.get("name") or "").lower() == "python.exe" and p.info.get("memory_info") is not None:
                out[int(p.info["pid"])] = int(p.info["memory_info"].rss)
        except (psutil.Error, KeyError, TypeError, ValueError):
            continue
    return out


def _psutil_pid_alive(pid) -> bool:
    """lease_vendored._pid_alive without `tasklist`; cannot tell -> alive (never reclaim on ignorance)."""
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True


def pdh_dedicated_gpu_bytes() -> dict:
    """pid -> dedicated GPU memory in bytes, from the same Windows counter lease_vendored reads through PowerShell
    ('\\GPU Process Memory(*)\\Dedicated Usage'), read in-process through pdh.dll. OSError if unavailable."""
    import ctypes
    from ctypes import byref, c_void_p, wintypes

    class _FMT(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

    class _ITEM(ctypes.Structure):
        _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _FMT)]

    lib = ctypes.WinDLL("pdh.dll")
    q, c = c_void_p(), c_void_p()
    if lib.PdhOpenQueryW(None, None, byref(q)) != 0:
        raise OSError("PdhOpenQueryW failed")
    try:
        if lib.PdhAddEnglishCounterW(q, ctypes.c_wchar_p(r"\GPU Process Memory(*)\Dedicated Usage"), None,
                                     byref(c)) != 0:
            raise OSError("PdhAddEnglishCounterW failed")
        if lib.PdhCollectQueryData(q) != 0:
            raise OSError("PdhCollectQueryData failed")
        size, count, fmt = wintypes.DWORD(0), wintypes.DWORD(0), wintypes.DWORD(0x00000400)   # PDH_FMT_LARGE
        st = lib.PdhGetFormattedCounterArrayW(c, fmt, byref(size), byref(count), None) & 0xFFFFFFFF
        if st != 0x800007D2:                                                              # PDH_MORE_DATA
            if st == 0 and size.value == 0:
                return {}
            raise OSError(f"PdhGetFormattedCounterArrayW size query {st:#x}")
        buf = (ctypes.c_byte * size.value)()
        if lib.PdhGetFormattedCounterArrayW(c, fmt, byref(size), byref(count), buf) != 0:
            raise OSError("PdhGetFormattedCounterArrayW failed")
        items = ctypes.cast(buf, ctypes.POINTER(_ITEM))
        res: dict = {}
        for i in range(count.value):
            name = items[i].szName or ""
            if items[i].FmtValue.CStatus not in (0, 1) or not name.startswith("pid_"):
                continue
            try:
                pid = int(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            res[pid] = res.get(pid, 0) + int(items[i].FmtValue.largeValue)
        return res
    finally:
        lib.PdhCloseQuery(q)


@contextmanager
def noproc_lease():
    """For this process only, replace rcwmt.lease_vendored's spawning helpers (`tasklist` for the python process
    table and for pid liveness, PowerShell Get-Counter for dedicated GPU memory - 2-3 processes per 20 s poll while
    waiting, I-27) with psutil and an in-process PDH read of the same counter. Semantics are unchanged (I-17: a
    foreign python.exe with > 256 MiB dedicated GPU memory still blocks). If PDH is unavailable, the original
    PowerShell read is used at most once per 2 minutes, its result cached in between."""
    sys.path.insert(0, str(ROOT))
    from rcwmt import lease_vendored as LV
    names = ("_heavy_python_pids", "_pid_alive", "_dedicated_gpu_bytes")
    orig = {k: getattr(LV, k) for k in names}
    last = {"t": -1e18, "v": {}}

    def dedicated():
        try:
            return pdh_dedicated_gpu_bytes()
        except Exception:
            if time.time() - last["t"] >= 120:
                last["t"], last["v"] = time.time(), orig["_dedicated_gpu_bytes"]()
            return last["v"]
    LV._heavy_python_pids, LV._pid_alive, LV._dedicated_gpu_bytes = _psutil_python_rss, _psutil_pid_alive, dedicated
    try:
        yield LV
    finally:
        for k, v in orig.items():
            setattr(LV, k, v)


@contextmanager
def noproc_sensors(runtime):
    """Replace runtime.gpu_temp_c (an nvidia-smi spawn every ~10 s) with the in-process NVML read of
    scripts/sensors_noproc.py for this process (I-27); nvidia-smi only as a <= 2/min fallback."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import sensors_noproc
    orig = runtime.gpu_temp_c
    last = [0.0, None]

    def temp():
        t, _, _ = sensors_noproc.gpu()
        if t is not None:
            return t
        if time.time() - last[0] >= 30:
            last[0], last[1] = time.time(), orig()
        return last[1]
    runtime.gpu_temp_c = temp
    try:
        yield
    finally:
        runtime.gpu_temp_c = orig


def render(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode(tok, messages) -> list[int]:
    return list(tok(render(tok, messages), add_special_tokens=False)["input_ids"])


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(resolve("supa_dir")), local_files_only=True)


def eos_ids(tok) -> set[int]:
    raw = read_json(resolve("supa_dir") / "generation_config.json")
    eos = raw.get("eos_token_id", tok.eos_token_id)
    values = eos if isinstance(eos, list) else [eos]
    if not values or any(type(v) is not int or not 0 <= v < len(tok) for v in values):
        raise SanityError("official generation EOS IDs are missing or invalid")
    return set(values)


def greedy(be, m, ids: list[int], max_new: int, eos: set[int], pace: Pace) -> tuple[list[int], bool]:
    torch = be.torch
    current = torch.tensor([ids], dtype=torch.long, device=be.dev)
    past, gen, term = None, [], False
    with torch.inference_mode():
        for i in range(max_new):
            logits, past = be.step(m, current, past)
            token = int(logits.argmax())
            if token in eos:
                term = True
                break
            gen.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=be.dev)
            pace.token(i)
    del past
    return gen, term


# ============================================================================================ run
def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def arm_files() -> dict:
    return {"supa": resolve("supa_dir") / "model.safetensors", "sparse": resolve("sparse_dir") / "model.safetensors",
            "l8_15": None}


def stage_done(d: Path, stage: str, ident: str) -> bool:
    try:
        st, arm = read_json(d / "STAGES.json"), read_json(d / "ARM.json")
    except (OSError, ValueError):
        return False
    if st.get("identity") != ident or arm.get("identity") != ident or stage not in st.get("stages", {}):
        return False
    files = st["stages"][stage].get("sha256", {})
    return all((d / f).exists() and files.get(f) == sha256_file(d / f) for f in STAGE_FILES[stage])


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _progress(arm: str, stage: str, done: int, total: int, t0: float) -> None:
    if done % PROGRESS_EVERY == 0 and done < total:
        log(f"{arm}: {stage} {done}/{total} rows ({time.time() - t0:.0f}s)")


def run_stage(stage: str, be, m, tok, eos: set[int], d: Path, inputs: dict, pace: Pace, arm: str = "") -> dict:
    torch = be.torch
    t0 = time.time()
    rows = inputs[stage]
    info = {"rows": len(rows)}
    if stage == "acq":
        recs = []
        for j, row in enumerate(rows):
            ids = encode(tok, row["prompt"])
            gen, term = greedy(be, m, ids, ACQ_MAX_NEW, eos, pace)
            recs.append({"id": row["id"], "family": row["family"], "input_tokens": len(ids), "generated_tokens": len(gen),
                         "terminated": term, "prediction": tok.decode(gen, skip_special_tokens=True).strip()})
            pace.row()
            _progress(arm, stage, j + 1, len(rows), t0)
        _write_jsonl(d / "acq.jsonl", recs)
        info["generations"] = len(recs)
    elif stage == "ood":                                   # C6: Titan's one-step frame, as stored in the bank
        recs = []
        for j, row in enumerate(rows):
            ids = encode(tok, row["messages"])
            gen, term = greedy(be, m, ids, OOD_MAX_NEW, eos, pace)
            recs.append({"id": row["id"], "stratum": row["stratum"], "family": row["family"], "input_tokens": len(ids),
                         "generated_tokens": len(gen), "terminated": term,
                         "prediction": tok.decode(gen, skip_special_tokens=True).strip()})
            pace.row()
            _progress(arm, stage, j + 1, len(rows), t0)
        _write_jsonl(d / "ood.jsonl", recs)
        info["generations"] = len(recs)
    elif stage == "ret":
        mx, ctx = inputs["ret_decoding"]["max_new_tokens"], inputs["ret_decoding"]["max_context"]
        recs = []
        for j, row in enumerate(rows):
            ids = encode(tok, row["messages"])
            if not 1 <= len(ids) <= ctx:
                raise SanityError(f"G3: probe prompt {row['id']} has {len(ids)} tokens, outside [1, {ctx}]; "
                                  f"truncation is forbidden")
            gen, term = greedy(be, m, ids, mx, eos, pace)
            recs.append({"id": row["id"], "reply_class": row["reply_class"], "input_tokens": len(ids),
                         "new_tokens": len(gen), "terminated": term,
                         "text": tok.decode(gen, skip_special_tokens=True).strip()})
            pace.row()
            _progress(arm, stage, j + 1, len(rows), t0)
        _write_jsonl(d / "ret.jsonl", recs)
        info["generations"] = len(recs)
    elif stage == "lc":
        recs, skipped = [], 0
        for j, row in enumerate(rows):
            ids = encode(tok, lc_messages(row))
            if len(ids) + LC_MAX_NEW > LC_MAX_CONTEXT:
                skipped += 1
                recs.append({"id": row["id"], "status": "skipped_context_overflow", "prediction": "",
                             "input_tokens": len(ids), "context_limit": LC_MAX_CONTEXT})
                continue
            gen, term = greedy(be, m, ids, LC_MAX_NEW, eos, pace)
            recs.append({"id": row["id"], "status": "ok", "input_tokens": len(ids), "generated_tokens": len(gen),
                         "terminated": term, "prediction": tok.decode(gen, skip_special_tokens=True).strip()})
            pace.row()
            _progress(arm, stage, j + 1, len(rows), t0)
        _write_jsonl(d / "lc.jsonl", recs)
        info.update(generations=len(recs) - skipped, rows_skipped_context_overflow=skipped)
    elif stage in ("drift_fresh", "drift_bridge"):
        npy, idf = STAGE_FILES[stage]
        V = None
        tmp = d / (npy + ".tmp")
        arr = None
        with torch.inference_mode():
            for j, row in enumerate(rows):
                ids = encode(tok, row["messages"])
                lg = be.first_logits(m, torch.tensor([ids], dtype=torch.long, device=be.dev)).float()
                lp = torch.log_softmax(lg, -1).clamp(min=-60.0).to(torch.float16).cpu().numpy()
                if arr is None:
                    V = lp.shape[-1]
                    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(rows), V))
                arr[j] = lp
                pace.row()
                _progress(arm, stage, j + 1, len(rows), t0)
        arr.flush()
        del arr
        os.replace(tmp, d / npy)
        write_json(d / idf, {"ids": [r["id"] for r in rows], "reply_class": [r.get("reply_class") for r in rows],
                             "vocab": V, "dtype": "float16", "clamp_min": -60.0,
                             "what": "first-answer-token log-probabilities (log_softmax of FP32 logits; DenseReference, "
                                     "batch 1, no padding; storage exactly as in R14)"})
        info["forwards"] = len(rows)
    else:
        raise ValueError(stage)
    info.update(seconds=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                sha256={f: sha256_file(d / f) for f in STAGE_FILES[stage]})
    return info


def stage_inputs() -> dict:
    views = load_views()
    check_floors(views)
    policy, decoding = protocol_policy()
    probe = load_probe_rows(policy["probe_classes"])
    lc_rows, _ = load_lc()
    return {"acq": subset_rows("acq", views), "ret": subset_rows("ret", probe),
            "lc": subset_rows("lc", lc_rows), "drift_fresh": subset_rows("drift_fresh", load_prompts("fresh")),
            "drift_bridge": subset_rows("drift_bridge", load_prompts("bridge")),
            "ood": subset_rows("ood", load_ood_bank()), "ret_decoding": decoding}


def run_keys(inputs: dict) -> dict:
    return {s: {"n": len(inputs[s]), "ids_sha256": sha256_text("\n".join(r["id"] for r in inputs[s]))} for s in STAGES}


def finalize(out_dir: Path, idents: dict) -> bool:
    status = {arm: {s: stage_done(out_dir / arm, s, idents[arm]) for s in STAGES} for arm in ARMS}
    if not all(all(v.values()) for v in status.values()):
        log("not complete yet: " + ", ".join(f"{a}:{s}" for a, v in status.items() for s, ok in v.items() if not ok))
        return False
    arms = {}
    for arm in ARMS:
        st, rec = read_json(out_dir / arm / "STAGES.json"), read_json(out_dir / arm / "ARM.json")
        arms[arm] = {"identity": rec["identity"], "weight_sha256": rec["weight_sha256"],
                     "files_sha256": {f: h for s in STAGES for f, h in st["stages"][s]["sha256"].items()},
                     "stages": {s: {k: v for k, v in st["stages"][s].items() if k != "sha256"} for s in STAGES}}
    gate = read_json(out_dir / "PARITY_GATE.json") if (out_dir / "PARITY_GATE.json").exists() else {}
    write_json(out_dir / "DONE.json", {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "finished_utc": _utc(),
                                       "arms": arms, "run": read_json(out_dir / "RUN.json").get("keys"),
                                       "parity_gate": {k: gate.get(k) for k in ("pass", "enforced", "failed_legs")}})
    log(f"wrote {out_dir / 'DONE.json'}")
    return True


# ---------------------------------------------------------------------------------------------- environment (G0)
def env_record() -> dict:
    """The software environment the freeze pins. Reads package metadata only (no torch import)."""
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except Exception:
            return None
    return {"python_executable": os.path.normcase(os.path.abspath(sys.executable)), "python": sys.version.split()[0],
            "torch": ver("torch"), "transformers": ver("transformers"), "tokenizers": ver("tokenizers"),
            "safetensors": ver("safetensors"), "numpy": np.__version__}


def nvml_info() -> dict:
    """NVIDIA driver version and GPU name through NVML (the management API: no CUDA context, no process)."""
    import ctypes
    lib = None
    for path in ("nvml.dll", r"C:\Windows\System32\nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"):
        try:
            lib = ctypes.WinDLL(path)
            break
        except OSError:
            continue
    if lib is None or lib.nvmlInit_v2() != 0:
        return {"nvidia_driver": None, "gpu_name": None}
    try:
        b = ctypes.create_string_buffer(96)
        drv = b.value.decode() if lib.nvmlSystemGetDriverVersion(b, ctypes.c_uint(96)) == 0 else None
        h, name = ctypes.c_void_p(), None
        if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(h)) == 0:
            nb = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(h, nb, ctypes.c_uint(96)) == 0:
                name = nb.value.decode()
        return {"nvidia_driver": drv, "gpu_name": name}
    finally:
        lib.nvmlShutdown()


# ---------------------------------------------------------------------------------------------- one run per directory
def _proc_create_time(pid) -> float | None:
    try:
        import psutil
        return round(psutil.Process(int(pid)).create_time(), 1)
    except Exception:
        return None


def _lock_alive(rec: dict) -> bool:
    try:
        import psutil
        pid = int(rec.get("pid"))
        if not psutil.pid_exists(pid):
            return False
        ct, now = rec.get("create_time"), _proc_create_time(pid)
        return ct is None or now is None or abs(now - ct) < 2.0          # a reused pid is not the lock's owner
    except (TypeError, ValueError):
        return False
    except Exception:
        return True


@contextmanager
def run_lock(out_dir: Path):
    """Create-only out_dir/.run.lock (pid, process start time). A live owner -> SanityError (one waiter, R14); a dead
    owner's lock is reclaimed. Released on every exit path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / ".run.lock"
    me = {"pid": os.getpid(), "create_time": _proc_create_time(os.getpid()), "started_utc": _utc(), "argv": sys.argv[1:]}
    fd = None
    for attempt in (0, 1):
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                rec = read_json(p)
            except (OSError, ValueError):
                rec = {}
            try:
                age = time.time() - p.stat().st_mtime
            except OSError:
                age = 0.0
            stale = (not _lock_alive(rec)) if rec else age > 60
            if attempt == 0 and stale:
                log(f"reclaiming a stale run lock (pid {rec.get('pid')})")
                p.unlink(missing_ok=True)
                continue
            raise SanityError(f"another l1_screen run (pid {rec.get('pid')}) holds {p}; one waiter only (R14)")
    with os.fdopen(fd, "wb") as fh:
        fh.write((json.dumps(me) + "\n").encode("utf-8"))
    try:
        yield
    finally:
        try:
            if read_json(p).get("pid") == os.getpid():
                p.unlink()
        except (OSError, ValueError):
            pass


def _anchors_complete(out_dir: Path, idents: dict) -> bool:
    """The anchors' pre-gate stages are done: everything the parity gate reads exists (never the OOD stage)."""
    return all(stage_done(out_dir / arm, s, idents[arm]) for arm in ("supa", "sparse") for s in PRE_GATE_STAGES)


def run_passes(pending: dict) -> list[tuple[str, list[str]]]:
    """The registered order as a list of passes (arm, stages); ("GATE", []) is the anchor parity gate (CPU).
    1. Supa, then Titan-Sparse, on their pending pre-gate stages (everything except the OOD bank);
    2. the gate, whenever any post-gate work is pending;
    3. only after a passed gate: Titan-Sparse's and Supa's OOD stage (C6's comparator and its report-only anchor),
       then every pending L8-15 stage.
    So no model reads the OOD bank, and L8-15 is never loaded, unless the gate passed. Titan-Sparse's post-gate pass
    directly follows the gate, so the runner keeps that model loaded across the gate instead of reloading it."""
    passes = [(arm, [s for s in pending[arm] if s in PRE_GATE_STAGES]) for arm in ("supa", "sparse")]
    post = [(arm, [s for s in pending[arm] if s in POST_GATE_STAGES]) for arm in ("sparse", "supa")]
    post.append(("l8_15", list(pending["l8_15"])))
    post = [p for p in post if p[1]]
    return [p for p in passes if p[1]] + ([("GATE", [])] + post if post else [])


def run_parity_gate(out_dir: Path, subset: bool) -> dict:
    """PG: the anchor parity gate on the anchors' saved pre-gate outputs (CPU, no model). Written to PARITY_GATE.json;
    a failed, enforced gate also writes PARITY_FAIL.json (create-only): L8-15 is never loaded and no model reads the
    OOD bank."""
    log("PG: anchor parity gate on supa and sparse (CPU) ...")
    gate = parity_gate(out_dir, subset)
    if _PARITY_HOOK is not None:
        gate = _PARITY_HOOK(gate)
    write_json(out_dir / "PARITY_GATE.json", gate)
    if gate["enforced"] and not gate["pass"]:
        write_create(out_dir / "PARITY_FAIL.json", json.dumps({
            "record": "L1_PARITY_FAIL", "written_utc": _utc(), "failed_legs": gate["failed_legs"],
            "consequence": ("C0 fails: NON-COMPARABLE. L8-15 was never loaded and no L8-15 endpoint exists; no model "
                            "generated on the OOD bank (C6), whose stage runs only after a passed gate. This "
                            "registration ends; a rerun in another environment is a new registration with a fresh "
                            "drift draw that may not reuse the L1 prompts.")}, indent=1) + "\n")
        log(f"PG FAILED ({', '.join(gate['failed_legs'][:6])}): wrote PARITY_FAIL.json; L8-15 is not loaded and no "
            f"model reads the OOD bank")
    else:
        log("PG passed" if gate["pass"] else "PG not enforced (row subset): " + ", ".join(gate["failed_legs"][:6]))
    return gate


def cmd_run(a) -> int:
    if a.wait_for and not wait_for_files(a.wait_for, a.wait_poll, a.wait_max_h):
        return EXIT_WAIT
    cuda = a.device == "cuda"
    real_gpu = cuda and _TENSOR_DEVICE is None
    freeze = None
    if a.no_freeze:
        if real_gpu:
            raise SanityError("G0: --no-freeze is for the tiny-model test only; it refuses a real GPU")
    else:
        bad = verify_freeze()
        if bad:
            raise SanityError(f"freeze check failed: {bad[:6]}")
        freeze = read_json(resolve("freeze"))
        env_now = env_record()
        if freeze.get("environment") != env_now:
            raise SanityError(f"G0: the environment differs from the frozen one: "
                              f"{ {k: (freeze.get('environment', {}).get(k), v) for k, v in env_now.items() if freeze.get('environment', {}).get(k) != v} }")
    torch = _torch(a.device)
    if cuda and _TENSOR_DEVICE is None:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") != "", "--device cuda in a CUDA-hidden process"
    if not cuda:
        torch.set_num_threads(a.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _lower_priority()
    out_dir = Path(a.out_dir) if a.out_dir else OUT_DEFAULT
    if (out_dir / "DONE.json").exists():
        log(f"{out_dir / 'DONE.json'} exists: nothing pending")
        return 0
    if (out_dir / "PARITY_FAIL.json").exists():
        log(f"{out_dir / 'PARITY_FAIL.json'} exists: the anchors failed parity; L8-15 is never run under this registration")
        return 0
    arm_dir = out_dir / "arm-l8_15"
    files = arm_files()
    files["l8_15"] = arm_dir / "model.safetensors"
    if not files["l8_15"].exists():
        raise SanityError(f"no built L8-15 arm at {arm_dir}; run `l1_screen.py build` first")
    with run_lock(out_dir):
        return _run_locked(a, torch, cuda, real_gpu, freeze, out_dir, arm_dir, files)


def _run_locked(a, torch, cuda: bool, real_gpu: bool, freeze, out_dir: Path, arm_dir: Path, files: dict) -> int:
    inputs = stage_inputs()
    keys = {"stages": run_keys(inputs), "subset": _SUBSET, "forward": a.forward,
            "inputs_sha256": {k: SHA[k] for k in ("views", "floors", "lc_prompts", "lc_key", "probe", "audit",
                                                   "protocol", "b28", "offtask_used", "hskill_views_build",
                                                   "ood_bank", "rcwmt_env")},
            "fresh_prompts_sha256": sha256_file(resolve("offtask_fresh"))}
    subset = bool(_SUBSET)
    env = env_record()
    if real_gpu:
        env.update(torch_cuda=torch.version.cuda, **nvml_info())
    old = read_json(out_dir / "RUN.json") if (out_dir / "RUN.json").exists() else None
    if old and old.get("keys") != keys:
        raise SystemExit(f"{out_dir} was started with different inputs/stages/subset; use another --out-dir")
    if old and old.get("invocations") and old["invocations"][0].get("env") != env:
        raise SanityError(f"G0: the environment differs from the first invocation's; stages from two environments are "
                          f"never stitched: {old['invocations'][0].get('env')} vs {env}")
    # G1 before the lease: every arm file's identity
    log("G1: hashing the three arm files (sha256 + per-group FP32 hashes) before the lease ...")
    r = r14_hashes()
    want_sha = {"supa": SHA["supa_weights"], "sparse": SHA["sparse_weights"], "l8_15": None}
    exp = resolve("expected")
    if exp.exists():
        want_sha["l8_15"] = read_json(exp)["out_weight_sha256"]
    elif not a.no_freeze:
        raise SanityError("no frozen L8-15 identity (docs/l1/L8_15_EXPECTED.json)")
    idents, fid = {}, {}
    for arm in ARMS:
        ident = file_identity(files[arm])
        if a.no_freeze and ident["sha256"] in REAL_WEIGHT_SHA:
            raise SanityError(f"G0: --no-freeze is for the tiny-model test only; {arm} is a real core "
                              f"({ident['sha256'][:12]}...)")
        if want_sha[arm] is not None and ident["sha256"] != want_sha[arm]:
            raise SanityError(f"G1: {arm} weights {ident['sha256'][:12]}... are not the pinned {want_sha[arm][:12]}...")
        if ident["group_hashes"] != r[arm]:
            raise SanityError(f"G1: {arm}'s per-group hashes differ from R14's record")
        fid[arm] = ident
        idents[arm] = sha256_text(json.dumps({"sha256": ident["sha256"], "groups": ident["group_hashes"],
                                              "forward": a.forward}, sort_keys=True))
    if arm_dir.exists():
        verify_arm_dir(arm_dir)
    script_sha = sha256_file(Path(__file__))
    freeze_sha = sha256_file(resolve("freeze")) if (freeze is not None and resolve("freeze").exists()) else None
    run = {**(old or {}), "record": "L1_RUN", "keys": keys, "arms": list(ARMS), "lease_label": LEASE_LABEL, "device": a.device,
           "dtype": "float32", "tf32": False, "forward": a.forward, "registered": REGISTERED,
           "arm_identity": {arm: {"sha256": fid[arm]["sha256"], "group_hashes": fid[arm]["group_hashes"]} for arm in ARMS}}
    run.setdefault("invocations", []).append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "started_utc": _utc(),
                                              "argv": sys.argv[1:], "no_freeze": bool(a.no_freeze),
                                              "script_sha256": script_sha, "freeze_sha256": freeze_sha, "env": env})
    inv_no = len(run["invocations"]) - 1
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "RUN.json", run)

    def pending_now() -> dict:
        return {arm: [s for s in STAGES if not stage_done(out_dir / arm, s, idents[arm])] for arm in ARMS}
    pending = pending_now()
    log("pending: " + (", ".join(f"{k}:{'/'.join(v)}" for k, v in pending.items() if v) or "none"))
    if not any(pending.values()):
        finalize(out_dir, idents)
        return 0
    if ("GATE", []) in run_passes(pending) and _anchors_complete(out_dir, idents):
        g = run_parity_gate(out_dir, subset)        # resume after the anchors' pre-gate stages: the gate, no lease
        if g["enforced"] and not g["pass"]:
            return 0
    tok = load_tokenizer()
    eos = eos_ids(tok)
    runtime = None
    if cuda:
        sys.path.insert(0, str(ROOT))
        from rcwmt import runtime as _rt
        runtime = _rt
    dev = torch.device(_TENSOR_DEVICE or a.device)
    be = (DenseBackend if a.forward == "dense" else HFBackend)(torch, dev)
    guard = (_GUARD_FACTORY or HostGuard)(real_gpu)
    guard.pre_lease()                                                  # RAM + live watchdog, nothing held
    quiet = noproc_lease() if cuda else nullcontext()                  # the lease's own helpers spawn nothing (I-27)
    lease = runtime.gpu_lease(label=LEASE_LABEL, job=LEASE_JOB) if cuda else nullcontext()
    # in-process temperature for breathe()/_duty_ratio/thermal_guard (they look up runtime.gpu_temp_c at call time);
    # the lease's own wait_for_resources() reads only VRAM and host RAM
    sensors = noproc_sensors(runtime) if real_gpu else nullcontext()
    gate_failed = False
    with sensors, quiet, lease:
        if (out_dir / "DONE.json").exists() or (out_dir / "PARITY_FAIL.json").exists():
            log("finished by another process while this one waited; nothing to do")
            return 0
        pending = pending_now()                                        # re-derived after the (possibly long) wait
        if not any(pending.values()):
            finalize(out_dir, idents)
            return 0
        if cuda:
            runtime.set_numerics() if _TENSOR_DEVICE is None else None
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        numerics = {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "float32_matmul_precision": torch.get_float32_matmul_precision()}
        if numerics["cuda_matmul_allow_tf32"] or numerics["float32_matmul_precision"] != "highest":
            raise SanityError(f"G3: numerics {numerics}")
        pace = Pace(cuda, torch, runtime, guard)
        held: dict = {"arm": None, "m": None}      # the loaded model (kept across the CPU gate for Titan-Sparse only)
        gate_rec = None

        def release() -> None:
            if held["m"] is not None:
                held["arm"], held["m"] = None, None
                import gc
                gc.collect()
                if cuda and _TENSOR_DEVICE is None:
                    torch.cuda.empty_cache()
        try:
            for arm, stages in run_passes(pending):
                if arm == "GATE":
                    # PG: after both anchors' pre-gate stages; before L8-15 is loaded and before any model reads the
                    # OOD bank
                    gate = run_parity_gate(out_dir, subset)
                    gate_rec = {"pass": gate["pass"], "enforced": gate["enforced"]}
                    if gate["enforced"] and not gate["pass"]:
                        gate_failed = True
                        break
                    continue
                post_gate = arm == "l8_15" or any(s in POST_GATE_STAGES for s in stages)
                if post_gate and gate_rec is None:
                    raise SanityError(f"internal: {arm} {stages} scheduled before the anchor parity gate")
                d = out_dir / arm
                d.mkdir(exist_ok=True)
                if held["arm"] != arm:
                    release()
                    guard.before_load()
                    t0 = time.time()
                    m = be.load(files[arm], resolve("supa_dir") / "config.json")
                    G = be.group_hashes(m)
                    if G != fid[arm]["group_hashes"]:                               # G2
                        raise SanityError(f"G2: {arm}: loaded weights differ from the file in "
                                          f"{[g for g in G if G[g] != fid[arm]['group_hashes'].get(g)][:6]}")
                    held["arm"], held["m"] = arm, m
                    del m
                    old_arm = read_json(d / "ARM.json") if (d / "ARM.json").exists() else None
                    if old_arm and old_arm.get("identity") != idents[arm]:
                        raise SanityError(f"{arm}: {d} holds outputs of another build")
                    load_s = round(time.time() - t0, 1)
                    rec = {"record": "L1_ARM", "arm": arm, "identity": idents[arm], "weight_file": str(files[arm]),
                           "weight_sha256": fid[arm]["sha256"], "group_hashes": G, "forward": be.name,
                           "numerics": numerics, "load_seconds": load_s,
                           "gates": ["G0 environment / runner / lock", "G1 file identity == pin and R14",
                                     "G2 loaded == file (per-group FP32 hashes)", "G3 FP32, TF32 off"]
                           + (["PG anchor parity gate passed before this load"] if post_gate else []),
                           "loads": (old_arm or {}).get("loads", []) + [{"utc": _utc(), "invocation": inv_no,
                                                                         "seconds": load_s, "for_stages": stages}]}
                    write_json(d / "ARM.json", rec)
                    log(f"loaded {arm} ({be.name}, {fid[arm]['sha256'][:12]}..., {load_s}s); G2 passed")
                else:
                    log(f"{arm} stays loaded across the parity gate for {', '.join(stages)}")
                st = (read_json(d / "STAGES.json") if (d / "STAGES.json").exists()
                      else {"record": "L1_STAGES", "arm": arm, "identity": idents[arm], "stages": {}})
                for s in stages:
                    log(f"{arm}: {s} ({len(inputs[s])} rows) ...")
                    info = run_stage(s, be, held["m"], tok, eos, d, inputs, pace, arm)
                    info.update(invocation=inv_no, script_sha256=script_sha, freeze_sha256=freeze_sha)
                    if post_gate:
                        info["parity_gate_before"] = gate_rec
                    st["stages"][s] = info
                    write_json(d / "STAGES.json", st)
                    log(f"{arm}: {s} done in {info['seconds']:.0f}s")
                    if _on_stage_done is not None:
                        _on_stage_done(arm, s)
        finally:
            release()
    if gate_failed:
        return 0
    finalize(out_dir, idents)
    return 0


# ============================================================================================ CPU checks
LC_ROLE = "report-only (SPENT long-context bank)"


def _check(results: list, name: str, ok: bool, detail: str = "", role: str = "decisive") -> None:
    """role 'decisive': part of C0's scorer parity; anything on the SPENT long-context bank is report-only."""
    results.append({"check": name, "pass": bool(ok), "detail": detail, "role": role})
    tag = "" if role == "decisive" else f" [{role}]"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{tag}" + (f"  ({detail})" if detail else ""), flush=True)


def parity_cpu(tokenizer_check: bool = True, quiet: bool = False) -> list[dict]:
    """Scorer parity on the PUBLISHED raw predictions (no model), plus tokenizer/template/decode parity. Checks that
    touch the SPENT long-context bank are labelled report-only and never enter C0."""
    res: list[dict] = []
    views = load_views()
    check_floors(views)
    _check(res, "SELECT-VIEWS-001: sha bound to FLOORS.json, role guard, leak checks, floors recomputed == published",
           True, f"{len(views)} views")
    pub_det = {a["arm"]: a for a in read_json(pinned("frontier_det"))["arms"]}
    s25 = pub_det["s25"]          # FRONTIER_DET.json labels the deterministic re-measurement s25d as 's25'
    published = {}
    for arm, pkey, lkey in (("sparse", "anchor_process_sparse", "anchor_lc_sparse"),
                            ("supa", "anchor_process_supa", "anchor_lc_supa")):
        P = read_json(pinned(pkey))
        preds = {d["id"]: d for d in P["predictions"]}
        S = score_acq(views, preds)
        row_eq = all(d["strict"] == preds[d["id"]]["strict"] and d["canonical"] == preds[d["id"]]["canonical"]
                     and d["parsed"] == preds[d["id"]]["parsed"] for d in S["detail"])
        fam_eq = all(all(S["per_family"][f][k] == P["per_family"][f][k] for k in (
            "n", "strict_correct", "canonical_correct", "parsed", "terminated", "binding_floor_count",
            "binding_floor_rate", "canonical_rate", "presentation_only_gain", "clopper_pearson_95", "clears_floor",
            "binding_floor_predictor")) and abs(S["per_family"][f]["exact_binomial_p_vs_floor_one_sided"]
                                                 - P["per_family"][f]["exact_binomial_p_vs_floor_one_sided"]) < 1e-12
            for f in FAMILIES)
        agg_eq = all(S["aggregate"][k] == P["aggregate"][k] for k in ("n", "strict_correct", "canonical_correct",
                                                                      "canonical_rate", "presentation_only_gain"))
        _check(res, f"acquisition scorer reproduces {arm}'s published PROCESS file row by row", row_eq and fam_eq and agg_eq,
               f"canonical {S['aggregate']['canonical_correct']}/384, floors {S['floors_cleared']}/3")
        L = read_json(pinned(lkey))
        _, key = load_lc()
        lp = {i: r for i, r in L["predictions"].items()}
        SL = score_lc(key, lp)
        ok = (SL["per_row_correct"] == L["per_row_correct"] and SL["all_hits"] == L["all256_hits"]
              and SL["showcase_hits"] == L["showcase64_hits"] and SL["stability_hits"] == L["stability192_hits"]
              and SL["rows_skipped_context_overflow"] == L["rows_skipped_context_overflow"] == 0)
        if "says_unknown_on_unanswerable" in L:
            ok = ok and SL["says_unknown_substring"] == L["says_unknown_on_unanswerable"]
        _check(res, f"long-context scorer reproduces {arm}'s published LONGCTX file row by row", ok,
               f"{SL['all_hits']}/256 (showcase {SL['showcase_hits']}, stability {SL['stability_hits']}); UNKNOWN "
               f"substring {SL['says_unknown_substring']}, literal {SL['says_unknown_literal']}, false abstentions "
               f"{SL['false_abstention_answerable']}", role=LC_ROLE)
        published[arm] = {"acq": S, "lc": SL, "lc_raw": L}
    m = mcnemar_exact(published["supa"]["lc"]["per_row_correct"], published["sparse"]["lc"]["per_row_correct"])
    _check(res, "exact McNemar Titan-Sparse vs Supa reproduces the published 7 lost / 10 gained / p 0.6291",
           (m["lost"], m["gained"], round(m["p"], 4)) == (s25["longctx_lost"], s25["longctx_gained"],
                                                          round(s25["longctx_paired_p"], 4)), str(m), role=LC_ROLE)
    ctx = ret_context()
    _check(res, "frozen B28 source answers re-score to exactly the frozen reference sets (permissive 47, strict 41)",
           {k: len(v) for k, v in ctx["references"].items()} == {"permissive": 47, "strict": 41})
    RT = read_json(pinned("anchor_ret_sparse"))
    R = ctx["R"]
    okv = True
    for name in ("permissive", "strict"):
        b = RT["by_bare_policy"][name]
        v = R.monitor_verdict(b["per_class"], ctx["policy"], ctx["policy"]["probe_classes"])
        okv = okv and v["status"] == b["status"] and v["failed_classes"] == b["failed_classes"]
    _check(res, "monitor_verdict on Titan-Sparse's published per-class counts reproduces FAIL / failed classes", okv)
    samples = {s["id"]: s for s in RT["samples"]}
    ok_s = True
    for name, ov in ctx["overlays"].items():
        for i, s in samples.items():
            src = R.score_prediction(s["source"], s["source_terminated"], ov["targets"][i])["correct"]
            ok_s = ok_s and src == (i in ctx["references"][name])
    _check(res, "the 24 published retention samples' source texts score consistently with the frozen references", ok_s)
    ok_a, n_a = True, 0
    for name, ov in ctx["overlays"].items():
        lost = {i for c in RT["by_bare_policy"][name]["per_class"].values() for i in c["lost_ids"]}
        for i, s in samples.items():
            if i in ctx["references"][name]:
                n_a += 1
                arm_ok = R.score_prediction(s["arm"], s["arm_terminated"], ov["targets"][i])["correct"]
                ok_a = ok_a and arm_ok == (i not in lost)
    _check(res, "the 24 published Titan-Sparse arm samples score consistently with its published lost ids (both "
                "policies)", ok_a, f"{n_a} source-correct sample rows checked")
    perm = RT["by_bare_policy"]["permissive"]
    retained = perm["source_correct_total"] - perm["lost_total"]
    frontier_ok = (published["sparse"]["acq"]["aggregate"]["canonical_correct"] == s25["acquisition"]
                   and published["sparse"]["acq"]["floors_cleared"] == s25["floors_cleared"]
                   and perm["arm_correct_total"] == s25["arithmetic_permissive"]
                   and RT["by_bare_policy"]["strict"]["arm_correct_total"] == s25["arithmetic_strict"]
                   and retained == s25["retained"]
                   and (perm["arm_correct_total"] >= ARITH_BAR and retained >= RETAINED_BAR
                        and published["sparse"]["acq"]["floors_cleared"] == 3) == s25["gate_v2"])
    _check(res, "tabulator parity: Titan-Sparse's FRONTIER_DET row (232, 3/3, 72, 63, 41, Gate v2 PASS)",
           frontier_ok, f"acq {s25['acquisition']}, arith {s25['arithmetic_permissive']}/{s25['arithmetic_strict']}, "
                        f"retained {s25['retained']}")
    _check(res, "tabulator parity: Titan-Sparse's FRONTIER_DET long-context columns (164, UNKNOWN 12)",
           published["sparse"]["lc"]["all_hits"] == s25["long_context"]
           and published["sparse"]["lc"]["says_unknown_substring"] == s25["literal_unknown"],
           f"lc {s25['long_context']}, UNK {s25['literal_unknown']}", role=LC_ROLE)
    if tokenizer_check:
        _torch("cpu")
        tok = load_tokenizer()
        rows, _ = load_lc()
        rec = published["sparse"]["lc_raw"]["predictions"]
        diff = [r["id"] for r in rows if len(encode(tok, lc_messages(r))) != rec[r["id"]]["input_tokens"]]
        _check(res, "tokenizer + chat template of this environment reproduce the 256 recorded long-context input-token "
                    "counts (the only recorded input counts)", not diff, f"{len(diff)} differ" + (f": {diff[:5]}" if diff else ""),
               role=LC_ROLE)
        policy, decoding = protocol_policy()
        lens = [len(encode(tok, r["messages"])) for r in load_probe_rows(policy["probe_classes"])]
        _check(res, "every arithmetic-panel prompt fits the registered 512-token context (no truncation)",
               max(lens) <= decoding["max_context"], f"max {max(lens)} tokens")
        vl = [len(encode(tok, r["prompt"])) for r in views]
        _check(res, "SELECT-VIEWS-001 prompts render and tokenize", min(vl) > 0, f"{min(vl)}-{max(vl)} tokens")
        def round_trip(texts):
            return [t for t in texts if tok.decode(tok(t, add_special_tokens=False)["input_ids"],
                                                   skip_special_tokens=True).strip() != t]
        texts = [d["prediction"] for x in ("supa", "sparse") for d in read_json(pinned(f"anchor_process_{x}"))["predictions"]]
        texts += [s["arm"] for s in RT["samples"]] + [r["source"]["raw"] for r in read_json(pinned("b28"))["rows"]]
        bad = round_trip(texts)
        _check(res, "decode path of this environment: every published anchor text on SELECT-VIEWS-001 and the "
                    "arithmetic panel survives encode -> decode(skip_special_tokens) -> strip unchanged", not bad,
               f"{len(texts)} texts, {len(bad)} differ" + (f": {bad[:2]}" if bad else ""))
        lct = [r["prediction"] for x in ("supa", "sparse") for r in published[x]["lc_raw"]["predictions"].values()]
        bad = round_trip(lct)
        _check(res, "decode path: the published long-context texts survive the same round trip", not bad,
               f"{len(lct)} texts, {len(bad)} differ", role=LC_ROLE)
    return res


def integrity() -> list[dict]:
    res: list[dict] = []
    for key in sorted(SHA):
        if key in ("supa_weights", "sparse_weights"):
            continue
        try:
            pinned(key)
            _check(res, f"pinned input {key}", True)
        except SanityError as e:
            _check(res, f"pinned input {key}", False, str(e))
    try:
        vendor_retention()
        _vendor_check(list(VENDOR_SHA))
        _check(res, "vendored 5b4c22d code == VENDOR.json == pinned hashes", True)
    except SanityError as e:
        _check(res, "vendored 5b4c22d code == VENDOR.json == pinned hashes", False, str(e))
    b = read_json(pinned("sv_build"))
    q = read_json(pinned("quarantine"))
    entries = q.get("entries", [])
    sv_entries = [e for e in entries if any(s in (str(e.get("key", "")) + str(e.get("path", ""))).lower()
                                            for s in ("select-views", "select_views"))]
    _check(res, "SELECT-VIEWS-001 declares SELECTION admissible and CONFIRMATION/TRAIN inadmissible (BUILD.json)",
           "SELECTION" in b["admissible_uses"] and {"TRAIN", "CONFIRMATION"} <= set(b["inadmissible_uses"]),
           f"admissible {b['admissible_uses']}, inadmissible {b['inadmissible_uses']}")
    _check(res, "SELECT-VIEWS-001 has no QUARANTINE.json entry (GATE_V2 requirement 5's SELECTION_EXPOSED entry is "
                "overdue; it would bar TRAIN/CONFIRMATION only)", not sv_entries, f"{len(sv_entries)} entries")
    lc_e = [e for e in entries if e.get("key") == "Rineya_Long_Context_Demo_v1/all"]
    _check(res, "long-context suite is SPENT and barred TRAIN/SELECTION/CONFIRMATION, entry sha == the answer key",
           bool(lc_e) and lc_e[0]["reason"] == "SPENT" and "SELECTION" in lc_e[0]["bars"]
           and lc_e[0]["sha256"] == SHA["lc_key"], json.dumps({k: lc_e[0][k] for k in ("bars", "reason")}) if lc_e else "")
    fl = read_json(pinned("floors"))
    _check(res, "FLOORS.json published before any model call", fl.get("published_before_any_model_call") is True)
    views = load_views()
    check_floors(views)
    _check(res, "floors recomputed from the views == published", True)
    sys.path.insert(0, str(ROOT))
    from rcwmt import env
    svk = {env.dumps(json.loads(r["prompt"][1]["content"])) for r in views}
    bank = read_json(pinned("pilot_bank"))
    keys = set()
    for p in bank["parents"]:
        for c in p["candidates"] + ([p["twin"]] if p.get("twin") else []):
            st = env.rollout(p["family"], p["s0"], c["events"])
            keys.update(env.dumps({"before": st[t], "event": ev}) for t, ev in enumerate(c["events"]))
    _check(res, "SELECT-VIEWS-001 shares no (before, event) payload with the RCWM pilot bank that chose the cut",
           not (keys & svk), f"{len(keys)} pilot payloads, {len(keys & svk)} shared")
    fr = None
    if resolve("offtask_fresh").exists():
        try:
            fr = load_prompts("fresh")
        except SanityError as e:
            _check(res, "fresh prompts == their SOURCE record", False, str(e))
    used = load_prompts("used")
    pools = hskill_pools()
    if fr is not None:
        sl = {r["id"]: r for r in read_jsonl(pinned("slice"))}
        tf_used = {sl[u["id"]]["metadata"].get("template_family") for u in used}
        per = Counter(r["reply_class"] for r in fr)
        ok = (not ({r["id"] for r in fr} & {u["id"] for u in used})
              and not ({r["template_family"] for r in fr} & tf_used)
              and not ({canonical(r["messages"]) for r in fr} & {canonical(u["messages"]) for u in used})
              and len(fr) == FRESH_N and len(per) == FRESH_N // FRESH_PER_CLASS and set(per.values()) == {FRESH_PER_CLASS})
        _check(res, f"fresh prompts: {FRESH_N} = {FRESH_N // FRESH_PER_CLASS} classes x {FRESH_PER_CLASS}; disjoint in id, "
                    f"template family and text from the 248 R14 prompts", ok, f"{len(fr)} prompts, {len(per)} classes")
        hit = [r["id"] for r in fr if in_pools(r, pools)]
        _check(res, "fresh prompts: no id and no user text of the HSKILL-002 source pools, hence no row of HSKILL-VIEWS-002 "
                    "or of the reserved HSKILL-CONFIRM-002 (never opened)", not hit,
               f"{len(pools['ids'])} pool ids, {len(pools['text'])} pool user texts; {len(hit)} hits")
        hv = read_jsonl(pinned("hskill_views"))
        hv_src = {r.get("source_id") for r in hv} | {r.get("parent_id") for r in hv}
        _check(res, "fresh prompts: no source row of HSKILL-VIEWS-002, read directly (source_id / parent_id)",
               not ({r["id"] for r in fr} & hv_src), f"{len({r['id'] for r in fr} & hv_src)} shared")
    elif not resolve("offtask_fresh").exists():
        _check(res, "fresh prompts built", False, "run `l1_screen.py fresh-prompts`")
    bridge = load_prompts("bridge")
    _check(res, f"bridge = the 248 R14 prompts minus every HSKILL-002 pool row = {BRIDGE_N}",
           len(bridge) == BRIDGE_N and {u["id"] for u in bridge} <= {u["id"] for u in used}
           and not any(in_pools(u, pools) for u in bridge), f"{len(bridge)} prompts")
    R = r14_bridge()
    meta_ok = all(read_json(ROOT / "runs" / "bank_pilot" / "localized" / d / "offtask_meta.json")["ids"]
                  == read_json(pinned("r14_meta"))["ids"] for d in ("sparse", "loc8_15"))
    idx = [R["row"][u["id"]] for u in bridge]
    k_l = lp_compare(np.asarray(R["l8_15"][idx]), np.asarray(R["supa"][idx]))[0]
    k_t = lp_compare(np.asarray(R["sparse"][idx]), np.asarray(R["supa"][idx]))[0]
    _check(res, f"R14's stored log-probs: the three arms share one prompt order, and their KL ratio on the {BRIDGE_N} "
                f"bridge prompts is the registered {BRIDGE_R14_RATIO} (P6's reference)",
           meta_ok and round(float(k_l.mean() / k_t.mean()), 4) == BRIDGE_R14_RATIO,
           f"{k_l.mean() / k_t.mean():.4f}")
    hq = read_json(pinned("hskill_quarantine"))["instruments"]
    _check(res, "HSKILL-CONFIRM-002 is barred TRAIN/SELECTION/DEV and UNSPENT_AND_UNREAD; HSKILL-VIEWS-002 is "
                "selection-admissible and UNSPENT (report-only use would be an amendment, Q2)",
           set(hq["HSKILL-CONFIRM-002"]["barred"]) == {"TRAIN", "SELECTION", "DEV"}
           and hq["HSKILL-CONFIRM-002"]["status"] == "UNSPENT_AND_UNREAD"
           and hq["HSKILL-VIEWS-002"]["status"] == "UNSPENT" and "SELECTION" not in hq["HSKILL-VIEWS-002"]["barred"])
    seals = {k: read_json(pinned(k))["calibration_bank_is_now_spent"] for k in ("rci001_seal", "rci002_seal")}
    _check(res, "RCI-001 / RCI-002 calibration banks: SPENT for confirmation, remain admissible for DEV and SELECTION "
                "(listed in the PREREG inventory; not used)",
           all(v["status"].startswith("SPENT for confirmation") and set(v["remains_admissible_for"]) == {"DEV", "SELECTION"}
               for v in seals.values()))
    g2 = pinned("gate_v2_prereg").read_text(encoding="utf-8")
    _check(res, "GATE_V2_PREREG carries the sentences the PREREG quotes (selection-exposed reuse; requirement 5)",
           "a split already spent on selection takes no further harm from selection" in " ".join(g2.split())
           and "A registered read budget." in g2)
    pr = resolve("prereg")
    sec = pr.read_text(encoding="utf-8").split("\n## 5.", 1)[-1].split("\n## 6.", 1)[0] if pr.exists() else ""
    quote, started = [], False
    for line in sec.splitlines():
        if line.startswith(">"):
            quote.append(line[1:].strip())
            started = True
        elif started:
            break
    _check(res, "PREREG section 5 quotes the registered DECISION_RULE verbatim (whitespace-normalised)",
           " ".join(" ".join(quote).split()) == " ".join(DECISION_RULE.split()), f"{len(quote)} quoted lines")
    ro = read_json(resolve("vendor_refonly"))
    _check(res, "reference-only vendored run.py / hf_checkpoint.py (5b4c22d) hash as recorded",
           all(sha256_file(VENDOR / f["path"]) == f["sha256"] for f in ro["files"]), f"{len(ro['files'])} files")
    r = r14_hashes()
    _check(res, "R14 loc8_15 group hashes == composition of R14's Supa and Titan-Sparse hashes",
           composed_l8_15(r["supa"], r["sparse"]) == r["l8_15"])
    if resolve("expected").exists():
        e = read_json(resolve("expected"))
        _check(res, "recorded L8-15 identity (L8_15_EXPECTED.json) carries R14 loc8_15's group hashes",
               e["group_hashes"] == r["l8_15"], e["out_weight_sha256"][:16] + "...")
    integrity_ood(res, pr.read_text(encoding="utf-8") if pr.exists() else "")
    return res


def integrity_ood(res: list, prereg_text: str) -> None:
    """Amendment A-OOD (C6): the bank re-checked independently from its file by the builder's own verifier (frame,
    oracle gold, strata, uniqueness, and zero payload / operative-content overlap with every exclusion source,
    including a fresh harvest of the historical set that skips the reserved HSKILL_CONFIRM_V2* files by name), plus
    the runner's own loader and scorer on it."""
    try:
        rows = load_ood_bank()
        _check(res, f"A-OOD bank: pinned, equal to its SOURCE record, {len(OOD_STRATA)} strata x {OOD_N_PER_STRATUM}",
               True, SHA["ood_bank"][:16] + "...")
    except SanityError as e:
        _check(res, "A-OOD bank: pinned, equal to its SOURCE record", False, str(e))
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    import l1_make_ood as MO
    if Path(MO.__file__).resolve() != resolve("ood_builder").resolve():
        _check(res, "A-OOD builder imported from scripts/l1_make_ood.py", False, MO.__file__)
        return
    for name, ok, detail in MO.verify(resolve("ood_bank"), resolve("ood_bank_source"), historical=True, quiet=True):
        _check(res, f"A-OOD bank (independent re-check): {name}", ok, detail)
    bank = {r["id"]: r for r in rows}
    gold = score_ood({i: {"prediction": r["gold"], "terminated": True} for i, r in bank.items()}, bank)
    copy = score_ood({i: {"prediction": ood_env().dumps(r["before"]), "terminated": True} for i, r in bank.items()}, bank)
    cp = {s: copy["per_stratum"][s]["correct"] for s in OOD_STRATA}
    _check(res, "A-OOD scorer (rcwmt.env.parse_state, lenient): the gold texts score 600/600; copying the input state "
                "scores 100 on the no-op strata L_void and P_inapp and 0 on the four must-change strata (by design)",
           gold["correct"] == OOD_N and cp == {"L_big": 0, "L_void": 100, "R_wrap": 0, "R_boundary": 0, "P_inapp": 100,
                                               "P_appdel": 0}, f"copy-before per stratum {cp}")
    _check(res, "PREREG names the A-OOD bank's sha256 (section 15) and its section 5 carries C6",
           SHA["ood_bank"] in prereg_text and "C6 out-of-distribution rule extrapolation" in " ".join(prereg_text.split()))
    full = run_passes({arm: list(STAGES) for arm in ARMS})
    pre = full[:full.index(("GATE", []))] if ("GATE", []) in full else full
    _check(res, "A-OOD run order: supa then sparse on the pre-gate stages, the parity gate, then sparse's and supa's OOD "
                "stage and every L8-15 stage; no OOD stage and no L8-15 stage is scheduled before the gate",
           [a_ for a_, _ in full] == ["supa", "sparse", "GATE", "sparse", "supa", "l8_15"]
           and all(a_ != "l8_15" and not set(st) & set(POST_GATE_STAGES) for a_, st in pre)
           and set(POST_GATE_STAGES) == {"ood"} and LOADS_PER_RUN == 4,
           " -> ".join(f"{a_}:{'/'.join(st)}" if st else a_ for a_, st in full))
    need = {n: c6_min_lost_to_fail(n) for n in (100, 60, 40, 8, 1)}
    _check(res, "C6 binding (report-only): at n = 100 a stratum can fail only if Titan-Sparse is correct on >= 6 of its "
                "rows (the -5 margin and the p < 0.05 detection both first bite at 6 lost, 0 gained)",
           need == {100: 6, 60: 4, 40: 4, 8: 4, 1: 4} and not c6_binding(5, 100)["can_fail"]
           and c6_binding(6, 100)["can_fail"], str(need))


# ============================================================================================ fresh prompts
def build_fresh_prompts(out: Path, src_out: Path) -> dict:
    rows = read_jsonl(pinned("slice"))
    used = read_jsonl(pinned("offtask_used"))
    by_id = {r["id"]: r for r in rows}
    used_ids = {u["id"] for u in used}
    used_tf = {by_id[i]["metadata"].get("template_family") for i in used_ids}
    used_text = {canonical(u["messages"]) for u in used}
    pools = hskill_pools()
    by_class, dropped = defaultdict(list), Counter()
    for r in rows:
        md = r.get("metadata", {})
        if r["id"] in used_ids:
            dropped["id_in_the_248"] += 1
            continue
        if in_pools(r, pools):
            dropped["id_or_user_text_in_the_hskill_002_pools"] += 1
            continue
        if md.get("template_family") in used_tf:
            dropped["template_family_of_the_248"] += 1
            continue
        msgs = r.get("messages") or []
        if len(msgs) < 2 or msgs[-1].get("role") != "assistant" or msgs[-2].get("role") != "user":
            dropped["not_ending_user_then_assistant"] += 1
            continue
        prompt = [{"role": m["role"], "content": m["content"]} for m in msgs[:-1]]
        if canonical(prompt) in used_text:
            dropped["text_equal_to_one_of_the_248"] += 1
            continue
        by_class[md["reply_class"]].append({"id": r["id"], "reply_class": md["reply_class"],
                                            "template_family": md.get("template_family"), "messages": prompt})
    classes = sorted({u["reply_class"] for u in used})
    rng = random.Random(FRESH_SEED)
    out_rows, seen_text, no_rows = [], set(), []
    for c in classes:
        if not by_class[c]:
            no_rows.append(c)                # every row of the class is in the 248 / their families / the pools
            continue
        cand = sorted(by_class[c], key=lambda x: x["id"])
        uniq, here = [], set()
        for x in cand:                       # first row (by id) of each distinct prompt text, none drawn before
            t = canonical(x["messages"])
            if t not in seen_text and t not in here:
                uniq.append(x)
                here.add(t)
            elif t in here:
                dropped["duplicate_text_within_class"] += 1
        if len(uniq) < FRESH_PER_CLASS:
            raise SanityError(f"class {c}: only {len(uniq)} eligible rows")
        pick = sorted(rng.sample(uniq, FRESH_PER_CLASS), key=lambda x: x["id"])
        for x in pick:
            seen_text.add(canonical(x["messages"]))
        out_rows.extend(pick)
    if len(out_rows) != FRESH_N:
        raise SanityError(f"the draw has {len(out_rows)} prompts, registered {FRESH_N}")
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows)
    write_create(out, text)
    src = {"record": "L1_OFFTASK_FRESH_PROMPTS", "schema": 2, "source": str(resolve("slice")),
           "source_sha256": SHA["slice"], "excluded_prompt_set": str(resolve("offtask_used")),
           "excluded_prompt_set_sha256": SHA["offtask_used"], "seed": FRESH_SEED, "per_class": FRESH_PER_CLASS,
           "classes": len(classes) - len(no_rows), "classes_without_an_eligible_row": no_rows,
           "n": len(out_rows), "dropped": dict(dropped),
           "excluded_hskill_002_pools": {"build_record": str(resolve("hskill_views_build")),
                                         "build_record_sha256": SHA["hskill_views_build"],
                                         "sources": pools["sources"], "source_sha256": pools["source_sha256"],
                                         "pool_rows_by_class": pools["rows_by_class"],
                                         "why": ("both HSKILL-VIEWS-002 and the operator-reserved HSKILL-CONFIRM-002 "
                                                 "were drawn from these pools; a prompt outside them is outside both "
                                                 "banks, and the CONFIRM file is never opened")},
           "supersedes": {"file_sha256": "62faf04dc3532788c8d839f819179731d816316217dec99b63a1647bac9047a6",
                          "n": 496, "why": ("the schema-1 draw did not exclude the HSKILL-002 pools (review of "
                                            "PREREG-L1, post-hoc lens, MUST-FIX 1); it was never read by any model "
                                            "and was deleted before the freeze")},
           "exclusions": ["ids of the 248 R14 prompts", "every id and user text of the HSKILL-002 source pools",
                          "every template_family of the 248", "prompt text equal to one of the 248",
                          "duplicate prompt text within the draw"],
           "prompt": "the slice row's messages without the final assistant turn (as data/offtask_prompts.jsonl)",
           "use": ("PREREG-L1 endpoint C5 (selection) and nothing else; built with no model call before any L1 output. "
                   "Spent for confirmation by that use."),
           "file": str(out), "file_sha256": sha256_text(text)}
    write_create(src_out, json.dumps(src, indent=1) + "\n")
    return src


# ============================================================================================ freeze
def freeze_record() -> dict:
    files = {"prereg": resolve("prereg"), "script": Path(__file__).resolve(), "test_script": resolve("test_script"),
             "vendor_manifest": VENDOR / "VENDOR.json", "fresh_prompts": resolve("offtask_fresh"),
             "fresh_prompts_source": resolve("offtask_fresh_source"), "expected_identity": resolve("expected"),
             "vendor_refonly_manifest": resolve("vendor_refonly"), "queue_hook_draft": resolve("queue_hook"),
             "ood_bank": resolve("ood_bank"), "ood_bank_source": resolve("ood_bank_source"),
             "ood_builder": resolve("ood_builder"), "rcwmt_env": resolve("rcwmt_env")}
    rec = {k: sha256_file(p) for k, p in files.items()}
    rec.update({f"vendor/{n}": sha256_file(AB / n) for n in VENDOR_SHA})
    return {"files": rec, "paths": {k: str(p) for k, p in files.items()}, "inputs_sha256": dict(SHA),
            "registered": json.loads(json.dumps(REGISTERED)), "predictions": PREDICTIONS}


def verify_freeze() -> list[str]:
    p = resolve("freeze")
    if not p.exists():
        return [f"no freeze record at {p}"]
    old = read_json(p)
    new = freeze_record()
    bad = [k for k in set(old["files"]) | set(new["files"]) if old["files"].get(k) != new["files"].get(k)]
    bad += [f"input:{k}" for k in set(old["inputs_sha256"]) | set(new["inputs_sha256"])
            if old["inputs_sha256"].get(k) != new["inputs_sha256"].get(k)]
    if old["registered"] != new["registered"]:
        bad.append("registered constants")
    return bad


L1_OUTPUT_NAMES = ("RUN.json", "ARM.json", "STAGES.json", "DONE.json", "PARITY_GATE.json", "PARITY_FAIL.json",
                   "L1_RESULTS.json")


def l1_outputs_under(root: Path) -> list[Path]:
    """Every L1 output record under root (one os.walk): a file with an L1 output name whose record is an L1 record
    or whose lease label is L1's. The build's MATERIALIZATION/ARM receipt (record CHECKPOINT_MATERIALIZATION) is a
    build artefact, not an output."""
    found = []
    if not Path(root).exists():
        return found
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn in L1_OUTPUT_NAMES:
                p = Path(dp) / fn
                try:
                    j = read_json(p)
                except (OSError, ValueError):
                    continue
                if not isinstance(j, dict):
                    continue
                if (str(j.get("record", "")).startswith(("L1_", "PREREG_L1_")) or j.get("lease_label") == LEASE_LABEL
                        or (fn == "STAGES.json" and "identity" in j and "stages" in j)):
                    found.append(p)
    return found


def cmd_freeze(a) -> int:
    out_dir = Path(a.out_dir) if a.out_dir else OUT_DEFAULT
    existing = l1_outputs_under(out_dir) + l1_outputs_under(resolve("runs_root"))
    if existing:
        raise SanityError(f"L1 outputs already exist ({existing[0]}); a freeze after any output is not a preregistration")
    for k in ("offtask_fresh", "offtask_fresh_source", "expected", "prereg", "ood_bank", "ood_bank_source",
              "ood_builder"):
        if not resolve(k).exists():
            raise SanityError(f"cannot freeze without {resolve(k)}")
    load_ood_bank()                                                     # C6's bank is the pinned one
    body = {"record": "PREREG_L1_FREEZE", "schema": 1, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frozen_before_any_l1_output": True, "approved_by": a.approved_by, "environment": env_record(),
            **freeze_record()}
    write_create(resolve("freeze"), json.dumps(body, indent=1) + "\n")
    log(f"wrote {resolve('freeze')}")
    return 0


# ============================================================================================ anchor parity gate (C0)
def load_arm_outputs(out_dir: Path, arm: str, ood: bool = True) -> dict:
    """An arm's saved outputs. ood=False reads the pre-gate stages only (the parity gate never reads OOD outputs; after
    a failed gate none exist)."""
    d = out_dir / arm
    lp = {s: np.load(d / STAGE_FILES[s][0], mmap_mode="r") for s in ("drift_fresh", "drift_bridge")}
    ids = {s: read_json(d / STAGE_FILES[s][1])["ids"] for s in ("drift_fresh", "drift_bridge")}
    O = {"acq": {r["id"]: r for r in read_jsonl(d / "acq.jsonl")}, "ret": read_jsonl(d / "ret.jsonl"),
         "lc": {r["id"]: r for r in read_jsonl(d / "lc.jsonl")}, "lp": lp, "lp_ids": ids}
    if ood:
        O["ood"] = {r["id"]: r for r in read_jsonl(d / "ood.jsonl")}
    return O


def score_arm(O: dict, vmap: dict, ctx: dict, key: dict, obank: dict | None = None) -> dict:
    E = {"acq": score_acq([vmap[i] for i in O["acq"]], O["acq"]), "ret": score_ret(ctx, O["ret"]),
         "lc": score_lc(key, O["lc"])}
    if "ood" in O:
        obank = obank if obank is not None else {r["id"]: r for r in load_ood_bank()}
        E["ood"] = score_ood(O["ood"], obank)
    E["gate_v2"] = (E["ret"]["arithmetic_permissive"] >= ARITH_BAR and E["ret"]["retained"] >= RETAINED_BAR
                    and E["acq"]["floors_cleared"] == 3)
    return E


def acq_leg(arm: str, E_acq: dict, O_acq: dict, subset: bool) -> dict:
    """C0: SELECT-VIEWS-001 texts (all but max(3, floor(3% n)) rows), canonical total +-3, floor verdicts identical."""
    P = read_json(pinned(f"anchor_process_{arm}"))
    tp = text_parity({i: r["prediction"] for i, r in O_acq.items()}, {d["id"]: d["prediction"] for d in P["predictions"]})
    pub = {"acq_total": P["aggregate"]["canonical_correct"],
           "floors": {f: bool(P["per_family"][f]["clears_floor"]) for f in FAMILIES}}
    ours = {"acq_total": E_acq["aggregate"]["canonical_correct"],
            "floors": {f: bool(E_acq["per_family"][f]["clears_floor"]) for f in E_acq["per_family"]}}
    stats = None if subset else (abs(ours["acq_total"] - pub["acq_total"]) <= PARITY_TOL["acq_total"]
                                 and ours["floors"] == pub["floors"])
    return {"text": tp, "published": pub, "ours": ours, "stats_pass": stats,
            "pass": bool(tp["pass"] and stats is not False)}


def ret_leg_sparse(E_ret: dict, O_ret: list, ctx: dict, subset: bool) -> dict:
    """C0, Titan-Sparse on the arithmetic panel (CUDA-vs-CUDA: both measurements on the RTX 4070): arithmetic
    permissive / strict within +-2 of 72 / 63 and retained within +-1 of 41; on the 47 permissive and 41 strict
    source-correct rows the re-measured correct set equals (source-correct minus the published lost ids) up to 1 row
    per policy; per class, arm_correct and terminated within +-1; the 24 published samples equal in (text,
    terminated) on all but 3."""
    RT = read_json(pinned("anchor_ret_sparse"))
    ours_ids = {o["id"] for o in O_ret}
    out, ok = {"policies": {}}, True
    for pol in ("permissive", "strict"):
        b = RT["by_bare_policy"][pol]
        S = set(ctx["references"][pol]) & ours_ids
        pub_lost = {i for c in b["per_class"].values() for i in c["lost_ids"]} & ours_ids
        want, got = S - pub_lost, set(E_ret["by_bare_policy"][pol]["correct_ids"]) & S
        diff = sorted(want ^ got)
        pc_ours = E_ret["by_bare_policy"][pol]["per_class"]
        cls = {c: {k: [pc_ours[c][k], b["per_class"][c][k]] for k in ("arm_correct", "terminated")}
               for c in b["per_class"]}
        cls_ok = None if subset else all(abs(x - y) <= RET_CLASS_TOL for v in cls.values() for x, y in v.values())
        rows_ok = len(diff) <= RET_ROW_TOL
        out["policies"][pol] = {"source_correct_rows": len(S), "published_lost_ids": sorted(pub_lost),
                                "differ_ids": diff, "rows_pass": rows_ok, "per_class_ours_published": cls,
                                "per_class_pass": cls_ok}
        ok = ok and rows_ok and cls_ok is not False
    samp = {s["id"]: [s["arm"], bool(s["arm_terminated"])] for s in RT["samples"]}
    out["samples"] = text_parity({o["id"]: [o["text"], bool(o["terminated"])] for o in O_ret if o["id"] in samp},
                                 samp)
    perm, strict = RT["by_bare_policy"]["permissive"], RT["by_bare_policy"]["strict"]
    pub = {"arith_permissive": perm["arm_correct_total"], "arith_strict": strict["arm_correct_total"],
           "retained": perm["source_correct_total"] - perm["lost_total"]}
    ours = {"arith_permissive": E_ret["arithmetic_permissive"], "arith_strict": E_ret["arithmetic_strict"],
            "retained": E_ret["retained"]}
    stats = None if subset else all(abs(ours[k] - pub[k]) <= PARITY_TOL[k] for k in pub)
    out.update(published=pub, ours=ours, stats_pass=stats)
    out["pass"] = bool(ok and out["samples"]["pass"] and stats is not False)
    return out


def ret_report_supa(E_ret: dict, O_ret: list) -> dict:
    """REPORT-ONLY: Supa's CUDA arithmetic-panel texts against the B28 source answers, which were measured on the CPU
    (G1_RETENTION_B28.json context: device cpu, torch 2.10.0, transformers 4.51.3). A CPU-to-GPU FP32 argmax flip is
    not the environment change C0 tests, so this never enters C0."""
    B = read_json(pinned("anchor_ret_supa"))
    tp = text_parity({o["id"]: o["text"] for o in O_ret}, {r["id"]: r["source"]["raw"] for r in B["rows"]})
    return {"text_vs_b28_cpu": tp, "arith_permissive_ours_b28": [E_ret["arithmetic_permissive"], 47],
            "arith_strict_ours_b28": [E_ret["arithmetic_strict"], 41],
            "pass_report_only": bool(tp["pass"] and abs(E_ret["arithmetic_permissive"] - 47) <= PARITY_TOL["arith_permissive"])}


def lc_report(arm: str, E_lc: dict, O_lc: dict, key: dict, subset: bool) -> dict:
    """REPORT-ONLY (the SPENT bank never enters C0): long-context texts, total +-3, UNKNOWN counters +-1."""
    L = read_json(pinned(f"anchor_lc_{arm}"))
    PS = score_lc(key, {i: r for i, r in L["predictions"].items()})
    tp = text_parity({i: r["prediction"] for i, r in O_lc.items()}, {i: r["prediction"] for i, r in L["predictions"].items()})
    pub = {"lc_total": PS["all_hits"], "unknown_substring": PS["says_unknown_substring"],
           "unknown_literal": PS["says_unknown_literal"]}
    ours = {"lc_total": E_lc["all_hits"], "unknown_substring": E_lc["says_unknown_substring"],
            "unknown_literal": E_lc["says_unknown_literal"]}
    stats = None if subset else all(abs(ours[k] - pub[k]) <= PARITY_TOL_REPORT[k] for k in pub)
    return {"text": tp, "published": pub, "ours": ours, "stats_pass": stats,
            "rows_skipped_context_overflow": E_lc["rows_skipped_context_overflow"],
            "pass_report_only": bool(tp["pass"] and stats is not False and not E_lc["rows_skipped_context_overflow"])}


_R14: dict = {}


def r14_bridge() -> dict:
    """R14's stored first-token log-probs (248 x V float16; HF Qwen3 sdpa, FP32, batch 8, left padding) by prompt id."""
    if not _R14:
        meta = read_json(pinned("r14_meta"))
        _R14["row"] = {i: k for k, i in enumerate(meta["ids"])}
        for arm in ARMS:
            _R14[arm] = np.load(pinned(f"r14_lp_{arm}"), mmap_mode="r")
    return _R14


def bridge_leg(arm: str, lp, ids: list[str]) -> dict:
    """C0: this harness's first-token log-probs for an anchor on the bridge prompts against R14's stored ones for the
    same arm (top-1 agreement >= 0.99 and mean KL(L1 || R14) < 0.001 nats)."""
    R = r14_bridge()
    missing = [i for i in ids if i not in R["row"]]
    if missing or not ids:
        return {"n": len(ids), "missing": missing[:5], "pass": False}
    ref = np.asarray(R[arm][[R["row"][i] for i in ids]])
    kl, a, b = lp_compare(np.asarray(lp), ref)
    top1 = float((a == b).mean())
    return {"n": len(ids), "top1_agreement": top1, "kl_l1_r14_mean": float(kl.mean()),
            "kl_l1_r14_max": float(kl.max()), "top1_min": BRIDGE_TOP1_MIN, "kl_max": BRIDGE_KL_MAX,
            "pass": bool(top1 >= BRIDGE_TOP1_MIN and float(kl.mean()) < BRIDGE_KL_MAX)}


def parity_gate(out_dir: Path, subset: bool) -> dict:
    """The anchor parity gate (C0), on the anchors' saved outputs only (CPU, no model). Decisive legs: Supa and
    Titan-Sparse on SELECT-VIEWS-001, Titan-Sparse on the arithmetic panel, both on the bridge log-probs. Report-only:
    long context and UNKNOWN (SPENT bank), Supa's panel texts against the CPU-measured B28. On a row subset (tests)
    the statistics legs are skipped and the gate is computed but not enforced. It reads the anchors' pre-gate stages
    only: the OOD stage (C6) runs after the gate has passed."""
    views = load_views()
    vmap = {r["id"]: r for r in views}
    _, key = load_lc()
    ctx = ret_context()
    legs, report = {}, {}
    for arm in ("supa", "sparse"):
        O = load_arm_outputs(out_dir, arm, ood=False)
        E = score_arm(O, vmap, ctx, key)
        legs[f"{arm}.select_views_001"] = acq_leg(arm, E["acq"], O["acq"], subset)
        if arm == "sparse":
            legs["sparse.arithmetic_panel"] = ret_leg_sparse(E["ret"], O["ret"], ctx, subset)
        else:
            report["supa.arithmetic_panel_vs_b28_cpu"] = ret_report_supa(E["ret"], O["ret"])
        legs[f"{arm}.bridge_logprobs"] = bridge_leg(arm, O["lp"]["drift_bridge"], O["lp_ids"]["drift_bridge"])
        report[f"{arm}.long_context"] = lc_report(arm, E["lc"], O["lc"], key, subset)
        del O
    failed = [k for k, v in legs.items() if not v["pass"]]
    return {"record": "L1_PARITY_GATE", "evaluated_utc": _utc(), "enforced": not subset, "pass": not failed,
            "failed_legs": failed, "legs": legs, "report_only": report,
            "tolerances": {"text_rows": "max(3, floor(0.03 n))", "parity_tol": PARITY_TOL,
                           "parity_tol_report": PARITY_TOL_REPORT, "ret_row_tol": RET_ROW_TOL,
                           "ret_class_tol": RET_CLASS_TOL, "bridge_top1_min": BRIDGE_TOP1_MIN,
                           "bridge_kl_max": BRIDGE_KL_MAX}}


# ============================================================================================ analyze
def bind_run_to_freeze(run: dict, freeze: dict, freeze_sha: str, out_dir: Path) -> list[str]:
    """The run is the registered run only if EVERY invocation and every stage was made after the freeze, with the
    frozen runner and the freeze record, without --no-freeze, in the frozen environment."""
    bad = []
    inv = run.get("invocations") or []
    if not inv:
        bad.append("RUN.json records no invocation")
    t_freeze, script = freeze.get("frozen_utc", "9999"), freeze["files"]["script"]
    env_f = freeze.get("environment") or {}
    for k, x in enumerate(inv):
        if x.get("no_freeze"):
            bad.append(f"invocation {k} ran with --no-freeze")
        if x.get("freeze_sha256") != freeze_sha:
            bad.append(f"invocation {k} carries freeze {str(x.get('freeze_sha256'))[:12]}, not the record")
        if x.get("script_sha256") != script:
            bad.append(f"invocation {k} ran runner {str(x.get('script_sha256'))[:12]}, not the frozen one")
        if not x.get("started_utc") or x["started_utc"] < t_freeze:
            bad.append(f"invocation {k} started {x.get('started_utc')}, before the freeze {t_freeze}")
        if {kk: (x.get("env") or {}).get(kk) for kk in env_f} != env_f:
            bad.append(f"invocation {k} ran in another environment")
    for arm in ARMS:
        p = out_dir / arm / "STAGES.json"
        if p.exists():
            for s, info in read_json(p).get("stages", {}).items():
                if info.get("freeze_sha256") != freeze_sha or info.get("script_sha256") != script:
                    bad.append(f"{arm}/{s} was not produced by the frozen runner under the freeze record")
    return bad


def _check_stage_files(out_dir: Path, arms, stages=STAGES) -> None:
    for arm in arms:
        st = read_json(out_dir / arm / "STAGES.json")
        for s in stages:
            for f, h in st["stages"][s]["sha256"].items():
                if sha256_file(out_dir / arm / f) != h:
                    raise SanityError(f"{arm}/{f} changed after the run")


def arm_summary(E: dict, D: dict | None, arm: str) -> dict:
    s = {"acquisition": E["acq"]["aggregate"]["canonical_correct"],
         "acquisition_strict": E["acq"]["aggregate"]["strict_correct"],
         "per_family": {f: {k: E["acq"]["per_family"][f][k] for k in (
             "canonical_correct", "canonical_rate", "clopper_pearson_95", "binding_floor_rate", "clears_floor")}
             for f in E["acq"]["per_family"]},
         "floors_cleared": E["acq"]["floors_cleared"],
         "arithmetic_permissive": E["ret"]["arithmetic_permissive"],
         "arithmetic_strict": E["ret"]["arithmetic_strict"], "retained": E["ret"]["retained"],
         "retained_strict": E["ret"]["retained_strict"],
         "legacy_monitor": {p: E["ret"]["by_bare_policy"][p]["status"] for p in ("permissive", "strict")},
         "gate_v2": E["gate_v2"], "long_context": E["lc"]["all_hits"],
         "showcase": E["lc"]["showcase_hits"], "stability": E["lc"]["stability_hits"],
         "unknown_substring": E["lc"]["says_unknown_substring"], "unknown_literal": E["lc"]["says_unknown_literal"],
         "false_abstention_answerable": E["lc"]["false_abstention_answerable"],
         "long_context_rows_skipped": E["lc"]["rows_skipped_context_overflow"]}
    if "ood" in E:                                   # absent after a failed gate: no model read the OOD bank
        s.update(ood_correct=E["ood"]["correct"], ood_rows=E["ood"]["rows"], ood_per_stratum=E["ood"]["per_stratum"])
    if D is not None:
        s.update(drift_fresh_kl_arm_supa=D["drift_fresh"]["arms"][arm]["kl_arm_supa_mean"],
                 drift_bridge_kl_arm_supa=D["drift_bridge"]["arms"][arm]["kl_arm_supa_mean"])
    return s


REPORT_ONLY = ["long context /256 (SPENT; never in C0-C6)", "literal UNKNOWN /64 (same bank; both counters)",
               "arithmetic strict (as an endpoint)", "legacy per-class monitor",
               "Supa's arithmetic-panel texts against the CPU-measured B28 source answers",
               "arm losses against the same-run CUDA Supa", "bridge drift on the 228 R14 prompts (P6)",
               "SPARSE-001's selection rule applied to {Titan-Sparse, L8-15}",
               "Supa's OOD scores and every arm's OOD contrast against Supa; the OOD aggregate over strata",
               "C6 binding flags: whether each OOD stratum can fail C6 at all, given Titan-Sparse's correct count"]


def cmd_analyze(a) -> int:
    out_dir = Path(a.out_dir) if a.out_dir else OUT_DEFAULT
    res_json, res_md = out_dir / "L1_RESULTS.json", out_dir / "L1_RESULTS.md"
    if res_json.exists() or res_md.exists():
        raise SanityError(f"{res_json} exists; the analysis is create-only")
    run = read_json(out_dir / "RUN.json")
    if a.no_freeze:
        binding = "not checked (--no-freeze: tests only)"
    else:
        bad = verify_freeze()
        if bad:
            raise SanityError(f"freeze check failed: {bad[:6]}")
        bad = bind_run_to_freeze(run, read_json(resolve("freeze")), sha256_file(resolve("freeze")), out_dir)
        if bad:
            raise SanityError(f"this run is not the registered run: {bad[:6]}")
        binding = "every invocation and stage: after the freeze, frozen runner, freeze record, frozen environment"
    subset = bool(run["keys"].get("subset"))
    if subset and not a.allow_subset:
        raise SanityError("the run used a row subset; no decision is issued on a subset")
    parity_failed = (out_dir / "PARITY_FAIL.json").exists()
    if parity_failed:
        if any((out_dir / "l8_15" / f).exists() for s in STAGES for f in STAGE_FILES[s]):
            raise SanityError("L8-15 outputs exist next to PARITY_FAIL.json")
        if any((out_dir / arm / f).exists() or s in read_json(out_dir / arm / "STAGES.json").get("stages", {})
               for arm in ("supa", "sparse") for s in POST_GATE_STAGES for f in STAGE_FILES[s]):
            raise SanityError("OOD outputs exist next to PARITY_FAIL.json: a model read the OOD bank without a "
                              "passed parity gate")
        _check_stage_files(out_dir, ("supa", "sparse"), PRE_GATE_STAGES)
    else:
        done = read_json(out_dir / "DONE.json")
        for arm in ARMS:
            for f, h in done["arms"][arm]["files_sha256"].items():
                if sha256_file(out_dir / arm / f) != h:
                    raise SanityError(f"{arm}/{f} changed after the run")
    log("scorer parity on the published artifacts ...")
    pc = parity_cpu(tokenizer_check=not a.skip_tokenizer_check)
    scorer_ok = all(r["pass"] for r in pc if r["role"] == "decisive")
    log("anchor parity gate, re-derived ...")
    gate = parity_gate(out_dir, subset)
    recorded = read_json(out_dir / "PARITY_GATE.json") if (out_dir / "PARITY_GATE.json").exists() else None
    if recorded is None or recorded.get("pass") != gate["pass"] or recorded.get("failed_legs") != gate["failed_legs"]:
        raise SanityError("the recorded parity gate is missing or differs from its re-derivation")
    if parity_failed and gate["pass"]:
        raise SanityError("PARITY_FAIL.json exists but the gate re-derives as passing")
    views = load_views()
    vmap = {r["id"]: r for r in views}
    _, key = load_lc()
    ctx = ret_context()
    arms_scored = ("supa", "sparse") if parity_failed else ARMS
    obank = {r["id"]: r for r in load_ood_bank()}
    O = {arm: load_arm_outputs(out_dir, arm, ood=not parity_failed) for arm in arms_scored}
    E = {arm: score_arm(O[arm], vmap, ctx, key, obank) for arm in arms_scored}
    reasons = []
    if not scorer_ok:
        reasons.append("scorer parity on the published artifacts failed")
    if gate["enforced"] and not gate["pass"]:
        reasons.append("the anchor parity gate failed: " + ", ".join(gate["failed_legs"]))
    if subset:
        reasons.append("row subset (test run)")
    if run.get("forward") != "dense":
        reasons.append("the forward path is not the programme's DenseReference (needs its own amendment)")
    base = {"run": {k: run.get(k) for k in ("forward", "device", "arm_identity")},
            "invocations": run.get("invocations"), "freeze_binding": binding, "scorer_parity": pc,
            "parity_gate": gate, "subset": run["keys"].get("subset"), "registered": REGISTERED,
            "report_only": REPORT_ONLY}
    if parity_failed:
        result = {"record": "PREREG_L1_RESULTS", "schema": 2, "analysed_utc": _utc(), "verdict": NON_COMPARABLE,
                  "decision": {"verdict": NON_COMPARABLE, "conditions": {"C0": False}, "failed": ["C0"],
                               "inputs": {"comparable_reasons": reasons}, "rule": DECISION_RULE},
                  "l8_15": "never loaded: the anchor parity gate failed before it (PARITY_FAIL.json); no L8-15 "
                           "endpoint exists. This registration ends.",
                  "summary": {arm: arm_summary(E[arm], None, arm) for arm in arms_scored}, **base}
        write_create(res_json, json.dumps(result, indent=1, default=float) + "\n")
        write_create(res_md, render_md(result))
        log(f"verdict: {NON_COMPARABLE} (anchor parity failed before L8-15); wrote {res_json.name}, {res_md.name}")
        return 0
    exp_sha = read_json(resolve("expected"))["out_weight_sha256"] if resolve("expected").exists() else None
    if exp_sha is None or done["arms"]["l8_15"]["weight_sha256"] != exp_sha:
        reasons.append("L8-15 weights are not the frozen bytes")
    late = [f"{arm}/{s}" for arm in ARMS for s, info in read_json(out_dir / arm / "STAGES.json")["stages"].items()
            if (arm == "l8_15" or s in POST_GATE_STAGES) and not (
                isinstance(info.get("parity_gate_before"), dict)
                and (info["parity_gate_before"].get("pass") or not info["parity_gate_before"].get("enforced")))]
    if late:
        reasons.append(f"stages that must follow a passed parity gate record none: {late[:6]}")
    if not subset:
        full = {"acq": len(views), "ret": len(ctx["rows"]), "drift_fresh": FRESH_N, "ood": OOD_N}
        got = {"acq": len(O["l8_15"]["acq"]), "ret": len(O["l8_15"]["ret"]),
               "drift_fresh": len(O["l8_15"]["lp_ids"]["drift_fresh"]), "ood": len(O["l8_15"]["ood"])}
        if (any(got[k] != full[k] for k in full) or any(len(O[x]["acq"]) != len(views) for x in ARMS)
                or any(set(O[x]["ood"]) != set(obank) for x in ARMS)):
            reasons.append(f"a decisive instrument was not run in full: {got} vs {full}")
    D = {}
    for s in ("drift_fresh", "drift_bridge"):
        ids0 = O["supa"]["lp_ids"][s]
        if any(O[arm]["lp_ids"][s] != ids0 for arm in ARMS):
            raise SanityError(f"{s}: arms cover different prompts")
        D[s] = drift_block({arm: np.asarray(O[arm]["lp"][s]) for arm in ARMS})
    ci_up = D["drift_fresh"]["ratio_l8_15_over_sparse"]["kl_arm_supa"]["ci95"][1]
    Ld, Td = E["l8_15"], E["sparse"]

    def correct_map(arm, fam=None):
        return {d["id"]: d["canonical"] for d in E[arm]["acq"]["detail"] if fam is None or d["family"] == fam}
    acq_family = {f: mcnemar_exact(correct_map("sparse", f), correct_map("l8_15", f)) for f in FAMILIES}

    def ood_map(arm, stratum=None):
        return {i: v for i, v in E[arm]["ood"]["per_row_correct"].items()
                if stratum is None or obank[i]["stratum"] == stratum}

    def ood_pair(base, arm, stratum):               # C6 input; a stratum with no rows stays absent (C6 then fails)
        b, x = ood_map(base, stratum), ood_map(arm, stratum)
        return {**mcnemar_exact(b, x), "n": len(set(b) & set(x))} if b or x else None
    ood_stratum = {s: m for s in OOD_STRATA if (m := ood_pair("sparse", "l8_15", s)) is not None}
    ps = E["sparse"]["ood"]["per_stratum"]
    c6_bind = {s: c6_binding(ps[s]["correct"], ps[s]["n"]) for s in OOD_STRATA if s in ps}       # report-only
    Ein = {"comparable": not reasons, "comparable_reasons": reasons, "floors_cleared": Ld["acq"]["floors_cleared"],
           "acq_family": acq_family, "arith_perm_l": Ld["ret"]["arithmetic_permissive"],
           "retained_l": Ld["ret"]["retained"], "retained_t": Td["ret"]["retained"], "drift_ci_upper": ci_up,
           "ood_stratum": ood_stratum,
           "acq_canonical_l": {f: Ld["acq"]["per_family"][f]["canonical_correct"] for f in Ld["acq"]["per_family"]}}
    decision = decide(Ein)
    fragile = fragile_flags(Ein, {f: min_count_clearing(f) for f in FAMILIES})

    def ret_map(arm, pol="permissive"):
        return {i: (i in set(E[arm]["ret"]["by_bare_policy"][pol]["correct_ids"])) for i in (o["id"] for o in O[arm]["ret"])}
    contrasts = {"acquisition_l8_15_vs_sparse": {"total": mcnemar_exact(correct_map("sparse"), correct_map("l8_15")),
                                                 **acq_family},
                 "retained_lost_ids": {arm: E[arm]["ret"]["by_bare_policy"]["permissive"]["lost_ids"] for arm in ARMS},
                 "arithmetic_vs_same_run_cuda_supa_report_only": {
                     arm: {pol: mcnemar_exact(ret_map("supa", pol), ret_map(arm, pol)) for pol in ("permissive", "strict")}
                     for arm in ("sparse", "l8_15")},
                 "arithmetic_l8_15_vs_sparse": {pol: mcnemar_exact(ret_map("sparse", pol), ret_map("l8_15", pol))
                                                for pol in ("permissive", "strict")},
                 "long_context_vs_supa_same_run": {arm: mcnemar_exact(E["supa"]["lc"]["per_row_correct"],
                                                                      E[arm]["lc"]["per_row_correct"])
                                                   for arm in ("sparse", "l8_15")},
                 "long_context_l8_15_vs_sparse": mcnemar_exact(E["sparse"]["lc"]["per_row_correct"],
                                                               E["l8_15"]["lc"]["per_row_correct"]),
                 "ood_l8_15_vs_sparse": {"total_report_only": mcnemar_exact(ood_map("sparse"), ood_map("l8_15")),
                                         **ood_stratum},
                 "ood_vs_supa_report_only": {arm: {s: m for s in OOD_STRATA if (m := ood_pair("supa", arm, s))}
                                             for arm in ("sparse", "l8_15")}}
    if not subset:
        pubS = read_json(pinned("anchor_lc_supa"))["per_row_correct"]
        contrasts["long_context_vs_published_supa"] = {arm: mcnemar_exact(pubS, E[arm]["lc"]["per_row_correct"])
                                                       for arm in ARMS}
    sp_pref = sorted([x for x in ("sparse", "l8_15") if E[x]["acq"]["floors_cleared"] == 3
                      and E[x]["ret"]["arithmetic_permissive"] >= ARITH_BAR],
                     key=lambda x: (E[x]["ret"]["retained"], E[x]["acq"]["aggregate"]["canonical_correct"]), reverse=True)
    net = {f: acq_family[f]["net"] for f in FAMILIES}
    agg = Ld["acq"]["aggregate"]["canonical_correct"] - Td["acq"]["aggregate"]["canonical_correct"]
    fr = D["drift_fresh"]["ratio_l8_15_over_sparse"]["kl_arm_supa"]["ratio"]
    br = D["drift_bridge"]["ratio_l8_15_over_sparse"]["kl_arm_supa"]["ratio"]
    agree = [gate["legs"][f"{x}.select_views_001"]["text"]["agreement"] for x in ("supa", "sparse")]
    agree += [gate["legs"]["sparse.arithmetic_panel"]["samples"]["agreement"],
              gate["report_only"]["supa.arithmetic_panel_vs_b28_cpu"]["text_vs_b28_cpu"]["agreement"]]
    agree += [gate["report_only"][f"{x}.long_context"]["text"]["agreement"] for x in ("supa", "sparse")]
    pred = {"P1": Ld["acq"]["floors_cleared"] == 3,
            "P2": (-12 <= net["ledger_replay"] <= 0 and -4 <= net["prerequisite_plan"] <= 4
                   and -6 <= net["register_machine"] <= 2 and -16 <= agg <= 2),
            "P3": Ld["ret"]["retained"] >= Td["ret"]["retained"] - 1,
            "P4": Ld["ret"]["arithmetic_permissive"] >= ARITH_BAR,
            "P5": 0.60 <= fr <= 0.76, "P6": abs(br - BRIDGE_R14_RATIO) <= 0.05,
            "P7": Ld["lc"]["all_hits"] >= Td["lc"]["all_hits"] - 3,
            "P8": Ld["lc"]["says_unknown_substring"] >= Td["lc"]["says_unknown_substring"] - 1,
            "P9": all(x >= 0.99 for x in agree),
            "P10": all(gate["legs"][f"{x}.bridge_logprobs"].get("kl_l1_r14_mean", 1.0) < 1e-4 for x in ("supa", "sparse"))}
    result = {"record": "PREREG_L1_RESULTS", "schema": 2, "analysed_utc": _utc(), "verdict": decision["verdict"],
              "decision": decision, "fragile_report_only": fragile,
              "c6_binding_report_only": {
                  "what": ("per stratum: can C6 fail there at all? L8-15 can lose only rows Titan-Sparse got right, so a "
                           "stratum can fail only if Titan-Sparse is correct on at least min_lost_to_fail rows "
                           "(6 at n = 100); where it cannot, that stratum's pass carries no information"),
                  "strata": c6_bind, "non_binding": [s for s, v in c6_bind.items() if not v["can_fail"]]},
              "long_context_comparable": not any(E[x]["lc"]["rows_skipped_context_overflow"] for x in ARMS),
              "summary": {arm: arm_summary(E[arm], D, arm) for arm in ARMS}, "contrasts": contrasts, "drift": D,
              "predictions": {k: {"text": PREDICTIONS[k], "held": bool(v)} for k, v in pred.items()},
              "sparse001_rule_preference_report_only": sp_pref, **base}
    write_create(res_json, json.dumps(result, indent=1, default=float) + "\n")
    write_create(res_md, render_md(result))
    log(f"verdict: {decision['verdict']} (failed: {decision['failed'] or 'none'}); wrote {res_json.name}, {res_md.name}")
    return 0


def render_md(r: dict) -> str:
    S, dec = r["summary"], r["decision"]
    arms = [x for x in ARMS if x in S]
    L = [f"# PREREG-L1 results: {r['verdict']}", "",
         f"Analysed {r['analysed_utc']}; forward {r['run']['forward']}; freeze binding: {r['freeze_binding']}. "
         f"Failed conditions: {', '.join(dec['failed']) or 'none'}.", "",
         "## Decision rule (registered, verbatim)", "", "> " + dec["rule"], "",
         "| condition | holds |", "|---|---|"] + [f"| {k} | {v} |" for k, v in dec["conditions"].items()]
    if dec["inputs"].get("comparable_reasons"):
        L += ["", "Comparability failures: " + "; ".join(dec["inputs"]["comparable_reasons"])]
    if "l8_15" in r:
        L += ["", f"**L8-15:** {r['l8_15']}"]
    for k, v in (dec.get("c2_by_family") or {}).items():
        L.append(f"- C2 {k}: net {v['net']:+d} (lost {v['lost']}, gained {v['gained']}, p {v['p']:.4g}); tolerance "
                 f"{v['tolerance']}; detected loss {v['detected_loss']}; {'pass' if v['pass'] else 'FAIL'}")
    for k, v in (dec.get("c6_by_stratum") or {}).items():
        if "net" not in v:
            L.append(f"- C6 {k}: {v.get('error')}; FAIL")
            continue
        L.append(f"- C6 {k} (n {v['n']}): net {v['net']:+d} (lost {v['lost']}, gained {v['gained']}, p {v['p']:.4g}); "
                 f"tolerance {v['tolerance']}; detected loss {v['detected_loss']}; {'pass' if v['pass'] else 'FAIL'}")
    missing6 = [s for s in OOD_STRATA if s not in (dec.get("c6_by_stratum") or {})]
    if "C6" in dec["conditions"] and missing6:
        L.append(f"- C6: no rows for stratum {', '.join(missing6)}; FAIL")
    cb = (r.get("c6_binding_report_only") or {}).get("strata") or {}
    if cb:
        L.append("- C6 binding (report-only; a stratum can fail only if Titan-Sparse is correct on at least the "
                 "fewest rows whose loss fails it): " + "; ".join(
                     f"{s} Titan-Sparse {v['sparse_correct']}/{v['n']}, "
                     + ("can fail" if v["can_fail"] else f"CANNOT FAIL (needs >= {v['min_lost_to_fail']})")
                     for s, v in cb.items()))
    if r.get("fragile_report_only"):
        L += ["", "Fragile (within the anchor-parity tolerance of a threshold, report-only): "
              + "; ".join(r["fragile_report_only"])]
    L += ["", "## Endpoints", "", "| | " + " | ".join({"supa": "Supa 1.7", "sparse": "Titan-Sparse", "l8_15": "L8-15"}[x]
                                                     for x in arms) + " |", "|---|" + "---:|" * len(arms)]
    rows = [("Acquisition /384 (canonical)", "acquisition"), ("Families clearing floor", "floors_cleared"),
            ("Arithmetic /150 permissive", "arithmetic_permissive"), ("Arithmetic /150 strict (report)", "arithmetic_strict"),
            ("Source answers retained /47", "retained"), ("Gate v2", "gate_v2"),
            ("Long context /256 (SPENT, report)", "long_context"), ("UNKNOWN substring /64 (report)", "unknown_substring"),
            ("UNKNOWN literal /64 (report)", "unknown_literal"),
            ("Mean first-token KL vs Supa, fresh prompts", "drift_fresh_kl_arm_supa"),
            ("Mean first-token KL vs Supa, 228 bridge prompts (report)", "drift_bridge_kl_arm_supa")]
    for label, k in rows:
        if all(k in S[x] for x in arms):
            L.append(f"| {label} | " + " | ".join(str(round(S[x][k], 4) if isinstance(S[x][k], float) else S[x][k])
                                                  for x in arms) + " |")
    L.append("| Legacy per-class monitor (report) | " + " | ".join(
        f"{S[x]['legacy_monitor']['permissive']}/{S[x]['legacy_monitor']['strict']}" for x in arms) + " |")
    if all("ood_per_stratum" in S[x] for x in arms):
        L.append(f"| OOD exact state, all strata (aggregate report-only) | "
                 + " | ".join(f"{S[x]['ood_correct']}/{S[x]['ood_rows']}" for x in arms) + " |")
        for s in OOD_STRATA:
            if all(s in S[x]["ood_per_stratum"] for x in arms):
                L.append(f"| OOD {s} (C6 input; Supa report-only) | " + " | ".join(
                    f"{S[x]['ood_per_stratum'][s]['correct']}/{S[x]['ood_per_stratum'][s]['n']}" for x in arms) + " |")
    if "contrasts" in r:
        c = r["contrasts"]["acquisition_l8_15_vs_sparse"]
        L += ["", "## Paired contrasts (L8-15 vs Titan-Sparse, same run)", "",
              "Acquisition (net, discordant, exact two-sided McNemar p): " + "; ".join(
                  f"{k} {v['net']:+d} / {v['discordant']} / {v['p']:.4g}" for k, v in c.items()), ""]
        co = r["contrasts"].get("ood_l8_15_vs_sparse") or {}
        if co:
            L += ["OOD rule extrapolation (net, discordant, exact two-sided McNemar p): " + "; ".join(
                f"{k} {v['net']:+d} / {v['discordant']} / {v['p']:.4g}" for k, v in co.items() if "net" in v), ""]
        for s in ("drift_fresh", "drift_bridge"):
            q = r["drift"][s]["ratio_l8_15_over_sparse"]
            L.append(f"- {s}: KL(arm||Supa) ratio {q['kl_arm_supa']['ratio']:.3f} [{q['kl_arm_supa']['ci95'][0]:.3f}, "
                     f"{q['kl_arm_supa']['ci95'][1]:.3f}]; KL(Supa||arm) {q['kl_supa_arm']['ratio']:.3f} "
                     f"[{q['kl_supa_arm']['ci95'][0]:.3f}, {q['kl_supa_arm']['ci95'][1]:.3f}]; flags "
                     f"{r['drift'][s]['flags'] or 'none'}")
        L += ["", "## Registered predictions (scored, not decisive)", ""] + [
            f"- {k}: {v['text']} -> {'held' if v['held'] else 'FALSIFIED'}" for k, v in r["predictions"].items()]
        L += ["", f"SPARSE-001's selection rule applied to {{Titan-Sparse, L8-15}} would prefer (report-only): "
                  f"{r['sparse001_rule_preference_report_only'][:1] or 'neither'}"]
    g = r["parity_gate"]
    L += ["", "## Anchor parity gate (C0)", "",
          f"Enforced {g['enforced']}; pass {g['pass']}; failed legs {g['failed_legs'] or 'none'}."]
    for k, v in g["legs"].items():
        L.append(f"- {k}: {'pass' if v['pass'] else 'FAIL'}")
    for k, v in g["report_only"].items():
        L.append(f"- (report-only) {k}: {'within tolerance' if v['pass_report_only'] else 'outside tolerance'}")
    dec_ = [x for x in r["scorer_parity"] if x["role"] == "decisive"]
    rep_ = [x for x in r["scorer_parity"] if x["role"] != "decisive"]
    L += [f"- scorer parity on published artifacts: decisive {sum(x['pass'] for x in dec_)}/{len(dec_)}, "
          f"report-only {sum(x['pass'] for x in rep_)}/{len(rep_)}", ""]
    return "\n".join(L) + "\n"


# ============================================================================================ plan
def cmd_plan(a) -> int:
    views = read_jsonl(resolve("views"))
    probe = read_jsonl(resolve("probe"), encoding="utf-8-sig")
    lc = read_jsonl(resolve("lc_prompts"))
    fresh = read_jsonl(resolve("offtask_fresh")) if resolve("offtask_fresh").exists() else [None] * FRESH_N
    ood = read_jsonl(resolve("ood_bank")) if resolve("ood_bank").exists() else []
    n = {"acq": len(views), "ret": len(probe), "lc": len(lc), "drift": len(fresh) + BRIDGE_N, "ood": len(ood) or OOD_N}
    tok_avg = {"acq": 20.0, "ret": 6.0, "lc": 6.0}
    # OOD rows: the acquisition workload (same families, frame, decoder, cap), scaled by the ratio of mean answer
    # lengths (OOD gold vs SELECT-VIEWS-001 references), never below 1
    ood_scale = 1.0
    if ood:
        ood_scale = max(1.0, float(np.mean([len(r["gold"]) for r in ood])) / float(np.mean([len(v["reference"])
                                                                                            for v in views])))
    rate = {**SEC_PER_ROW, "ood": tuple(x * ood_scale for x in SEC_PER_ROW["ood"])}
    try:
        P = read_json(resolve("anchor_process_sparse"))
        tok_avg["acq"] = float(np.mean([d["generated_tokens"] for d in P["predictions"]]))
        Lc = read_json(resolve("anchor_lc_sparse"))
        tok_avg["lc"] = float(np.mean([r.get("generated_tokens", 0) for r in Lc["predictions"].values()]))
        B = read_json(resolve("b28"))
        tok_avg["ret"] = float(np.mean([r["source"].get("new_tokens", 6) for r in B["rows"]]))
    except (OSError, ValueError, KeyError):
        pass
    tok_avg["ood"] = tok_avg["acq"] * ood_scale
    per_arm, rows_per_arm = {}, sum(n.values())
    for band, r in DUTY.items():
        for lo_hi in (0, 1):
            tot = 0.0
            for s, cnt in n.items():
                c = rate[s][lo_hi]
                pause = (tok_avg.get(s, 0.0) / TOKEN_PAUSE_EVERY) * TOKEN_PAUSE_S
                tot += cnt * (c + max(BREATHE_MIN_S, min(r * c, 5.0)) + pause)
            per_arm[(band, lo_hi)] = tot
    compute = [sum(n[s] * rate[s][i] for s in n) for i in (0, 1)]
    ood_compute = [n["ood"] * rate["ood"][i] for i in (0, 1)]
    # in-lease overhead: per streamed load about 60 s plus 15 s for its G2 group hashes; LOADS_PER_RUN loads (the
    # anchors' OOD stage after the gate costs one extra load: Supa's; Titan-Sparse stays loaded across the gate)
    overhead_s = LOADS_PER_RUN * (60 + 15)
    out = {"arms": list(ARMS), "rows_per_arm": n, "generations_per_arm": n["acq"] + n["ret"] + n["lc"] + n["ood"],
           "forward_only_prompts_per_arm": n["drift"],
           "avg_generated_tokens_from_published_anchors": {k: round(v, 1) for k, v in tok_avg.items()},
           "compute_seconds_per_arm_no_duty_cycle": [round(x) for x in compute],
           "of_which_ood_c6_seconds_per_arm": [round(x) for x in ood_compute],
           "ood_rate_scale_vs_acquisition": round(ood_scale, 3),
           "gpu_hours_3_arms": {band: [round((3 * per_arm[(band, i)] + overhead_s) / 3600, 2) for i in (0, 1)]
                                for band in DUTY},
           "duty_ratio_bands": DUTY, "overhead_seconds_in_lease": overhead_s, "streamed_loads": LOADS_PER_RUN,
           "order": ("supa, sparse (pre-gate stages) -> parity gate (CPU) -> only if it passed: sparse ood (kept "
                     "loaded), supa ood (reloaded), l8_15 (every stage)"),
           "pre_lease_cpu": "hash the three arm files (~6 GB sequential read, about 1-3 min); no lease held",
           "build_cpu": "`build`: 2 derivations of 3.6 GB input + 1 re-read of the 2.38 GB output, about 2-5 min",
           "one_lease": LEASE_LABEL,
           "job_class": ("one I-24 single job (one lease, three arms); the D-22 4.5 h cap is scoped to T2 pairs "
                         "and does not apply"),
           "not_included": ("watchdog pauses (I-25/I-26 stalled jobs 7-30 min), waits for the peer's turn and "
                            "for the lease, and the pre-lease host gate"),
           "disk_GiB": round((2.384e9 + 3 * n["drift"] * 151936 * 2 + 3 * 2e6) / 2 ** 30, 2),
           "rate_provenance": "probe_terminal.json of the SPARSE-001/REP-001 batteries (process 270-325 s, retention "
                              "46-59 s, long context 151-192 s per arm, RTX 4070 Laptop, FP32 DenseReference); drift "
                              "forwards estimated 0.03-0.08 s per prompt; OOD rows at the process rate x the answer-"
                              "length ratio",
           "torch_imported": "torch" in sys.modules}
    print(json.dumps(out, indent=1), flush=True)
    return 0


# ============================================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("integrity")
    pc = sub.add_parser("parity-cpu")
    pc.add_argument("--out", default=None, help="also write the check list here (create-only)")
    sub.add_parser("fresh-prompts")
    b = sub.add_parser("build")
    b.add_argument("--dry-hash", action="store_true")
    b.add_argument("--record", action="store_true", help="with --dry-hash: write docs/l1/L8_15_EXPECTED.json")
    b.add_argument("--out-dir", default=None)
    f = sub.add_parser("freeze")
    f.add_argument("--approved-by", required=True, help="who approved the freeze (the operator)")
    f.add_argument("--out-dir", default=None)
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cuda", "cpu"), required=True)
    r.add_argument("--forward", choices=("dense", "hf"), default="dense")
    r.add_argument("--out-dir", default=None)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--no-freeze", action="store_true", help="tests only: skip the freeze check")
    r.add_argument("--wait-for", action="append", default=None)
    r.add_argument("--wait-poll", type=float, default=60.0)
    r.add_argument("--wait-max-h", type=float, default=0.0)
    z = sub.add_parser("analyze")
    z.add_argument("--out-dir", default=None)
    z.add_argument("--no-freeze", action="store_true", help="tests only")
    z.add_argument("--allow-subset", action="store_true", help="tests only")
    z.add_argument("--skip-tokenizer-check", action="store_true", help="tests only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "integrity":
        res = integrity()
        print(f"integrity: {sum(r['pass'] for r in res)}/{len(res)} checks pass", flush=True)
        return 0 if all(r["pass"] for r in res) else 1
    if a.cmd == "parity-cpu":
        res = parity_cpu()
        print(f"parity-cpu: {sum(r['pass'] for r in res)}/{len(res)} checks pass", flush=True)
        if a.out:
            write_create(Path(a.out), json.dumps(res, indent=1) + "\n")
        return 0 if all(r["pass"] for r in res) else 1
    if a.cmd == "fresh-prompts":
        src = build_fresh_prompts(resolve("offtask_fresh"), resolve("offtask_fresh_source"))
        print(json.dumps({k: src[k] for k in ("n", "classes", "per_class", "dropped", "file_sha256")}, indent=1))
        return 0
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "freeze":
        return cmd_freeze(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "analyze":
        return cmd_analyze(a)
    raise SystemExit(a.cmd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SanityError as e:
        log(f"SANITY FAILED, aborting: {e}")
        sys.exit(EXIT_SANITY)
    except ResumableExit as e:
        log(f"RESUMABLE STOP (lease released; rerun resumes from the saved stages): {e}")
        sys.exit(EXIT_RESUMABLE)
