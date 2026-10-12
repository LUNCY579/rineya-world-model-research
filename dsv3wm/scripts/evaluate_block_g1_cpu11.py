"""Frozen original-delta BW qualification with explicit raw-before-filter diagnostics."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src")); sys.path.insert(0, str(PACKAGE / "scripts"))
from build_block_g1 import (BW_ADAPTER, BW_SHA, BW_CONFIG_SHA, BlockG1Config, B, VERSION,
                            canonical_parent, structural_valid, required_sources, score_case, summarize)
from dsv3wm.native import sha256, verify_freeze
from dsv3wm.provenance import verify_bound_inputs
from dsv3wm.native_cpu11 import admission_profile, last_admission_check


def cpu11_sources():
    return [*required_sources(), Path(__file__), PACKAGE / "src/dsv3wm/native_cpu11.py",
            PACKAGE / "scripts/freeze_block_g1_cpu11.py", PACKAGE / "prereg/BLOCK_G1_CPU11_V1.md"]


def validate_cases(cases):
    # No exact transitions or correctness-based filtering in this input gate.
    if len(cases) != 16 or len({c["id"] for c in cases}) != 16:
        raise ValueError("The complete fixed twelve-primary/four-diagnostic list is required")
    primary = [c for c in cases if c["role"] == "primary"]
    diagnostics = [c for c in cases if c["role"] == "diagnostic"]
    if len(primary) != 12 or len(diagnostics) != 4 or len({c["parent_world_id"] for c in primary}) != 12:
        raise ValueError("Independent primary cases mismatch")
    parents = {c["id"]: c["parent_world_id"] for c in primary}
    for case in cases:
        facts = frozenset(tuple(f) for f in case["before"])
        if not structural_valid(facts, case["objects"]) or canonical_parent(facts, case["objects"]) != case["parent_world_id"]:
            raise ValueError("Constructed canonical parent identity mismatch")
        if case["role"] == "diagnostic" and parents.get(case.get("source_case_id")) != case["parent_world_id"]:
            raise ValueError("Diagnostic must remain a paired primary-parent variant")
        if case["role"] == "primary" and case["prompt"] != B().forward_prompt_delta(facts, tuple(case["action"])):
            raise ValueError("Canonical source prompt parity failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cases", "manifest", "freeze", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(); cfg = BlockG1Config.read(args.config)
    digest = verify_freeze(args.freeze)
    verify_bound_inputs(args.freeze, [*cpu11_sources(), args.config, args.cases, args.manifest,
                        BW_ADAPTER / "adapter_model.safetensors", BW_ADAPTER / "adapter_config.json"])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("generator") != VERSION or manifest["cases_sha256"] != sha256(args.cases) or manifest["config"] != cfg.dict():
        raise ValueError("Prospective manifest/cases/config mismatch")
    if sha256(BW_ADAPTER / "adapter_model.safetensors") != BW_SHA or sha256(BW_ADAPTER / "adapter_config.json") != BW_CONFIG_SHA:
        raise ValueError("Registered private BW initializer identity mismatch")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_cases(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix(".raw.json")
    result = {"resource_admission_profile": admission_profile(), "status": "started", "command": sys.argv, "config": cfg.dict(), "config_sha256": sha256(args.config),
              "cases_sha256": sha256(args.cases), "manifest_sha256": sha256(args.manifest), "freeze_sha256": digest,
              "adapter_sha256": BW_SHA, "adapter_config_sha256": BW_CONFIG_SHA, "oracle_feedback": False,
              "source_sha256": {str(p.relative_to(PACKAGE.parent)): sha256(p) for p in cpu11_sources()},
              "scope": "constructed original-interface G1; not teacher-unseen, confirmation or efficacy",
              "retention_pass": None}
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result) + "\n")
    raw_records = [{"id": c["id"], "parent_world_id": c["parent_world_id"], "role": c["role"], "status": "started"} for c in cases]
    with raw_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(raw_records) + "\n")
    started = time.perf_counter(); counters = {"decoder_forwards": 0, "decoder_token_positions": 0}
    try:
        from dsv3wm.native import runtime, EXPECTED_CORE_SHA
        from dsv3wm.native_cpu11 import load
        with runtime(cfg, "block-g1-B0"):
            import torch
            from peft import PeftModel
            model, tokenizer, lm = load(cfg)
            result["admission_check"] = last_admission_check()
            model = PeftModel.from_pretrained(model, str(BW_ADAPTER), is_trainable=False)
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("FP32 parent/adapter required")
            model.eval(); lm.DUTY_RATIO = 0.0
            prompts = [case["prompt"] for case in cases]
            lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
            if any(n > cfg.max_length for n in lengths):
                raise ValueError("Overlength input; no silent truncation")
            def count_forward(module, positional, keyword):
                ids = keyword.get("input_ids", positional[0] if positional else None)
                if ids is None:
                    raise ValueError("Unable to identify decoder token positions")
                counters["decoder_forwards"] += 1; counters["decoder_token_positions"] += int(ids.numel())
            hook = model.register_forward_pre_hook(count_forward, with_kwargs=True)
            try:
                generated = lm.sample_crn(model, tokenizer, prompts, [None] * len(prompts), max_new_tokens=cfg.generation_tokens)
            finally:
                hook.remove()
            if len(generated) != len(cases):
                raise ValueError("Generation batch mismatch")
            # Finish and persist every raw output BEFORE exact state/action scoring.
            for record, value in zip(raw_records, generated):
                record.update(status="generated", generated=value)
            raw_path.write_text(json.dumps(raw_records, indent=2) + "\n", encoding="utf-8")
            outcomes = [score_case(case, value["text"]) for case, value in zip(cases, generated)]
            result.update(status="development-evaluated", summary=summarize(outcomes), outcomes=outcomes,
                          parent_sha256=EXPECTED_CORE_SHA, model_queries=len(cases), input_tokens=sum(lengths),
                          output_tokens=sum(int(value["n_tokens"]) for value in generated),
                          versions={"python": sys.version, **{name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "peft", "tokenizers")}},
                          peak_rss_bytes=getattr(__import__("psutil").Process().memory_info(), "peak_wset", None))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["admission_check"] = last_admission_check()
        result.update(duration_s=time.perf_counter() - started, raw_sha256=sha256(raw_path), **counters,
                      counter_scope="actual padded generation positions/model forwards; no FLOP or peak-GPU claim")
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "out": str(args.out), "g1_qualified": result.get("summary", {}).get("g1_qualified")}), flush=True)


if __name__ == "__main__":
    main()
