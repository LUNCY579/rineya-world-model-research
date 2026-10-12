"""Bound Ledger B0/adapted DEV readout; generation precedes offline scoring."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "scripts"))
from train_ledger_recovery import (LedgerConfig, bind_execution, load_ledger_rows, required_sources,
                                  save_json, validate_adapter_config, version_record)
from dsv3wm.ledger_recovery import T2_ADAPTER, T2_SHA, T2_CONFIG_SHA, ledger_prompt, ledger_target, parse_ledger_response
from dsv3wm.native import sha256


class LedgerNativePredictor:
    """Generation-only adapter with raw trace IDs and real decoder counters."""
    def __init__(self, model, tokenizer, config, trace_sink=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.calls = self.input_tokens = self.output_tokens = 0
        self.decoder_forwards = self.decoder_token_positions = 0
        self.traces, self.trace_sink = [], trace_sink

    def predict_batch(self, inputs, contexts):
        if len(inputs) != len(contexts):
            raise ValueError("One parent/phase/step context per request required")
        if not inputs:
            return []
        from dsv3wm.native import boundary
        from bwmn import lm
        boundary(self.config)
        prompts = [ledger_prompt(item) for item in inputs]
        lengths = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
        if any(n > self.config.max_length for n in lengths):
            raise ValueError("Overlength Ledger prediction input")
        pending = []
        for item, prompt, count, context in zip(inputs, prompts, lengths, contexts):
            if (set(context) != {"parent_world_id", "phase", "step"} or not isinstance(context["parent_world_id"], str)
                    or context["phase"] not in {"rollout", "direct"} or type(context["step"]) is not int):
                raise ValueError("Explicit causal trace context required")
            trace = {"trace_id": f"{context['parent_world_id']}:{context['phase']}:{context['step']}:{len(self.traces)}",
                     **context, "status": "started", "conditioning": copy.deepcopy(item), "input_tokens": count}
            self.traces.append(trace); pending.append(trace)
        if self.trace_sink:
            self.trace_sink(self.traces)
        def count_forward(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None:
                raise ValueError("Decoder counter cannot identify input token positions")
            self.decoder_forwards += 1
            self.decoder_token_positions += int(ids.numel())
        self.model.eval()
        if self.config.device == "cpu":
            lm.DUTY_RATIO = 0.0
        hook = self.model.register_forward_pre_hook(count_forward, with_kwargs=True)
        try:
            generated = lm.sample_crn(self.model, self.tokenizer, prompts, [None] * len(prompts),
                                      max_new_tokens=self.config.generation_tokens)
        finally:
            hook.remove()
        if len(generated) != len(inputs):
            raise ValueError("Native generation batch length mismatch")
        self.calls += len(inputs); self.input_tokens += sum(lengths)
        outputs = []
        for item, raw, trace in zip(inputs, generated, pending):
            parsed = parse_ledger_response(raw["text"], item)
            self.output_tokens += int(raw["n_tokens"])
            trace.update(status="complete", generated=raw, parsed=parsed)
            outputs.append(parsed)
        if self.trace_sink:
            self.trace_sink(self.traces)
        return outputs


def generate_dynamics(predictor, rows, *, baseline_only):
    """No labels inspected here. All rollout inputs use current own predictions."""
    states = [copy.deepcopy(row["s0"]) for row in rows]
    outputs, dead = [[] for _ in rows], [False] * len(rows)
    for step in range(4):
        active = [i for i in range(len(rows)) if not dead[i]]
        batch = [{"domain": "ledger", "rules": copy.deepcopy(rows[i]["rules"]), "s0": copy.deepcopy(states[i]),
                  "actions": [copy.deepcopy(rows[i]["actions"][step])]} for i in active]
        contexts = [{"parent_world_id": rows[i]["parent_world_id"], "phase": "rollout", "step": step + 1} for i in active]
        values = predictor.predict_batch(batch, contexts) if active else []
        for i, value in zip(active, values):
            outputs[i].append(value)
            if value.get("malformed"):
                dead[i] = True
            else:
                states[i] = copy.deepcopy(value["state"])
    direct = {}
    for horizon in ((1,) if baseline_only else (1, 2, 4)):
        batch = [{"domain": "ledger", "rules": copy.deepcopy(row["rules"]), "s0": copy.deepcopy(row["s0"]),
                  "actions": copy.deepcopy(row["actions"][:horizon])} for row in rows]
        contexts = [{"parent_world_id": row["parent_world_id"], "phase": "direct", "step": horizon} for row in rows]
        direct[horizon] = predictor.predict_batch(batch, contexts)
    return outputs, direct


def score_dynamics(rows, outputs, direct):
    per_parent = []
    for i, row in enumerate(rows):
        entry = {"parent_world_id": row["parent_world_id"], "cohort": row["provenance"]["cohort"], "horizons": {}}
        for horizon in (1, 2, 4):
            roll = outputs[i][horizon - 1] if len(outputs[i]) >= horizon else {"malformed": True}
            prediction = direct[horizon][i] if horizon in direct else None
            gold = row["labels"]["states"][horizon - 1]
            before_gold = row["s0"] if horizon == 1 else row["labels"]["states"][horizon - 2]
            attempted = len(outputs[i]) >= horizon
            previous = row["s0"] if horizon == 1 else (outputs[i][horizon - 2].get("state") if len(outputs[i]) >= horizon - 1 else None)
            entry["horizons"][str(horizon)] = {
                "rollout_correct": not roll.get("malformed") and roll.get("state") == gold,
                "rollout_malformed": bool(roll.get("malformed")),
                "direct_correct": None if prediction is None else not prediction.get("malformed") and prediction.get("state") == gold,
                "direct_malformed": None if prediction is None else bool(prediction.get("malformed")),
                "reference_noop": gold == before_gold,
                "false_apply": attempted and not roll.get("malformed") and gold == before_gold and roll.get("state") != previous,
                "false_copy": attempted and not roll.get("malformed") and gold != before_gold and roll.get("state") == previous,
                "rollout_attempted": attempted}
        per_parent.append(entry)
    summary = {}
    cohorts = sorted({row["provenance"]["cohort"] for row in rows}) + ["all"]
    for cohort in cohorts:
        selected = [row for row in per_parent if cohort == "all" or row["cohort"] == cohort]
        summary[cohort] = {}
        for horizon in ("1", "2", "4"):
            metrics = {"n": len(selected)}
            for field in ("rollout_correct", "rollout_malformed", "direct_correct", "direct_malformed",
                          "reference_noop", "false_apply", "false_copy", "rollout_attempted"):
                values = [row["horizons"][horizon][field] for row in selected]
                metrics[field] = None if all(value is None for value in values) else sum(bool(value) for value in values)
            summary[cohort][horizon] = metrics
    return {"summary": summary, "per_parent": per_parent,
            "flag_metrics_scope": "legality/termination flags are known input-contract values; no learned flag accuracy",
            "false_copy_apply_scope": "report-only attempted well-formed steps; malformed and unattempted denominators are reported separately"}


def validate_child_lineage(adapter, cfg, train):
    adapter = Path(adapter)
    if adapter.resolve() == T2_ADAPTER.resolve():
        if sha256(adapter / "adapter_model.safetensors") != T2_SHA:
            raise ValueError("Registered T2 weights changed")
        return None
    record = json.loads((adapter.parent / "run.json").read_text(encoding="utf-8"))
    if (record.get("status") != "pilot-trained" or record.get("config") != cfg.dict()
            or record.get("data_sha256") != sha256(train) or record.get("initial_adapter_sha256") != T2_SHA
            or record.get("adapter_sha256") != sha256(adapter / "adapter_model.safetensors")
            or record.get("adapter_config_sha256") != sha256(adapter / "adapter_config.json")):
        raise ValueError("Adapted Ledger training lineage mismatch")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "out", "freeze"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--adapter", type=Path, default=T2_ADAPTER)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    cfg = LedgerConfig.read(args.config)
    train = args.data.with_name("train.jsonl")
    bound = [args.config, args.data, train, args.adapter / "adapter_model.safetensors", args.adapter / "adapter_config.json"]
    if args.adapter.resolve() != T2_ADAPTER.resolve():
        bound.append(args.adapter.parent / "run.json")
    freeze_sha = bind_execution(args.freeze, bound)
    if args.baseline_only and args.adapter.resolve() != T2_ADAPTER.resolve():
        raise ValueError("B0 qualification must use the unchanged original T2 artifact")
    child = validate_child_lineage(args.adapter, cfg, train)
    validate_adapter_config(json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8")))
    rows = load_ledger_rows(args.data, "dev")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces = args.out.with_suffix(".traces.json")
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write("{\"status\":\"reserved\"}\n")
    with traces.open("x", encoding="utf-8") as stream:
        stream.write("[]\n")
    result = {"status": "started", "config": cfg.dict(), "command": sys.argv, "baseline_only": args.baseline_only,
              "config_sha256": sha256(args.config), "data_sha256": sha256(args.data), "freeze_sha256": freeze_sha,
              "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
              "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
              "training_run_sha256": None if child is None else sha256(args.adapter.parent / "run.json"),
              "source_sha256": {str(path.relative_to(PACKAGE.parent)): sha256(path) for path in required_sources()},
              "instrument": "T2-compatible Ledger bare balances / own-state rollout / batched greedy standard KV",
              "oracle_feedback": False, "retention_pass": None, "baseline_qualified": False,
              "flags_source": "input syntax/contract only; no learned legality/termination metric", "started_unix": time.time()}
    save_json(args.out, result)
    started = time.perf_counter(); predictor = None
    try:
        from dsv3wm.native import runtime, load, boundary, EXPECTED_CORE_SHA
        with runtime(cfg, "ledger-evaluate-" + ("B0" if args.baseline_only else cfg.arm)):
            import torch
            from peft import PeftModel
            from dsv3wm.data import conditioning
            model, tok, lm = load(cfg)
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
            validate_adapter_config(model.peft_config["default"])
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("FP32 parent and Ledger adapter required")
            model.eval()
            predictor = LedgerNativePredictor(model, tok, cfg, trace_sink=lambda value: save_json(traces, value))
            # Legitimate response teacher forcing is separate from generated-state scoring.
            ce_rows = []
            for row in rows:
                boundary(cfg)
                prompt = ledger_prompt(conditioning(row, 1))
                target = json.dumps(ledger_target(row, 1), sort_keys=True, separators=(",", ":"))
                encoded = lm.encode_pair(tok, prompt, target, cfg.max_length)
                if encoded is None:
                    raise ValueError("Overlength DEV response; no truncation/drop")
                ids, labels, attention = (value.to(next(model.parameters()).device) for value in lm.collate([encoded], tok.pad_token_id))
                with torch.no_grad():
                    loss, count = lm.chunked_ce(model, ids, attention, labels, chunk=cfg.ce_chunk)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DEV CE")
                ce_rows.append({"parent_world_id": row["parent_world_id"], "cohort": row["provenance"]["cohort"],
                                "token_mean_ce": float(loss), "supervised_tokens": int(count)})
            outputs, direct = generate_dynamics(predictor, rows, baseline_only=args.baseline_only)
            # All generated outputs are fixed before exact labels are scored.
            result.update(score_dynamics(rows, outputs, direct))
            result.update(status="development-evaluated", versions=version_record(), core_sha256=EXPECTED_CORE_SHA,
                          parent_hash_and_sidecars_verified=True, teacher_forced_one_step=ce_rows,
                          teacher_forced_one_step_ce=sum(row["token_mean_ce"] * row["supervised_tokens"] for row in ce_rows)
                            / sum(row["supervised_tokens"] for row in ce_rows),
                          ce_scope="token-weighted response+EOS CE; teacher-forced, separate from native state outcomes",
                          peak_rss_bytes=getattr(__import__("psutil").Process().memory_info(), "peak_wset", None))
            canonical = result["summary"]["canonical_id"]["1"]
            result["baseline_qualified"] = (args.baseline_only and canonical["n"] == 12
                                            and canonical["rollout_correct"] >= 10 and canonical["rollout_malformed"] <= 1)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if predictor is not None:
            result.update(model_queries=predictor.calls, input_tokens=predictor.input_tokens, output_tokens=predictor.output_tokens,
                          decoder_forwards=predictor.decoder_forwards, decoder_token_positions=predictor.decoder_token_positions,
                          decoder_counter_scope="actual model forward-pre-hook counts, padded token positions, generation only; no FLOP claim")
        result.update(duration_s=time.perf_counter() - started, traces_sha256=sha256(traces))
        save_json(args.out, result)
        print(json.dumps({"out": str(args.out), "status": result["status"], "baseline_qualified": result["baseline_qualified"]}), flush=True)


if __name__ == "__main__":
    main()
