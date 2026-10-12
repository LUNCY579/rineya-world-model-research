"""Prospectively bound native T2 warm-start Ledger training; never merges weights."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from dsv3wm import REPO
from dsv3wm.recovery import RecoveryConfig
from dsv3wm.ledger_recovery import (T2_ADAPTER, T2_SHA, T2_CONFIG_SHA, COHORTS,
    ledger_prompt, ledger_target, validate_ledger_record)
from dsv3wm.native import sha256, verify_freeze
from dsv3wm.native_cpu11 import admission_profile, last_admission_check
from dsv3wm.provenance import verify_bound_inputs

LORA_PARAMETERS = 10092544
OPTIMIZER = {"name": "AdamW", "lr": 2e-5, "betas": [0.9, 0.999], "eps": 1e-8,
             "weight_decay": 0.01, "amsgrad": False, "scheduler": "none", "grad_clip": 1.0}


@dataclass(frozen=True)
class LedgerConfig(RecoveryConfig):
    generation_tokens: int = 48
    max_length: int = 512
    def __post_init__(self):
        super().__post_init__()
        if self.arm not in {"A0", "A1", "A2", "A5"}:
            raise ValueError("Ledger recovery permits A0/A1/A2/A5 only")
        expected_steps = 96 if self.arm == "A5" else 32
        if (self.steps != expected_steps or self.seed != 0 or self.batch_size != 1
                or self.lr != 2e-5 or self.generation_tokens != 48 or self.max_length != 512
                or self.legality_weight != 0 or self.mtp_tokens != 2 or self.mtp_rank != 16
                or self.mtp_weight != 0.1 or self.lambda2 != 0.5 or self.lambda4 != 0.5
                or self.share_aux or self.shuffled_aux):
            raise ValueError("Ledger pilot configuration differs from the fixed operational recipe")


def normalize_adapter_config(value):
    """Ignore serialization/runtime metadata; compare every behavioral setting."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    value = dict(value)
    targets = value.get("target_modules")
    if targets is not None:
        if (not isinstance(targets, (list, tuple, set, frozenset))
                or any(not isinstance(item, str) or not item for item in targets)):
            raise ValueError("target_modules must contain only explicit nonempty projection names")
        value["target_modules"] = sorted(set(targets))
    ignored = {"inference_mode", "peft_version", "base_model_name_or_path", "revision"}
    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in sorted(item.items())}
        if isinstance(item, (set, frozenset)):
            return sorted(normalize(val) for val in item)
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if hasattr(item, "value"):
            return normalize(item.value)
        return item
    result = {key: normalize(val) for key, val in value.items() if key not in ignored}
    return result


def canonical_adapter_config(value):
    """JSON preserves bool/int/float types that Python dictionary equality aliases."""
    return json.dumps(normalize_adapter_config(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_adapter_config(actual):
    original = T2_ADAPTER / "adapter_config.json"
    if sha256(original) != T2_CONFIG_SHA:
        raise ValueError("Registered T2 configuration bytes changed")
    expected = json.loads(original.read_text(encoding="utf-8"))
    if canonical_adapter_config(actual) != canonical_adapter_config(expected):
        raise ValueError("Adapter behavior differs from registered T2 rank/alpha/dropout/targets/configuration")
    return expected


def load_ledger_rows(path, split):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError("Expected labelled Ledger TRAIN/DEV split")
    for row in rows:
        # This dedicated replay accepts neutral tick; the old generic replay does not.
        validate_ledger_record(row, replay=True)
    if len({row["parent_world_id"] for row in rows}) != len(rows):
        raise ValueError("One canonical balance parent per row is required")
    if split == "train" and len(rows) != 128:
        raise ValueError("Frozen Ledger training cohort requires 128 rows")
    if split == "dev":
        counts = {name: sum(r["provenance"]["cohort"] == name for r in rows) for name in COHORTS}
        if len(rows) != 24 or counts != COHORTS:
            raise ValueError("Frozen Ledger primary DEV cohort requires 12/4/4/4 parents")
    return rows


def required_sources():
    paths = [Path(__file__), PACKAGE / "scripts/evaluate_ledger_recovery.py",
             PACKAGE / "src/dsv3wm/ledger_recovery.py", PACKAGE / "src/dsv3wm/objectives.py",
             PACKAGE / "src/dsv3wm/native.py", PACKAGE / "src/dsv3wm/recovery.py",
             PACKAGE / "src/dsv3wm/config.py", PACKAGE / "src/dsv3wm/provenance.py",
             PACKAGE / "src/dsv3wm/data.py", PACKAGE / "src/dsv3wm/domains.py",
             REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py", REPO / "bwm/code/bwmn/rw2.py",
             REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/gpu_lease.py",
             REPO / "bwm/code/scripts/planbench_qwen.py"]
    paths += [PACKAGE / "scripts/train_ledger_recovery.py",
              PACKAGE / "scripts/evaluate_ledger_recovery_cpu11.py",
              PACKAGE / "scripts/evaluate_rcwm_retention_v2_cpu11.py",
              PACKAGE / "scripts/freeze_ledger_cpu11.py", PACKAGE / "src/dsv3wm/native_cpu11.py",
              PACKAGE / "prereg/RECOVERY_RESOURCE_CLARIFICATION_3.md",
              PACKAGE / "prereg/RECOVERY_RESOURCE_CLARIFICATION_4.md",
              PACKAGE / "prereg/LEDGER_RESOURCE_CLARIFICATION_CPU11_V1.md",
              PACKAGE / "prereg/EXECUTION_SOURCE_CLOSURE_V1.md",
              PACKAGE / "scripts/freeze_execution_dependencies.py", PACKAGE / "src/dsv3wm/__init__.py",
              REPO / "bwm/code/bwmn/__init__.py", REPO / "bwm/code/bwmn/blocks.py",
              REPO / "bwm/code/bwmn/flops.py", REPO / "bwm/code/bwmn/planbench_extract.py"]
    return paths


def bind_execution(freeze, inputs):
    digest = verify_freeze(freeze)
    verify_bound_inputs(freeze, [*required_sources(), *inputs])
    return digest


def qualified_baseline(result, qualification_freeze_sha, dev_sha):
    expected = result.get("summary", {}).get("canonical_id", {}).get("1", {})
    counts = [expected.get(key) for key in ("n", "rollout_correct", "rollout_malformed")]
    if (any(type(value) is not int for value in counts) or counts[0] != 12
            or not 0 <= counts[1] <= 12 or not 0 <= counts[2] <= 12 or counts[1] + counts[2] > 12):
        return False
    return (result.get("status") == "development-evaluated" and result.get("baseline_only") is True
            and result.get("baseline_qualified") is True and result.get("adapter_sha256") == T2_SHA
            and result.get("adapter_config_sha256") == T2_CONFIG_SHA and result.get("freeze_sha256") == qualification_freeze_sha
            and result.get("data_sha256") == dev_sha and expected.get("n") == 12
            and expected.get("rollout_correct", -1) >= 10 and expected.get("rollout_malformed", 13) <= 1)


def bind_qualification(training_freeze, qualification_path, dev_path):
    """Bind the generated B0 result in a new training freeze, retaining its old freeze.

    This resolves the necessary prospective chronology: the B0 source/data freeze
    precedes its outputs; a later training freeze includes those immutable outputs.
    """
    qualification_path = Path(qualification_path)
    verify_bound_inputs(training_freeze, [qualification_path])
    frozen = json.loads(Path(training_freeze).read_text(encoding="utf-8"))
    reference = frozen.get("qualification_freeze_sha256")
    if (not isinstance(reference, str) or len(reference) != 64
            or set(reference) - set("0123456789abcdef")):
        raise ValueError("Training freeze must name the original B0 qualification_freeze_sha256")
    result = json.loads(qualification_path.read_text(encoding="utf-8"))
    trace_digest = result.get("traces_sha256")
    if trace_digest is not None:
        trace_path = qualification_path.with_suffix(".traces.json")
        verify_bound_inputs(training_freeze, [trace_path])
        if sha256(trace_path) != trace_digest:
            raise ValueError("Qualified B0 trace digest differs from its bound artifact")
    if not qualified_baseline(result, reference, sha256(dev_path)):
        raise ValueError("Qualified B0 must match its original freeze and this fixed DEV/initializer contract")
    return result, reference


def version_record():
    return {"python": sys.version, **{name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "safetensors", "tokenizers")}}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train", "freeze", "qualification", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    cfg = LedgerConfig.read(args.config)
    dev = args.train.with_name("dev.jsonl")
    freeze_sha = bind_execution(args.freeze, [args.config, args.train, dev,
                          T2_ADAPTER / "adapter_model.safetensors", T2_ADAPTER / "adapter_config.json", args.qualification])
    qualification, qualification_freeze_sha = bind_qualification(args.freeze, args.qualification, dev)
    rows = load_ledger_rows(args.train, "train")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"resource_admission_profile": admission_profile(), "status": "started", "config": cfg.dict(), "command": sys.argv,
              "config_sha256": sha256(args.config), "data_sha256": sha256(args.train), "freeze_sha256": freeze_sha,
              "qualification_sha256": sha256(args.qualification), "initial_adapter_sha256": T2_SHA,
              "qualification_freeze_sha256": qualification_freeze_sha,
              "initial_adapter_config_sha256": T2_CONFIG_SHA, "optimizer": OPTIMIZER,
              "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in required_sources()},
              "counts": {"input_tokens": 0, "supervised_tokens": 0, "aux_tokens": 0, "forwards": 0,
                         "supervised_transitions": 0}, "loss_curve": [], "row_order": [], "parent_draws": [],
              "started_unix": time.time(), "target_schema": "bare integer A-D balances; no flag heads",
              "retention_pass": None}
    save_json(args.out / "run.json", result)
    started = time.perf_counter()
    try:
        from dsv3wm.native import runtime, boundary, EXPECTED_CORE_SHA
        from dsv3wm.native_cpu11 import load
        with runtime(cfg, "ledger-train-" + cfg.arm):
            import torch
            from peft import PeftModel
            from dsv3wm.data import conditioning
            from dsv3wm.objectives import SequentialMTP, token_objective, multi_horizon_loss
            model, tok, lm = load(cfg)
            result["admission_check"] = last_admission_check()
            if sha256(T2_ADAPTER / "adapter_model.safetensors") != T2_SHA:
                raise ValueError("Registered T2 initializer bytes changed")
            model = PeftModel.from_pretrained(model, str(T2_ADAPTER), is_trainable=True)
            validate_adapter_config(model.peft_config["default"])
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            decoder, head = lm._decoder_and_head(model)
            embedding, device = model.get_input_embeddings(), next(model.parameters()).device
            lora = [p for p in model.parameters() if p.requires_grad]
            if sum(p.numel() for p in lora) != LORA_PARAMETERS:
                raise ValueError("Unexpected trainable scope; parent must stay frozen and T2 capacity unchanged")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9000 + cfg.seed)
                mtp = (SequentialMTP(model.config.hidden_size, cfg.mtp_rank, cfg.mtp_tokens, cfg.share_aux)
                       .to(device=device, dtype=torch.float32)) if cfg.arm == "A1" else None
            parameters = lora + (list(mtp.parameters()) if mtp is not None else [])
            if any(p.dtype != torch.float32 for p in model.parameters()) or any(p.dtype != torch.float32 for p in parameters):
                raise ValueError("FP32 parent, adapters and auxiliary modules required")
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=0.01, amsgrad=False)
            rng = random.Random(177 + cfg.seed)
            order = [rng.randrange(len(rows)) for _ in range(cfg.steps * cfg.batch_size)]
            result.update(row_order=order, parent_draws=[rows[i]["parent_world_id"] for i in order],
                          lora_parameters=sum(p.numel() for p in lora),
                          auxiliary_parameters=sum(p.numel() for p in mtp.parameters()) if mtp is not None else 0,
                          trainable_parameters=sum(p.numel() for p in parameters), versions=version_record(),
                          core_sha256=EXPECTED_CORE_SHA, parent_hash_and_sidecars_verified=True)
            model.train(); training_started = time.perf_counter()
            for step in range(cfg.steps):
                boundary(cfg); lm.thermal_guard(); optimizer.zero_grad(set_to_none=True)
                value = 0.0
                for position, index in enumerate(order[step * cfg.batch_size:(step + 1) * cfg.batch_size]):
                    row, losses = rows[index], {}
                    for horizon in cfg.horizons:
                        torch.manual_seed(100000 + cfg.seed * 10000 + step * 100 + position * 10 + horizon)
                        prompt = ledger_prompt(conditioning(row, horizon))
                        target = json.dumps(ledger_target(row, horizon), sort_keys=True, separators=(",", ":"))
                        encoded = lm.encode_pair(tok, prompt, target, cfg.max_length)
                        if encoded is None:
                            raise ValueError("Overlength Ledger training record; no silent drop/truncation")
                        ids, labels, attention = (x.to(device) for x in lm.collate([encoded], tok.pad_token_id))
                        loss, stats, _ = token_objective(decoder, head, embedding, ids, attention, labels, mtp,
                                                        cfg.mtp_weight, cfg.ce_chunk, cfg.shuffled_aux)
                        losses[horizon] = loss
                        result["counts"]["input_tokens"] += int(ids.numel())
                        result["counts"]["supervised_tokens"] += stats["base_tokens"]
                        result["counts"]["aux_tokens"] += stats["aux_tokens"]
                        result["counts"]["forwards"] += 1
                        result["counts"]["supervised_transitions"] += 1
                    loss = multi_horizon_loss(losses, cfg.lambda2, cfg.lambda4)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite Ledger training loss")
                    (loss / cfg.batch_size).backward(); value += float(loss.detach()) / cfg.batch_size
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True); optimizer.step()
                result["loss_curve"].append({"step": step + 1, "loss": value,
                                             "elapsed_s": time.perf_counter() - training_started})
                save_json(args.out / "run.json", result)
                print(json.dumps(result["loss_curve"][-1]), flush=True)
                if (step + 1) % 16 == 0:
                    model.save_pretrained(args.out / f"checkpoint_step{step + 1}")
                if cfg.device == "cuda":
                    lm.gpu_yield()
            model.save_pretrained(args.out / "adapter")
            torch.save(mtp.state_dict() if mtp is not None else {}, args.out / "auxiliary.pt")
            validate_adapter_config(json.loads((args.out / "adapter/adapter_config.json").read_text(encoding="utf-8")))
            result.update(status="pilot-trained", optimizer_steps=cfg.steps,
                training_duration_s=time.perf_counter() - training_started,
                adapter_sha256=sha256(args.out / "adapter/adapter_model.safetensors"),
                adapter_config_sha256=sha256(args.out / "adapter/adapter_config.json"),
                peak_rss_bytes=getattr(__import__("psutil").Process().memory_info(), "peak_wset", None),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device == "cuda" else None,
                parameter_token_proxy=6 * 596049920 * result["counts"]["input_tokens"],
                proxy_scope="total-parent parameter-token proxy; excludes auxiliary/head/checkpoint work; not measured FLOPs",
                checkpoints_scope="immutable adapter snapshots; optimizer/RNG resume is not implemented")
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["admission_check"] = last_admission_check()
        result["duration_s"] = time.perf_counter() - started
        save_json(args.out / "run.json", result)


if __name__ == "__main__":
    main()
