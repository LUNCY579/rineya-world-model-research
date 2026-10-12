"""Frozen TRAIN-positive warm A3/A4 pilot; all admission precedes native load."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from dsv3wm import REPO
    from dsv3wm.recovery_student import (StudentConfig, load_teacher_corpus, warm_lineage, shared_orders,
                                         assert_adapter_configuration, continuation_slot, combine_student_losses)
    from dsv3wm.native import runtime, load, sha256, verify_freeze, boundary
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.data import load_records, conditioning
    from dsv3wm.recovery import load_warm, legacy_prompt, INITIAL_F, INITIAL_F_SHA
    from dsv3wm.recovery_data import legacy_target
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train", "tasks", "teacher-dir", "warm-dir", "qualification", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cfg = StudentConfig.read(args.config)
    freeze_sha = verify_freeze(args.freeze)
    sources = [Path(__file__), *[REPO / ("dsv3wm/src/dsv3wm/" + name + ".py") for name in
              ("recovery_student", "recovery", "recovery_data", "recovery_planning", "native", "provenance", "data", "domains", "planning", "objectives", "config")],
              REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py", REPO / "bwm/code/bwmn/gpu_lease.py",
              REPO / "bwm/code/scripts/planbench_qwen.py"]
    teacher_manifest = json.loads((args.teacher_dir / "manifest.json").read_text(encoding="utf-8"))
    outputs = [args.teacher_dir / name for name in teacher_manifest.get("output_sha256", {})]
    bound = [args.config, args.train, args.tasks, args.qualification, args.teacher_dir / "manifest.json", *outputs,
             args.warm_dir / "run.json", args.warm_dir / "adapter/adapter_model.safetensors",
             args.warm_dir / "adapter/adapter_config.json", INITIAL_F / "adapter_config.json",
             REPO / "dsv3wm/prereg/RECOVERY_H3_DRAFT_V1.md",
             REPO / "dsv3wm/prereg/RECOVERY_H3_CONTROL_CLARIFICATION_V1.md", *sources]
    verify_bound_inputs(args.freeze, bound)
    rows = load_records(args.train, expected_split="train")
    if any(row["domain"] != "register" for row in rows):
        raise ValueError("recovery RegisterWorld training only")
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Independent exact re-verification runs here, BEFORE runtime/model imports.
    positives, manifest, gate = load_teacher_corpus(args.teacher_dir, rows, tasks)
    if manifest["train_file_sha256"] != sha256(args.train) or manifest["tasks_file_sha256"] != sha256(args.tasks):
        raise ValueError("teacher data/task file lineage mismatch")
    warm = warm_lineage(args.warm_dir, cfg, args.train)
    warm_config = Path(warm["command"][warm["command"].index("--config") + 1])
    warm_config = warm_config if warm_config.is_absolute() else REPO / warm_config
    verify_bound_inputs(args.freeze, [warm_config])
    if sha256(warm_config) != warm.get("config_sha256"):
        raise ValueError("warm config lineage mismatch")
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    if (qualification.get("status") != "development-evaluated" or qualification.get("baseline_qualified") is not True
            or qualification.get("adapter_sha256") != INITIAL_F_SHA):
        raise ValueError("useful immutable B0 qualification required")
    trajectory_order, plan_order = shared_orders(len(rows), len(positives), cfg.steps, cfg.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    record = {"status": "started", "config": cfg.dict(), "command": sys.argv, "freeze_sha256": freeze_sha,
              "config_sha256": sha256(args.config), "train_sha256": sha256(args.train), "tasks_sha256": sha256(args.tasks),
              "teacher_manifest_sha256": sha256(args.teacher_dir / "manifest.json"),
              "verified_corpus_sha256": sha256(args.teacher_dir / "verified.jsonl"), "qualification_sha256": sha256(args.qualification),
              "warm_run_sha256": sha256(args.warm_dir / "run.json"), "warm_adapter_sha256": warm["adapter_sha256"],
              "warm_adapter_config_sha256": sha256(args.warm_dir / "adapter/adapter_config.json"),
              "registered_adapter_config_sha256": sha256(INITIAL_F / "adapter_config.json"),
              "warm_arm": cfg.warm_arm, "corpus_gate": gate, "source_sha256": {str(path): sha256(path) for path in sources},
              "training_kind": ("C3" if cfg.arm == "A3" else "C4") if cfg.control_extra_one_step else cfg.arm,
              "trajectory_order": trajectory_order, "plan_order": plan_order, "loss_curve": [],
              "counts": {"input_tokens": 0, "supervised_tokens": 0, "decoder_forwards": 0, "dynamics_examples": 0,
                         "plan_examples": 0, "extra_one_step_examples": 0},
              "planned_plan_forwards": cfg.steps if cfg.plan_weight else 0,
              "omitted_plan_forwards_in_control": 0 if cfg.plan_weight else cfg.steps,
              "planned_control_extra_one_step_forwards": cfg.steps if cfg.control_extra_one_step else 0,
              "nominal_total_forwards": cfg.steps * (len(cfg.horizons) + 1),
              "scope": "one-seed warm H3 feasibility; no efficacy/confirmation claim",
              "control_scope": "plan slot replaced by real same-row h1 CE: equal nominal forwards/updates, different tokens/labels/FLOPs" if cfg.control_extra_one_step else None}
    started = time.perf_counter()
    try:
        with runtime(cfg, "recovery-student-" + cfg.arm):
            import torch
            from dsv3wm.objectives import token_objective
            model, tokenizer, lm = load(cfg)
            model = load_warm(model, args.warm_dir / "adapter", trainable=True)
            peft = model.peft_config["default"]
            if peft.r != 16 or peft.lora_alpha != 32 or peft.lora_dropout != .05:
                raise ValueError("warm adapter capacity/dropout drift")
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            decoder, head = lm._decoder_and_head(model)
            embedding, device = model.get_input_embeddings(), next(model.parameters()).device
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .999), eps=1e-8, weight_decay=.01)
            def loss_pair(prompt, target, dropout_seed, kind):
                torch.manual_seed(dropout_seed)
                encoded = lm.encode_pair(tokenizer, prompt, target, cfg.max_length)
                if encoded is None:
                    raise ValueError("overlength H3 record; no silent truncation/drop")
                ids, labels, attention = (value.to(device) for value in lm.collate([encoded], tokenizer.pad_token_id))
                loss, stats, _ = token_objective(decoder, head, embedding, ids, attention, labels, chunk=cfg.ce_chunk)
                record["counts"]["input_tokens"] += ids.numel()
                record["counts"]["supervised_tokens"] += stats["base_tokens"]
                record["counts"]["decoder_forwards"] += 1
                record["counts"][kind + "_examples"] += 1
                return loss
            model.train()
            for step, (trajectory_index, plan_index) in enumerate(zip(trajectory_order, plan_order)):
                boundary(cfg)
                lm.thermal_guard()
                optimizer.zero_grad(set_to_none=True)
                row, losses = rows[trajectory_index], {}
                for horizon in cfg.horizons:
                    losses[horizon] = loss_pair(legacy_prompt(conditioning(row, horizon)),
                        json.dumps(legacy_target(row, horizon), sort_keys=True, separators=(",", ":")),
                        100000 + cfg.seed * 10000 + step * 100 + horizon, "dynamics")
                slot_prompt, slot_target, slot_kind = continuation_slot(row, positives[plan_index], cfg)
                slot_loss = loss_pair(slot_prompt, slot_target,
                                     100000 + cfg.seed * 10000 + step * 100 + 9, slot_kind)
                loss = combine_student_losses(losses, slot_loss, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite H3 training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                item = {"step": step + 1, "loss": float(loss.detach()), "slot_kind": slot_kind,
                        "slot_loss": float(slot_loss.detach()),
                        "plan_loss": float(slot_loss.detach()) if slot_kind == "plan" else None,
                        "extra_one_step_loss": float(slot_loss.detach()) if slot_kind == "extra_one_step" else None,
                        "elapsed_s": time.perf_counter() - started}
                record["loss_curve"].append(item)
                print(json.dumps(item), flush=True)
                if (step + 1) % 16 == 0:
                    model.save_pretrained(args.out / f"checkpoint_step{step + 1}")
            model.save_pretrained(args.out / "adapter")
            assert_adapter_configuration(args.out / "adapter/adapter_config.json")
            record.update(status="student-pilot-trained", optimizer_steps=cfg.steps,
                          trainable_parameters=sum(parameter.numel() for parameter in parameters),
                          adapter_sha256=sha256(args.out / "adapter/adapter_model.safetensors"),
                          adapter_config_sha256=sha256(args.out / "adapter/adapter_config.json"),
                          core_sha256=sha256(lm.MODELS[cfg.core] / "model.safetensors"),
                          optimizer={"kind": "AdamW", "lr": cfg.lr, "betas": [.9, .999], "eps": 1e-8,
                                     "weight_decay": .01, "clip": 1., "scheduler": None},
                          parameter_token_proxy=6 * 596049920 * record["counts"]["input_tokens"],
                          proxy_scope="parent parameter-token proxy, excludes recomputation/head work; not measured FLOPs",
                          peak_rss_bytes=getattr(__import__('psutil').Process().memory_info(), 'peak_wset', None))
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["duration_s"] = time.perf_counter() - started
        (args.out / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
