"""CPU native teacher-forced loss audit on six fresh historical-format fixtures.

This is a narrow loss-only audit, not a retention pass. No historical evaluation
bank, registered adapter or frozen test label is inspected. Collection semantics
are used once to construct fresh fixtures, before native loading or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixtures() -> dict:
    """Pure, deterministic, no-torch construction from read-only prompt/step APIs."""
    from dsv3wm import REPO
    from dsv3wm.domains import _semantics, canonical, exact_transition, rules_for

    register, ledger = _semantics("register"), _semantics("ledger")
    rows = []
    for identifier, x, action, limit in (("rw-odd-halving", 3, "halve", 32),
                                          ("rw-limit-boundary", 32, "add1", 32),
                                          ("rw-negative-double", -4, "double", 32)):
        after, legal, terminated, invalid = exact_transition("register", {"x": x}, action,
                                                           rules_for("register", limit=limit))
        if invalid:
            raise AssertionError("A retention fixture must use the historical valid-input contract")
        messages = [{"role": "system", "content": register.SYS_FORWARD},
                    {"role": "user", "content": register.canonical({"action": action, "limit": limit, "x": x})}]
        rows.append({"id": identifier, "domain": "register", "fresh_input": {"x": x, "action": action, "limit": limit},
                     "messages": messages, "prompt": register.forward_prompt(x, action, limit),
                     "target": register.forward_target(after["x"], legal, terminated),
                     "format": "bwmn.rw2.forward_prompt/forward_target"})
    ledger_cases = [
        ("ledger-posted-transfer", {"A": 9, "B": -2, "C": 3, "D": 0},
         {"kind": "transfer", "account": "A", "other": "B", "amount": 4, "status": "posted"}),
        ("ledger-cancelled-scale", {"A": 3, "B": 8, "C": -1, "D": 2},
         {"kind": "scale", "account": "A", "other": "C", "amount": -5, "status": "cancelled"}),
        ("ledger-posted-negative-scale", {"A": -3, "B": 1, "C": 7, "D": 0},
         {"kind": "scale", "account": "A", "other": "D", "amount": -2, "status": "posted"}),
    ]
    for identifier, state, event in ledger_cases:
        after, legal, terminated, invalid = exact_transition("ledger", state, event, rules_for("ledger"))
        if not legal or terminated or invalid:
            raise AssertionError("Historical ledger fixtures must be valid and nonterminating")
        user = ledger.dumps({"before": state, "event": event})
        messages = [{"role": "system", "content": ledger.SYSTEM_ONE["ledger_replay"]},
                    {"role": "user", "content": user}]
        rows.append({"id": identifier, "domain": "ledger", "fresh_input": {"before": state, "event": event},
                     "messages": messages, "prompt": register.render_chat(messages[0]["content"], user),
                     "target": canonical(after), "format": "rcwmt.env.SYSTEM_ONE[ledger_replay], compact chat"})
    paths = [Path(__file__), REPO / "dsv3wm/src/dsv3wm/domains.py", REPO / "bwm/code/bwmn/rw2.py",
             REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py"]
    return {"schema": "dsv3wm-retention-fixtures-v1", "construction": "six fresh fixed inputs; no bank or marginal access",
            "fixture_count": len(rows), "fixtures": rows,
            "source_sha256": {str(path.relative_to(REPO)).replace("\\", "/"): file_hash(path) for path in paths},
            "claim_scope": "Teacher-forced CE on old prompt formats only; no behavioral retention pass or skill-wide guarantee."}


def freeze_fixtures(path: str | Path) -> tuple[dict, str]:
    """Write once and hash before any model import, load, loss or selection."""
    document = build_fixtures()
    content = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
    return document, hashlib.sha256(content).hexdigest()


def summarize(rows: list[dict]) -> dict:
    """Token-weighted CE; means over six fixtures are descriptive, not statistical tests."""
    def aggregate(selected):
        count = sum(row["supervised_tokens"] for row in selected)
        if count <= 0 or any(not math.isfinite(row["token_mean_ce"]) for row in selected):
            raise ValueError("Invalid or nonfinite retention loss")
        return {"fixtures": len(selected), "supervised_tokens": count,
                "token_mean_ce": sum(row["token_mean_ce"] * row["supervised_tokens"] for row in selected) / count}
    return {"overall": aggregate(rows),
            "by_domain": {domain: aggregate([row for row in rows if row["domain"] == domain])
                          for domain in sorted({row["domain"] for row in rows})}}


def compare_baseline(result: dict, baseline: dict) -> dict:
    """Pure paired loss deltas; reject changes to fixtures, model or tokenization."""
    if baseline.get("status") != "complete" or baseline.get("model_kind") != "frozen_core":
        raise ValueError("Baseline must be a completed frozen-core audit")
    for key in ("fixture_sha256", "parent_checkpoint_sha256", "tokenizer_chat_template_sha256"):
        if result.get(key) != baseline.get(key):
            raise ValueError(f"Baseline mismatch: {key}")
    old = {row["id"]: row for row in baseline["rows"]}
    if set(old) != {row["id"] for row in result["rows"]}:
        raise ValueError("Baseline fixture IDs disagree")
    deltas = []
    for row in result["rows"]:
        base = old[row["id"]]
        for key in ("supervised_tokens", "input_tokens", "encoded_sha256"):
            if row[key] != base[key]:
                raise ValueError(f"Baseline tokenization mismatch: {row['id']}/{key}")
        deltas.append({"id": row["id"], "domain": row["domain"], "core_ce": base["token_mean_ce"],
                       "current_ce": row["token_mean_ce"], "delta_ce": row["token_mean_ce"] - base["token_mean_ce"]})
    return {"interpretation": "Positive CE delta means higher teacher-forced loss; this does not establish behavioral forgetting.",
            "rows": deltas, "overall_delta_ce": result["summary"]["overall"]["token_mean_ce"]
                                                   - baseline["summary"]["overall"]["token_mean_ce"],
            "retention_pass": None}


def native_provenance(model, lm) -> dict:
    """Versions and actual loaded source files; no model execution."""
    from dsv3wm import REPO
    from dsv3wm import native
    versions, files = {}, {}
    for distribution in ("torch", "transformers", "peft", "safetensors", "tokenizers"):
        versions[distribution] = importlib.metadata.version(distribution)
        module = sys.modules.get(distribution)
        location = getattr(module, "__file__", None)
        if location and Path(location).is_file():
            files[str(Path(location).resolve())] = file_hash(location)
    decoder, _ = lm._decoder_and_head(model)
    for item in (type(model), type(decoder)):
        location = inspect.getfile(item)
        if Path(location).is_file():
            files[str(Path(location).resolve())] = file_hash(location)
    for path in (Path(lm.__file__), Path(native.__file__), REPO / "dsv3wm/src/dsv3wm/config.py"):
        files[str(path.resolve())] = file_hash(path)
    return {"python": sys.version, "libraries": versions, "loaded_source_sha256": files,
            "parent_sidecars_sha256": dict(native.EXPECTED_SIDECARS),
            "parent_hash_and_sidecars_verified_by_native_load": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="unmerged adapter; omit to measure frozen-core baseline")
    parser.add_argument("--baseline", type=Path, help="completed core result for paired loss-only comparisons")
    parser.add_argument("--out", type=Path, required=True, help="new JSON result file")
    args = parser.parse_args()
    from dsv3wm.config import Config
    cfg = Config.read(args.config)
    if cfg.device != "cpu":
        raise ValueError("This narrow retention audit permits CPU execution only")
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite retention results")
    if args.adapter and not args.adapter.is_dir():
        raise ValueError("Adapter must be an existing unmerged adapter directory")
    fixtures_path = args.out.with_suffix(".fixtures.json")
    fixture_document, fixture_sha = freeze_fixtures(fixtures_path)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    if baseline is not None and (baseline.get("status") != "complete" or baseline.get("model_kind") != "frozen_core"
                                 or baseline.get("fixture_sha256") != fixture_sha):
        raise ValueError("Baseline must be completed core audit with identical frozen fixtures")
    result = {"schema": "dsv3wm-native-retention-loss-v1", "status": "started",
              "claim_scope": fixture_document["claim_scope"], "retention_pass": None,
              "config": cfg.dict(), "config_sha256": file_hash(args.config), "command": sys.argv,
              "model_kind": "unmerged_adapter" if args.adapter else "frozen_core",
              "adapter": str(args.adapter.resolve()) if args.adapter else None,
              "fixture_file": str(fixtures_path.resolve()), "fixture_sha256": fixture_sha,
              "fixtures_frozen_before_native_load": True, "teacher_forced": True,
              "generated_behavior_accuracy_measured": False, "precision": "FP32", "tf32": False,
              "rows": [], "fixture_source_sha256": fixture_document["source_sha256"]}
    if args.adapter:
        result["adapter_files_sha256"] = {path.name: file_hash(path) for path in sorted(args.adapter.iterdir())
                                          if path.is_file()}
    if args.baseline:
        result["baseline_file_sha256"] = file_hash(args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    started = time.perf_counter()
    try:
        from dsv3wm.native import EXPECTED_CORE_SHA, boundary, load, runtime
        with runtime(cfg, "retention-loss-" + (cfg.arm if args.adapter else "core")):
            import torch
            model, tokenizer, lm = load(cfg, args.adapter)
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise RuntimeError("FP32 parent and adapter required for retention loss audit")
            model.eval()
            model.config.use_cache = False
            result["parent_checkpoint_sha256"] = EXPECTED_CORE_SHA
            result["native_provenance"] = native_provenance(model, lm)
            template = tokenizer.chat_template
            result["tokenizer_chat_template_sha256"] = hashlib.sha256(
                json.dumps(template, sort_keys=True).encode("utf-8")).hexdigest()
            device = next(model.parameters()).device
            for fixture in fixture_document["fixtures"]:
                boundary(cfg)
                rendered = tokenizer.apply_chat_template(fixture["messages"], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                if rendered != fixture["prompt"]:
                    raise ValueError(f"Historical prompt/template drift: {fixture['id']}")
                encoded = lm.encode_pair(tokenizer, fixture["prompt"], fixture["target"], cfg.max_length)
                if encoded is None:
                    raise ValueError(f"Overlength fixture: {fixture['id']}; no silent truncation")
                ids, labels, attention = (tensor.to(device) for tensor in lm.collate([encoded], tokenizer.pad_token_id))
                with torch.no_grad():
                    loss, count = lm.chunked_ce(model, ids, attention, labels, chunk=cfg.ce_chunk)
                row = {"id": fixture["id"], "domain": fixture["domain"], "token_mean_ce": float(loss.item()),
                       "supervised_tokens": int(count), "input_tokens": len(encoded.input_ids),
                       "target_includes_eos": True,
                       "encoded_sha256": hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest()}
                if not math.isfinite(row["token_mean_ce"]):
                    raise ValueError("Nonfinite retention fixture loss")
                result["rows"].append(row)
                args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            result["summary"] = summarize(result["rows"])
            if baseline is not None:
                result["paired_core_loss_comparison"] = compare_baseline(result, baseline)
            result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["duration_seconds"] = time.perf_counter() - started
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "status": result["status"],
                      "teacher_forced_token_mean_ce": result["summary"]["overall"]["token_mean_ce"],
                      "retention_pass": None}, indent=2))


if __name__ == "__main__":
    main()
