"""Additive offline H3 readout costs; no generation, scoring or inference edits.

Proposal counters are recorded single-query formula estimates. Forward input
lengths can be reconstructed with the pinned tokenizer/legacy prompt, without a
model. Physical forward/token/FLOP counters are not present and remain null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
INITIAL_F_SHA = "c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466"
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
DEFAULT_TOKENIZER = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
BUDGET = {"proposal_requests": 4, "proposal_tokens_per_request": 80, "forward_queries": 26,
          "forward_tokens_per_query": 32, "total_model_queries": 30, "input_tokens_per_query": 512,
          "combined_processed_positions": 120000}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, name, *, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid nonnegative integer counter: {name}")
    return value


def proposal_ledger(outcome):
    candidates = outcome.get("candidates")
    if (not isinstance(candidates, list) or len(candidates) != 4
            or any(type(row.get("index")) is not int for row in candidates)
            or [row["index"] for row in candidates] != list(range(4))):
        raise ValueError("Fixed ordered four-candidate raw proposal trace required")
    totals = {key: 0 for key in ("model_queries", "input_tokens", "output_tokens", "processed_positions", "decoder_forwards_estimate")}
    for candidate in candidates:
        raw = candidate["raw"]
        calls = integer(raw.get("model_calls"), "proposal model_calls", maximum=1)
        n = integer(raw.get("input_tokens"), "proposal input_tokens", maximum=512)
        g = integer(raw.get("n_tokens"), "proposal n_tokens", maximum=80)
        if calls == 0:
            if n != 0 or g != 0:
                raise ValueError("Refused zero-call proposal cannot consume reported model tokens")
            positions = integer(raw.get("decoder_token_positions", 0), "refused proposal positions")
            forwards = integer(raw.get("decoder_forwards", 0), "refused proposal forwards")
            if positions or forwards:
                raise ValueError("Refused proposal counters must be zero")
        else:
            if n == 0 or type(raw.get("hit_eos")) is not bool:
                raise ValueError("Called proposal requires actual prompt tokens and EOS flag")
            extra = int(not raw["hit_eos"])
            positions = integer(raw.get("decoder_token_positions"), "proposal positions")
            forwards = integer(raw.get("decoder_forwards"), "proposal forwards")
            if positions != n + g - 1 + extra or forwards != g + extra:
                raise ValueError("Proposal raw single-query formula counters disagree")
        for key, value in (("model_queries", calls), ("input_tokens", n), ("output_tokens", g),
                           ("processed_positions", positions), ("decoder_forwards_estimate", forwards)):
            totals[key] += value
    aliases = {"model_queries": "model_queries", "input_tokens": "input_tokens", "output_tokens": "output_tokens",
               "processed_positions": "processed_positions", "decoder_forwards_estimate": "decoder_forwards"}
    if any(type(outcome.get(field)) is not int or outcome[field] != totals[key] for key, field in aliases.items()):
        raise ValueError("Proposal aggregate differs from preserved raw candidate counters")
    if (type(outcome.get("selected_index")) is not int or outcome["selected_index"] != 0
            or outcome.get("selected_plan") != candidates[0]["plan"] or outcome.get("oracle_feedback") is not False):
        raise ValueError("Pure top1 must remain fixed candidate index zero")
    totals["scope"] = "all four realized proposal requests; single-query n+g-1+cap estimates from NativePlanGenerator"
    return totals


def forward_ledger(outcome, proposals, *, token_length=None):
    rerank = outcome.get("learned_reranking")
    if rerank is None:
        return {"model_queries": 0, "input_tokens": 0, "output_tokens": 0, "processed_positions": 0,
                "decoder_forwards_estimate": 0, "reconstructed": True, "scope": "no forward reranking requested"}
    if rerank.get("task_id") != outcome["task_id"] or rerank.get("oracle_feedback") is not False:
        raise ValueError("Fixed-B0 learned rerank identity/boundary mismatch")
    traces = rerank.get("forward_traces")
    if not isinstance(traces, list):
        raise ValueError("Preserved per-task forward traces required")
    calls = integer(rerank.get("forward_queries"), "forward_queries", maximum=26)
    if calls != len(traces) or integer(rerank.get("total_model_queries"), "total_model_queries", maximum=30) != calls + proposals["model_queries"]:
        raise ValueError("Proposal-plus-forward query identity/budget differs from raw traces")
    outputs, forwards, inputs, positions, extras = 0, 0, 0, 0, 0
    for trace in traces:
        conditioning, generated = trace["conditioning"], trace["generated"]
        if (set(conditioning) != {"domain", "rules", "s0", "actions"} or conditioning["domain"] != "register"
                or not isinstance(conditioning["actions"], list) or len(conditioning["actions"]) != 1
                or conditioning["actions"][0] not in {"add1", "sub1", "double", "halve", "finish"}
                or set(conditioning["s0"]) != {"x"}
                or type(conditioning["s0"]["x"]) is not int):
            raise ValueError("State-only one-action forward trace required")
        g = integer(generated.get("n_tokens"), "forward output tokens", maximum=32)
        if type(generated.get("hit_eos")) is not bool:
            raise ValueError("Forward EOS flag required")
        extra = int(not generated["hit_eos"])
        extras += extra
        outputs += g
        forwards += g + extra
        if token_length is not None:
            n = integer(token_length(conditioning), "reconstructed forward input", maximum=512)
            if n == 0:
                raise ValueError("Called forward prompt cannot be empty")
            inputs += n
            positions += n + g - 1 + extra
    combined = integer(rerank.get("processed_positions"), "recorded combined positions", maximum=120000)
    recorded_forward = combined - proposals["processed_positions"]
    if recorded_forward < 0:
        raise ValueError("Combined position estimate cannot be below proposal work")
    if not outputs + extras <= recorded_forward <= calls * 512 + outputs - calls + extras:
        raise ValueError("Recorded forward position estimate cannot fit the preserved single-query trace caps")
    if token_length is not None and positions != recorded_forward:
        raise ValueError("Combined proposal/forward position formula disagrees with reconstructed raw traces")
    if calls == 0 and recorded_forward != 0:
        raise ValueError("Zero forward calls cannot consume forward positions")
    selected = rerank.get("selected_index")
    if selected is not None and (type(selected) is not int or not 0 <= selected < 4
                                or rerank.get("selected_plan") != outcome["candidates"][selected]["plan"]):
        raise ValueError("Learned rerank choice must stay within the fixed proposal set")
    if selected is None and rerank.get("selected_plan") is not None:
        raise ValueError("Unselected rerank cannot report a plan")
    return {"model_queries": calls, "input_tokens": inputs if token_length is not None or calls == 0 else None,
            "output_tokens": outputs, "processed_positions": recorded_forward, "decoder_forwards_estimate": forwards,
            "reconstructed": token_length is not None or calls == 0,
            "scope": "single-query LegacyPredictor forward traces; n+g-1+cap recorded estimate; input lengths unknown without pinned tokenization"}


def summary(rows, mode, success_field):
    solved = sum(row[success_field] for row in rows)
    ledgers = [row[mode] for row in rows]
    totals = {}
    for key in ("model_queries", "input_tokens", "output_tokens", "processed_positions", "decoder_forwards_estimate"):
        values = [ledger[key] for ledger in ledgers]
        totals[key] = None if any(value is None for value in values) else sum(values)
    return {"n": len(rows), "solved": solved, "totals_including_failed_tasks": totals,
            "model_queries_per_solved_including_failures": totals["model_queries"] / solved if solved else None,
            "physical_decoder_forwards": None, "physical_processed_token_positions": None, "measured_flops": None}


def reduce_record(record, *, token_length=None):
    if (record.get("status") != "development-evaluated" or record.get("oracle_feedback") is not False
            or record.get("confirmation_access") is not False
            or record.get("all_generation_and_learned_reranking_fixed_before_exact_scoring") is not True):
        raise ValueError("Completed DEV student readout with fixed generation/rerank boundary required")
    cfg = record["config"]
    if any(type(cfg.get(key)) is not int or cfg[key] != value for key, value in
           (("student_k", 4), ("student_tokens", 80), ("max_length", 512), ("generation_tokens", 32))):
        raise ValueError("Fixed K4/proposal80/dynamics32/input512 budget identity required")
    outcomes = record.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 12 or type(record.get("n")) is not int or record["n"] != 12:
        raise ValueError("Complete fixed twelve-parent student readout required")
    seen_tasks, seen_parents, rows, hybrid = set(), set(), [], None
    for outcome in outcomes:
        task, parent = outcome.get("task_id"), outcome.get("parent_world_id")
        if not isinstance(task, str) or not task or not isinstance(parent, str) or not parent or task in seen_tasks or parent in seen_parents:
            raise ValueError("Unique student task/parent identities required")
        seen_tasks.add(task); seen_parents.add(parent)
        for key in ("top1_success", "pass_at_4"):
            if type(outcome.get(key)) is not bool:
                raise ValueError("Boolean recorded student outcome required")
        proposals = proposal_ledger(outcome)
        scores = outcome.get("scores")
        if (not isinstance(scores, list) or any(type(score.get("index")) is not int for score in scores)
                or [score["index"] for score in scores] != list(range(4))
                or any(type(score.get("success")) is not bool for score in scores)
                or outcome["top1_success"] != scores[0]["success"] or outcome["pass_at_4"] != any(score["success"] for score in scores)):
            raise ValueError("Recorded top1/pass4 differ from preserved fixed-candidate scores")
        present = outcome.get("learned_reranking") is not None
        if hybrid is not None and hybrid != present:
            raise ValueError("Pure and hybrid modes cannot silently mix within one readout")
        hybrid = present
        forward = forward_ledger(outcome, proposals, token_length=token_length)
        combined = {key: None if proposals[key] is None or forward[key] is None else proposals[key] + forward[key]
                    for key in ("model_queries", "input_tokens", "output_tokens", "processed_positions", "decoder_forwards_estimate")}
        if combined["model_queries"] > 30 or combined["processed_positions"] > 120000:
            raise ValueError("Combined H3 query/position budget exceeded")
        selected_success = None
        if present:
            rerank = outcome["learned_reranking"]
            index = rerank["selected_index"]
            selected_success = False if index is None else scores[index]["success"]
            if type(rerank.get("success")) is not bool or rerank["success"] != selected_success:
                raise ValueError("Hybrid success must score only the fixed learned-selected proposal")
        rows.append({"task_id": task, "parent_world_id": parent, "top1_success": outcome["top1_success"],
                     "pass_at_4": outcome["pass_at_4"], "learned_rerank_success": selected_success,
                     "proposal": proposals, "forward": forward, "combined": combined,
                     "combined_counter_scope": {
                         "processed_positions": "recorded proposal single-query estimates plus recorded rerank n+g-1+cap ledger; optional tokenizer reconstruction validates the forward component",
                         "decoder_forwards_estimate": "proposal raw NativePlanGenerator formula counts plus forward-trace g+cap estimates; no physical forward hook measurement",
                         "input_tokens": "proposal raw prompt lengths plus optionally reconstructed forward lengths; null if reconstruction is unavailable"},
                     "combined_positions_reconstructed": forward["reconstructed"],
                     "physical_decoder_forwards": None, "physical_processed_token_positions": None, "measured_flops": None})
    proposal_totals = summary(rows, "proposal", "top1_success")["totals_including_failed_tasks"]
    aliases = {"proposal_queries": "model_queries", "proposal_input_tokens": "input_tokens",
               "proposal_output_tokens": "output_tokens", "proposal_processed_positions": "processed_positions"}
    if any(type(record.get(key)) is not int or record[key] != proposal_totals[field] for key, field in aliases.items()):
        raise ValueError("Readout proposal totals differ from preserved raw traces")
    if hybrid:
        if (record.get("forward_adapter_sha256") != INITIAL_F_SHA
                or record.get("adapter_switch", {}).get("student_adapter_restored") is not True
                or type(record.get("reranking_forward_queries")) is not int
                or record["reranking_forward_queries"] != sum(row["forward"]["model_queries"] for row in rows)):
            raise ValueError("Fixed-B0 forward identity/restoration/query totals required")
    elif record.get("reranking_forward_queries") is not None or record.get("with_search_successes") is not None:
        raise ValueError("Pure mode cannot claim reranking cost/success")
    if (type(record.get("top1_successes")) is not int or type(record.get("pass_at_4_successes")) is not int
            or record["top1_successes"] != sum(row["top1_success"] for row in rows)
            or record["pass_at_4_successes"] != sum(row["pass_at_4"] for row in rows)):
        raise ValueError("Recorded aggregate student successes differ")
    hybrid_summary = summary(rows, "combined", "learned_rerank_success") if hybrid else None
    if hybrid and (type(record.get("with_search_successes")) is not int or record["with_search_successes"] != hybrid_summary["solved"]):
        raise ValueError("Recorded learned rerank success total differs")
    return {"schema": "dsv3wm-h3-student-realized-costs-v1", "status": "offline reducer; no new native outcome",
            "per_parent": rows, "budget": BUDGET, "pure_top1_readout": summary(rows, "proposal", "top1_success"),
            "offline_pass4_coverage": summary(rows, "proposal", "pass_at_4"), "learned_reranked_readout": hybrid_summary,
            "scope": ["All four generated proposals are charged as realized readout work, including failed tasks.",
                      "pass@4 is offline coverage and never an exact-verifier deployment selector.",
                      "Hybrid costs add fixed-B0 learned queries; exact scoring supplies no inference feedback.",
                      "Proposal/forward decoder counts are formula estimates with explicit single-query scopes; physical counters remain null.",
                      "Processed positions are cached-decoder formula ledgers, not measured FLOPs or GPU memory.",
                      "No student outcome exists yet unless a genuine completed input artifact is supplied."]}


def make_token_length(tokenizer_json, record):
    """Tokenizer-only reconstruction under matching frozen prompt source bytes."""
    if file_hash(tokenizer_json) != TOKENIZER_SHA:
        raise ValueError("Pinned parent tokenizer JSON required")
    sources = {(REPO / name).resolve(): value for name, value in record["source_sha256"].items()}
    for path in (ROOT / "src/dsv3wm/recovery.py", ROOT / "src/dsv3wm/domains.py", REPO / "bwm/code/bwmn/rw2.py"):
        if sources.get(path.resolve()) != file_hash(path):
            raise ValueError("Frozen forward prompt source drift prevents token reconstruction")
    from tokenizers import Tokenizer
    from dsv3wm.recovery import legacy_prompt
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    return lambda inputs: len(tokenizer.encode(legacy_prompt(inputs), add_special_tokens=False).ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, help="optional pinned tokenizer-only forward input reconstruction")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Student cost reduction output exists")
    record = json.loads(args.input.read_text(encoding="utf-8"))
    counter = make_token_length(args.tokenizer_json, record) if args.tokenizer_json else None
    output = reduce_record(record, token_length=counter)
    output.update(input_path=str(args.input.resolve()), input_sha256=file_hash(args.input),
                  source_sha256={str(Path(__file__).resolve()): file_hash(__file__)},
                  tokenizer_json_sha256=file_hash(args.tokenizer_json) if args.tokenizer_json else None,
                  inherited_execution_provenance={key: record.get(key) for key in ("freeze_sha256", "tasks_sha256", "config_sha256", "run_sha256",
                                                                                 "adapter_sha256", "adapter_config_sha256", "source_sha256", "forward_adapter_sha256")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write("\n")
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
