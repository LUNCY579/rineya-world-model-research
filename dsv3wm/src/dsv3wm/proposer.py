"""Direct plan student: fixed candidate order, no oracle selection or repair.

Generation and exact scoring are separate APIs. pass@k is an offline coverage
metric, not deployable verifier-guided selection. Native A3/A4 efficacy remains
unmeasured until real trained adapters and permitted runs exist.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from .planning import planning_prompt, validate_problem


def validate_task(task):
    if (not isinstance(task, dict) or set(task) != {"domain", "rules", "s0", "goal", "parent_world_id", "split"}
            or task["split"] != "dev" or not isinstance(task["parent_world_id"], str) or not task["parent_world_id"]):
        raise ValueError("explicit DEV-only initial-state/goal task required; labels/plans forbidden")
    inputs = {key: copy.deepcopy(task[key]) for key in ("domain", "rules", "s0")}
    inputs["actions"] = []
    validate_problem(inputs, task["goal"])
    from .domains import parent_world_key
    if task['parent_world_id']!=parent_world_key(task['domain'],task['s0'],task['rules']):
        raise ValueError('canonical parent identity required')
    return inputs


def _action_syntax(domain, action, rules):
    if domain == "register":
        return isinstance(action, str) and action in {"add1", "sub1", "double", "halve", "finish"}
    if domain == "blocks":
        parts = action.split() if isinstance(action, str) else []
        return (bool(parts) and len(parts) == {"pick-up": 2, "put-down": 2, "stack": 3, "unstack": 3}.get(parts[0], -1)
                and all(obj in rules["objects"] for obj in parts[1:]))
    if not isinstance(action, dict):
        return False
    return (set(action) <= {"kind", "account", "other", "amount", "status"}
            and {"kind", "account", "amount", "status"} <= set(action)
            and action["kind"] in {"add", "scale", "transfer"}
            and action["account"] in rules["objects"] and type(action["amount"]) is int
            and isinstance(action["status"], str)
            and (action["kind"] != "transfer" or action.get("other") in rules["objects"])
            and ("other" not in action or action["other"] in rules["objects"]))


def parse_plan(text, domain, rules, max_actions=8):
    """JSON/action syntax only. Syntactically valid illegal plans remain plans."""
    if type(max_actions) is not int or max_actions <= 0:
        raise ValueError("positive action limit required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("nonfinite constant")
    try:
        if not isinstance(text, str) or domain not in {"register", "ledger", "blocks"}:
            raise ValueError("text/domain required")
        plan = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
        if not isinstance(plan, list) or len(plan) > max_actions or not all(_action_syntax(domain, action, rules) for action in plan):
            raise ValueError("invalid plan array/action syntax")
        return {"malformed": False, "plan": plan}
    except (ValueError, TypeError, KeyError):
        return {"malformed": True, "plan": None, "text": text}


def candidate_seed(base_seed, parent, index):
    if type(base_seed) is not int or base_seed < 0 or type(index) is not int or index < 0:
        raise ValueError("nonnegative seeds/indices required")
    payload = json.dumps([base_seed, parent, index], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def generate_plans(task, generator, *, k=4, base_seed=0, max_new_tokens=128, max_actions=8):
    inputs = validate_task(task)
    if type(k) is not int or not 1 <= k <= 64 or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("bounded k in 1..64 and positive token cap required")
    prompt = planning_prompt(inputs, task["goal"])
    candidates = []
    for index in range(k):
        seed = candidate_seed(base_seed, task["parent_world_id"], index)
        raw = generator(prompt, seed, max_new_tokens)
        parsed = parse_plan(raw.get("text"), task["domain"], task["rules"], max_actions)
        candidates.append({"index": index, "seed": seed, "raw": raw, **parsed})
    plans = [json.dumps(c["plan"], sort_keys=True, separators=(",", ":")) for c in candidates if not c["malformed"]]
    return {"parent_world_id": task["parent_world_id"], "prompt": prompt, "candidates": candidates,
            "selected_index": 0, "selected_plan": candidates[0]["plan"], "k": k,
            "generator_requests": k, "model_queries": sum(c["raw"].get("model_calls", 1) for c in candidates),
            "input_tokens": sum(c["raw"].get("input_tokens", 0) for c in candidates),
            "output_tokens": sum(c["raw"].get("n_tokens", 0) for c in candidates),
            "decoder_forwards": sum(c["raw"].get("decoder_forwards", 0) for c in candidates),
            "decoder_token_positions": sum(c["raw"].get("decoder_token_positions", 0) for c in candidates),
            "unique_valid_plans": len(set(plans)), "valid_syntax_count": len(plans),
            "diversity_fraction": len(set(plans)) / k,
            "oracle_feedback": False, "instrument": "pure direct proposer; fixed index-0 deployment choice"}


def score_plans(task, generated):
    """Call only once all generation has finished. Never change selected_index."""
    inputs = validate_task(task)
    if (generated.get("oracle_feedback") is not False or generated.get("selected_index") != 0
            or generated["parent_world_id"] != task["parent_world_id"]
            or generated["prompt"] != planning_prompt(inputs, task["goal"])):
        raise ValueError("unaltered direct-proposer record required")
    from .planning import verify_plan
    scored = []
    for candidate in generated["candidates"]:
        verification = (verify_plan(inputs, task["goal"], candidate["plan"]) if not candidate["malformed"]
                        else {"success": False, "all_legal": False, "malformed": True})
        scored.append({"index": candidate["index"], **verification})
    return {**generated, "top1_success": scored[0]["success"], "pass_at_k": any(row["success"] for row in scored),
            "valid_plan_count": sum(row["all_legal"] for row in scored), "scores": scored,
            "pass_at_k_scope": "offline coverage; no verifier reranking/fallback", "selected_index": 0}


class NativePlanGenerator:
    """Existing BWM seeded sampler, FP32 native model loaded by runtime/load."""
    def __init__(self, model, tokenizer, config, *, temperature=1., top_p=.95):
        if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("valid temperature/top_p required")
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.temperature, self.top_p = temperature, top_p

    def __call__(self, prompt, seed, max_new_tokens):
        from .native import boundary
        from bwmn import lm
        boundary(self.config)
        n = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
        if n > self.config.max_length:
            return {"text": None, "error": "overlength", "model_calls": 0, "n_tokens": 0, "input_tokens": 0}
        self.model.eval()
        if self.config.device == "cpu":
            lm.DUTY_RATIO = 0.
        result = lm.sample_crn(self.model, self.tokenizer, [prompt], [seed],
                               max_new_tokens=max_new_tokens, temperature=self.temperature, top_p=self.top_p)[0]
        g, extra = int(result["n_tokens"]), int(not result["hit_eos"])
        return {**result, "model_calls": 1, "input_tokens": n,
                "decoder_forwards": g + extra, "decoder_token_positions": n + g - 1 + extra}
