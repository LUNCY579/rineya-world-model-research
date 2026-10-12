"""Construct a tiny original-delta Blocksworld G1; no banks, plans or models."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from dsv3wm import REPO
from dsv3wm.domains import _semantics
from dsv3wm.native import EXPECTED_CORE_SHA, EXPECTED_SIDECARS, sha256

BW_ADAPTER = REPO.parent / "Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_Fdelta_s0/adapter"
BW_SHA = "2ed0bb0064bc6b11ec9c661840a38094e468462cea2ff833cda3bf234ead65ae"
BW_CONFIG_SHA = "3038c77961f33461e1bef736959765434c0be473d991b0ef92c4813d269b3b8c"
CORE = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
VERSION = "dsv3wm-block-delta-g1-v1"


@dataclass(frozen=True)
class BlockG1Config:
    core: str = "titan-sparse"
    device: str = "cpu"
    threads: int = 4
    seed: int = 0
    max_length: int = 256
    generation_tokens: int = 48
    def __post_init__(self):
        if self.core != "titan-sparse" or self.device != "cpu" or type(self.threads) is not int or self.threads != 4:
            raise ValueError("This G1 binds frozen Titan-Sparse on the coordinated four-thread CPU lane")
        if type(self.seed) is not int or self.seed != 0:
            raise ValueError("Fixed greedy qualification seed0 required")
        for key in ("max_length", "generation_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("Positive typed token caps required")
        if self.max_length > 512 or self.generation_tokens > 128:
            raise ValueError("Bounded G1 token budget required")
    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
    def dict(self):
        return asdict(self)


def B():
    return _semantics("blocks")


def _state(towers, holding=None):
    source = B()
    facts = set(source.towers_to_state(towers))
    if holding is not None:
        facts.discard(("handempty",)); facts.add(("holding", holding))
    return frozenset(facts)


def structural_valid(facts, objects):
    """State invariants only, never action applicability or exact transition."""
    objects = set(objects)
    arity = {"clear": 2, "ontable": 2, "holding": 2, "handempty": 1, "on": 3}
    if any(not fact or len(fact) != arity.get(fact[0], -1) or any(x not in objects for x in fact[1:]) for fact in facts):
        return False
    held = {fact[1] for fact in facts if fact[0] == "holding"}
    if len(held) > 1 or bool(held) == (("handempty",) in facts):
        return False
    locations = {obj: [] for obj in objects}
    supports, below = set(), {}
    for fact in facts:
        if fact[0] in {"ontable", "holding", "on"}:
            locations[fact[1]].append(fact)
        if fact[0] == "on":
            top, base = fact[1:]
            if top == base or base in supports or top in below:
                return False
            supports.add(base); below[top] = base
    if any(len(value) != 1 for value in locations.values()):
        return False
    clear = {fact[1] for fact in facts if fact[0] == "clear"}
    if clear != objects - supports - held or held & supports:
        return False
    for obj in objects:
        current, visited = obj, set()
        while current in below:
            if current in visited:
                return False
            visited.add(current); current = below[current]
    return True


def canonical_parent(facts, objects):
    """New bounded <=6 permutation canonicalizer, without changing old <=5 code."""
    objects = sorted(objects)
    if not 1 <= len(objects) <= 6 or not structural_valid(facts, objects):
        raise ValueError("Valid one-to-six-object construction required")
    best = None
    for indices in itertools.permutations(range(len(objects))):
        mapping = {obj: f"o{indices[i]}" for i, obj in enumerate(objects)}
        renamed = sorted([fact[0], *[mapping[x] for x in fact[1:]]] for fact in facts)
        value = json.dumps({"domain": "blocks", "n": len(objects), "facts": renamed}, sort_keys=True, separators=(",", ":"))
        best = value if best is None or value < best else best
    return hashlib.sha256(best.encode()).hexdigest()


def _case(identifier, towers, action, legal, holding=None):
    source = B()
    facts = _state(towers, holding)
    objects = sorted({x for tower in towers for x in tower} | ({holding} if holding else set()))
    if not structural_valid(facts, objects):
        raise ValueError("Invalid constructed initial state")
    after, actual_legal = source.step(facts, tuple(action))
    if actual_legal is not legal or (legal and after == facts) or (not legal and after != facts):
        raise ValueError("Construction failed the prospectively chosen legal-changed/no-op stratum")
    return {"id": identifier, "role": "primary", "diagnostic": None, "objects": objects,
            "before": [list(f) for f in sorted(facts)], "action": list(action),
            "parent_world_id": canonical_parent(facts, objects),
            "prompt": source.forward_prompt_delta(facts, tuple(action)),
            "target": source.forward_target_delta(facts, after, actual_legal),
            "collection_legal": actual_legal,
            "exposure": "constructed original-interface validation; no teacher-unseen-state or historical nonoverlap claim"}


def construct_cases():
    cases = [
        _case("legal_pickup_3", [["a"], ["b"], ["c"]], ("pick-up", "a"), True),
        _case("legal_unstack_3", [["a", "b"], ["c"]], ("unstack", "b", "a"), True),
        _case("legal_putdown_4", [["b"], ["c"], ["d"]], ("put-down", "a"), True, "a"),
        _case("legal_stack_4", [["b", "c"], ["d"]], ("stack", "a", "c"), True, "a"),
        _case("legal_unstack_5", [["a", "b", "c"], ["d", "e"]], ("unstack", "c", "b"), True),
        _case("legal_stack_6", [["b", "c", "d"], ["e", "f"]], ("stack", "a", "d"), True, "a"),
        _case("illegal_pickup_3", [["a", "b", "c"]], ("pick-up", "a"), False),
        _case("illegal_unstack_4", [["a", "b", "c", "d"]], ("unstack", "a", "b"), False),
        _case("illegal_putdown_4", [["a", "b"], ["c", "d"]], ("put-down", "a"), False),
        _case("illegal_stack_5", [["a"], ["b"], ["c"], ["d"], ["e"]], ("stack", "a", "b"), False),
        _case("illegal_pickup_5", [["b", "c", "d", "e"]], ("pick-up", "b"), False, "a"),
        _case("illegal_unstack_6", [["a", "b"], ["c", "d"], ["e", "f"]], ("unstack", "b", "c"), False),
    ]
    if len({case["parent_world_id"] for case in cases}) != 12:
        raise ValueError("Primary constructions must have twelve distinct canonical initial parents")
    diagnostics = []
    for index, names in ((0, ["u", "v", "w"]), (4, ["reed", "stone", "ash", "leaf", "fern"])):
        base = cases[index]; mapping = dict(zip(base["objects"], names))
        facts = frozenset((f[0], *[mapping[x] for x in f[1:]]) for f in base["before"])
        action = (base["action"][0], *[mapping[x] for x in base["action"][1:]])
        after, legal = B().step(facts, action)
        diagnostics.append({**base, "id": "names_" + base["id"], "role": "diagnostic", "diagnostic": "renamed_objects",
            "source_case_id": base["id"], "objects": names, "before": [list(f) for f in sorted(facts)], "action": list(action),
            "prompt": B().forward_prompt_delta(facts, action), "target": B().forward_target_delta(facts, after, legal)})
    for index in (4, 8):
        base = cases[index]; facts = frozenset(tuple(f) for f in base["before"])
        state = B().state_json(facts)
        for key in ("clear", "table", "on"):
            state[key] = list(reversed(state[key]))
        prompt = B().render_chat(B().SYS_FORWARD_DELTA, B().canonical({"action": B().action_str(tuple(base["action"])), "state": state}))
        diagnostics.append({**base, "id": "atoms_" + base["id"], "role": "diagnostic", "diagnostic": "reversed_atom_lists",
                            "source_case_id": base["id"], "prompt": prompt})
    return [*cases, *diagnostics]


def diagnose_delta(text, before, objects):
    """Retain raw predictions and naive set state before support/invariant checks.

    This function never calls step/applicable. Raw set arithmetic is a diagnostic,
    not a repaired/accepted model state when a deletion was unsupported.
    """
    source = B()
    before = frozenset(tuple(fact) for fact in before)
    result = {"raw_text": text, "format_valid": False, "raw_legal": None, "raw_delta": None,
              "raw_state_facts": None, "unsupported_deletions": [], "support_ok": False,
              "structural_state_valid": False, "type_or_format_failure": None}
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate-json-key")
            value[key] = item
        return value
    def reject_constant(value):
        raise ValueError("nonfinite-json-constant")
    try:
        raw = json.loads(text.strip(), object_pairs_hook=unique, parse_constant=reject_constant)
        result["raw_json"] = raw
        if isinstance(raw, dict):
            result["raw_delta"] = {key: raw.get(key) for key in ("add", "del")}
            result["raw_legal"] = raw.get("legal") if type(raw.get("legal")) is bool else None
        if not isinstance(raw, dict) or set(raw) != {"add", "del", "legal"} or type(raw["legal"]) is not bool:
            raise ValueError("delta-schema-or-legal-type")
        if any(not isinstance(raw[key], list) or any(not isinstance(value, str) for value in raw[key]) for key in ("add", "del")):
            raise ValueError("delta-facts-must-be-string-lists")
        add, delete = ({tuple(value.split()) for value in raw[key]} for key in ("add", "del"))
        arities = {"clear": 2, "ontable": 2, "holding": 2, "handempty": 1, "on": 3}
        if any(not fact or len(fact) != arities.get(fact[0], -1) or any(x not in objects for x in fact[1:]) for fact in add | delete):
            raise ValueError("unknown-or-malformed-fact")
        result.update(format_valid=True, raw_add_facts=[list(f) for f in sorted(add)], raw_del_facts=[list(f) for f in sorted(delete)],
                      add_delete_overlap=[list(f) for f in sorted(add & delete)])
        naive = (before - delete) | add
        missing = delete - before
        result.update(raw_state_facts=[list(f) for f in sorted(naive)],
                      unsupported_deletions=[list(f) for f in sorted(missing)], support_ok=not missing,
                      structural_state_valid=structural_valid(naive, objects))
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        result["type_or_format_failure"] = str(error)
    # Preserve the exact registered parser's answer separately, without silently
    # replacing the recorded raw prediction with its filtered interpretation.
    parsed = source.parse_forward_delta(text, before)
    result["source_parser_accepted"] = parsed is not None
    result["source_parser_state_facts"] = None if parsed is None else [list(f) for f in sorted(parsed[0])]
    result["source_parser_legal"] = None if parsed is None else parsed[1]
    result["accepted"] = bool(parsed is not None and result["format_valid"] and result["support_ok"] and result["structural_state_valid"])
    return result


def score_case(case, text):
    """Offline only: invoke after every generated text is fixed."""
    before = frozenset(tuple(f) for f in case["before"])
    expected, legal = B().step(before, tuple(case["action"]))
    result = diagnose_delta(text, case["before"], case["objects"])
    expected_add, expected_del = expected - before, before - expected
    raw_state = None if result["raw_state_facts"] is None else frozenset(tuple(f) for f in result["raw_state_facts"])
    raw_add = frozenset(tuple(f) for f in result.get("raw_add_facts", []))
    raw_del = frozenset(tuple(f) for f in result.get("raw_del_facts", []))
    result.update(id=case["id"], role=case["role"], diagnostic=case["diagnostic"], parent_world_id=case["parent_world_id"],
                  expected_legal=legal, raw_legal_correct=result["raw_legal"] is not None and result["raw_legal"] == legal,
                  raw_candidate_state_exact=raw_state == expected if raw_state is not None else False,
                  raw_delta_exact=bool(result["format_valid"] and raw_add == expected_add and raw_del == expected_del),
                  accepted_state_exact=bool(result["accepted"] and frozenset(tuple(f) for f in result["source_parser_state_facts"]) == expected),
                  accepted_tuple_exact=bool(result["accepted"] and result["source_parser_legal"] == legal
                                           and frozenset(tuple(f) for f in result["source_parser_state_facts"]) == expected),
                  oracle_feedback=False,
                  raw_state_scope="naive decoded set arithmetic before deletion/invariant filters; not an accepted/repair guarantee")
    return result


def summarize(outcomes):
    primary = [row for row in outcomes if row["role"] == "primary"]
    if len(primary) != 12 or sum(row["expected_legal"] for row in primary) != 6:
        raise ValueError("Exactly twelve fixed six-legal/six-illegal primary cases required")
    groups = {"primary": primary, "legal_changed": [row for row in primary if row["expected_legal"]],
              "illegal_noop": [row for row in primary if not row["expected_legal"]],
              "diagnostic": [row for row in outcomes if row["role"] == "diagnostic"]}
    result = {}
    for name, rows in groups.items():
        result[name] = {"n": len(rows), **{field: sum(bool(row[field]) for row in rows) for field in
                       ("format_valid", "raw_legal_correct", "raw_candidate_state_exact", "raw_delta_exact",
                        "support_ok", "structural_state_valid", "accepted", "accepted_state_exact", "accepted_tuple_exact")},
                        "unsupported_deletion_count": sum(bool(row["unsupported_deletions"]) for row in rows),
                        "type_or_format_failure_count": sum(row["type_or_format_failure"] is not None for row in rows)}
    gate = result["primary"]["format_valid"] >= 11 and result["primary"]["accepted_tuple_exact"] >= 10
    gate = gate and result["legal_changed"]["accepted_tuple_exact"] >= 5
    result["g1_qualified"] = gate
    result["scope"] = "constructed inherited-interface G1 only; no unseen-state, raw-rule-learning or efficacy claim"
    return result


def required_sources():
    return [Path(__file__), PACKAGE / "scripts/evaluate_block_g1.py", PACKAGE / "src/dsv3wm/native.py",
            PACKAGE / "reports/BLOCKSWORLD_G1_DRAFT_V1.md",
            PACKAGE / "src/dsv3wm/provenance.py", PACKAGE / "src/dsv3wm/domains.py",
            REPO / "bwm/code/bwmn/blocks.py", REPO / "bwm/code/bwmn/rw2.py", REPO / "bwm/code/bwmn/lm.py",
            REPO / "bwm/code/bwmn/gpu_lease.py", REPO / "bwm/code/scripts/planbench_qwen.py"]


def build(output, tokenizer=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Block G1 destination already exists")
    if sha256(BW_ADAPTER / "adapter_model.safetensors") != BW_SHA or sha256(BW_ADAPTER / "adapter_config.json") != BW_CONFIG_SHA:
        raise ValueError("Private registered Blocksworld initializer changed")
    if sha256(CORE / "tokenizer.json") != EXPECTED_SIDECARS["tokenizer.json"]:
        raise ValueError("Tokenizer identity changed")
    if tokenizer is None:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(CORE / "tokenizer.json"))
    cases = construct_cases()
    counts = [{"id": c["id"], "prompt_tokens": len(tokenizer.encode(c["prompt"], add_special_tokens=False).ids),
               "target_tokens_including_eos": len(tokenizer.encode(c["target"], add_special_tokens=False).ids) + 1} for c in cases]
    maximum = max(c["target_tokens_including_eos"] for c in counts)
    cap = ((maximum + 16 + 7) // 8) * 8
    length = ((max(c["prompt_tokens"] for c in counts) + cap + 63) // 64) * 64
    cfg = BlockG1Config(max_length=length, generation_tokens=cap)
    content = ("".join(json.dumps(c, sort_keys=True, separators=(",", ":")) + "\n" for c in cases)).encode()
    manifest = {"generator": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "primary_cases": 12, "legal_changed": 6, "illegal_noop": 6, "primary_independent_parents": 12,
                "diagnostic_cases": 4, "diagnostic_independence": "paired variants reuse primary parents; never pooled as new independent cases",
                "initializer": {"path": str(BW_ADAPTER), "weight_sha256": BW_SHA, "config_sha256": BW_CONFIG_SHA,
                                "parent_weight_sha256": EXPECTED_CORE_SHA},
                "tokenizer_sha256": EXPECTED_SIDECARS["tokenizer.json"], "token_counts": counts,
                "budget_rule": "output cap = ceil8(max gold target+EOS +16); context cap = ceil64(max prompt+output cap), <=512",
                "config": cfg.dict(), "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in required_sources()},
                "cases_sha256": hashlib.sha256(content).hexdigest(), "confirmation_access": False,
                "gate": {"primary_format_at_least": 11, "primary_accepted_tuple_at_least": 10,
                         "legal_changed_accepted_tuple_at_least": 5},
                "limitations": ["Constructed validation; inherited training exposure and historical evaluation overlap unknown.",
                    "Raw legality and delta support are separately reported; accepted accuracy includes symbolic filters.",
                    "No historical generator/signature bank/solver/reference plan was invoked.",
                    "New <=6 canonicalization exists only in this collection; frozen <=5 domains.py is unchanged."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases.jsonl").write_bytes(content)
    (output / "config.json").write_text(json.dumps(cfg.dict(), indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({"output": str(args.output), "primary_cases": 12, "diagnostic_cases": 4,
                      "config": manifest["config"], "native_models_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
