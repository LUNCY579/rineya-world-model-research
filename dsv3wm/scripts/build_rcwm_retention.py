"""Freeze fresh original-format T2 register/prerequisite retention cases.

Only SHA-bound original TRAIN input fields may be audited. No prior targets,
historical evaluation rows, marginals, solver plans or sealed banks are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsv3wm import REPO
from dsv3wm.domains import _semantics, canonical, sha256

TRAIN_SHA = "bf5a22765757856517496b64e54d4d7b307dbb50af8f4e30540ff69937a1d744"
DEFAULT_TRAIN = REPO.parent / "Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/data/t2_train.jsonl"
DEFAULT_TOKENIZER = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
FAMILIES = ("register_machine", "prerequisite_plan")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_key(family, state):
    return sha256({"family": family, "state": state, "modulus": 97 if family == "register_machine" else 256})


def input_key(family, state, event):
    """Semantic one-step key: names/costs/ticks and unused event fields are nuisance."""
    if family == "prerequisite_plan":
        law = {key: event[key] for key in ("pre_on", "pre_off", "delete", "add")}
    else:
        kind = event["kind"]
        fields = {"swap": ("register", "other"), "add": ("register", "value"),
                  "mix": ("register", "other", "value"),
                  "conditional": ("register", "other", "value", "threshold")}[kind]
        law = {"kind": kind, **{key: event[key] for key in fields}}
        if "value" in law:
            law["value"] %= 97
    return sha256({"family": family, "before": state, "event": law})


def training_input_audit(path=None, *, expected_sha=TRAIN_SHA):
    empty = {"available": False, "input_keys": {family: set() for family in FAMILIES},
             "parent_keys": {family: set() for family in FAMILIES}, "source": None}
    if path is None:
        empty["limitation"] = "Original TRAIN input audit unavailable; prior exposure is unknown."
        return empty
    path = Path(path).resolve()
    if path.name != "t2_train.jsonl":
        raise PermissionError("Only the registered t2_train.jsonl input surface may be audited")
    if not path.exists():
        empty["limitation"] = "Original TRAIN input audit unavailable; prior exposure is unknown."
        return empty
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError("Original TRAIN SHA mismatch; no input keys are consumed")
    counts = {family: 0 for family in FAMILIES}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        family = record["family"]
        if family not in FAMILIES:
            continue
        # Deliberately never reference record['target'], labels or old outputs.
        state, event = record["before"], record["event"]
        empty["input_keys"][family].add(input_key(family, state, event))
        empty["parent_keys"][family].add(parent_key(family, state))
        counts[family] += 1
    empty.update(available=True, source={"path": str(path), "sha256": actual, "bytes": len(raw),
                                        "input_rows_by_family": counts,
                                        "semantic_input_keys": {f: len(empty["input_keys"][f]) for f in FAMILIES},
                                        "initial_parent_keys": {f: len(empty["parent_keys"][f]) for f in FAMILIES},
                                        "labels_used": False})
    return empty


def _register_input(index, rng):
    state = {name: rng.randrange(97) for name in ("X", "Y", "Z")}
    if index == 2:
        state["X"] = rng.randrange(80, 97)
    if index == 3:
        state["Z"] = rng.randrange(9)
    if index == 4:
        state["X"] = rng.randrange(48, 97)
    if index == 6:
        state["Y"] = rng.randrange(1, 97)
    if index == 7:
        state["X"] = rng.randrange(96)
    specifications = [
        ("swap", "X", "Y", 0, 0, "swap_xy"), ("swap", "Y", "Z", 0, 0, "swap_yz"),
        ("add", "X", "Y", 17, 0, "positive_modular_wrap"),
        ("add", "Z", "X", -9, 0, "negative_modular_wrap"),
        ("mix", "Y", "X", 3, 0, "mix_modular_wrap"),
        ("mix", "Z", "Y", -11, 0, "mix_negative_value"),
        ("conditional", "X", "Y", 7, state["Y"] - 1, "conditional_true"),
        ("conditional", "Z", "X", 5, state["X"] + 1, "conditional_false"),
        ("conditional", "Y", "Z", -4, state["Z"], "conditional_equality_false"),
        ("conditional", "X", "Y", 0, state["Y"], "conditional_zero_value_noop"),
        ("add", "Y", "X", 0, 0, "add_zero_noop"),
        ("swap", "X", "X", 0, 0, "self_swap_noop"),
    ]
    kind, register, other, value, threshold, cohort = specifications[index]
    return state, {"kind": kind, "register": register, "other": other, "value": value, "threshold": threshold}, cohort


def _prerequisite_input(index, state):
    on = [1 << bit for bit in range(8) if state & (1 << bit)]
    off = [1 << bit for bit in range(8) if not state & (1 << bit)]
    if not on or not off or (index == 7 and len(on) < 2) or (index == 8 and len(off) < 2):
        return None
    first_on, first_off = on[0], off[0]
    variants = [
        (first_on, first_on, "delete_then_add_overlap_noop"),
        (0, 0, "empty_effect_noop"), (0, first_off, "add_absent_bit"),
        (first_on, 0, "delete_present_bit"),
        (0, sum(off[:2]), "add_multiple_bits"), (sum(on[:2]), 0, "delete_multiple_bits"),
        (sum(on[:2]), first_off | first_on, "mixed_overlap_delete_then_add"),
        (first_on, first_off, "pre_on_conjunction"), (first_on, first_off, "pre_off_conjunction"),
        (0, first_on, "add_existing_bit_noop"), (first_off, 0, "delete_absent_bit_noop"),
        (first_on, first_off, "mixed_delete_and_add"),
    ]
    delete, add, cohort = variants[index]
    event = {"kind": "action", "name": f"A{index + 1}", "cost": 1 + index % 5,
             "pre_on": sum(on[:2]) if index == 7 else first_on,
             "pre_off": sum(off[:2]) if index == 8 else first_off, "delete": delete, "add": add}
    return state, event, cohort


def build_fixtures(seed=202610125, *, audit=None, tokenizer=None):
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative deterministic fixture seed required")
    audit = audit if audit is not None else training_input_audit()
    env, rng, fixtures, used = _semantics("ledger"), random.Random(seed), [], set()
    def retain(family, index, candidate):
        state, event, cohort = candidate
        parent, key = parent_key(family, state), input_key(family, state, event)
        if parent in used or key in audit["input_keys"][family]:
            return False
        messages = env.one_step_messages(family, state, event)
        prompt = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)
        prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        label = env.step(family, state, event)
        if family == "prerequisite_plan" and not env.applicable(state, event):
            raise AssertionError("Original prerequisite frame presumes applicable actions")
        row = {"id": f"{family}-{index:02d}", "family": family, "cohort": cohort,
               "input": {"before": state, "event": event}, "messages": messages, "prompt": prompt,
               "target": env.dumps(label), "parent_world_id": parent, "semantic_input_id": key,
               "graph_group_id": sha256({"family": family, "system_rule": env.SYSTEM_ONE[family],
                                          "modulus": 97 if family == "register_machine" else 256}),
               "changed": label != state, "applicable": True,
               "original_train_input_exposure": "audited_semantically_novel" if audit["available"] else "unknown",
               "original_train_parent_exposure": ("audited_seen" if parent in audit["parent_keys"][family] else "audited_novel")
                                                  if audit["available"] else "unknown"}
        if tokenizer is not None:
            row["prompt_tokens"] = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            row["target_tokens_including_eos"] = len(tokenizer.encode(row["target"], add_special_tokens=False).ids) + 1
            if row["prompt_tokens"] > 512 or row["target_tokens_including_eos"] > 64:
                raise ValueError("Fixture exceeds the prospective input/output caps; no truncation")
        fixtures.append(row)
        used.add(parent)
        return True
    for index in range(12):
        for attempt in range(10000):
            candidate = _register_input(index, rng)
            parent = parent_key("register_machine", candidate[0])
            if attempt < 5000 and parent in audit["parent_keys"]["register_machine"]:
                continue
            if retain("register_machine", index, candidate):
                break
        else:
            raise RuntimeError("Cannot construct a fresh register input under the frozen novelty rule")
    for index in range(12):
        states = list(range(256))
        rng.shuffle(states)
        states.sort(key=lambda state: parent_key("prerequisite_plan", state) in audit["parent_keys"]["prerequisite_plan"])
        for state in states:
            candidate = _prerequisite_input(index, state)
            if candidate is not None and retain("prerequisite_plan", index, candidate):
                break
        else:
            raise RuntimeError("Cannot construct twelve unique prerequisite parents without input overlap")
    sources = [Path(__file__), REPO / "dsv3wm/scripts/evaluate_rcwm_retention.py", REPO / "dsv3wm/src/dsv3wm/domains.py",
               REPO / "rcwm/experiments/RCWM-T1/rcwmt/env.py"]
    return {"schema": "dsv3wm-rcwm-retention-v1", "seed": seed, "fixtures": fixtures,
            "family_counts": {family: sum(row["family"] == family for row in fixtures) for family in FAMILIES},
            "parent_counts": {family: len({row["parent_world_id"] for row in fixtures if row["family"] == family}) for family in FAMILIES},
            "graph_group_counts": {family: len({row["graph_group_id"] for row in fixtures if row["family"] == family}) for family in FAMILIES},
            "train_input_audit": audit["source"] if audit["available"] else {"available": False, "limitation": audit["limitation"]},
            "source_sha256": {str(path): file_hash(path) for path in sources},
            "inference_contract": {"greedy": True, "max_input_tokens": 512, "max_new_tokens": 64,
                                   "output": "bare complete register JSON object or prerequisite JSON integer",
                                   "cap_rationale": "64 is fixed for both families/arms; builder checks actual target token lengths including EOS, never adapts caps to model outcomes."},
            "claim_scope": "Supplementary R/P behavior during Ledger adaptation; no broad retention pass or independent confirmation.",
            "coverage_limitations": [
                "Exposure classification references only the SHA-bound original t2_train input surface, not every predecessor corpus.",
                "Known semantic one-step keys are excluded. Novel initial parents are preferred; any finite-mask fallback is explicitly labelled audited_seen.",
                "Prerequisite cases are applicable under original SYSTEM_ONE; inapplicable no-ops require a separately qualified instruction frame.",
                "Parent IDs describe initial-state identities; graph_group_id records the shared modular-97 or 8-bit rule graph. Each family has one graph group, so these are not twelve independent worlds per family.",
                "Self-swap and zero-effect actions are legal original-rule probes; their one-step TRAIN keys may be novel.",
                "This supplement does not replace the separately required registered six-fixture audit.",
            ]}


def main():
    from dsv3wm.native import EXPECTED_SIDECARS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new frozen fixture JSON file")
    parser.add_argument("--seed", type=int, default=202610125)
    parser.add_argument("--t2-train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("RCWM retention fixtures already exist")
    if file_hash(args.tokenizer_json) != EXPECTED_SIDECARS["tokenizer.json"]:
        raise ValueError("Registered tokenizer JSON hash mismatch")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    document = build_fixtures(args.seed, audit=training_input_audit(args.t2_train), tokenizer=tokenizer)
    document["tokenizer_json_sha256"] = file_hash(args.tokenizer_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "sha256": file_hash(args.out), "family_counts": document["family_counts"],
                      "parent_counts": document["parent_counts"],
                      "max_prompt_tokens": max(row["prompt_tokens"] for row in document["fixtures"]),
                      "max_target_tokens_including_eos": max(row["target_tokens_including_eos"] for row in document["fixtures"])}), flush=True)


if __name__ == "__main__":
    main()
