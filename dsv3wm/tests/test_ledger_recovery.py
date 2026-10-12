"""Pure six-field contract, causal interfaces, exposure and fresh-data tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dsv3wm.domains import _semantics
from dsv3wm.ledger_recovery import (ACCOUNTS, T2_TRAIN, TrainingSupport, batched_ledger_rollout,
    build_ledger_recovery, exact_ledger_transition, ledger_parent_id, ledger_prompt, ledger_rollout_labels,
    ledger_rules, ledger_target, load_training_support, make_ledger_record, operative_key,
    parse_ledger_response, prior_public_parents, validate_ledger_record, write_ledger_recovery)


def event(**changes):
    value = {"account": "A", "amount": 2, "kind": "add", "other": "B", "status": "posted", "tick": 1}
    value.update(changes)
    return value


def inputs(events=None):
    return {"domain": "ledger", "rules": ledger_rules(), "s0": {"A": 2, "B": -1, "C": 3, "D": 0},
            "actions": events if events is not None else [event()]}


class LedgerInterfaceTests(unittest.TestCase):
    def test_six_field_tick_executes_and_nonposted_is_legal_noop(self):
        before = inputs()["s0"]
        after, legal, terminated, invalid = exact_ledger_transition(before, event(tick=9901))
        self.assertEqual(after, {"A": 4, "B": -1, "C": 3, "D": 0})
        self.assertEqual((legal, terminated, invalid), (True, False, False))
        self.assertEqual(exact_ledger_transition(before, event(status="pending")), (before, True, False, False))
        self.assertEqual(exact_ledger_transition(before, event(kind="transfer", other="A")), (before, True, False, False))

    def test_malformed_contract_is_explicit_noop_extension(self):
        before = inputs()["s0"]
        for ev in (event(account="missing"), event(kind="teleport"), event(amount=True), event(tick=1.2)):
            self.assertEqual(exact_ledger_transition(before, ev), (before, False, False, True))
        ev = event(); ev.pop("tick")
        self.assertEqual(exact_ledger_transition(before, ev), (before, False, False, True))

    def test_h1_prompt_parity_and_multihorizon_causal_projection(self):
        inp = inputs()
        env, rw = _semantics("ledger"), _semantics("register")
        messages = env.one_step_messages("ledger_replay", inp["s0"], inp["actions"][0])
        self.assertEqual(ledger_prompt(inp), rw.render_chat(messages[0]["content"], messages[1]["content"]))
        inp["actions"] += [event(status="void", tick=2)]
        rendered = ledger_prompt(inp)
        self.assertIn('"events":[', rendered)
        self.assertNotIn('"labels"', rendered); self.assertNotIn('"goal"', rendered)
        with patch.object(env, "step", side_effect=AssertionError("oracle during prediction")):
            self.assertEqual(ledger_prompt(inp), rendered)
            self.assertEqual(parse_ledger_response(json.dumps(inp["s0"]), inp)["state"], inp["s0"])
        poisoned = copy.deepcopy(inp)
        poisoned["actions"][0]["target"] = "FUTURE_SECRET"
        with self.assertRaises(ValueError):
            ledger_prompt(poisoned)

    def test_strict_parser_and_contract_flags_are_not_learned(self):
        inp = inputs([event(status="cancelled")])
        parsed = parse_ledger_response(json.dumps(inp["s0"]), inp)
        self.assertEqual((parsed["legal"], parsed["terminated"], parsed["failed"]), (True, False, False))
        self.assertIn("not model", parsed["flags_source"])
        self.assertEqual(set(parse_ledger_response(json.dumps(inp["s0"]), inp, include_flags=False)), {"state"})
        for text in ('{"A":true,"B":0,"C":0,"D":0}', '{"A":1,"A":2,"B":0,"C":0,"D":0}',
                     '{"A":1,"B":0,"C":0,"D":0} prose', '{"A":NaN,"B":0,"C":0,"D":0}',
                     '{"state":{"A":1,"B":0,"C":0,"D":0}}'):
            self.assertTrue(parse_ledger_response(text, inp)["malformed"])
        changed = copy.deepcopy(inp); changed["s0"] = {"E": 2, "F": -1, "G": 3, "H": 0}
        with self.assertRaises(ValueError):
            ledger_prompt(changed)

    def test_own_states_rollout_never_uses_teacher_labels(self):
        seen = []
        class Mock:
            def predict_batch(self, batch):
                seen.append([x["s0"]["A"] for x in batch])
                return [{"state": {**x["s0"], "A": x["s0"]["A"] + 7}} for x in batch]
        row = {**inputs([event(tick=i) for i in range(4)]), "labels": {"states": [{"A": 999}] * 4}}
        outputs = batched_ledger_rollout(Mock(), [row])
        self.assertEqual(seen, [[2], [9], [16], [23]])
        self.assertEqual(outputs[0][-1]["state"]["A"], 30)
        class Bad:
            def predict_batch(self, batch):
                return [{"malformed": True} for _ in batch]
        self.assertEqual(len(batched_ledger_rollout(Bad(), [row])[0]), 1)

    def test_keys_ignore_tick_unused_other_and_account_renaming(self):
        before = inputs()["s0"]
        self.assertEqual(operative_key(before, event()), operative_key(before, event(tick=9999, other="D")))
        self.assertEqual(ledger_parent_id(before), ledger_parent_id({"A": 3, "B": 0, "C": 2, "D": -1}))


class LedgerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_ledger_recovery(912)

    def test_collection_is_reproducible_independent_and_informative(self):
        data = self.data
        self.assertEqual(data, build_ledger_recovery(912))
        self.assertEqual([len(data[k]) for k in ("train", "dev", "dev_invalid")], [128, 24, 4])
        rows = [*data["train"], *data["dev"], *data["dev_invalid"]]
        self.assertEqual(len({r["parent_world_id"] for r in rows}), 156)
        for row in [*data["train"], *data["dev"]]:
            validate_ledger_record(row, replay=True)
            self.assertNotEqual(row["s0"], row["labels"]["states"][-1])
            self.assertTrue(all(row["labels"]["legal"]))
            self.assertFalse(any(row["labels"]["terminated"]))
            self.assertEqual(sum(e["status"] == "posted" for e in row["actions"]), 2)
            self.assertTrue(all(abs(v) <= 98 for s in row["labels"]["states"] for v in s.values()))
        self.assertTrue(all(r["labels"]["invalid"][0] for r in data["dev_invalid"]))
        self.assertNotIn("confirmation", data)

    def test_status_key_and_tick_stresses_are_explicit(self):
        data = self.data["dev"]
        self.assertEqual([sum(r["provenance"]["cohort"] == name for r in data) for name in
                          ("canonical_id", "held_status", "reordered_keys", "shifted_tick")], [12, 4, 4, 4])
        for row in data:
            cohort = row["provenance"]["cohort"]
            inp = {key: row[key] for key in ("domain", "rules", "s0", "actions")}
            inp["actions"] = inp["actions"][:1]
            if cohort == "shifted_tick":
                self.assertTrue(all(e["tick"] >= 1001 for e in row["actions"]))
            if cohort == "held_status":
                self.assertTrue(all(e["status"] in {"posted", "pending", "draft"} for e in row["actions"]))
            if cohort == "reordered_keys":
                self.assertIn('{"event":', ledger_prompt(inp))

    def test_training_support_hash_and_tick_compatible_exclusion(self):
        before = inputs()["s0"]
        row = {"family": "ledger_replay", "before": before, "event": event(), "target": "NOT_USED"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t2_train.jsonl"
            raw = (json.dumps(row) + "\n").encode(); path.write_bytes(raw)
            support = load_training_support(path, expected_sha256=hashlib.sha256(raw).hexdigest())
            self.assertTrue(support.metadata["available"])
            self.assertIn(ledger_parent_id(before), support.parents)
            self.assertIn(operative_key(before, event(tick=100)), support.keys)
            with self.assertRaises(ValueError):
                load_training_support(path)
        restricted = build_ledger_recovery(913, exclusions={self.data["train"][0]["parent_world_id"]})
        self.assertNotIn(self.data["train"][0]["parent_world_id"], restricted["parent_registry"])

    def test_targets_and_public_exclusions_refuse_confirmation_labels(self):
        row = self.data["train"][0]
        self.assertEqual(ledger_target(row, 4), row["labels"]["states"][3])
        with self.assertRaises(PermissionError):
            ledger_target(dict(row, split="confirmation"), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confirmation.jsonl"
            path.write_text(json.dumps(dict(row, split="confirmation")) + "\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                prior_public_parents([path])
            with self.assertRaises(ValueError):
                prior_public_parents([Path(directory) / "confirmation.labels.jsonl"])

    def test_prior_public_rename_exclusion_does_not_broaden_native_contract(self):
        row = copy.deepcopy(self.data["train"][0])
        mapping = dict(zip(ACCOUNTS, ("u", "v", "w", "z")))
        row["s0"] = {mapping[k]: v for k, v in row["s0"].items()}
        row["rules"]["objects"] = list(mapping.values())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            parents, _ = prior_public_parents([path])
            self.assertEqual(parents, {row["parent_world_id"]})
        with self.assertRaises(ValueError):
            ledger_prompt({key: row[key] for key in ("domain", "rules", "s0", "actions")})

    def test_writer_does_not_create_confirmation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ledger"
            missing = Path(directory) / "unavailable" / "t2_train.jsonl"
            manifest = write_ledger_recovery(output, seed=914, prior_train=missing)
            self.assertEqual(manifest["confirmation_rows"], 0)
            self.assertFalse(manifest["training_support"]["available"])
            for name, metadata in manifest["files"].items():
                self.assertEqual(file_sha(output / name), metadata["sha256"])
            with self.assertRaises(FileExistsError):
                write_ledger_recovery(output, prior_train=missing)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
