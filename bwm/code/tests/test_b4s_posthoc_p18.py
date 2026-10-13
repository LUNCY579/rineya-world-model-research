"""scripts/b4s_posthoc_p18.py: the post-hoc seeds_fresh/0 note never runs before the registered report and is exact."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def m(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("b4s_posthoc_p18", ROOT / "scripts" / "b4s_posthoc_p18.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "BW_B4_SCREEN"
    (out / "a_beam").mkdir(parents=True)
    (tmp_path / "R0").mkdir()
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "A_PLANS", out / "a_beam" / "plans.jsonl")
    monkeypatch.setattr(mod, "R0_PLANS", tmp_path / "R0" / "plans.jsonl")
    monkeypatch.setattr(mod, "REPORT", out / "B4S_REPORT.json")
    return mod


def write(p, outcomes, val=True):
    rows = [{"id": f"seeds_fresh/{i}", "success": o, **({"val": o} if val else {})} for i, o in enumerate(outcomes)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_refuses_before_the_registered_report(m):
    write(m.A_PLANS, [True] * 40)
    write(m.R0_PLANS, [True] * 40)
    with pytest.raises(SystemExit):
        m.main()


def test_concordant_instance_leaves_net_unchanged(m):
    a = [True, True, False] + [False] * 37
    c = [True, False, True] + [False] * 37
    rec = m.analyse({f"seeds_fresh/{i}": o for i, o in enumerate(a)}, {f"seeds_fresh/{i}": o for i, o in enumerate(c)})
    assert rec["concordant"] and rec["all_40"]["net"] == 0 and "without_instance" not in rec


def test_discordant_instance_reports_both_nets(m):
    a = [True, True, True] + [False] * 37
    c = [False, False, True] + [False] * 37
    rec = m.analyse({f"seeds_fresh/{i}": o for i, o in enumerate(a)}, {f"seeds_fresh/{i}": o for i, o in enumerate(c)})
    assert not rec["concordant"]
    assert rec["all_40"]["net"] == 2 and rec["without_instance"]["net"] == 1 and rec["without_instance"]["n"] == 39
    assert "DISCORDANT" in m.render(rec)


def test_val_field_wins_over_success(m, tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": "seeds_fresh/0", "success": True, "val": False}) + "\n", encoding="utf-8")
    assert m.outcomes(p) == {"seeds_fresh/0": False}


def test_end_to_end_writes_once_and_quotes_the_decision(m):
    write(m.A_PLANS, [True] + [False] * 39)
    write(m.R0_PLANS, [True] + [False] * 39)
    m.REPORT.write_text(json.dumps({"status": "NO-GO", "other": 1}), encoding="utf-8")
    assert m.main() == 0
    rec = json.loads((m.OUT / "POSTHOC_P18_SENSITIVITY.json").read_text())
    assert rec["registered_decision_quoted"] == {"status": "NO-GO"} and rec["concordant"]
    with pytest.raises(SystemExit):
        m.main()


def test_mismatched_ids_refused(m):
    with pytest.raises(ValueError):
        m.analyse({"seeds_fresh/0": True}, {"seeds_fresh/0": True, "seeds_fresh/1": False})
