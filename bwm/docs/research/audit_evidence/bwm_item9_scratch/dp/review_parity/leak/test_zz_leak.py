import sys


def test_no_stub_leak():
    import bwmn
    bad = {m: repr(sys.modules.get(m)) for m in ("bwmn.lm", "bwmn.blocks_planner", "bwmn.domain_planner")
           if m in sys.modules}
    attrs = [a for a in ("lm", "blocks_planner", "domain_planner") if a in vars(bwmn)]
    assert not bad and not attrs, (bad, attrs)
    assert "torch" not in sys.modules
