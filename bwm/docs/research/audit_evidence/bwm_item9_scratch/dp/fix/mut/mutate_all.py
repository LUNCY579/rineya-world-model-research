"""Scratch mutation check for domain_planner.py fixes: each mutant must make tests/test_domain_planner.py fail."""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "dp_orig.py").read_text(encoding="utf-8")
M = {
    "PAR1_prefix_ignores_mode": ('if self.prefix_ok and self.mode == "actions"', 'if self.prefix_ok'),
    "PAR2a_hardcoded_sampling": ('temperature=self.temperature, top_p=self.top_p,', 'temperature=1.0, top_p=0.95,'),
    "PAR2b_hardcoded_fwd_adapter": ('            self.model.set_adapter(self.adapter)\n            prompts = [self.dom',
                                    '            self.model.set_adapter("fwd")\n            prompts = [self.dom'),
    "PAR2c_hardcoded_pa_adapter": ('        self.model.set_adapter(self.adapter)\n        rows =',
                                   '        self.model.set_adapter("pa")\n        rows ='),
    "LG1_no_render_check": ('return r if self.renders(r[0]) else None', 'return r'),
    "LG1_crash_guard_only": ('return self.dom.state_from_json(self.dom.state_json(s)) == s',
                             'return bool(self.dom.state_json(s))'),
    "LG1_roundtrip_for_blocks_too": ('self.roundtrip = dom is not B', 'self.roundtrip = True'),
    "LG1_no_key_check": ('if need and self.roundtrip:  #', 'if False:  #'),
    "LG2_canonicalise_instead": ('return r if self.renders(r[0]) else None',
                                 'return r if self.renders(r[0]) else ((self.dom.state_from_json(self.dom.state_json('
                                 'r[0])), r[1]) if () not in r[0] else None)'),
    "LG3_no_objs_check": ('if self.dom is not B:  # B2 parity: Blocksworld items', 'if False:  #'),
    "LG3_objs_check_for_blocks": ('if self.dom is not B:  # B2 parity: Blocksworld items', 'if True:  #'),
    "LG3_size_fallback": ('    if objs is None:\n        raise ValueError(f"score_d',
                          '    if objs is None:\n        objs = dom.objs_of(it["init"])\n    if objs is None:\n'
                          '        raise ValueError(f"score_d'),
}
target = HERE / "bwmn" / "domain_planner.py"
res = {}
try:
    for name, (old, new) in [(k, v) for k, v in M.items() if k in ('LG1_no_render_check', 'LG1_roundtrip_for_blocks_too', 'LG1_no_key_check', 'PAR1_prefix_ignores_mode')]:
        assert src.count(old) == 1, (name, src.count(old))
        target.write_text(src.replace(old, new), encoding="utf-8")
        p = subprocess.run([sys.executable, "-B", "-m", "pytest", "tests/test_domain_planner.py", "-q", "-p",
                            "no:cacheprovider", "--no-header"], cwd=HERE, capture_output=True, text=True)
        last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1]
        failed = [ln.split("::")[1].split(" ")[0] for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
        res[name] = (p.returncode != 0, last, failed)
        print(f"{name:32s} caught={p.returncode != 0}  {last}  {failed}", flush=True)
finally:
    target.write_text(src, encoding="utf-8")
print("all caught:", all(v[0] for v in res.values()))
