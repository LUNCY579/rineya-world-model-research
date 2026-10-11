import sys, pathlib
src = pathlib.Path("dp_orig.py").read_text(encoding="utf-8")
M = {
 "M1_prefix_ignores_mode": ('if self.prefix_ok and self.mode == "actions"', 'if self.prefix_ok'),
 "M5_hardcoded_sampling": ('temperature=self.temperature, top_p=self.top_p,', 'temperature=1.0, top_p=0.95,'),
 "M8_hardcoded_fwd_adapter": ('            self.model.set_adapter(self.adapter)\n            prompts = [self.dom.forward_prompt_delta',
                              '            self.model.set_adapter("fwd")\n            prompts = [self.dom.forward_prompt_delta'),
 "M9_hardcoded_pa_adapter": ('        self.model.set_adapter(self.adapter)\n        rows =', '        self.model.set_adapter("pa")\n        rows ='),
}
name = sys.argv[1]
old, new = M[name]
assert src.count(old) == 1, (name, src.count(old))
pathlib.Path("bwmn/domain_planner.py").write_text(src.replace(old, new), encoding="utf-8")
