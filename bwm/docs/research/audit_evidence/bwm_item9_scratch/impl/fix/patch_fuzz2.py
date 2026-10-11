s = open("fuzz_reader.py").read()
old = "    if rng.random() < 0.15:\n        ms = [m.start() for m in re.finditer(r\"[ (][a-z]\", text)]"
new = ("    if rng.random() < 0.2:\n        cs = [m.start() for m in re.finditer(r\"\)\", text)]\n"
       "        if cs:\n            k = rng.choice(cs); text = text[:k] + ' ' + rng.choice(['c0', 'zzz', 'p0 t0', 'l0-0', 'a1']) + text[k:]; tags.append('extra_arg')\n"
       + old)
assert old in s
s = s.replace(old, new, 1)
open("fuzz_reader.py", "w").write(s)
