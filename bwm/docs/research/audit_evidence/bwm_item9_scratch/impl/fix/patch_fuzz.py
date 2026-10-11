s = open("fuzz_reader.py").read()
old = "    if rng.random() < 0.25:\n        parts = text.split"
new = ("    if rng.random() < 0.15:\n        ms = [m.start() for m in re.finditer(r\"[ (][a-z]\", text)]\n"
       "        if ms:\n            k = rng.choice(ms) + 1; text = text[:k] + rng.choice(['#', '#', '\"', '@']) + text[k:]; tags.append('glue')\n"
       + old)
assert old in s
s = s.replace(old, new, 1)
s = s.replace('"1_", "x1-0"]', '"1_", "x1-0", "#t", "#T1"]')
open("fuzz_reader.py", "w").write(s)
