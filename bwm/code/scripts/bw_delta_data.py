"""Delta-format forward rows from the SAME transitions as runs/BW_data/forward_{train,val}.jsonl (state format), so the
two world-model formats differ only in their output. Writes runs/BW_data/forward_delta_{train,val}.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402


def main():
    for split in ("train", "val"):
        src = ROOT / f"runs/BW_data/forward_{split}.jsonl"
        dst = ROOT / f"runs/BW_data/forward_delta_{split}.jsonl"
        n = 0
        with open(dst, "w", encoding="utf-8") as out:
            for line in open(src, encoding="utf-8"):
                r = json.loads(line)
                user = json.loads(r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
                s = B.state_from_json(user["state"])
                a = tuple(user["action"].split())
                ns, ok = B.step(s, a)
                assert B.forward_target(ns, ok) == r["target"] and ok == r["legal"]
                assert B.forward_prompt(s, a) == r["prompt"]
                out.write(json.dumps({"prompt": B.forward_prompt_delta(s, a), "target": B.forward_target_delta(s, ns, ok),
                                      "legal": ok, "n_blocks": r["n_blocks"]}) + "\n")
                n += 1
        print(split, n, "rows ->", dst.name)


if __name__ == "__main__":
    main()
