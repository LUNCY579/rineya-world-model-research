"""Render-only entry point for S-roll: merge the existing per-family outputs into
runs/bank_pilot/analysis/s_roll.{json,log} and docs/s_roll/S_ROLL_PILOT.md, computing nothing.

`python scripts/s_roll.py --report-only` does the same but, because of a flag-handling quirk in its main()
(report-only still counts as a full run), it also re-runs the deterministic depth check s_roll_deep.py
(about 6 min on 2 threads). s_roll.py is left unchanged so the code hashes stored in the per-family outputs
stay valid; use this script when only the report needs re-rendering.

  python scripts/s_roll_render.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s_roll  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="", patches torch.cuda, 2 threads; no training on import)

if __name__ == "__main__":
    s_roll.report("")
