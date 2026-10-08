"""Off-task capture only, for arms whose main capture already finished (e.g. sparse, supa).

  python scripts/run_offtask.py --bank bank_pilot --arm supa
"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from rcwmt import offtask, runtime  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="bank_pilot")
ap.add_argument("--arm", required=True)
a = ap.parse_args()
out = ROOT / "runs" / a.bank / a.arm
out.mkdir(parents=True, exist_ok=True)
if (out / "offtask_meta.json").exists():
    raise SystemExit(f"{a.arm}: off-task already captured")
tok = runtime.tokenizer()
with runtime.gpu_lease(label=f"rcwm-t1:offtask:{a.arm}", job=f"RCWM-T1 off-task capture {a.arm}"):
    t0 = time.time()
    model, prov = runtime.load_arm(a.arm)
    offtask.capture_offtask(model, tok, out)
    runtime.unload(model)
print(f"[{a.arm}] off-task capture done in {time.time() - t0:.0f}s fingerprint {prov['param_fingerprint'][:12]}", flush=True)
