import os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("which nvidia-smi:", shutil.which("nvidia-smi"))
print("PATH has System32:", any("system32" in p.lower() for p in os.environ.get("PATH", "").split(os.pathsep)))
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    print("rc", r.returncode, "out", repr(r.stdout[:80]), "err", repr(r.stderr[:120]))
except Exception as e:
    print("EXC", type(e).__name__, e)
try:
    import torch
    print("torch.cuda.temperature:", torch.cuda.temperature())
except Exception as e:
    print("torch temp EXC", type(e).__name__, str(e)[:120])
import rcwmt.runtime as rt
print("gpu_temp_c():", rt.gpu_temp_c())
