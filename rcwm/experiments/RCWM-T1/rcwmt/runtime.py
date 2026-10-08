"""Model arms, resource guards and the shared-GPU lease.

Every arm is materialized in FP32 on the GPU. Delta arms are built in place on the device from a
loaded FP32 core and per-tensor reads of the BF16 Supa source through safetensors mmap, so host
RAM never holds a second copy of the model.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch

CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
QWEN3 = Path(r"C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
PATHS = {
    "supa": CORES / "Rineya-Supa-1.7.0-devfreeze",
    "sparse": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
    "exp": CORES / "Rineya-Titan-Experimental-1.0.0-devfreeze",
    "transition": CORES / "Rineya-Titan-Transition-1.0.0-devfreeze",
    "qwen3": QWEN3,
}
# Delta arms: base + alpha * (donor - base), built on device.
#   sparse_a{x}: Supa + x * dSparse      (dose line along the magnitude-pruned transition update)
#   randdelta:   Supa + random vector with dSparse's exact per-tensor support and norm (null)
DELTA_ARMS = {
    "sparse_a0.5": ("sparse", 0.5),
    "sparse_a2.0": ("sparse", 2.0),
}
TOKENIZER_DIR = PATHS["sparse"]           # byte-identical across all Titan cores and Supa
LEASE_DIR = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1")
HOST_FLOOR_GIB = 2.0                      # device_map streaming keeps host RSS well below model size
VRAM_FLOOR_GIB = 3.6


BREATHE_S = float(os.environ.get("RCWM_BREATHE_S", "0.12"))   # minimum pause between batches
DUTY_RATIO = float(os.environ.get("RCWM_DUTY_RATIO", "1.5"))   # sleep >= ratio x the batch's compute time
_last_breathe = [None]


def breathe() -> None:
    """Bound the GPU duty cycle between batches (DECISIONS I-10, I-12, I-13).

    The RTX 4070 Laptop drives the desktop and reached 95 C at 109 W within 10 s of sustained FP32
    load. After each batch: synchronize, then sleep max(BREATHE_S, DUTY_RATIO x time since the
    previous breathe), which caps the duty cycle near 1/(1+DUTY_RATIO) (40% at 1.5). Also blocks
    while the independent watchdog's pause flag exists."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.time()
    busy = 0.0 if _last_breathe[0] is None else now - _last_breathe[0]
    time.sleep(max(BREATHE_S, min(_duty_ratio() * busy, 5.0)))
    flag = Path(__file__).resolve().parents[1] / "runs" / ".thermal_pause"
    if flag.exists():                         # set by scripts/thermal_watchdog.py (I-12)
        print("[thermal] watchdog pause flag set; waiting", flush=True)
        while flag.exists():
            time.sleep(2)
        print("[thermal] watchdog flag cleared; resuming", flush=True)
    thermal_guard()
    _last_breathe[0] = time.time()


class ThrottleProcessor:
    """LogitsProcessor that pauses every `every` decode steps (generate() otherwise runs its whole
    decode loop as one uninterrupted GPU burst; DECISIONS I-14). Returns scores unchanged."""

    def __init__(self, every: int = 8, sleep_s: float = 0.06):
        self.every, self.sleep_s, self.n = every, sleep_s, 0

    def __call__(self, input_ids, scores):
        self.n += 1
        if self.n % self.every == 0:
            torch.cuda.synchronize()
            time.sleep(self.sleep_s)
            if (Path(__file__).resolve().parents[1] / "runs" / ".thermal_pause").exists():
                while (Path(__file__).resolve().parents[1] / "runs" / ".thermal_pause").exists():
                    time.sleep(2)
        return scores


def throttle_list():
    from transformers import LogitsProcessorList
    return LogitsProcessorList([ThrottleProcessor()])


_temp_cache = [0.0, None]


def _duty_ratio() -> float:
    """Temperature-adaptive pause ratio (I-15): RCWM_DUTY_FORCE=1 forces RCWM_DUTY_RATIO;
    otherwise 0.5 below 70 C, 1.5 at 70-74 C, 3.0 at >= 75 C (temperature cached for 10 s)."""
    if os.environ.get("RCWM_DUTY_FORCE") == "1":
        return DUTY_RATIO
    now = time.time()
    if now - _temp_cache[0] > 10:
        _temp_cache[0], _temp_cache[1] = now, gpu_temp_c()
    t = _temp_cache[1]
    if t is None or t >= 75:
        return 3.0
    return 1.5 if t >= 70 else 0.5


_last_temp_check = [0.0]


def gpu_temp_c() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def thermal_guard(hot: int = 80, cool: int = 72, every_s: float = 10.0) -> None:
    """In-process backup to the watchdog: if the GPU is >= 80 C, pause until <= 72 C."""
    now = time.time()
    if now - _last_temp_check[0] < every_s:
        return
    _last_temp_check[0] = now
    t = gpu_temp_c()
    if t is not None and t >= hot:
        print(f"[thermal] GPU {t} C >= {hot}; pausing", flush=True)
        while (t := gpu_temp_c()) is not None and t > cool:
            time.sleep(3)
        print(f"[thermal] resumed at {t} C", flush=True)


def set_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70, 0)   # never spill into shared memory (I-07, I-10)


def free_vram_gib() -> float:
    free, _ = torch.cuda.mem_get_info(0)
    return free / 2 ** 30


def host_free_gib() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def wait_for_resources(poll: int = 30, timeout: int = 6 * 3600) -> None:
    t0 = time.time()
    announced = False
    while True:
        v, h = free_vram_gib(), host_free_gib()
        if v >= VRAM_FLOOR_GIB and h >= HOST_FLOOR_GIB:
            return
        if not announced:
            print(f"[resources] waiting: free VRAM {v:.2f} GiB (need {VRAM_FLOOR_GIB}), "
                  f"host {h:.2f} GiB (need {HOST_FLOOR_GIB})", flush=True)
            announced = True
        if time.time() - t0 > timeout:
            raise RuntimeError("resources never became available")
        time.sleep(poll)


YIELD_FILE = Path(__file__).resolve().parents[1] / "runs" / ".yield.json"
LOCK_PATH = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")


def _lock_label():
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8")).get("label")
    except (OSError, ValueError):
        return None


def yield_to_peer(my_label: str, poll: int = 10) -> None:
    """One-shot courtesy yield agreed with the peer session: if runs/.yield.json applies to this
    label prefix, wait until the named peer lease has been seen holding the lock and then released
    (or max_wait_s), then mark the file consumed."""
    for f in sorted(YIELD_FILE.parent.glob(".yield*.json")):
        y = json.loads(f.read_text(encoding="utf-8"))
        if not y.get("consumed") and my_label.startswith(y["applies_to_prefix"]):
            break
    else:
        return
    print(f"[yield] waiting for peer lease '{y['peer_label']}' to run and release (max {y['max_wait_s']}s)", flush=True)
    t0, seen = time.time(), False
    while time.time() - t0 < y["max_wait_s"]:
        lab = _lock_label()
        if lab == y["peer_label"]:
            seen = True
        elif seen:
            break
        try:                                   # another of this session's jobs already honoured it
            if json.loads(f.read_text(encoding="utf-8")).get("consumed"):
                print("[yield] already honoured by another job of this session", flush=True)
                return
        except (OSError, ValueError):
            pass
        time.sleep(poll)
    y.update(consumed=True, consumed_by=my_label, peer_seen=seen, waited_s=round(time.time() - t0))
    f.write_text(json.dumps(y, indent=1) + "\n", encoding="utf-8")
    print(f"[yield] done (peer seen={seen}, waited {y['waited_s']}s)", flush=True)


TURN = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.turn.json")
WAITING = {"rcwm": Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.waiting.rcwm.json"),
           "bwm": Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.waiting.bwm.json")}


def _pid_alive(pid) -> bool:
    try:
        return psutil.pid_exists(int(pid))
    except (TypeError, ValueError):
        return False


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def alternate_turn(label: str, max_wait_s: int = 2700, poll: int = 15) -> None:
    """One-for-one alternation with the peer session (DECISIONS I-16): if this side held the GPU
    last AND the peer has a live waiting marker, wait until the peer's job has taken a turn."""
    me, peer = "rcwm", "bwm"
    WAITING[me].write_text(json.dumps({"pid": os.getpid(), "label": label, "since": time.time()}), encoding="utf-8")
    t0, announced = time.time(), False
    while time.time() - t0 < max_wait_s:
        turn = _read_json(TURN) or {}
        w = _read_json(WAITING[peer])
        peer_waiting = bool(w) and _pid_alive(w.get("pid"))
        if turn.get("last") != me or not peer_waiting:
            break
        if not announced:
            print(f"[turn] last turn was ours and the peer is waiting ({w.get('label')}); yielding", flush=True)
            announced = True
        time.sleep(poll)


def _released(label: str) -> None:
    TURN.write_text(json.dumps({"last": "rcwm", "label": label, "released_unix": time.time()}), encoding="utf-8")


@contextmanager
def gpu_lease(label: str, job: str):
    from .lease_vendored import lease          # dedicated-GPU-memory holder check (I-17)
    yield_to_peer(label)
    alternate_turn(label)
    flag = Path(__file__).resolve().parents[1] / "runs" / ".thermal_pause"
    if flag.exists():
        print("[thermal] watchdog pause flag set before acquiring the GPU; waiting", flush=True)
        while flag.exists():
            time.sleep(5)
    try:
        with lease(label=label, job=job, timeout_seconds=6 * 3600):
            try:
                WAITING["rcwm"].unlink()
            except OSError:
                pass
            try:
                wait_for_resources()
                yield
            finally:
                _released(label)
    finally:
        mk = _read_json(WAITING["rcwm"])
        if mk and mk.get("pid") == os.getpid():
            try:
                WAITING["rcwm"].unlink()
            except OSError:
                pass


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)


def _from_pretrained(path: Path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.float32, local_files_only=True, trust_remote_code=False,
        low_cpu_mem_usage=True, device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    assert all(p.dtype == torch.float32 for p in m.parameters())
    return m


def _safetensor_path(path: Path) -> Path:
    return path / "model.safetensors"


@torch.no_grad()
def _apply_delta(model, donor: str, alpha: float | None, random_null: bool, seed: int = 17) -> dict:
    """Model currently holds DONOR weights (FP32). Rewrite every parameter as
    base + alpha*(donor-base), or base + random-with-same-support-and-norm."""
    from safetensors import safe_open
    base = _safetensor_path(PATHS["supa"])
    stats = {"tensors": 0, "changed": 0, "delta_norm_sq": 0.0}
    g = torch.Generator(device="cuda").manual_seed(seed)
    names = dict(model.named_parameters())
    with safe_open(str(base), framework="pt", device="cuda") as f:
        keys = set(f.keys())
        for name, p in names.items():
            key = name if name in keys else None
            if key is None:
                if name == "lm_head.weight":
                    continue                       # tied to embed_tokens
                raise KeyError(name)
            b = f.get_tensor(key).to(torch.float32)
            d = p.data - b
            stats["tensors"] += 1
            if torch.count_nonzero(d) == 0:
                continue
            stats["changed"] += 1
            if random_null:
                mask = d != 0
                r = torch.randn(d.shape, generator=g, device="cuda") * mask
                r = r * (d.norm() / r.norm())
                d = r
            else:
                d = d * alpha
            stats["delta_norm_sq"] += float(d.double().pow(2).sum())
            p.data.copy_(b + d)
            del b, d
    torch.cuda.empty_cache()
    stats["delta_norm"] = stats.pop("delta_norm_sq") ** 0.5
    return stats


def load_arm(arm: str):
    """Return (model, provenance dict)."""
    set_numerics()
    prov = {"arm": arm, "dtype": "float32", "tf32": False}
    if arm in PATHS:
        m = _from_pretrained(PATHS[arm])
        prov["source"] = str(PATHS[arm])
    elif arm in DELTA_ARMS:
        donor, alpha = DELTA_ARMS[arm]
        m = _from_pretrained(PATHS[donor])
        prov.update(source=str(PATHS[donor]), base=str(PATHS["supa"]), alpha=alpha,
                    delta_stats=_apply_delta(m, donor, alpha, random_null=False))
    elif arm == "randdelta":
        m = _from_pretrained(PATHS["sparse"])
        prov.update(source=str(PATHS["sparse"]), base=str(PATHS["supa"]),
                    construction="Supa + gaussian delta on dSparse support, per-tensor norm-matched, seed 17",
                    delta_stats=_apply_delta(m, "sparse", None, random_null=True))
    else:
        raise KeyError(arm)
    prov["param_fingerprint"] = param_fingerprint(m)
    return m, prov


@torch.no_grad()
def param_fingerprint(model) -> str:
    """Cheap identity: sha256 over a fixed strided sample of every parameter (FP32 bytes)."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        flat = p.detach().reshape(-1)
        steps = min(4096, flat.numel())
        # integer arithmetic: a float32 linspace rounds the last index of a 155M-element tensor to n
        idx = torch.arange(steps, device=flat.device, dtype=torch.int64) * (flat.numel() - 1) // max(steps - 1, 1)
        h.update(name.encode())
        h.update(flat[idx].float().cpu().numpy().tobytes())
    return h.hexdigest()


def unload(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def runtime_info() -> dict:
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0), "tf32": False, "use_cache": False,
            "executable": sys.executable}
