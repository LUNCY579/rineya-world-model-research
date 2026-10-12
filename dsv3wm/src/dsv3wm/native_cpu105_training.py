"""Training-only CPU10.5 admission for the remaining fixed H3 A4/C3/C4 pilot.

Every earlier loader is unchanged. Callers acquire the original CPU lane and
retain its four-GiB work boundary. This API does not admit inference or A3 repeats.
"""
import json
from pathlib import Path
from . import REPO
from .native import EXPECTED_CORE_SHA, EXPECTED_SIDECARS, sha256, runtime, boundary
from .native_cpu11 import validate_environment

REGISTERED_CONFIGS = {
    "A4": ("recovery_student_A4.json", "e7f7992585e8029ab07d3bea410ac3b16d21166e280dd486475fe4cdd63b5dd5"),
    "C3": ("recovery_student_C3.json", "e0a2d3da3aa66a3a88cada47d5b9124c0c4260fe1ff4a2a757a3e74b831782d6"),
    "C4": ("recovery_student_C4.json", "88db721d4749abb5b36d7331d1324b008474f24f9979cb2d9c440cb6da7ca942"),
}
ADMISSION_PROFILE = {
    "name": "dsv3wm-cpu105-h3-training-v1", "device": "cpu", "threads": 4,
    "load_free_host_ram_floor_gib": 10.5, "admission_memory_kinds": ["physical", "commit"],
    "historical_load_free_host_ram_floor_gib": 12, "previous_cpu11_load_floor_gib": 11,
    "unchanged_boundary_free_host_ram_floor_gib": 4, "training_only": True,
    "allowed_training_kinds": ["A4", "C3", "C4"], "seed": 0, "batch_size": 1, "optimizer_steps": 32,
    "training_cache": False, "runtime": "dsv3wm.native.runtime", "boundary": "dsv3wm.native.boundary",
    "sampler_cpu_batch_tokens_parameter": 8000, "sampler_ram_floor_gib": 8,
    "sampler_batch_geometry": "unchanged inherited heuristic; not a hard padded-token cap",
    "prediction_change": False, "objective_change": False,
}
_LAST_ADMISSION_CHECK = None


def registered_config_paths():
    return [REPO / "dsv3wm/configs" / filename for filename, _ in REGISTERED_CONFIGS.values()]


def validate_training_config(config):
    """Only complete, type-exact registered remaining-training recipes qualify."""
    from .recovery_student import StudentConfig
    if type(config) is not StudentConfig:
        raise ValueError("CPU10.5 requires an explicit registered H3 training StudentConfig")
    render = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = render(config.dict())
    for kind, (filename, expected_sha) in REGISTERED_CONFIGS.items():
        path = REPO / "dsv3wm/configs" / filename
        if sha256(path) != expected_sha:
            raise ValueError("Registered remaining H3 training configuration bytes changed")
        expected = StudentConfig.read(path)
        if actual == render(expected.dict()):
            return kind
    raise ValueError("CPU10.5 permits only fixed A4/C3/C4 training; ordinary A3 repeats and changed recipes refuse")


def admission_profile():
    import os
    validate_environment()
    return {**ADMISSION_PROFILE, "environment": {name: os.environ.get(name) for name in
            ("BWM_CPU_BATCH_TOKENS", "BWM_RAM_FLOOR_GB")}}


def last_admission_check():
    return None if _LAST_ADMISSION_CHECK is None else dict(_LAST_ADMISSION_CHECK)


def load(config, adapter=None, *, training_only=False):
    global _LAST_ADMISSION_CHECK
    _LAST_ADMISSION_CHECK = None
    if training_only is not True or adapter is not None:
        raise ValueError("CPU10.5 requires training_only=True and core-only load; inference adapter loading is refused")
    kind = validate_training_config(config)
    validate_environment()
    from bwmn import lm
    import torch
    physical, commit = lm.host_ram_free_gb()
    passed = physical >= 10.5 and commit >= 10.5
    _LAST_ADMISSION_CHECK = {
        "training_kind": kind, "physical_available_gib": physical, "commit_available_gib": commit,
        "required_physical_gib": 10.5, "required_commit_gib": 10.5,
        "physical_qualified": physical >= 10.5, "commit_qualified": commit >= 10.5,
        "passed": passed, "status": "admitted" if passed else "refused",
        "measurement_stage": "after BWM/torch imports; before core hash/load",
    }
    if not passed:
        raise RuntimeError("native CPU10.5 training load requires at least 10.5 GiB physical and commit availability")
    torch.set_num_threads(config.threads)
    lm.set_determinism(config.seed)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=True)
    core_path = lm.MODELS[config.core]
    actual = sha256(core_path / 'model.safetensors')
    if actual != EXPECTED_CORE_SHA:
        raise RuntimeError('parent checkpoint changed; audit and preregistration required')
    for name, expected in EXPECTED_SIDECARS.items():
        if sha256(core_path / name) != expected:
            raise RuntimeError(f'parent sidecar changed: {name}')
    model, tokenizer = lm.load_core(config.core)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('FP32 parent required')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, tokenizer, lm
