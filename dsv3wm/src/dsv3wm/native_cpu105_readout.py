"""CPU10.5 admission for fixed singleton H3 proposer/guard readout only.

All previous loaders remain unchanged. The original CPU lane/four-GiB boundary
and fixed numerical contract apply. No training or broad-batch admission occurs.
"""
import json
from pathlib import Path
from . import REPO
from .native import EXPECTED_CORE_SHA, EXPECTED_SIDECARS, sha256, runtime, boundary
from .native_cpu11 import validate_environment

REGISTERED_CONFIGS = {
    "A3": ("recovery_student_A3.json", "e5de29d701cf0cbc596c84785e5235e79c3ab57a641a9d4a3c3d6ea5e6e05104"),
    "A4": ("recovery_student_A4.json", "e7f7992585e8029ab07d3bea410ac3b16d21166e280dd486475fe4cdd63b5dd5"),
    "C3": ("recovery_student_C3.json", "e0a2d3da3aa66a3a88cada47d5b9124c0c4260fe1ff4a2a757a3e74b831782d6"),
    "C4": ("recovery_student_C4.json", "88db721d4749abb5b36d7331d1324b008474f24f9979cb2d9c440cb6da7ca942"),
}
SINGLETON_GEOMETRY = "h3-singleton-requests-v1"
ADMISSION_PROFILE = {
    "name": "dsv3wm-cpu105-h3-singleton-readout-v1", "device": "cpu", "threads": 4,
    "load_free_host_ram_floor_gib": 10.5, "admission_memory_kinds": ["physical", "commit"],
    "historical_load_free_host_ram_floor_gib": 12, "previous_cpu11_load_floor_gib": 11,
    "unchanged_boundary_free_host_ram_floor_gib": 4, "readout_only": True,
    "allowed_config_kinds": ["A3", "A4", "C3", "C4"], "seed": 0,
    "readout_geometry": SINGLETON_GEOMETRY, "max_sampler_requests_per_call": 1,
    "sampler_cpu_batch_tokens_parameter": 8000, "sampler_ram_floor_gib": 8,
    "sampler_batch_geometry": "one unchanged sampler request per call; inherited heuristic unmodified",
    "runtime": "dsv3wm.native.runtime", "boundary": "dsv3wm.native.boundary",
    "prediction_contract_change": False, "scoring_change": False,
    "geometry_change": "guard batches become singleton; proposer already singleton",
}
_LAST_ADMISSION_CHECK = None


def registered_config_paths():
    return [REPO / "dsv3wm/configs" / filename for filename, _ in REGISTERED_CONFIGS.values()]


def validate_readout_config(config):
    """Only complete, type-exact registered readout recipes qualify."""
    from .recovery_student import StudentConfig
    if type(config) is not StudentConfig:
        raise ValueError("CPU10.5 requires an explicit registered H3 readout StudentConfig")
    render = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = render(config.dict())
    for kind, (filename, expected_sha) in REGISTERED_CONFIGS.items():
        path = REPO / "dsv3wm/configs" / filename
        if sha256(path) != expected_sha:
            raise ValueError("Registered H3 readout configuration bytes changed")
        expected = StudentConfig.read(path)
        if actual == render(expected.dict()):
            return kind
    raise ValueError("CPU10.5 singleton readout permits only fixed A3/A4/C3/C4 recipes")


def admission_profile():
    import os
    validate_environment()
    return {**ADMISSION_PROFILE, "environment": {name: os.environ.get(name) for name in
            ("BWM_CPU_BATCH_TOKENS", "BWM_RAM_FLOOR_GB")}}


def last_admission_check():
    return None if _LAST_ADMISSION_CHECK is None else dict(_LAST_ADMISSION_CHECK)


def load(config, adapter=None, *, readout_only=False):
    global _LAST_ADMISSION_CHECK
    _LAST_ADMISSION_CHECK = None
    if readout_only is not True or adapter is not None:
        raise ValueError("CPU10.5 requires readout_only=True and core-only load; adapter and training loading are refused")
    kind = validate_readout_config(config)
    validate_environment()
    from bwmn import lm
    import torch
    physical, commit = lm.host_ram_free_gb()
    passed = physical >= 10.5 and commit >= 10.5
    _LAST_ADMISSION_CHECK = {
        "config_kind": kind, "physical_available_gib": physical, "commit_available_gib": commit,
        "required_physical_gib": 10.5, "required_commit_gib": 10.5,
        "physical_qualified": physical >= 10.5, "commit_qualified": commit >= 10.5,
        "passed": passed, "status": "admitted" if passed else "refused",
        "measurement_stage": "after BWM/torch imports; before core hash/load",
    }
    if not passed:
        raise RuntimeError("native CPU10.5 singleton readout load requires at least 10.5 GiB physical and commit availability")
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


def validate_singleton_baseline(record):
    profile = record.get("resource_admission_profile", {})
    if (record.get("warm_baseline") is not True or record.get("status") != "development-guards-evaluated"
            or profile.get("name") != ADMISSION_PROFILE["name"]
            or profile.get("readout_geometry") != SINGLETON_GEOMETRY
            or profile.get("readout_only") is not True
            or type(profile.get("max_sampler_requests_per_call")) is not int
            or profile.get("max_sampler_requests_per_call") != 1):
        raise ValueError("New completed singleton warm guard reference required; old broad-batch results stay separate")


from .recovery import LegacyPredictor


class SingletonLegacyPredictor(LegacyPredictor):
    """Only resource geometry changes; all inherited parsing/traces/counters remain."""
    def predict_batch(self, inputs):
        outputs = []
        for item in inputs:
            outputs.extend(super().predict_batch([item]))
        return outputs
