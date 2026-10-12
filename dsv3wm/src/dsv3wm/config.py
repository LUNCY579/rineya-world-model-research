"""Fail-closed, typed experiment configuration."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    arm: str = "A0"
    seed: int = 0
    steps: int = 12
    batch_size: int = 1
    lr: float = 0.0001
    rank: int = 4
    max_length: int = 384
    ce_chunk: int = 8
    mtp_tokens: int = 2  # includes ordinary next-token prediction
    mtp_rank: int = 16
    mtp_weight: float = 0.3
    lambda2: float = 0.5
    lambda4: float = 0.5
    legality_weight: float = 0.1
    target_mode: str = "full"
    shuffled_aux: bool = False
    share_aux: bool = False
    device: str = "cpu"
    threads: int = 4
    core: str = "titan-sparse"
    generation_tokens: int = 128

    def __post_init__(self):
        if self.arm not in {"A0", "A1", "A2", "A3", "A4", "A5"}:
            raise ValueError("unknown arm")
        if self.device not in {"cpu", "cuda"} or self.target_mode not in {"full", "delta", "mixed"}:
            raise ValueError("invalid device/target_mode")
        if self.mtp_tokens not in {1, 2, 4}:
            raise ValueError("mtp_tokens counts main head and must be 1, 2 or 4")
        for key in ("steps", "batch_size", "rank", "max_length", "ce_chunk", "mtp_rank", "threads", "generation_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        import math
        for key in ("lr", "mtp_weight", "lambda2", "lambda4", "legality_weight"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if self.lr == 0:
            raise ValueError("lr must be positive")
        for key in ("shuffled_aux", "share_aux"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.core != "titan-sparse":
            raise ValueError("this protocol binds the Titan-Sparse parent")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def dict(self):
        return asdict(self)

    @property
    def horizons(self):
        return (1, 2, 4) if self.arm in {"A2", "A3"} else (1,)
