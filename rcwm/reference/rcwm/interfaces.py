"""Native integration boundary only. No model loading is implemented here."""
from __future__ import annotations
from typing import Protocol
import torch


class FrozenHistoryEncoder(Protocol):
    checkpoint_sha256: str
    capture_version: str

    def encode_available_history(self, histories: list[dict]) -> torch.Tensor:
        """Return [batch,slots,width] from observed history only.

        Implementations must pin tokenizer/mask/precision, pass parity tests,
        remain read-only, and store cache provenance. No target leakage allowed.
        """
        ...
