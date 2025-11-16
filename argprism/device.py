"""Device selection utilities for ARGprism."""

from __future__ import annotations

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Return the torch device to use for computation."""
    if preferred:
        normalized = preferred.lower()
        if normalized not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device '{preferred}'")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        return torch.device(normalized)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
