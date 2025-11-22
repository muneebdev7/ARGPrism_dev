"""Device selection utilities for ARGprism."""

from __future__ import annotations

import os
import torch


def configure_cpu_threading() -> None:
    """Configure PyTorch threading for CPU operations."""
    # Set PyTorch to use single thread to avoid conflicts
    torch.set_num_threads(1)


def select_device(preferred: str | None = None) -> torch.device:
    """Return the torch device to use for computation."""
    if preferred:
        normalized = preferred.lower()
        if normalized not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device '{preferred}'")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        device = torch.device(normalized)
        if normalized == "cpu":
            configure_cpu_threading()
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    configure_cpu_threading()
    return torch.device("cpu")
