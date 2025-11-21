"""Core pipeline and orchestration components."""

from .device_manager import select_device
from .pipeline import PipelineResult, run_pipeline

__all__ = ["select_device", "PipelineResult", "run_pipeline"]