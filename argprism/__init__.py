"""ARGprism package."""

from ._version import __version__
from .core.pipeline import PipelineResult, run_pipeline

__all__ = ["run_pipeline", "PipelineResult", "__version__"]
