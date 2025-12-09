"""Shared console utilities for ARGPrism.

This module provides a centralized console interface for consistent
status message formatting across all ARGPrism modules.
"""

from __future__ import annotations

from rich.console import Console

_console = Console()


def print_status(message: str, style: str = "green") -> None:
    """Print status messages with Rich formatting.
    
    Args:
        message: The status message to print
        style: The color style (green, cyan, yellow, red, magenta)
    """
    _console.print(f"[+] {message}", style=f"bold {style}")


def get_console() -> Console:
    """Get the shared Rich console instance.
    
    Returns:
        The shared Console instance
    """
    return _console
