"""PolarQuant CLI — one command for everything.

Usage:
    polarquant chat google/gemma-4-31B-it
    polarquant quantize google/gemma-4-31B-it
    polarquant serve caiovicentino1/model
    polarquant bench google/gemma-4-31B-it
    polarquant info google/gemma-4-31B-it
    polarquant gguf caiovicentino1/model
    polarquant monitor
"""

from .main import main

__all__ = ["main"]
