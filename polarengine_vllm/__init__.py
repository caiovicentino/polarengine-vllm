"""
PolarEngine vLLM Plugin
========================

Quantization plugin for vLLM that enables inference with PolarQuant-compressed
models (mixed-bit, block-structured quantization with entropy-aware bit assignment).

Registration happens automatically when this package is imported. The vLLM plugin
entry point calls ``register_polar_quant()``, which triggers the import of
``PolarQuantConfig`` and its ``@register_quantization_config`` decorator.
"""

__version__ = "0.1.0"


def register_polar_quant() -> None:
    """Entry point called by vLLM's plugin system.

    Importing ``config`` is sufficient to register the quantization method
    because ``PolarQuantConfig`` uses the ``@register_quantization_config``
    decorator, which adds it to vLLM's global registry on class creation.
    """
    try:
        from polarengine_vllm.config import PolarQuantConfig  # noqa: F401
    except ImportError as exc:
        import warnings
        warnings.warn(
            f"Failed to register PolarEngine quantization with vLLM. "
            f"Ensure vLLM >= 0.8.0 is installed. Original error: {exc}"
        )
