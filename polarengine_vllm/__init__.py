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

from polarengine_vllm.polar_model import PolarQuantModel  # noqa: F401


def register_polar_quant() -> None:
    """Entry point called by vLLM's plugin system.

    Registers PolarQuantConfig and patches vLLM's model weight loading
    to dequant PolarQuant codes on-the-fly during model loading.
    """
    try:
        from polarengine_vllm.config import PolarQuantConfig  # noqa: F401
        _patch_weight_loading()
    except ImportError as exc:
        import warnings
        warnings.warn(
            f"Failed to register PolarEngine quantization with vLLM. "
            f"Ensure vLLM >= 0.8.0 is installed. Original error: {exc}"
        )


def _patch_weight_loading():
    """Monkey-patch vLLM's DefaultModelLoader to dequant PolarQuant codes."""
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        _original_load = DefaultModelLoader.load_model

        def _patched_load(self, *args, **kwargs):
            # Check if this is a PolarQuant model
            model_config = self.load_config
            quant = getattr(getattr(self, 'model_config', None), 'quantization', None)
            if quant is None:
                # Try to find quant config from vllm_config
                try:
                    vllm_config = getattr(self, 'vllm_config', None)
                    if vllm_config:
                        quant = getattr(vllm_config.model_config, 'quantization', None)
                except:
                    pass

            if quant == 'polarengine':
                import logging
                logger = logging.getLogger("polarengine_vllm")
                logger.info("PolarEngine: patching weight iterator for on-the-fly dequant")
                # Patch _get_all_weights to wrap with dequant converter
                _orig_get_weights = self._get_all_weights

                def _wrapped_get_weights(*a, **kw):
                    from polarengine_vllm.weight_converter import polar_dequant_iterator
                    weights = _orig_get_weights(*a, **kw)
                    # Get model directory
                    model_dir = None
                    try:
                        model_dir = getattr(self, '_model_dir', None)
                        if model_dir is None:
                            from huggingface_hub import snapshot_download
                            model_name = a[0] if a else kw.get('model_name_or_path', '')
                            if model_name:
                                model_dir = snapshot_download(model_name,
                                    allow_patterns=["polar_config.json"])
                    except:
                        pass
                    if model_dir:
                        return polar_dequant_iterator(weights, model_dir)
                    return weights

                self._get_all_weights = _wrapped_get_weights

            return _original_load(self, *args, **kwargs)

        DefaultModelLoader.load_model = _patched_load
    except ImportError:
        pass  # vLLM version doesn't have DefaultModelLoader
