"""
PolarQuant Weight Converter — dequants codes to BF16 on-the-fly during vLLM loading.

Wraps the safetensors weight iterator to convert PolarQuant codes back to
standard weight tensors. This lets vLLM's standard weight loading pipeline
handle name mapping (hf_to_vllm_mapper) without knowing about PolarQuant.

Usage in model loading:
    weights = polar_dequant_iterator(original_weights, model_dir)
    model.load_weights(weights)
"""

import torch
import math
import os
import json
import logging
from typing import Iterator, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


def _build_H(n: int) -> torch.Tensor:
    """Build Walsh-Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]])
    h = _build_H(n // 2)
    return torch.cat([
        torch.cat([h, h], dim=1),
        torch.cat([h, -h], dim=1),
    ], dim=0) / math.sqrt(2)


def _dequant_weight(codes: torch.Tensor, norms: torch.Tensor,
                     ct: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequant PolarQuant codes → BF16 weight tensor.

    Args:
        codes: (out_f, in_f_padded) int8 quantization codes
        norms: (out_f, n_blocks) fp16 per-block L2 norms
        ct: (n_levels,) fp32 Lloyd-Max centroids
        block_size: PolarQuant block size (default 128)

    Returns:
        weight: (out_f, in_f_padded) BF16 dequantized weight
    """
    out_f = codes.shape[0]
    in_f_padded = codes.shape[1]
    n_blocks = in_f_padded // block_size
    scale = math.sqrt(block_size)

    H = _build_H(block_size)

    # Centroid lookup
    values = ct[codes.long()] / scale  # (out_f, in_f_padded)
    values = values.view(out_f, n_blocks, block_size)

    # Inverse Hadamard (chunked for memory)
    for i in range(0, out_f, 64):
        end = min(i + 64, out_f)
        v = values[i:end].reshape(-1, block_size)
        values[i:end] = (v @ H).reshape(end - i, n_blocks, block_size)

    # Scale by norms
    values = values * norms.float().unsqueeze(2)

    return values.reshape(out_f, in_f_padded).to(torch.bfloat16)


def polar_dequant_iterator(
    weights: Iterator[Tuple[str, torch.Tensor]],
    model_dir: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Wrap a weight iterator to dequant PolarQuant codes on-the-fly.

    For each layer that has .codes/.norms/.ct:
    - Buffers all 3 tensors
    - Dequants to BF16
    - Yields as .weight (standard format)

    For regular tensors: yields as-is.

    Args:
        weights: Original (name, tensor) iterator from safetensors
        model_dir: Path to model directory (for polar_config.json)

    Yields:
        (name, tensor) pairs with PolarQuant codes replaced by BF16 weights
    """
    # Load polar config for metadata
    config_path = os.path.join(model_dir, "polar_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            polar_config = json.load(f)
        block_size = polar_config.get("block_size", 128)
        layers_meta = polar_config.get("layers", {})
    else:
        # No polar config — pass through everything
        yield from weights
        return

    # Buffer for collecting codes/norms/ct per layer
    pending: Dict[str, Dict[str, torch.Tensor]] = {}
    # Track which layers we've yielded
    yielded_layers = set()

    n_dequanted = 0
    n_passthrough = 0

    for name, tensor in weights:
        # Check if this is a PolarQuant component
        if name.endswith(".codes"):
            prefix = name[:-6]
            pending.setdefault(prefix, {})["codes"] = tensor
        elif name.endswith(".norms"):
            prefix = name[:-6]
            pending.setdefault(prefix, {})["norms"] = tensor
        elif name.endswith(".ct"):
            prefix = name[:-3]
            pending.setdefault(prefix, {})["ct"] = tensor
        elif name.endswith(".ct_scaled"):
            prefix = name[:-10]
            pending.setdefault(prefix, {})["ct_scaled"] = tensor
        else:
            # Regular tensor — pass through
            yield name, tensor
            n_passthrough += 1
            continue

        # Check if we have all 3 components for this layer
        prefix_check = name.rsplit(".", 1)[0] if "." in name else name
        for pfx in list(pending.keys()):
            components = pending[pfx]
            has_codes = "codes" in components
            has_norms = "norms" in components
            has_ct = "ct" in components or "ct_scaled" in components

            if has_codes and has_norms and has_ct:
                # Get metadata
                meta = layers_meta.get(pfx, {})
                bs = meta.get("block_size", block_size)
                in_f = meta.get("in_features", None)

                # Get centroids (handle both ct and ct_scaled)
                if "ct_scaled" in components:
                    ct = components["ct_scaled"] * math.sqrt(bs)
                else:
                    ct = components["ct"]

                # Dequant
                weight = _dequant_weight(
                    components["codes"], components["norms"],
                    ct, block_size=bs
                )

                # Trim padding if we know original size
                if in_f is not None and weight.shape[1] > in_f:
                    weight = weight[:, :in_f]

                # Yield as .weight
                yield pfx + ".weight", weight
                n_dequanted += 1
                del pending[pfx]

    # Handle any remaining pending layers (shouldn't happen normally)
    for pfx, components in pending.items():
        if "codes" in components and "norms" in components:
            ct = components.get("ct", components.get("ct_scaled"))
            if ct is not None:
                if "ct_scaled" in components and "ct" not in components:
                    ct = ct * math.sqrt(block_size)
                meta = layers_meta.get(pfx, {})
                in_f = meta.get("in_features", None)
                weight = _dequant_weight(
                    components["codes"], components["norms"],
                    ct, block_size=block_size
                )
                if in_f is not None and weight.shape[1] > in_f:
                    weight = weight[:, :in_f]
                yield pfx + ".weight", weight
                n_dequanted += 1

    logger.info(f"PolarQuant weight converter: {n_dequanted} layers dequanted, "
                f"{n_passthrough} passed through")
