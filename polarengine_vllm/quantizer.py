"""
PolarQuant model quantizer: converts FP16 HuggingFace models to PolarQuant format.

Offline tool -- run once to produce a quantized checkpoint that is then served
by vLLM with the PolarEngine plugin.

Usage (CLI)::

    python -m polarengine_vllm.quantizer \\
        --model Qwen/Qwen3.5-9B \\
        --output ./polar-9b/

Usage (Python)::

    from polarengine_vllm.quantizer import PolarQuantizer
    quantizer = PolarQuantizer(block_size=128)
    quantizer.quantize_model("Qwen/Qwen3.5-9B", output_dir="./polar-9b/")
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from polarengine_vllm.utils import (
    DEFAULT_BIT_ASSIGNMENT,
    get_bits_for_layer,
    get_centroids,
    pack_codes_half_block,
)
from polarengine_vllm.packing import pack_codes_q5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hadamard matrix builder
# ---------------------------------------------------------------------------

# Try to import from Agent 3's kernels module first; fall back to a pure-Python
# recursive builder if the kernel module is not yet available.

try:
    from polarengine_vllm.kernels.fwht import build_hadamard
except ImportError:
    logger.debug(
        "polarengine_vllm.kernels.fwht not found; using built-in Hadamard builder."
    )

    _hadamard_cache: dict[int, torch.Tensor] = {}

    def build_hadamard(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Build a normalised Walsh-Hadamard matrix of size *n* (power of 2).

        The matrix is orthogonal: H @ H^T = I.
        Cached so that repeated calls with the same *n* reuse the same tensor.
        """
        if n not in _hadamard_cache:
            if n == 1:
                H = torch.tensor([[1.0]])
            else:
                h = build_hadamard(n // 2)
                H = torch.cat(
                    [torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0
                ) / math.sqrt(2)
            _hadamard_cache[n] = H

        H = _hadamard_cache[n]
        if device is not None:
            H = H.to(device)
        return H


# ---------------------------------------------------------------------------
# Safetensors helpers
# ---------------------------------------------------------------------------

def _save_sharded_safetensors(
    tensors: Dict[str, torch.Tensor],
    output_dir: str,
    max_shard_bytes: int = 5 * 1024**3,  # 5 GB
) -> Dict[str, Any]:
    """Save tensors as sharded safetensors files with an index.

    Returns the weight map (tensor-name -> filename) for the index file.
    """
    from safetensors.torch import save_file

    # Estimate shard boundaries
    shards: list[tuple[str, Dict[str, torch.Tensor]]] = []
    current_shard: Dict[str, torch.Tensor] = {}
    current_bytes = 0
    shard_idx = 1

    for name, tensor in tensors.items():
        nbytes = tensor.nelement() * tensor.element_size()
        if current_bytes + nbytes > max_shard_bytes and current_shard:
            fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            shards.append((fname, current_shard))
            current_shard = {}
            current_bytes = 0
            shard_idx += 1
        current_shard[name] = tensor
        current_bytes += nbytes

    if current_shard:
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shards.append((fname, current_shard))

    total_shards = len(shards)
    weight_map: Dict[str, str] = {}

    for i, (_, shard_tensors) in enumerate(shards):
        real_fname = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard_tensors, os.path.join(output_dir, real_fname))
        for tname in shard_tensors:
            weight_map[tname] = real_fname
        logger.info("  Saved shard %s (%d tensors)", real_fname, len(shard_tensors))

    # Write index
    index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in tensors.values())},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return weight_map


# ---------------------------------------------------------------------------
# PolarQuantizer
# ---------------------------------------------------------------------------

class PolarQuantizer:
    """Quantize a HuggingFace model to PolarQuant format.

    The quantizer walks every parameter of the model, applies PolarQuant
    block-structured quantization (normalize -> Hadamard rotate -> Lloyd-Max),
    and writes the result as sharded safetensors alongside a ``polar_config.json``
    that describes the quantization metadata for each layer.

    Usage::

        quantizer = PolarQuantizer(block_size=128)
        quantizer.quantize_model("Qwen/Qwen3.5-9B", output_dir="./polar-9b/")

    Args:
        block_size:     Number of weights per quantization block. Must be a power
                        of 2 (default 128).
        bit_assignment: Optional custom mapping of layer-name patterns to bit widths.
                        Defaults to ``DEFAULT_BIT_ASSIGNMENT``.
    """

    def __init__(
        self,
        block_size: int = 128,
        bit_assignment: Optional[Dict[str, int]] = None,
    ) -> None:
        if block_size & (block_size - 1) != 0 or block_size < 1:
            raise ValueError(f"block_size must be a positive power of 2, got {block_size}")

        self.block_size = block_size
        self.bit_assignment = bit_assignment or dict(DEFAULT_BIT_ASSIGNMENT)

    # ------------------------------------------------------------------
    # Single-tensor quantization
    # ------------------------------------------------------------------

    def quantize_tensor(
        self,
        weight: torch.Tensor,
        bits: int,
    ) -> Dict[str, torch.Tensor]:
        """PolarQuant one tensor: normalize -> Hadamard -> Lloyd-Max.

        The full pipeline for a single weight matrix:

          1. Pad the in-features dimension to a multiple of ``block_size``.
          2. View as ``(out_features, n_blocks, block_size)``.
          3. Extract L2 norms per block (clamp min 1e-10).
          4. Normalize blocks to the unit sphere.
          5. Hadamard rotate: ``blocks @ H``.
          6. Scale by ``sqrt(block_size)`` so coordinates are ~ N(0,1).
          7. Quantize: find nearest Lloyd-Max centroid (argmin over levels).
          8. Return codes (int8), norms (fp16), ct_scaled (centroids / sqrt(block_size)).

        Args:
            weight: 2-D float tensor of shape ``(out_features, in_features)``.
            bits:   Quantization bit width (2-8).

        Returns:
            Dict with keys:
              - ``"codes"``:     int8 tensor ``(out_features, in_features_padded)``
              - ``"norms"``:     fp16 tensor ``(out_features, n_blocks)``
              - ``"ct_scaled"``: fp32 tensor ``(n_levels,)`` -- centroids / sqrt(block_size)
        """
        bs = self.block_size
        device = weight.device
        ct = get_centroids(bits).to(device)
        H = build_hadamard(bs, device)

        out_f, in_f = weight.shape
        in_f_padded = ((in_f + bs - 1) // bs) * bs
        n_blocks = in_f_padded // bs
        pad = in_f_padded - in_f

        w = weight.detach().float()
        if pad > 0:
            w = F.pad(w, (0, pad))

        # (out_f, n_blocks, block_size)
        blocks = w.view(out_f, n_blocks, bs)

        # Per-block L2 norms: (out_f, n_blocks)
        norms = blocks.norm(dim=2).clamp(min=1e-10)

        # Normalize to unit sphere
        blocks_norm = blocks / norms.unsqueeze(2)

        # Allocate output codes
        all_codes = torch.empty(
            out_f, n_blocks, bs, dtype=torch.int8, device=device
        )

        # Chunk over rows (64 at a time) to avoid OOM on the argmin broadcast.
        # The argmin creates a (chunk, n_blocks, block_size, n_levels) diff tensor.
        chunk_size = 64
        scale = math.sqrt(bs)
        ct_view = ct.view(1, 1, 1, -1)

        for i in range(0, out_f, chunk_size):
            end = min(i + chunk_size, out_f)
            b = blocks_norm[i:end].reshape(-1, bs)            # (chunk*n_blocks, bs)
            b_rot = (b @ H) * scale                            # Hadamard + scale to N(0,1)
            b_rot = b_rot.view(end - i, n_blocks, bs)          # (chunk, n_blocks, bs)
            # Find nearest centroid: broadcast (chunk, n_blocks, bs, 1) vs (1,1,1, n_levels)
            diffs = (b_rot.unsqueeze(-1) - ct_view).abs()
            all_codes[i:end] = diffs.argmin(dim=-1).to(torch.int8)

        # Flatten codes back to 2-D: (out_f, in_f_padded)
        codes = all_codes.reshape(out_f, in_f_padded)
        ct_scaled = ct / scale

        return {
            "codes": codes,
            "norms": norms.half(),
            "ct_scaled": ct_scaled,
        }

    # ------------------------------------------------------------------
    # Full model quantization
    # ------------------------------------------------------------------

    def quantize_model(
        self,
        model_name: str,
        output_dir: str,
        pack_int4: bool = True,
        pack_q5: bool = True,
    ) -> None:
        """Full pipeline: load FP16 model -> quantize layers -> save safetensors.

        Steps:
          1. Load the model with ``AutoModelForCausalLM`` in FP16.
          2. Iterate ``named_parameters``.
          3. Skip layers with ``bits == 16`` (norms, biases, etc.) -- kept as FP16.
          4. Quantize each weight with :meth:`quantize_tensor`.
          5. Optionally pack Q3/Q4 codes as nibbles (``pack_int4``).
          5b. Optionally pack Q5 codes as 5-bit packed (``pack_q5``).
          6. Save as sharded safetensors (5 GB per shard).
          7. Save ``polar_config.json`` with layer metadata.
          8. Copy tokenizer and model config from the source model.

        Args:
            model_name: HuggingFace model ID or local path.
            output_dir: Directory to write the quantized model.
            pack_int4:  If True (default), nibble-pack codes for layers with bits <= 4.
            pack_q5:    If True (default), 5-bit pack codes for layers with bits == 5.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.makedirs(output_dir, exist_ok=True)

        # --- 1. Load model ---
        logger.info("Loading model: %s", model_name)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        load_s = time.time() - t0
        logger.info("  Model loaded in %.1f s", load_s)

        # --- 2/3/4. Quantize ---
        output_tensors: Dict[str, torch.Tensor] = {}
        layers_meta: Dict[str, Dict[str, Any]] = {}
        total_params = 0
        quant_params = 0
        layer_count = 0

        param_list = list(model.named_parameters())
        logger.info("  %d parameters to process", len(param_list))

        for idx, (name, param) in enumerate(param_list):
            bits = get_bits_for_layer(name, param.data, self.bit_assignment)
            total_params += param.numel()

            if bits == 16:
                # Keep as FP16
                output_tensors[name] = param.data.half().contiguous()
                continue

            if param.ndim != 2:
                # Safety: only quantize 2-D weights
                output_tensors[name] = param.data.half().contiguous()
                continue

            layer_count += 1
            quant_params += param.numel()
            out_f, in_f = param.shape
            in_f_padded = ((in_f + self.block_size - 1) // self.block_size) * self.block_size
            n_blocks = in_f_padded // self.block_size

            result = self.quantize_tensor(param.data, bits)
            codes = result["codes"]
            norms = result["norms"]
            ct_scaled = result["ct_scaled"]

            # Determine storage key prefix (drop trailing ".weight")
            key_prefix = name
            if key_prefix.endswith(".weight"):
                key_prefix = key_prefix[:-7]

            # Optionally nibble-pack codes for bits <= 4
            packed = False
            packed_q5_flag = False
            if pack_int4 and bits <= 4:
                codes = pack_codes_half_block(codes, self.block_size)
                packed = True
            elif pack_q5 and bits == 5:
                codes = pack_codes_q5(codes, self.block_size)
                packed_q5_flag = True

            output_tensors[f"{key_prefix}.codes"] = codes.contiguous()
            output_tensors[f"{key_prefix}.norms"] = norms.contiguous()
            output_tensors[f"{key_prefix}.ct_scaled"] = ct_scaled.contiguous()

            layer_meta_entry = {
                "in_features": in_f,
                "out_features": out_f,
                "in_features_padded": in_f_padded,
                "n_blocks": n_blocks,
                "bits": bits,
                "block_size": self.block_size,
                "packed": packed,
                "scale_dtype": "float16",
            }
            if packed_q5_flag:
                layer_meta_entry["packed_q5"] = True
            layers_meta[key_prefix] = layer_meta_entry

            if layer_count % 20 == 0:
                logger.info("  Quantized %d layers...", layer_count)

        logger.info(
            "  Quantization complete: %d layers quantized (%.1f M / %.1f M params)",
            layer_count,
            quant_params / 1e6,
            total_params / 1e6,
        )

        # --- 6. Save sharded safetensors ---
        logger.info("Saving sharded safetensors to %s", output_dir)
        _save_sharded_safetensors(output_tensors, output_dir, max_shard_bytes=5 * 1024**3)

        # --- 7. Save polar_config.json ---
        # Use v5 format if any layer uses Q5 packing, otherwise v4 for backward compat.
        has_q5_packed = any(
            meta.get("packed_q5", False) for meta in layers_meta.values()
        )
        format_version = "polar_engine_v5" if has_q5_packed else "polar_engine_v4"
        polar_config = {
            "format": format_version,
            "quantization": "polarengine",
            "block_size": self.block_size,
            "bit_assignment": self.bit_assignment,
            "layers": layers_meta,
        }
        config_path = os.path.join(output_dir, "polar_config.json")
        with open(config_path, "w") as f:
            json.dump(polar_config, f, indent=2)
        logger.info("  Saved %s", config_path)

        # --- 8. Save tokenizer and model config ---
        logger.info("Saving tokenizer and model config...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            tokenizer.save_pretrained(output_dir)
            logger.info("  Tokenizer saved.")
        except Exception as e:
            logger.warning("  Could not save tokenizer: %s", e)

        # Copy the model's config.json (the architecture description, not weights)
        if hasattr(model, "config"):
            model.config.save_pretrained(output_dir)
            logger.info("  Model config saved.")

        logger.info("Done. Quantized model written to: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace model to PolarQuant format."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path (e.g. Qwen/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the quantized model.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for quantization (power of 2, default 128).",
    )
    parser.add_argument(
        "--no-pack",
        action="store_true",
        help="Disable nibble-packing of Q3/Q4 codes (store as int8 instead).",
    )
    parser.add_argument(
        "--no-pack-q5",
        action="store_true",
        help="Disable 5-bit packing of Q5 codes (store as int8 instead).",
    )
    args = parser.parse_args()

    quantizer = PolarQuantizer(block_size=args.block_size)
    quantizer.quantize_model(
        args.model,
        args.output,
        pack_int4=not args.no_pack,
        pack_q5=not args.no_pack_q5,
    )
