"""PolarQuant → CompressedTensors converter.

Converts PQ5 codes → dequant BF16 → INT4 symmetric group128 → pack-quantized INT32.
Output loads natively in vLLM via CompressedTensorsWNA16 → Marlin kernel (741 tok/s).

Usage:
    from polarengine_vllm.compressed_tensors_export import convert_pq5_to_compressed_tensors
    convert_pq5_to_compressed_tensors("caiovicentino1/Model-PQ5", "caiovicentino1/Model-CT-INT4")
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# INT4 Symmetric Group Quantization
# ═══════════════════════════════════════════════════════════════

def quantize_symmetric_int4_group(
    weight: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 weight → INT4 symmetric with per-group scales.

    Args:
        weight: (out_features, in_features) BF16/FP32 tensor
        group_size: quantization group size (default 128)

    Returns:
        quantized: (out_features, in_features) int8 in [-8, 7]
        scales: (out_features, num_groups) BF16 per-group scale factors
    """
    out_f, in_f = weight.shape
    num_groups = (in_f + group_size - 1) // group_size

    # Pad to multiple of group_size
    if in_f % group_size != 0:
        pad = group_size * num_groups - in_f
        weight = torch.nn.functional.pad(weight, (0, pad))

    w = weight.float().reshape(out_f, num_groups, group_size)

    # Symmetric: scale = max(|w|) / q_max
    q_max = 7  # INT4 signed: [-8, 7]
    max_abs = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-10)
    scales = (max_abs / q_max).squeeze(2)  # (out_f, num_groups)

    # Quantize
    quantized = (w / max_abs * q_max).round().clamp(-8, 7).to(torch.int8)
    quantized = quantized.reshape(out_f, -1)[:, :in_f]  # unpad

    return quantized, scales.to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════
# Pack INT8 → INT32 (CompressedTensors format)
# ═══════════════════════════════════════════════════════════════

def pack_to_int32(value: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """Pack int8 values into int32 (CompressedTensors pack-quantized format).

    Matches compressed_tensors.compressors.pack_quantized.helpers.pack_to_int32 exactly.

    Args:
        value: int8 tensor of shape (rows, cols)
        num_bits: bits per value (4 for INT4)

    Returns:
        int32 tensor of shape (rows, cols // pack_factor)
    """
    pack_factor = 32 // num_bits
    rows, cols = value.shape

    # Pad cols to multiple of pack_factor
    if cols % pack_factor != 0:
        pad = pack_factor - cols % pack_factor
        value = torch.nn.functional.pad(value, (0, pad))
        cols = value.shape[1]

    # Convert signed to unsigned: offset by 2^(num_bits-1)
    offset = 1 << (num_bits - 1)
    unsigned = (value.to(torch.int32) + offset).to(torch.uint8)

    # Reshape to groups of pack_factor
    unsigned = unsigned.reshape(rows, -1, pack_factor)

    # Pack: shift each value by i*num_bits and sum
    shifts = torch.arange(pack_factor, device=value.device) * num_bits
    packed = (unsigned.to(torch.int32) << shifts.view(1, 1, -1)).sum(dim=2)

    return packed.to(torch.int32)


# ═══════════════════════════════════════════════════════════════
# PQ5 Dequant (reused from weight_converter)
# ═══════════════════════════════════════════════════════════════

def _get_centroids(bits: int) -> torch.Tensor:
    from scipy.stats import norm as sp_norm
    n = 1 << bits
    bd = torch.linspace(-4.0, 4.0, n + 1)
    ct = torch.zeros(n)
    for _ in range(100):
        for i in range(n):
            a, b = bd[i].item(), bd[i + 1].item()
            pa, pb = sp_norm.cdf(a), sp_norm.cdf(b)
            ct[i] = (sp_norm.pdf(a) - sp_norm.pdf(b)) / (pb - pa) if pb - pa > 1e-12 else (a + b) / 2
        for i in range(1, n):
            bd[i] = (ct[i - 1] + ct[i]) / 2
    return ct


def _build_hadamard(n: int) -> torch.Tensor:
    if n == 1:
        return torch.tensor([[1.0]])
    h = _build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def _unpack_5bit(packed: torch.Tensor, total: int) -> torch.Tensor:
    p = packed.long().reshape(-1, 5)
    b0, b1, b2, b3, b4 = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
    return torch.stack([
        (b0 >> 3) & 31, ((b0 & 7) << 2) | ((b1 >> 6) & 3),
        (b1 >> 1) & 31, ((b1 & 1) << 4) | ((b2 >> 4) & 15),
        ((b2 & 15) << 1) | ((b3 >> 7) & 1), (b3 >> 2) & 31,
        ((b3 & 3) << 3) | ((b4 >> 5) & 7), b4 & 31,
    ], dim=-1).reshape(-1)[:total].to(torch.uint8)


def dequant_pq5_weight(
    packed: torch.Tensor,
    norms: torch.Tensor,
    meta: torch.Tensor,
    centroids: torch.Tensor,
    H: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequant PQ5 bit-packed codes → BF16 weight."""
    out_f = int(meta[0].item()) if meta.numel() >= 1 else norms.shape[0]
    n_blocks = int(meta[1].item()) if meta.numel() >= 2 else norms.shape[1] if norms.ndim == 2 else 1
    total_codes = int(meta[3].item()) if meta.numel() >= 4 else out_f * n_blocks * block_size

    # Unpack 5-bit codes
    codes = _unpack_5bit(packed, total_codes)
    codes = codes.reshape(out_f, n_blocks, block_size)

    scale = math.sqrt(block_size)
    values = centroids[codes.long()] / scale
    values = values.reshape(out_f, n_blocks, block_size)

    # Inverse Hadamard (chunked)
    for i in range(0, out_f, 256):
        e = min(i + 256, out_f)
        v = values[i:e].reshape(-1, block_size)
        values[i:e] = (v @ H).reshape(e - i, n_blocks, block_size)

    # Scale by norms
    if norms.ndim == 2:
        values = values * norms.float().unsqueeze(2)
    else:
        values = values * norms.float().view(out_f, n_blocks, 1)

    return values.reshape(out_f, n_blocks * block_size).to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════
# Main Converter
# ═══════════════════════════════════════════════════════════════

def convert_pq5_to_compressed_tensors(
    pq5_model_id: str,
    output_dir: str,
    *,
    num_bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,
    upload_repo: Optional[str] = None,
) -> str:
    """Convert PQ5 HuggingFace model → CompressedTensors INT4 for native vLLM.

    Args:
        pq5_model_id: HF repo with PQ5 codes (e.g. "caiovicentino1/Model-PQ5")
        output_dir: Local directory to save CompressedTensors model
        num_bits: Target quantization bits (default 4)
        group_size: INT4 group size (default 128)
        block_size: PQ5 Hadamard block size (default 128)
        upload_repo: If set, upload to this HF repo after conversion

    Returns:
        Path to output directory
    """
    from huggingface_hub import snapshot_download, HfApi
    from safetensors import safe_open

    os.makedirs(output_dir, exist_ok=True)

    # Download PQ5 model
    logger.info(f"Downloading {pq5_model_id}...")
    model_dir = snapshot_download(pq5_model_id)

    # Load polar_config
    pc_path = os.path.join(model_dir, "polar_config.json")
    polar_config = {}
    if os.path.isfile(pc_path):
        with open(pc_path) as f:
            polar_config = json.load(f)
    bs = polar_config.get("block_size", block_size)

    # Precompute PQ5 math
    ct5 = _get_centroids(5)
    H = _build_hadamard(bs)

    # Iterate safetensors
    sf_files = sorted(Path(model_dir).glob("*.safetensors"))
    logger.info(f"Processing {len(sf_files)} safetensors files...")

    ct_state = {}  # CompressedTensors output
    n_converted = 0
    n_passthrough = 0
    shard_idx = 0
    SHARD_LIMIT = 5_000_000_000  # 5 GB per shard

    # Collect PQ5 components
    pending = {}

    for sf_file in sf_files:
        with safe_open(str(sf_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # PQ5 components
                if key.endswith("__packed"):
                    prefix = key[:-8]
                    pending.setdefault(prefix, {})["packed"] = tensor
                elif key.endswith("__norms"):
                    prefix = key[:-7]
                    pending.setdefault(prefix, {})["norms"] = tensor
                elif key.endswith("__meta"):
                    prefix = key[:-6]
                    pending.setdefault(prefix, {})["meta"] = tensor
                elif key.endswith(".codes"):
                    prefix = key[:-6]
                    pending.setdefault(prefix, {})["codes"] = tensor
                elif key.endswith(".norms"):
                    prefix = key[:-6]
                    pending.setdefault(prefix, {})["norms_dot"] = tensor
                elif key.endswith(".ct"):
                    prefix = key[:-3]
                    pending.setdefault(prefix, {})["ct"] = tensor
                else:
                    # BF16 passthrough
                    ct_state[key] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                    n_passthrough += 1

    # Process PQ5 → dequant → INT4 → pack
    logger.info(f"Converting {len(pending)} PQ5 layers → INT4 CompressedTensors...")

    for prefix, components in pending.items():
        # Recover layer name
        layer_name = prefix.replace("__", ".")

        # Dequant PQ5 → BF16
        if "packed" in components:
            weight = dequant_pq5_weight(
                components["packed"], components["norms"],
                components.get("meta", torch.tensor([0])),
                ct5, H, bs,
            )
        elif "codes" in components:
            codes = components["codes"]
            norms = components.get("norms_dot", components.get("norms"))
            ct_local = components.get("ct", ct5)
            if ct_local.ndim == 2:
                ct_local = ct_local[0] if ct_local.shape[0] == 1 else ct_local.mean(0)
            out_f = codes.shape[0]
            in_f_padded = codes.shape[1]
            n_blocks = in_f_padded // bs
            scale = math.sqrt(bs)
            values = ct_local[codes.long()] / scale
            values = values.reshape(out_f, n_blocks, bs)
            for i in range(0, out_f, 256):
                e = min(i + 256, out_f)
                v = values[i:e].reshape(-1, bs)
                values[i:e] = (v @ H).reshape(e - i, n_blocks, bs)
            values = values * norms.float().unsqueeze(2) if norms.ndim == 2 else values * norms.float().view(out_f, -1, 1)
            weight = values.reshape(out_f, -1).to(torch.bfloat16)
        else:
            continue

        # Trim padding
        if polar_config:
            layers_meta = polar_config.get("layers", {})
            meta = layers_meta.get(layer_name, {})
            in_f = meta.get("in_features")
            if in_f and weight.shape[1] > in_f:
                weight = weight[:, :in_f]

        out_f, in_f = weight.shape

        # Check Marlin compatibility: both dims must be divisible by 64
        # Also skip lm_head, embed_tokens, vision layers
        marlin_skip = (
            out_f % 64 != 0 or in_f % 64 != 0
            or "lm_head" in layer_name
            or "embed_tokens" in layer_name
            or "visual" in layer_name
            or "vision" in layer_name
        )

        if marlin_skip:
            # Keep as BF16 (Marlin-incompatible dimensions or special layer)
            ct_state[f"{layer_name}.weight"] = weight
            n_passthrough += 1
            continue

        # Quantize → INT4 symmetric group
        quantized, scales = quantize_symmetric_int4_group(weight, group_size)

        # Pack into INT32
        packed = pack_to_int32(quantized, num_bits)

        # Store in CompressedTensors naming
        ct_state[f"{layer_name}.weight_packed"] = packed
        ct_state[f"{layer_name}.weight_scale"] = scales
        ct_state[f"{layer_name}.weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)

        n_converted += 1
        if n_converted % 50 == 0:
            logger.info(f"  {n_converted} layers converted")

    logger.info(f"Converted {n_converted} layers, {n_passthrough} passthrough")

    # Save safetensors (shard if large)
    total_bytes = sum(t.numel() * t.element_size() for t in ct_state.values())

    if total_bytes > SHARD_LIMIT:
        # Shard
        shard = {}
        shard_bytes = 0
        shard_idx = 0
        weight_map = {}

        for key in sorted(ct_state.keys()):
            tensor = ct_state[key]
            t_bytes = tensor.numel() * tensor.element_size()

            if shard_bytes + t_bytes > SHARD_LIMIT and shard:
                fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                save_file(shard, os.path.join(output_dir, fname))
                for k in shard:
                    weight_map[k] = fname
                shard = {}
                shard_bytes = 0
                shard_idx += 1

            shard[key] = tensor
            shard_bytes += t_bytes

        if shard:
            fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(shard, os.path.join(output_dir, fname))
            for k in shard:
                weight_map[k] = fname
            shard_idx += 1

        # Fix shard names
        total_shards = shard_idx
        fixed_map = {}
        for k, v in weight_map.items():
            fixed_map[k] = v.replace("XXXXX", f"{total_shards:05d}")

        # Rename files
        for i in range(total_shards):
            old = os.path.join(output_dir, f"model-{i:05d}-of-XXXXX.safetensors")
            new = os.path.join(output_dir, f"model-{i:05d}-of-{total_shards:05d}.safetensors")
            os.rename(old, new)

        # Save index
        index = {"metadata": {"total_size": total_bytes}, "weight_map": fixed_map}
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    else:
        save_file(ct_state, os.path.join(output_dir, "model.safetensors"))

    # Copy config + tokenizer from source
    import shutil
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "generation_config.json"]:
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_dir, fname))

    # Write CompressedTensors config.json
    cfg_src = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_src):
        with open(cfg_src) as f:
            config = json.load(f)
    else:
        config = {}

    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": round(16.0 / num_bits, 3),
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                    "dynamic": False,
                    "block_structure": None,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ["lm_head", "embed_tokens", "re:visual\\..*", "re:vision\\..*",
                   "re:.*\\.in_proj_a$", "re:.*\\.in_proj_b$"],
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved CompressedTensors model to {output_dir}")
    logger.info(f"  {n_converted} layers as INT{num_bits} pack-quantized")
    logger.info(f"  {n_passthrough} layers as BF16")
    logger.info(f"  Total: {total_bytes / 1e9:.1f} GB")

    # Upload
    if upload_repo:
        api = HfApi()
        api.create_repo(upload_repo, exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=upload_repo,
            commit_message=f"PolarQuant PQ5 → CompressedTensors INT{num_bits} (native vLLM, Marlin kernel)",
        )
        logger.info(f"Uploaded to https://huggingface.co/{upload_repo}")

    return output_dir
