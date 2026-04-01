#!/usr/bin/env python3
"""
PolarQuant CLI — convert PolarQuant codes to BF16 for vLLM.

Usage:
    polarquant-convert <hf_repo_or_path> <output_dir>

Downloads PolarQuant Q5 codes (31 GB for Nemotron 30B), dequants to BF16
weights, saves as standard safetensors. Output is loadable by vLLM directly.

Example:
    polarquant-convert caiovicentino1/Nemotron-Cascade-2-30B-A3B-PolarQuant-Q5 /tmp/nemotron
    vllm serve /tmp/nemotron --trust-remote-code --dtype bfloat16
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
import torch
from pathlib import Path


def build_H(n):
    if n == 1:
        return torch.tensor([[1.0]])
    h = build_H(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def dequant_codes(codes, norms, ct, block_size=128):
    """Dequant PolarQuant codes → BF16."""
    out_f = codes.shape[0]
    in_f_padded = codes.shape[1]
    n_blocks = in_f_padded // block_size
    scale = math.sqrt(block_size)
    H = build_H(block_size)

    values = ct[codes.long()] / scale
    values = values.view(out_f, n_blocks, block_size)
    for i in range(0, out_f, 64):
        end = min(i + 64, out_f)
        v = values[i:end].reshape(-1, block_size)
        values[i:end] = (v @ H).reshape(end - i, n_blocks, block_size)
    values = values * norms.float().unsqueeze(2)
    return values.reshape(out_f, -1).to(torch.bfloat16)


def convert(source: str, output_dir: str):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"PolarQuant Converter")
    print(f"  Source: {source}")
    print(f"  Output: {output_dir}")

    # Download
    print(f"\n  Downloading PolarQuant codes...", flush=True)
    t0 = time.time()
    if os.path.isdir(source):
        model_dir = source
    else:
        model_dir = snapshot_download(source)
    print(f"  Downloaded in {time.time()-t0:.0f}s")

    # Load config
    config_path = os.path.join(model_dir, "polar_config.json")
    with open(config_path) as f:
        polar_config = json.load(f)
    block_size = polar_config.get("block_size", 128)
    layers_meta = polar_config.get("layers", {})
    print(f"  {len(layers_meta)} quantized layers, block_size={block_size}")

    # Load weight index
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    # Group keys by shard file
    shard_keys = {}
    for key, shard in weight_map.items():
        shard_keys.setdefault(shard, []).append(key)

    # Process each shard
    os.makedirs(output_dir, exist_ok=True)
    new_weight_map = {}
    total_dequanted = 0
    total_passthrough = 0

    print(f"\n  Converting...", flush=True)
    t0 = time.time()

    for shard_name, keys in sorted(shard_keys.items()):
        shard_path = os.path.join(model_dir, shard_name)
        output_tensors = {}

        # Load all tensors from this shard
        shard_tensors = {}
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                shard_tensors[key] = f.get_tensor(key)

        # Process: group codes/norms/ct by prefix, dequant
        processed_prefixes = set()

        for key in list(shard_tensors.keys()):
            if key.endswith(".codes"):
                prefix = key[:-6]
                codes = shard_tensors[key]
                norms_key = prefix + ".norms"
                ct_key = prefix + ".ct"
                ct_scaled_key = prefix + ".ct_scaled"

                norms = shard_tensors.get(norms_key)
                ct = shard_tensors.get(ct_key)
                ct_scaled = shard_tensors.get(ct_scaled_key)

                if norms is None:
                    # norms might be in a different shard — load it
                    norms_shard = weight_map.get(norms_key)
                    if norms_shard:
                        with safe_open(os.path.join(model_dir, norms_shard), framework="pt") as f2:
                            norms = f2.get_tensor(norms_key)

                if ct is None and ct_scaled is None:
                    ct_shard = weight_map.get(ct_key) or weight_map.get(ct_scaled_key)
                    if ct_shard:
                        with safe_open(os.path.join(model_dir, ct_shard), framework="pt") as f2:
                            ct = f2.get_tensor(ct_key) if ct_key in weight_map else None
                            ct_scaled = f2.get_tensor(ct_scaled_key) if ct_scaled_key in weight_map else None

                if norms is not None and (ct is not None or ct_scaled is not None):
                    if ct_scaled is not None and ct is None:
                        ct = ct_scaled * math.sqrt(block_size)

                    meta = layers_meta.get(prefix, {})
                    in_f = meta.get("in_features")

                    weight = dequant_codes(codes, norms, ct, block_size)
                    if in_f and weight.shape[1] > in_f:
                        weight = weight[:, :in_f]

                    output_tensors[prefix + ".weight"] = weight.contiguous()
                    processed_prefixes.add(prefix)
                    total_dequanted += 1

            elif key.endswith(".norms") or key.endswith(".ct") or key.endswith(".ct_scaled"):
                continue  # handled with codes
            else:
                if key not in output_tensors:
                    output_tensors[key] = shard_tensors[key].contiguous()
                    total_passthrough += 1

        # Save output shard
        if output_tensors:
            out_shard = os.path.join(output_dir, shard_name)
            save_file(output_tensors, out_shard)
            for k in output_tensors:
                new_weight_map[k] = shard_name
            print(f"    {shard_name}: {len(output_tensors)} tensors")

    # Download missing buffers from original base model
    base_model = polar_config.get("base_model")
    if base_model:
        print(f"\n  Checking for missing buffers from {base_model}...", flush=True)
        try:
            base_dir = snapshot_download(base_model,
                allow_patterns=["*.safetensors", "model.safetensors.index.json"])
            base_index_path = os.path.join(base_dir, "model.safetensors.index.json")
            if os.path.exists(base_index_path):
                with open(base_index_path) as f:
                    base_index = json.load(f)
                base_wmap = base_index["weight_map"]
                # Find keys in base that are NOT in our output
                missing = set(base_wmap.keys()) - set(new_weight_map.keys())
                if missing:
                    print(f"  Found {len(missing)} missing tensors (buffers), recovering...")
                    # Group by shard
                    miss_by_shard = {}
                    for k in missing:
                        miss_by_shard.setdefault(base_wmap[k], []).append(k)
                    extra_shard_name = "model-extra-buffers.safetensors"
                    extra_tensors = {}
                    for shard, keys in miss_by_shard.items():
                        shard_path = os.path.join(base_dir, shard)
                        with safe_open(shard_path, framework="pt") as f:
                            for k in keys:
                                if k in f.keys():
                                    extra_tensors[k] = f.get_tensor(k)
                    if extra_tensors:
                        save_file(extra_tensors, os.path.join(output_dir, extra_shard_name))
                        for k in extra_tensors:
                            new_weight_map[k] = extra_shard_name
                        print(f"  Recovered {len(extra_tensors)} buffers")
                else:
                    print(f"  No missing tensors")
        except Exception as e:
            print(f"  Warning: could not recover buffers: {e}")

    # Re-save index with any new entries
    new_index = {"metadata": weight_index.get("metadata", {}), "weight_map": new_weight_map}
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy config, tokenizer, custom code
    for fname in os.listdir(model_dir):
        if fname.endswith((".json", ".py", ".txt", ".model", ".jinja", ".tiktoken")):
            if fname not in ("model.safetensors.index.json", "polar_config.json"):
                src = os.path.join(model_dir, fname)
                dst = os.path.join(output_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # Fix tokenizer if needed
    tok_cfg = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        with open(tok_cfg) as f:
            tc = json.load(f)
        if tc.get("tokenizer_class") not in (None, "PreTrainedTokenizerFast"):
            tc["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(tok_cfg, "w") as f:
                json.dump(tc, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s!")
    print(f"  Dequanted: {total_dequanted} layers")
    print(f"  Passthrough: {total_passthrough} tensors")
    print(f"\n  Load with vLLM:")
    print(f"    vllm serve {output_dir} --trust-remote-code --dtype bfloat16")


def main():
    parser = argparse.ArgumentParser(description="Convert PolarQuant codes to BF16 for vLLM")
    parser.add_argument("source", help="HuggingFace repo ID or local path with PolarQuant codes")
    parser.add_argument("output", help="Output directory for BF16 weights")
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
