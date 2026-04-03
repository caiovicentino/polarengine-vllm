"""polarquant info — show model specs and VRAM estimate.

Usage:
    polarquant info google/gemma-4-31B-it
    polarquant info Qwen/Qwen3.5-9B --nbits 3
"""

from __future__ import annotations

import json
import math


def run_info(args):
    from huggingface_hub import HfApi, hf_hub_download

    model_id = args.model
    nbits = args.nbits
    api = HfApi()

    print(f"🧊 PolarQuant Model Info: {model_id}\n")

    # Fetch config
    try:
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error fetching config: {e}")
        return

    # Parse architecture
    text_config = config.get("text_config", config)
    num_layers = text_config.get("num_hidden_layers", "?")
    num_heads = text_config.get("num_attention_heads", "?")
    num_kv_heads = text_config.get("num_key_value_heads", num_heads)
    head_dim = text_config.get("head_dim", "?")
    hidden_size = text_config.get("hidden_size", "?")
    intermediate_size = text_config.get("intermediate_size", "?")
    vocab_size = text_config.get("vocab_size", config.get("vocab_size", "?"))
    model_type = config.get("model_type", "?")
    max_pos = text_config.get("max_position_embeddings", "?")

    # Check for vision
    has_vision = "vision_config" in config or "vision_tower" in str(config)
    vision_params = 0
    if has_vision and "vision_config" in config:
        vc = config["vision_config"]
        v_hidden = vc.get("hidden_size", 0)
        v_layers = vc.get("num_hidden_layers", 0)
        v_inter = vc.get("intermediate_size", 0)
        vision_params = v_layers * (4 * v_hidden * v_hidden + 2 * v_hidden * v_inter)

    # Estimate total params
    if isinstance(num_layers, int) and isinstance(hidden_size, int):
        attn_params = num_layers * (
            hidden_size * (num_heads * head_dim if isinstance(head_dim, int) else hidden_size)  # q
            + hidden_size * (num_kv_heads * head_dim if isinstance(head_dim, int) else hidden_size) * 2  # k, v
            + (num_heads * head_dim if isinstance(head_dim, int) else hidden_size) * hidden_size  # o
        )
        mlp_params = num_layers * hidden_size * intermediate_size * 3 if isinstance(intermediate_size, int) else 0
        embed_params = vocab_size * hidden_size if isinstance(vocab_size, int) else 0
        total_params = attn_params + mlp_params + embed_params + vision_params
    else:
        total_params = 0

    total_b = total_params / 1e9

    # VRAM estimates
    bf16_gb = total_params * 2 / 1e9
    int4_gb = total_params * 0.6 / 1e9  # INT4 + scales + overhead
    pq5_int4_gb = int4_gb * 1.05  # PQ5+INT4 ≈ INT4 + small overhead

    # KV cache per token
    if isinstance(num_layers, int) and isinstance(num_kv_heads, int) and isinstance(head_dim, int):
        kv_bytes_per_tok_fp16 = num_layers * 2 * num_kv_heads * head_dim * 2
        kv_bytes_per_tok_q3 = num_layers * 2 * num_kv_heads * (head_dim * 3 / 8 + 2)
        can_hadamard = isinstance(head_dim, int) and (head_dim & (head_dim - 1)) == 0
    else:
        kv_bytes_per_tok_fp16 = 0
        kv_bytes_per_tok_q3 = 0
        can_hadamard = False

    # GPU recommendations
    gpus = [
        ("T4", 16), ("RTX 4060", 8), ("RTX 4070", 12), ("RTX 4080", 16),
        ("RTX 4090", 24), ("RTX 5090", 32), ("L4", 24),
        ("A6000", 48), ("A100 40GB", 40), ("A100 80GB", 80), ("H100", 80),
    ]

    # Print report
    print(f"{'Architecture':<20} {model_type}")
    print(f"{'Params':<20} {total_b:.1f}B" if total_b > 0 else f"{'Params':<20} ?")
    print(f"{'Layers':<20} {num_layers}")
    print(f"{'Attention heads':<20} {num_heads}")
    print(f"{'KV heads':<20} {num_kv_heads}")
    print(f"{'Head dim':<20} {head_dim}")
    print(f"{'Hidden size':<20} {hidden_size}")
    print(f"{'Vocab size':<20} {vocab_size}")
    print(f"{'Max context':<20} {max_pos}")
    print(f"{'Multimodal':<20} {'Yes' if has_vision else 'No'}")
    if has_vision:
        print(f"{'Vision params':<20} {vision_params/1e6:.0f}M")

    print(f"\n{'─' * 50}")
    print(f"{'VRAM Estimates':^50}")
    print(f"{'─' * 50}")
    print(f"{'BF16 (original)':<25} {bf16_gb:.1f} GB")
    print(f"{'PolarQuant Q{nbits}+INT4':<25} {pq5_int4_gb:.1f} GB")
    if has_vision:
        vision_gb = vision_params * 2 / 1e9
        print(f"{'  + Vision BF16':<25} +{vision_gb:.1f} GB")
        pq5_int4_gb += vision_gb

    print(f"\n{'Hadamard compatible':<25} {'✅ Yes' if can_hadamard else '❌ No'} (head_dim={head_dim})")

    if kv_bytes_per_tok_fp16 > 0:
        print(f"\n{'KV Cache (4 GB budget)':}")
        for label, bpt in [("FP16", kv_bytes_per_tok_fp16), ("Q3", kv_bytes_per_tok_q3)]:
            max_ctx = int(4 * 1024**3 / bpt) if bpt > 0 else 0
            print(f"  {label:<10} {max_ctx:>8,} tokens ({max_ctx/1000:.0f}K)")

    print(f"\n{'GPU Compatibility (PQ{nbits}+INT4)':}")
    for gpu, vram in gpus:
        if pq5_int4_gb < vram * 0.85:
            status = "✅"
        elif pq5_int4_gb < vram:
            status = "⚠️ "
        else:
            status = "❌"
        headroom = vram - pq5_int4_gb
        if headroom > 0:
            print(f"  {status} {gpu:<15} {vram:>3} GB  ({headroom:.0f} GB free for KV)")

    print(f"\n{'Recommended command':}")
    if has_vision:
        print(f"  polarquant quantize {model_id} --vision --upload")
        print(f"  polarquant chat {model_id} --vision")
    else:
        print(f"  polarquant quantize {model_id} --upload")
        print(f"  polarquant chat {model_id}")
