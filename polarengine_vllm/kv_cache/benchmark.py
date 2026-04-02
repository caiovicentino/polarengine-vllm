"""Benchmark PolarQuant KV cache vs FP16.

Measures:
- Quantize/dequantize latency per layer
- Memory usage at different context lengths
- Throughput (simulated tok/s impact)
- Quality (cosine similarity, MSE)

Usage:
    python -m polarengine_vllm.kv_cache.benchmark --head-dim 256 --nbits 3
    python -m polarengine_vllm.kv_cache.benchmark --preset gemma4-31b
"""

from __future__ import annotations

import argparse
import time

import torch

from .config import PolarKVConfig
from .cache import PolarKVQuantizer, PolarKVCache, BitPacker


def benchmark_latency(config: PolarKVConfig, num_tokens: int = 32, warmup: int = 5, runs: int = 20):
    """Measure per-batch quantize and dequantize latency."""
    quantizer = PolarKVQuantizer(config.head_dim, config.nbits, "cuda")

    # Simulate a batch of KV vectors: (num_tokens * num_kv_heads, head_dim)
    N = num_tokens * config.num_kv_heads
    x = torch.randn(N, config.head_dim, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        p, n = quantizer.quantize(x)
        _ = quantizer.dequantize(p, n, x.shape)

    torch.cuda.synchronize()

    # Quantize latency
    t_quant = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p, n = quantizer.quantize(x)
        torch.cuda.synchronize()
        t_quant.append(time.perf_counter() - t0)

    # Dequantize latency
    t_dequant = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = quantizer.dequantize(p, n, x.shape)
        torch.cuda.synchronize()
        t_dequant.append(time.perf_counter() - t0)

    avg_q = sum(t_quant) / len(t_quant) * 1000
    avg_d = sum(t_dequant) / len(t_dequant) * 1000

    return {
        "quantize_ms": avg_q,
        "dequantize_ms": avg_d,
        "total_ms": avg_q + avg_d,
        "num_tokens": num_tokens,
        "num_vectors": N,
    }


def benchmark_quality(config: PolarKVConfig, num_tokens: int = 1024):
    """Measure quantization quality via cosine similarity and MSE."""
    quantizer = PolarKVQuantizer(config.head_dim, config.nbits, "cuda")

    B, H, S, D = 1, config.num_kv_heads, num_tokens, config.head_dim
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    flat = x.reshape(-1, D)
    packed, norms = quantizer.quantize(x)
    recon = quantizer.dequantize(packed, norms, (B, H, S, D))
    recon_flat = recon.reshape(-1, D)

    # Cosine similarity (per-vector, then average)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat.float(), recon_flat.float(), dim=1
    ).mean().item()

    # MSE
    mse = ((flat.float() - recon_flat.float()) ** 2).mean().item()

    # Relative error
    rel_err = (
        (flat.float() - recon_flat.float()).norm()
        / flat.float().norm()
    ).item()

    return {
        "cosine_similarity": cos_sim,
        "mse": mse,
        "relative_error": rel_err,
        "num_tokens": num_tokens,
    }


def benchmark_memory(config: PolarKVConfig, context_lengths: list[int] = None):
    """Measure memory at different context lengths."""
    if context_lengths is None:
        context_lengths = [1024, 4096, 16384, 65536]

    results = []
    for ctx_len in context_lengths:
        fp16_bytes = (
            config.num_layers * 2 * config.num_kv_heads
            * config.head_dim * ctx_len * 2
        )
        polar_bytes = (
            config.num_layers * 2 * config.num_kv_heads
            * ctx_len * (config.head_dim * config.nbits / 8 + 2)
        )

        results.append({
            "context_length": ctx_len,
            "fp16_gb": fp16_bytes / 1e9,
            "polar_gb": polar_bytes / 1e9,
            "compression": fp16_bytes / polar_bytes,
            "savings_gb": (fp16_bytes - polar_bytes) / 1e9,
        })

    return results


def benchmark_bitpacker(head_dim: int = 256, runs: int = 100):
    """Benchmark BitPacker pack/unpack speed."""
    results = {}
    for nbits in (2, 3, 4):
        N = 4096
        codes = torch.randint(0, 1 << nbits, (N, head_dim), dtype=torch.int8, device="cuda")

        # Warmup
        for _ in range(5):
            p = BitPacker.pack(codes, nbits)
            _ = BitPacker.unpack(p, nbits, head_dim)

        torch.cuda.synchronize()

        # Pack
        t0 = time.perf_counter()
        for _ in range(runs):
            p = BitPacker.pack(codes, nbits)
        torch.cuda.synchronize()
        pack_ms = (time.perf_counter() - t0) / runs * 1000

        # Unpack
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = BitPacker.unpack(p, nbits, head_dim)
        torch.cuda.synchronize()
        unpack_ms = (time.perf_counter() - t0) / runs * 1000

        # Verify roundtrip
        unpacked = BitPacker.unpack(p, nbits, head_dim)
        correct = (unpacked == codes.long()).all().item()

        results[f"Q{nbits}"] = {
            "pack_ms": pack_ms,
            "unpack_ms": unpack_ms,
            "roundtrip_correct": correct,
            "packed_size": p.shape,
            "compression": codes.numel() / p.numel(),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark PolarQuant KV Cache")
    parser.add_argument("--preset", choices=["gemma4-31b", "llama3-8b", "llama3-70b", "qwen35-9b"], default=None)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--nbits", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    args = parser.parse_args()

    if args.preset:
        presets = {
            "gemma4-31b": PolarKVConfig.for_gemma4_31b,
            "llama3-8b": lambda n=args.nbits: PolarKVConfig.for_llama3(n, "8b"),
            "llama3-70b": lambda n=args.nbits: PolarKVConfig.for_llama3(n, "70b"),
            "qwen35-9b": lambda n=args.nbits: PolarKVConfig.for_qwen35(n, "9b"),
        }
        config = presets[args.preset](args.nbits) if args.preset == "gemma4-31b" else presets[args.preset]()
    else:
        config = PolarKVConfig(
            nbits=args.nbits,
            head_dim=args.head_dim,
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
        )

    print("=" * 60)
    print(f"PolarQuant KV Cache Benchmark")
    print(f"  head_dim={config.head_dim}, nbits={config.nbits}")
    print(f"  layers={config.num_layers}, kv_heads={config.num_kv_heads}")
    print(f"  compression={config.compression_ratio:.1f}x")
    print("=" * 60)

    # Latency
    print("\n--- Latency (per batch) ---")
    for batch in [1, 8, 32]:
        lat = benchmark_latency(config, num_tokens=batch)
        print(f"  {batch:>3} tokens: quant={lat['quantize_ms']:.3f}ms  "
              f"dequant={lat['dequantize_ms']:.3f}ms  "
              f"total={lat['total_ms']:.3f}ms")

    # Quality
    print("\n--- Quality ---")
    for nbits in (2, 3, 4):
        cfg = PolarKVConfig(nbits=nbits, head_dim=config.head_dim,
                            num_layers=1, num_kv_heads=config.num_kv_heads)
        q = benchmark_quality(cfg)
        print(f"  Q{nbits}: cos_sim={q['cosine_similarity']:.6f}  "
              f"rel_err={q['relative_error']:.6f}")

    # Memory
    print("\n--- Memory ---")
    print(f"  {'Context':>8}  {'FP16':>8}  {'PQ Q{}'.format(config.nbits):>8}  {'Savings':>8}  {'Ratio':>6}")
    for m in benchmark_memory(config):
        print(f"  {m['context_length']:>7}  {m['fp16_gb']:>7.2f}G  "
              f"{m['polar_gb']:>7.2f}G  {m['savings_gb']:>7.2f}G  "
              f"{m['compression']:>5.1f}x")

    # BitPacker
    print("\n--- BitPacker ---")
    bp = benchmark_bitpacker(config.head_dim)
    for name, r in bp.items():
        print(f"  {name}: pack={r['pack_ms']:.3f}ms  unpack={r['unpack_ms']:.3f}ms  "
              f"correct={r['roundtrip_correct']}")

    # Max context projections
    print("\n--- Max Context (4 GB KV budget) ---")
    for nbits in (2, 3, 4):
        cfg = PolarKVConfig(nbits=nbits, head_dim=config.head_dim,
                            num_layers=config.num_layers, num_kv_heads=config.num_kv_heads)
        max_ctx = cfg.max_context(4.0)
        print(f"  Q{nbits}: {max_ctx:>8,} tokens ({max_ctx/1000:.0f}K)")

    fp16_cfg = PolarKVConfig(nbits=4, head_dim=config.head_dim,  # dummy
                              num_layers=config.num_layers, num_kv_heads=config.num_kv_heads)
    fp16_max = int(4 * 1024**3 / (fp16_cfg.bytes_per_token(fp16=True) * config.num_layers))
    print(f"  FP16: {fp16_max:>8,} tokens ({fp16_max/1000:.0f}K)")


if __name__ == "__main__":
    main()
