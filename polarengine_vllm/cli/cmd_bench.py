"""polarquant bench — benchmark quantization methods.

Usage:
    polarquant bench google/gemma-4-31B-it
"""

from __future__ import annotations


def run_bench(args):
    print(f"🧊 PolarQuant Benchmark: {args.model}")
    print(f"  Methods: {', '.join(args.methods)}")
    print(f"  Tokens: {args.tokens} × {args.runs} runs")
    print(f"\n  ⚠️  Full benchmark requires GPU. Use the Colab notebook:")
    print(f"  /polarquant-bench {args.model}")
    print(f"\n  Or run: polarquant info {args.model}")
