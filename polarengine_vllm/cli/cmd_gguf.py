"""polarquant gguf — convert to GGUF for ollama/llama.cpp.

Usage:
    polarquant gguf google/gemma-4-31B-it --quant Q4_K_M
"""

from __future__ import annotations


def run_gguf(args):
    print(f"🧊 PolarQuant → GGUF: {args.model}")
    print(f"  Quantizations: {', '.join(args.quant)}")
    print(f"\n  ⚠️  GGUF conversion requires llama.cpp. Use the Colab notebook:")
    print(f"  /polarquant-gguf {args.model}")
    print(f"\n  Steps:")
    print(f"  1. polarquant quantize {args.model} --no-int4 --save-dir ./bf16")
    print(f"  2. python llama.cpp/convert_hf_to_gguf.py ./bf16 --outfile model.gguf")
    print(f"  3. llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M")
    print(f"  4. ollama run model-Q4_K_M.gguf")
