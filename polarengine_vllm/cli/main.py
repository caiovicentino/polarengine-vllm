"""PolarQuant CLI — main entry point.

Usage:
    polarquant <command> [options]

Commands:
    chat        Start Gradio chat with a PolarQuant model
    quantize    Quantize a HuggingFace model with PolarQuant Q5+INT4
    serve       Start OpenAI-compatible API server
    bench       Benchmark against FP16/torchao/BnB
    info        Show model specs and VRAM estimate
    gguf        Convert to GGUF for ollama/llama.cpp
    monitor     Check for new models to quantize
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="polarquant",
        description="🧊 PolarQuant — LLM compression for consumer GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  polarquant chat google/gemma-4-31B-it
  polarquant quantize google/gemma-4-31B-it --output caiovicentino1/model
  polarquant info Qwen/Qwen3.5-27B --nbits 3
  polarquant serve caiovicentino1/model --port 8000
  polarquant bench google/gemma-4-31B-it
  polarquant gguf caiovicentino1/model --quant Q4_K_M
  polarquant monitor --check-new

📄 Paper: https://arxiv.org/abs/2603.29078
💻 GitHub: https://github.com/caiovicentino/polarengine-vllm
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── chat ──────────────────────────────────────────────────
    p_chat = subparsers.add_parser("chat", help="Start Gradio chat UI")
    p_chat.add_argument("model", help="HuggingFace model name or path")
    p_chat.add_argument("--nbits-kv", type=int, default=3, choices=[2, 3, 4],
                        help="KV cache quantization bits (default: 3)")
    p_chat.add_argument("--no-share", action="store_true",
                        help="Don't create a public Gradio link")
    p_chat.add_argument("--vision", action="store_true",
                        help="Enable multimodal (image+text) mode")
    p_chat.add_argument("--port", type=int, default=7860,
                        help="Gradio server port")
    p_chat.add_argument("--max-tokens", type=int, default=512,
                        help="Max generation tokens")
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top-p", type=float, default=0.9)

    # ── quantize ──────────────────────────────────────────────
    p_quant = subparsers.add_parser("quantize", help="Quantize a model with PolarQuant Q5+INT4")
    p_quant.add_argument("model", help="HuggingFace model name")
    p_quant.add_argument("--output", "-o", help="Output HF repo (default: auto)")
    p_quant.add_argument("--nbits", type=int, default=5, choices=[3, 4, 5],
                         help="Weight quantization bits (default: 5)")
    p_quant.add_argument("--no-int4", action="store_true",
                         help="Skip torchao INT4 (keep BF16 after PQ dequant)")
    p_quant.add_argument("--vision", action="store_true",
                         help="Keep vision encoder in BF16")
    p_quant.add_argument("--upload", action="store_true",
                         help="Upload to HuggingFace after quantization")
    p_quant.add_argument("--save-dir", default=None,
                         help="Local save directory")

    # ── serve ─────────────────────────────────────────────────
    p_serve = subparsers.add_parser("serve", help="Start OpenAI-compatible API server")
    p_serve.add_argument("model", help="HuggingFace model name or path")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--nbits-kv", type=int, default=3, choices=[2, 3, 4])
    p_serve.add_argument("--vision", action="store_true")

    # ── bench ─────────────────────────────────────────────────
    p_bench = subparsers.add_parser("bench", help="Benchmark quantization methods")
    p_bench.add_argument("model", help="HuggingFace model name")
    p_bench.add_argument("--methods", nargs="+",
                         default=["fp16", "polarquant", "torchao", "bnb"],
                         help="Methods to compare")
    p_bench.add_argument("--tokens", type=int, default=100,
                         help="Tokens per benchmark run")
    p_bench.add_argument("--runs", type=int, default=3)

    # ── info ──────────────────────────────────────────────────
    p_info = subparsers.add_parser("info", help="Show model specs and VRAM estimate")
    p_info.add_argument("model", help="HuggingFace model name")
    p_info.add_argument("--nbits", type=int, default=5, choices=[3, 4, 5])

    # ── gguf ──────────────────────────────────────────────────
    p_gguf = subparsers.add_parser("gguf", help="Convert to GGUF for ollama")
    p_gguf.add_argument("model", help="HuggingFace model name or PolarQuant repo")
    p_gguf.add_argument("--quant", nargs="+", default=["Q4_K_M", "Q5_K_M"],
                        help="GGUF quantization types")
    p_gguf.add_argument("--upload", action="store_true")

    # ── monitor ───────────────────────────────────────────────
    p_mon = subparsers.add_parser("monitor", help="Monitor models and find opportunities")
    p_mon.add_argument("--check-new", action="store_true",
                       help="Check for new base model releases")
    p_mon.add_argument("--stats", action="store_true",
                       help="Show download/like stats for all PolarQuant models")
    p_mon.add_argument("--opportunities", action="store_true",
                       help="Suggest high-value models to quantize")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to subcommand
    if args.command == "chat":
        from .cmd_chat import run_chat
        run_chat(args)
    elif args.command == "quantize":
        from .cmd_quantize import run_quantize
        run_quantize(args)
    elif args.command == "serve":
        from .cmd_serve import run_serve
        run_serve(args)
    elif args.command == "bench":
        from .cmd_bench import run_bench
        run_bench(args)
    elif args.command == "info":
        from .cmd_info import run_info
        run_info(args)
    elif args.command == "gguf":
        from .cmd_gguf import run_gguf
        run_gguf(args)
    elif args.command == "monitor":
        from .cmd_monitor import run_monitor
        run_monitor(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
