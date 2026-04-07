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

    # ── demo ─────────────────────────────────────────────────
    p_demo = subparsers.add_parser("demo", help="Launch interactive Gradio demo (chat + KV charts + info)")
    p_demo.add_argument("model", help="HuggingFace model name or path")
    p_demo.add_argument("--port", type=int, default=7860,
                        help="Gradio server port (default: 7860)")
    p_demo.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    p_demo.add_argument("--kv-nbits", type=int, default=None, choices=[2, 3, 4],
                        help="KV cache quantization bits for comparison chart")

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
    p_serve.add_argument("--nbits-kv", type=int, default=3, choices=[2, 3, 4],
                        help="KV cache quantization bits (default: 3 = 5.3x compression)")
    p_serve.add_argument("--no-kv-cache", action="store_true",
                        help="Disable PolarQuant KV cache compression")
    p_serve.add_argument("--vision", action="store_true")

    # ── bench ─────────────────────────────────────────────────
    p_bench = subparsers.add_parser("bench", help="Benchmark PolarQuant (PPL, lm-eval, comparisons)")
    p_bench.add_argument("model", help="HuggingFace model name or PolarQuant repo")
    p_bench.add_argument("--ppl", action="store_true",
                         help="Compute WikiText-2 perplexity (default if nothing else requested)")
    p_bench.add_argument("--eval-tasks", type=str, default=None,
                         help="Comma-separated lm-eval tasks (e.g. mmlu,hellaswag,arc_challenge)")
    p_bench.add_argument("--compare", type=str, default=None,
                         help="Comma-separated baselines to compare (fp16, awq, gguf)")
    p_bench.add_argument("--compare-ppl", type=str, default=None,
                         help="Pre-computed PPL values (e.g. gguf=8.5,awq=7.2)")
    p_bench.add_argument("--output", "-o", type=str, default=None,
                         help="Save results to file (.md or .json)")
    p_bench.add_argument("--chart", action="store_true",
                         help="Generate matplotlib comparison chart")
    p_bench.add_argument("--max-length", type=int, default=2048,
                         help="Sliding window max length for PPL (default: 2048)")
    p_bench.add_argument("--stride", type=int, default=512,
                         help="Sliding window stride for PPL (default: 512)")
    p_bench.add_argument("--fewshot", type=int, default=5,
                         help="Few-shot count for lm-eval tasks (default: 5)")
    p_bench.add_argument("--batch-size", type=int, default=4,
                         help="Batch size for lm-eval (default: 4)")

    # ── info ──────────────────────────────────────────────────
    p_info = subparsers.add_parser("info", help="Show model specs and VRAM estimate")
    p_info.add_argument("model", help="HuggingFace model name")
    p_info.add_argument("--nbits", type=int, default=5, choices=[3, 4, 5])

    # ── export-ct ─────────────────────────────────────────────
    p_ct = subparsers.add_parser("export-ct", help="Export PQ5 → CompressedTensors INT4 (native vLLM)")
    p_ct.add_argument("model", help="PolarQuant HF repo with PQ5 codes")
    p_ct.add_argument("--output", "-o", default=None, help="Output directory")
    p_ct.add_argument("--upload", default=None, help="Upload to this HF repo")
    p_ct.add_argument("--num-bits", type=int, default=4, choices=[4, 8])
    p_ct.add_argument("--group-size", type=int, default=128)

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

    # ── mlx ───────────────────────────────────────────────────
    p_mlx = subparsers.add_parser("mlx", help="Convert to MLX 4-bit (Apple Silicon)")
    p_mlx.add_argument("model", help="HuggingFace model name")
    p_mlx.add_argument("--output", "-o", help="Output directory")
    p_mlx.add_argument("--upload", action="store_true")

    # ── llamacpp ──────────────────────────────────────────────
    p_lc = subparsers.add_parser("llamacpp", help="PolarQuant Q3 KV cache for llama.cpp")
    p_lc.add_argument("--patch", help="Path to llama.cpp repo to patch")
    p_lc.add_argument("--run", help="GGUF model path to run with PQ3 KV cache")
    p_lc.add_argument("--context", "-c", type=int, default=131072)
    p_lc.add_argument("--llama-cli", default=None, help="Path to llama-cli binary")

    # ── vllm-kv ───────────────────────────────────────────────
    p_vkv = subparsers.add_parser("vllm-kv", help="PolarQuant KV cache for vLLM")
    p_vkv.add_argument("--benchmark", help="Run benchmark (preset name)")
    p_vkv.add_argument("--test", action="store_true", help="Run unit tests")
    p_vkv.add_argument("--info", action="store_true", help="Show available presets")

    # ── arena ─────────────────────────────────────────────────
    p_arena = subparsers.add_parser("arena", help="Run public benchmarks (MMLU, HumanEval)")
    p_arena.add_argument("model", help="HuggingFace model name")
    p_arena.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k", "arc_challenge"],
                         help="Benchmark tasks")
    p_arena.add_argument("--fewshot", type=int, default=5)
    p_arena.add_argument("--batch-size", type=int, default=4)

    # ── finetune ──────────────────────────────────────────────
    p_ft = subparsers.add_parser("finetune", help="QLoRA fine-tune + re-quantize")
    p_ft.add_argument("model", help="HuggingFace model name")
    p_ft.add_argument("--dataset", required=True, help="HF dataset name")
    p_ft.add_argument("--lora-rank", type=int, default=16)
    p_ft.add_argument("--epochs", type=int, default=1)
    p_ft.add_argument("--max-samples", type=int, default=10000)
    p_ft.add_argument("--output", "-o", help="Output directory")
    p_ft.add_argument("--merge", action="store_true", help="Merge LoRA after training")

    # ── collection ────────────────────────────────────────────
    p_col = subparsers.add_parser("collection", help="Manage HuggingFace collection")
    p_col.add_argument("action", choices=["sync", "audit", "stats"],
                       help="Action: sync, audit, or stats")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to subcommand
    if args.command == "chat":
        from .cmd_chat import run_chat
        run_chat(args)
    elif args.command == "demo":
        from .cmd_demo import run_demo
        run_demo(args)
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
    elif args.command == "export-ct":
        from polarengine_vllm.compressed_tensors_export import convert_pq5_to_compressed_tensors
        import logging; logging.basicConfig(level=logging.INFO)
        output = args.output or f"/tmp/ct_{args.model.split('/')[-1]}"
        convert_pq5_to_compressed_tensors(
            args.model, output,
            num_bits=args.num_bits, group_size=args.group_size,
            upload_repo=args.upload,
        )
    elif args.command == "gguf":
        from .cmd_gguf import run_gguf
        run_gguf(args)
    elif args.command == "monitor":
        from .cmd_monitor import run_monitor
        run_monitor(args)
    elif args.command == "mlx":
        from .cmd_mlx import run_mlx
        run_mlx(args)
    elif args.command == "llamacpp":
        from .cmd_llamacpp import run_llamacpp
        run_llamacpp(args)
    elif args.command == "vllm-kv":
        from .cmd_vllm_kv import run_vllm_kv
        run_vllm_kv(args)
    elif args.command == "arena":
        from .cmd_arena import run_arena
        run_arena(args)
    elif args.command == "finetune":
        from .cmd_finetune import run_finetune
        run_finetune(args)
    elif args.command == "collection":
        from .cmd_collection import run_collection
        run_collection(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
