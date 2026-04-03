"""polarquant vllm-kv — manage PolarQuant KV cache for vLLM.

Usage:
    polarquant vllm-kv benchmark --preset gemma4-31b
    polarquant vllm-kv test
"""

from __future__ import annotations


def run_vllm_kv(args):
    print(f"🧊 PolarQuant vLLM KV Cache")

    if args.benchmark:
        from polarengine_vllm.kv_cache.benchmark import main as bench_main
        import sys
        sys.argv = ["benchmark", "--preset", args.benchmark]
        bench_main()

    elif args.test:
        import subprocess
        import sys
        test_path = os.path.join(os.path.dirname(__file__), "..", "kv_cache", "test_polar_kv.py")
        subprocess.run([sys.executable, "-m", "pytest", test_path, "-v"])

    elif args.info:
        from polarengine_vllm.kv_cache.config import PolarKVConfig
        presets = {
            "gemma4-31b": PolarKVConfig.for_gemma4_31b,
            "llama3-8b": lambda: PolarKVConfig.for_llama3(3, "8b"),
            "llama3-70b": lambda: PolarKVConfig.for_llama3(3, "70b"),
            "qwen35-9b": lambda: PolarKVConfig.for_qwen35(3, "9b"),
        }
        print("\n  Available presets:")
        for name, factory in presets.items():
            cfg = factory() if callable(factory) else factory(3)
            max_ctx = cfg.max_context(4.0)
            print(f"    {name:<15} head_dim={cfg.head_dim}, {cfg.num_layers} layers, "
                  f"{cfg.compression_ratio:.1f}x compression, {max_ctx/1000:.0f}K ctx/4GB")

    else:
        print("  Usage:")
        print("    polarquant vllm-kv --benchmark gemma4-31b")
        print("    polarquant vllm-kv --test")
        print("    polarquant vllm-kv --info")
        print(f"\n  Module: polarengine_vllm.kv_cache")
        print(f"  Files:  cache.py, config.py, attention.py, triton_kernels.py")
