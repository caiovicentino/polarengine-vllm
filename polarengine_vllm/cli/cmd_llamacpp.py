"""polarquant llamacpp — apply PolarQuant Q3 KV cache to llama.cpp.

Usage:
    polarquant llamacpp --patch /path/to/llama.cpp
    polarquant llamacpp --run model.gguf -c 131072
"""

from __future__ import annotations

import os
import shutil


def run_llamacpp(args):
    print(f"🧊 PolarQuant llama.cpp KV Cache Integration")

    patch_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ggml-integration")

    if args.patch:
        llama_dir = args.patch
        if not os.path.isdir(llama_dir):
            print(f"  Error: {llama_dir} not found")
            return

        # Copy PolarQuant Q3 files
        for f in ["polar_quants.c", "polar_quants.h"]:
            src = os.path.join(patch_dir, f)
            if os.path.exists(src):
                dst = os.path.join(llama_dir, "ggml", "src", f)
                shutil.copy2(src, dst)
                print(f"  Copied {f} → {dst}")

        # Apply patch
        patch_file = os.path.join(patch_dir, "polarquant-q3.patch")
        if os.path.exists(patch_file):
            ret = os.system(f"cd {llama_dir} && git apply {patch_file}")
            if ret == 0:
                print("  Patch applied successfully!")
            else:
                print("  Patch failed (may already be applied)")

        print(f"\n  Build: cd {llama_dir} && cmake -B build && cmake --build build -j")
        print(f"  Run:   ./build/bin/llama-cli -m model.gguf --cache-type-k pq3_0 --cache-type-v pq3_0 -c 131072")

    elif args.run:
        model_path = args.run
        ctx = args.context or 131072
        llama_cli = args.llama_cli or "llama-cli"
        cmd = f"{llama_cli} -m {model_path} --cache-type-k pq3_0 --cache-type-v pq3_0 -c {ctx} -ngl 99 --interactive"
        print(f"  Running: {cmd}")
        os.system(cmd)

    else:
        print("  Usage:")
        print("    polarquant llamacpp --patch /path/to/llama.cpp")
        print("    polarquant llamacpp --run model.gguf")
        print(f"\n  PQ3 KV cache: 5.1x compression, 0.983 cosine similarity")
        print(f"  RTX 4090: ~160K context (vs 32K in FP16)")
