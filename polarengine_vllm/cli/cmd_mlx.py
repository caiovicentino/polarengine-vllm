"""polarquant mlx — convert model to MLX 4-bit for Apple Silicon.

Usage:
    polarquant mlx google/gemma-4-31B-it
"""

from __future__ import annotations


def run_mlx(args):
    import os
    print(f"🧊 PolarQuant MLX: {args.model}")

    try:
        from mlx_lm import convert
        print("Converting to MLX 4-bit...")
        output_dir = args.output or f"/tmp/polarquant-mlx-{args.model.split('/')[-1]}"
        convert(args.model, mlx_path=output_dir, quantize=True, q_bits=4)
        print(f"✅ Saved to {output_dir}")

        if args.upload:
            from huggingface_hub import HfApi
            api = HfApi()
            short = args.model.split("/")[-1]
            repo = f"caiovicentino1/{short}-PolarQuant-MLX-4bit"
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=output_dir, repo_id=repo, repo_type="model")
            print(f"✅ https://huggingface.co/{repo}")
    except ImportError:
        print("  MLX not available. Install: pip install mlx-lm")
        print("  Or use the Colab skill: /polarquant-mlx")
        print(f"\n  This command requires Apple Silicon (M1/M2/M3/M4).")
