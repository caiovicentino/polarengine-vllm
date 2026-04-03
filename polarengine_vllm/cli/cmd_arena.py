"""polarquant arena — run public benchmarks (MMLU, HumanEval, GSM8K).

Usage:
    polarquant arena google/gemma-4-31B-it --tasks mmlu gsm8k
"""

from __future__ import annotations


def run_arena(args):
    print(f"🧊 PolarQuant Arena: {args.model}")
    print(f"  Tasks: {', '.join(args.tasks)}")

    try:
        import lm_eval
    except ImportError:
        print("\n  Install lm-eval: pip install lm-eval")
        print("  Or use the Colab skill: /polarquant-arena")
        return

    from .cmd_chat import _load_model_streaming

    model, tokenizer, processor, num_layers, num_kv_heads, head_dim = \
        _load_model_streaming(args.model, vision=False)

    from lm_eval.models.huggingface import HFLM
    import torch

    print("\nWrapping model for lm-eval...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    print(f"Running benchmarks: {', '.join(args.tasks)}...")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.fewshot,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print(f"{'Task':<25} {'Score':>10} {'Stderr':>10}")
    print("-" * 60)
    for task, res in results["results"].items():
        metric = "acc,none" if "acc,none" in res else list(res.keys())[0]
        score = res.get(metric, 0)
        stderr = res.get(f"{metric}_stderr", 0)
        if isinstance(score, (int, float)):
            print(f"{task:<25} {score*100:>9.1f}% {stderr*100:>9.1f}%")
    print("=" * 60)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\nVRAM: {vram:.1f} GB")
