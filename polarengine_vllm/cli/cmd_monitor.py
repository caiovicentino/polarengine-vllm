"""polarquant monitor — track models and find opportunities.

Usage:
    polarquant monitor --stats
    polarquant monitor --check-new
    polarquant monitor --opportunities
"""

from __future__ import annotations


def run_monitor(args):
    from huggingface_hub import HfApi

    api = HfApi()

    if args.stats:
        print("🧊 PolarQuant Model Stats\n")
        models = list(api.list_models(author="caiovicentino1", sort="downloads", direction=-1))
        pq_models = [m for m in models if any(
            t in m.id.lower() for t in ["polarquant", "eoq", "polarengine"]
        )]

        print(f"{'Model':<55} {'Downloads':>10} {'Likes':>6}")
        print("─" * 75)
        for m in pq_models:
            try:
                info = api.model_info(m.id)
                dl = info.downloads or 0
                likes = info.likes or 0
            except Exception:
                dl, likes = 0, 0
            short = m.id.replace("caiovicentino1/", "")
            print(f"{short:<55} {dl:>10,} {likes:>6}")
        print(f"\nTotal: {len(pq_models)} models")

    elif args.check_new:
        print("🧊 Checking for new base models...\n")
        orgs = ["google", "Qwen", "meta-llama", "nvidia", "mistralai"]

        for org in orgs:
            try:
                models = list(api.list_models(author=org, sort="lastModified", direction=-1, limit=5))
                print(f"\n{org}:")
                for m in models:
                    print(f"  {m.id}  (updated: {str(m.lastModified)[:10]})")
            except Exception as e:
                print(f"  Error: {e}")

        print("\n  To quantize: polarquant quantize <model> --upload")

    elif args.opportunities:
        print("🧊 High-Value Quantization Opportunities\n")
        print("  Check trending models at https://huggingface.co/models?sort=trending")
        print("  Then run: polarquant info <model>")
        print("  If it fits: polarquant quantize <model> --upload")

    else:
        print("Usage: polarquant monitor [--stats|--check-new|--opportunities]")
