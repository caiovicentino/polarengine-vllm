"""polarquant collection — manage HuggingFace collection.

Usage:
    polarquant collection sync
    polarquant collection audit
    polarquant collection stats
"""

from __future__ import annotations

COLLECTION_SLUG = "caiovicentino1/polarquant-models-69cbc96292c5174df2088b08"
AUTHOR = "caiovicentino1"


def run_collection(args):
    from huggingface_hub import HfApi
    api = HfApi()

    if args.action == "sync":
        _sync(api)
    elif args.action == "audit":
        _audit(api)
    elif args.action == "stats":
        _stats(api)
    else:
        print("Usage: polarquant collection [sync|audit|stats]")


def _sync(api):
    """Ensure all PolarQuant models are in the collection."""
    print("🧊 Syncing collection...\n")
    col = api.get_collection(COLLECTION_SLUG)
    existing = {item.item_id for item in col.items}

    all_models = [m.id for m in api.list_models(author=AUTHOR)]
    pq_models = [m for m in all_models if any(
        t in m.lower() for t in ["polarquant", "eoq", "polarengine"]
    )]

    added = 0
    for m in pq_models:
        if m not in existing:
            try:
                api.add_collection_item(collection_slug=COLLECTION_SLUG,
                                        item_id=m, item_type="model")
                print(f"  + {m}")
                added += 1
            except Exception:
                pass

    print(f"\nSynced: {added} added, {len(existing)} existed, {len(pq_models)} total PQ models")


def _audit(api):
    """Check all models for missing files."""
    print("🧊 Auditing PolarQuant models...\n")
    all_models = [m.id for m in api.list_models(author=AUTHOR)]
    pq_models = [m for m in all_models if any(
        t in m.lower() for t in ["polarquant", "eoq", "polarengine"]
    )]

    for m in pq_models:
        files = set(api.list_repo_files(m))
        short = m.replace(f"{AUTHOR}/", "")
        issues = []
        if "README.md" not in files:
            issues.append("no README")
        if "polar_config.json" not in files:
            issues.append("no polar_config")
        has_weights = any(f.endswith((".pt", ".safetensors")) for f in files)
        if not has_weights:
            issues.append("no weights")
        has_notebook = any(f.endswith(".ipynb") for f in files)
        if not has_notebook:
            issues.append("no notebook")

        status = "✅" if not issues else "⚠️ "
        detail = ", ".join(issues) if issues else "complete"
        print(f"  {status} {short:<50} {detail}")


def _stats(api):
    """Show download/like stats."""
    print("🧊 PolarQuant Model Stats\n")
    all_models = list(api.list_models(author=AUTHOR, sort="downloads", direction=-1))
    pq_models = [m for m in all_models if any(
        t in m.id.lower() for t in ["polarquant", "eoq", "polarengine"]
    )]

    total_dl = 0
    print(f"  {'Model':<50} {'DL':>8} {'Likes':>6}")
    print(f"  {'─' * 68}")
    for m in pq_models:
        try:
            info = api.model_info(m.id)
            dl = info.downloads or 0
            likes = info.likes or 0
            total_dl += dl
        except Exception:
            dl, likes = 0, 0
        short = m.id.replace(f"{AUTHOR}/", "")
        print(f"  {short:<50} {dl:>8,} {likes:>6}")

    print(f"\n  Total: {len(pq_models)} models, {total_dl:,} downloads")
