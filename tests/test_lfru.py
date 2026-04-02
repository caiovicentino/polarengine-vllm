"""Test LFRU vs LRU cache eviction for MoE expert offloading."""

import random
import sys
sys.path.insert(0, '.')

from polarengine_vllm.expert_cache_lfru import LFRUCache


def simulate_moe_routing(num_experts=128, top_k=6, num_layers=23,
                         cache_size=8, num_tokens=500, seed=42):
    """Simulate MoE routing with Zipf-distributed expert popularity.

    In real MoE models, some experts are much more popular than others
    (Zipf distribution). Standard LRU fails because early layers evict
    late layers' popular experts.
    """
    random.seed(seed)

    # Create Zipf-distributed expert popularity per layer
    # (some experts are 10x more popular than others)
    layer_expert_probs = []
    for _ in range(num_layers):
        probs = [1.0 / (i + 1) ** 0.8 for i in range(num_experts)]
        total = sum(probs)
        probs = [p / total for p in probs]
        # Shuffle so popular experts are different per layer
        random.shuffle(probs)
        layer_expert_probs.append(probs)

    # Simulate: LRU vs LFRU
    for policy_name, CacheClass in [("LRU", None), ("LFRU", LFRUCache)]:
        if CacheClass is None:
            # Simple LRU simulation
            from collections import OrderedDict
            caches = [OrderedDict() for _ in range(num_layers)]
        else:
            caches = [CacheClass(cache_size, decay_interval=200) for _ in range(num_layers)]

        layer_hits = [0] * num_layers
        layer_total = [0] * num_layers

        for token in range(num_tokens):
            for layer_idx in range(num_layers):
                probs = layer_expert_probs[layer_idx]
                # Select top_k experts (weighted random)
                selected = random.choices(range(num_experts), weights=probs, k=top_k)

                for eid in set(selected):
                    layer_total[layer_idx] += 1

                    if CacheClass is None:
                        # LRU simulation
                        cache = caches[layer_idx]
                        if eid in cache:
                            cache.move_to_end(eid)
                            layer_hits[layer_idx] += 1
                        else:
                            if len(cache) >= cache_size:
                                cache.popitem(last=False)
                            cache[eid] = True
                    else:
                        # LFRU
                        is_hit, _ = caches[layer_idx].access(eid)
                        if is_hit:
                            layer_hits[layer_idx] += 1

        # Print results
        print(f"\n{'='*60}")
        print(f"  {policy_name} — cache_size={cache_size}, {num_experts} experts, top-{top_k}")
        print(f"{'='*60}")
        total_hits = sum(layer_hits)
        total_accesses = sum(layer_total)
        print(f"  Overall: {total_hits}/{total_accesses} = {total_hits/total_accesses*100:.1f}%")
        print()
        print(f"  {'Layer':<8} {'Hits':>8} {'Total':>8} {'Rate':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for i in range(num_layers):
            rate = layer_hits[i] / layer_total[i] * 100 if layer_total[i] > 0 else 0
            marker = " ←" if rate < 30 else ""
            print(f"  {i:<8} {layer_hits[i]:>8} {layer_total[i]:>8} {rate:>7.1f}%{marker}")

        # Early vs late layer comparison
        early = sum(layer_hits[:5]) / sum(layer_total[:5]) * 100
        late = sum(layer_hits[-5:]) / sum(layer_total[-5:]) * 100
        print(f"\n  Early layers (0-4):  {early:.1f}%")
        print(f"  Late layers (18-22): {late:.1f}%")
        print(f"  Gap: {early - late:+.1f}pp")


if __name__ == "__main__":
    print("Nemotron-like simulation: 128 experts, top-6, 23 layers, cache=8")
    simulate_moe_routing(num_experts=128, top_k=6, num_layers=23,
                         cache_size=8, num_tokens=500)

    print("\n\n" + "="*60)
    print("  With larger cache (cache=16)")
    print("="*60)
    simulate_moe_routing(num_experts=128, top_k=6, num_layers=23,
                         cache_size=16, num_tokens=500)
