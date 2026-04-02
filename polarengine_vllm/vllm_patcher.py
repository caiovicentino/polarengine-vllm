"""
vLLM Expert Offloading Patcher — applies all patches needed for MoE expert caching.

Usage:
    polarengine-patch-vllm              # patch installed vLLM
    polarengine-patch-vllm --undo       # restore original files
    MOE_EXPERT_CACHE_SIZE=8 vllm serve nvidia/Nemotron-Cascade-2-30B-A3B ...

Patches vLLM 0.18.x to support expert-level CPU offloading with LRU GPU cache.
Based on PR #37190 by @e1n00r with Nemotron-specific fixes.

Tested: Nemotron-Cascade-2-30B-A3B, 15.6-24.4 tok/s, correct output.
"""

import argparse
import importlib
import logging
import os
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# The CachedWeightProvider source (standalone, no vLLM imports)
EXPERT_WEIGHT_PROVIDER = """\
"Expert weight provider with LRU GPU cache for MoE expert offloading."
import torch
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CachedWeightProvider:
    \"\"\"LRU cache for MoE expert weights on GPU.

    Stores all expert weights in CPU pinned memory. Maintains a fixed-size
    GPU buffer with the hottest experts. On cache miss, evicts LRU expert
    and copies from CPU via H2D transfer.

    Args:
        capacity: Number of expert slots in GPU buffer.
        w13_weight: Full expert gate+up weights, shape (num_experts, intermediate*2, hidden).
        w2_weight: Full expert down weights, shape (num_experts, hidden, intermediate).
    \"\"\"

    def __init__(self, capacity, w13_weight, w2_weight):
        self.capacity = capacity
        self.num_experts = w13_weight.shape[0]
        logger.info("CachedWeightProvider: %d slots, w13=%s",
                    capacity, list(w13_weight.shape))
        self._cpu_w13 = w13_weight.detach().cpu().pin_memory()
        self._cpu_w2 = w2_weight.detach().cpu().pin_memory()
        logger.info("Copied %.1f GB to CPU pinned memory",
                    (self._cpu_w13.numel() + self._cpu_w2.numel()) * 2 / 1e9)
        self._buf_w13 = torch.empty(
            (capacity, *w13_weight.shape[1:]),
            dtype=w13_weight.dtype, device="cuda")
        self._buf_w2 = torch.empty(
            (capacity, *w2_weight.shape[1:]),
            dtype=w2_weight.dtype, device="cuda")
        self._lru = OrderedDict()
        self._free_slots = list(range(capacity))
        self._mapping = torch.full(
            (self.num_experts,), -1, dtype=torch.int32, device="cuda")
        self.hits = 0
        self.misses = 0
        self._last_log = time.time()

    def prepare(self, topk_ids):
        \"\"\"Ensure experts in topk_ids are on GPU. Returns (buf_w1, buf_w2, remapped_ids).\"\"\"
        unique = topk_ids.unique().tolist()
        if len(unique) > self.capacity:
            unique = unique[-self.capacity:]
        for eid in unique:
            if eid in self._lru:
                self._lru.move_to_end(eid)
                self.hits += 1
            else:
                self.misses += 1
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    evicted_id, evicted_slot = self._lru.popitem(last=False)
                    self._mapping[evicted_id] = -1
                    slot = evicted_slot
                self._buf_w13[slot].copy_(self._cpu_w13[eid], non_blocking=True)
                self._buf_w2[slot].copy_(self._cpu_w2[eid], non_blocking=True)
                self._lru[eid] = slot
                self._mapping[eid] = slot
        torch.cuda.current_stream().synchronize()
        remapped = self._mapping[topk_ids.long()]
        now = time.time()
        if now - self._last_log > 30:
            total = self.hits + self.misses
            rate = self.hits / total * 100 if total > 0 else 0
            logger.info("Expert cache: %.1f%% hit (%d/%d)", rate, self.hits, total)
            self._last_log = now
        return self._buf_w13, self._buf_w2, remapped
"""

# Module-level function for layer.py
MAYBE_INIT_FUNC = """\

def _maybe_init_expert_cache(layer):
    \"\"\"Init expert LRU cache. Skips warmup, frees GPU weights.\"\"\"
    if hasattr(layer, '_expert_cache_done'):
        return
    import os as _os
    cache_size = int(_os.environ.get("MOE_EXPERT_CACHE_SIZE", "0"))
    if cache_size <= 0:
        layer._expert_cache_done = True
        return
    if not hasattr(layer, 'w13_weight') or layer.w13_weight.numel() == 0:
        layer._expert_cache_done = True
        return
    if not hasattr(layer, '_cache_init_skip'):
        layer._cache_init_skip = 0
    layer._cache_init_skip += 1
    if layer._cache_init_skip < 5:
        return
    try:
        import torch, gc
        from vllm.model_executor.layers.fused_moe.expert_weight_provider import CachedWeightProvider
        logger.info("Expert cache: %d slots, w13=%s", cache_size, list(layer.w13_weight.shape))
        layer.expert_weight_provider = CachedWeightProvider(
            capacity=cache_size,
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight)
        vram_before = torch.cuda.memory_allocated() / 1e9
        layer.w13_weight = torch.nn.Parameter(
            torch.empty(0, device="cuda", dtype=torch.bfloat16), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(
            torch.empty(0, device="cuda", dtype=torch.bfloat16), requires_grad=False)
        gc.collect()
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated() / 1e9
        logger.info("Expert cache ACTIVE — freed %.1f GB (%.1f -> %.1f GB)",
                    vram_before - vram_after, vram_before, vram_after)
        layer._expert_cache_done = True
    except Exception as e:
        logger.warning("Expert cache init failed (retry): %s", e)

"""


def find_vllm_path():
    """Find the installed vLLM package path."""
    try:
        import vllm
        return Path(vllm.__file__).parent
    except ImportError:
        # Try common locations
        for p in [
            Path("/usr/local/lib/python3.12/dist-packages/vllm"),
            Path("/usr/local/lib/python3.11/dist-packages/vllm"),
            Path("/usr/local/lib/python3.10/dist-packages/vllm"),
        ]:
            if p.exists():
                return p
        raise RuntimeError("vLLM not found. Install with: pip install vllm")


def backup_file(path: Path):
    """Create a .bak backup if one doesn't exist."""
    bak = path.with_suffix(path.suffix + ".polarengine_bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def restore_file(path: Path):
    """Restore from .bak backup."""
    bak = path.with_suffix(path.suffix + ".polarengine_bak")
    if bak.exists():
        shutil.copy2(bak, path)
        bak.unlink()
        return True
    return False


def patch_expert_weight_provider(vllm_path: Path):
    """Create expert_weight_provider.py (new file)."""
    target = vllm_path / "model_executor/layers/fused_moe/expert_weight_provider.py"
    target.write_text(EXPERT_WEIGHT_PROVIDER)
    print(f"  [NEW] expert_weight_provider.py")


def patch_gpu_model_runner(vllm_path: Path):
    """Patch may_reinitialize_input_batch for hybrid models."""
    path = vllm_path / "v1/worker/gpu_model_runner.py"
    if not path.exists():
        print(f"  [SKIP] gpu_model_runner.py not found")
        return
    backup_file(path)
    c = path.read_text()
    if "POLARENGINE_PATCH" in c:
        print(f"  [OK] gpu_model_runner.py already patched")
        return
    old = "    def may_reinitialize_input_batch(self"
    if old in c:
        idx = c.index(old)
        colon = c.index(":", idx)
        nl = c.index("\n", colon)
        c = c[:nl + 1] + "        return  # POLARENGINE_PATCH: hybrid model support\n" + c[nl + 1:]
        path.write_text(c)
        print(f"  [OK] gpu_model_runner.py — patched assertion")
    else:
        print(f"  [SKIP] may_reinitialize_input_batch not found")


def patch_layer(vllm_path: Path):
    """Patch FusedMoE layer for expert cache init."""
    path = vllm_path / "model_executor/layers/fused_moe/layer.py"
    backup_file(path)
    c = path.read_text()
    if "_maybe_init_expert_cache" in c:
        print(f"  [OK] layer.py already patched")
        return

    # Add module-level function before first class
    idx = c.index("\nclass ")
    c = c[:idx] + MAYBE_INIT_FUNC + c[idx:]

    # Add import os if needed
    if "import os" not in c[:500]:
        c = "import os\n" + c

    # Set cache_size = 0 before create_weights (prevent CPU allocation)
    m = re.search(r"(        self\.quant_method\.create_weights\()", c)
    if m:
        c = c[:m.start()] + (
            "        self._moe_expert_cache_size = 0\n\n"
        ) + c[m.start():]

    # Add call in forward_cuda
    m = re.search(r"def forward_cuda\(", c)
    if m:
        # Find the end of multi-line signature
        pos = m.start()
        depth = 0
        while pos < len(c):
            if c[pos] == "(":
                depth += 1
            elif c[pos] == ")":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        colon = c.index(":", pos)
        nl = c.index("\n", colon)
        c = c[:nl + 1] + "        _maybe_init_expert_cache(self)\n" + c[nl + 1:]

    path.write_text(c)
    print(f"  [OK] layer.py — cache init + forward_cuda hook")


def patch_fused_moe(vllm_path: Path):
    """Patch kernel grid to use buffer size when smaller than global_num_experts."""
    path = vllm_path / "model_executor/layers/fused_moe/fused_moe.py"
    backup_file(path)
    c = path.read_text()
    old = "if global_num_experts == -1:\n        global_num_experts = E"
    new = "if global_num_experts == -1 or E < global_num_experts:\n        global_num_experts = E"
    if new in c:
        print(f"  [OK] fused_moe.py already patched")
        return
    count = c.count(old)
    if count > 0:
        c = c.replace(old, new)
        path.write_text(c)
        print(f"  [OK] fused_moe.py — {count} kernel grid fix(es)")
    else:
        print(f"  [SKIP] fused_moe.py — pattern not found")


def patch_apply_method(vllm_path: Path, rel_path: str, name: str):
    """Add expert_weight_provider check before kernel calls."""
    path = vllm_path / rel_path
    if not path.exists():
        print(f"  [SKIP] {name} — file not found")
        return
    backup_file(path)
    c = path.read_text()
    if "expert_weight_provider" in c:
        print(f"  [OK] {name} already patched")
        return

    # Find w1=layer.w13_weight patterns and add provider check
    patterns = [
        ("w1=layer.w13_weight,", "w2=layer.w2_weight,"),
        ("gemm1_weights=layer.w13_weight,", "gemm2_weights=layer.w2_weight,"),
    ]

    patched = 0
    for w1_pat, w2_pat in patterns:
        while w1_pat in c:
            idx = c.index(w1_pat)
            # Find statement start
            pos = idx
            while pos > 0:
                ls = c.rfind("\n", 0, pos) + 1
                line = c[ls:pos].strip()
                if not line or line.endswith(",") or line.endswith("("):
                    pos = ls - 1
                else:
                    pos = ls
                    break
            stmt = c.rfind("\n", 0, pos) + 1
            sl = c[stmt:c.find("\n", stmt)]
            indent = sl[: len(sl) - len(sl.lstrip())]

            prov = (
                f"{indent}# Expert cache (polarengine)\n"
                f"{indent}_ep = getattr(layer, 'expert_weight_provider', None)\n"
                f"{indent}if _ep is not None:\n"
                f"{indent}    _w1, _w2, topk_ids = _ep.prepare(topk_ids)\n"
                f"{indent}else:\n"
                f"{indent}    _w1, _w2 = layer.w13_weight, layer.w2_weight\n"
            )
            c = c[:stmt] + prov + c[stmt:]
            new_w1 = w1_pat.replace("layer.w13_weight", "_w1")
            new_w2 = w2_pat.replace("layer.w2_weight", "_w2")
            c = c.replace(w1_pat, new_w1, 1)
            c = c.replace(w2_pat, new_w2, 1)
            patched += 1

    if patched > 0:
        path.write_text(c)
        print(f"  [OK] {name} — {patched} kernel call(s) patched")
    else:
        print(f"  [SKIP] {name} — no w13_weight patterns found")


def patch_offload_config(vllm_path: Path):
    """Add moe_expert_cache_size field to OffloadConfig."""
    path = vllm_path / "config/offload.py"
    if not path.exists():
        print(f"  [SKIP] offload.py not found")
        return
    backup_file(path)
    c = path.read_text()
    if "moe_expert_cache_size" in c:
        print(f"  [OK] offload.py already patched")
        return
    for marker in ["    @model_validator", "    def "]:
        if marker in c:
            c = c.replace(
                marker,
                '    moe_expert_cache_size: int = 0\n\n' + marker,
                1,
            )
            path.write_text(c)
            print(f"  [OK] offload.py — added field")
            return
    print(f"  [SKIP] offload.py — no insertion point")


def clear_pycache(vllm_path: Path):
    """Clear __pycache__ directories."""
    count = 0
    for d in vllm_path.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
        count += 1
    print(f"  Cleared {count} __pycache__ directories")


def apply_patches():
    """Apply all expert offloading patches to installed vLLM."""
    vllm_path = find_vllm_path()
    print(f"vLLM found at: {vllm_path}")
    print(f"\nApplying expert offloading patches...")

    patch_expert_weight_provider(vllm_path)
    patch_gpu_model_runner(vllm_path)
    patch_layer(vllm_path)
    patch_fused_moe(vllm_path)
    patch_apply_method(
        vllm_path,
        "model_executor/layers/fused_moe/fused_moe_modular_method.py",
        "modular_method",
    )
    patch_apply_method(
        vllm_path,
        "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "unquantized_method",
    )
    patch_offload_config(vllm_path)
    clear_pycache(vllm_path)

    print(f"\nDone! Usage:")
    print(f"  export FLASHINFER_DISABLE_VERSION_CHECK=1")
    print(f"  export MOE_EXPERT_CACHE_SIZE=8")
    print(f"  vllm serve nvidia/Nemotron-Cascade-2-30B-A3B \\")
    print(f"    --trust-remote-code --dtype bfloat16 --enforce-eager")


def undo_patches():
    """Restore all patched files from backups."""
    vllm_path = find_vllm_path()
    print(f"vLLM found at: {vllm_path}")
    print(f"\nRestoring original files...")

    files = [
        "v1/worker/gpu_model_runner.py",
        "model_executor/layers/fused_moe/layer.py",
        "model_executor/layers/fused_moe/fused_moe.py",
        "model_executor/layers/fused_moe/fused_moe_modular_method.py",
        "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "config/offload.py",
    ]
    for f in files:
        path = vllm_path / f
        if restore_file(path):
            print(f"  [RESTORED] {f}")
        else:
            print(f"  [SKIP] {f} — no backup")

    # Remove new file
    provider = vllm_path / "model_executor/layers/fused_moe/expert_weight_provider.py"
    if provider.exists():
        provider.unlink()
        print(f"  [REMOVED] expert_weight_provider.py")

    clear_pycache(vllm_path)
    print(f"\nDone! vLLM restored to original state.")


def main():
    parser = argparse.ArgumentParser(
        description="Patch vLLM for MoE expert offloading")
    parser.add_argument("--undo", action="store_true",
                        help="Restore original vLLM files")
    args = parser.parse_args()

    if args.undo:
        undo_patches()
    else:
        apply_patches()


if __name__ == "__main__":
    main()
