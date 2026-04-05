"""polarquant bench — benchmark PolarQuant models (PPL, lm-eval, comparisons).

Usage:
    polarquant bench google/gemma-4-31B-it --ppl
    polarquant bench caiovicentino1/model --ppl --eval-tasks mmlu,hellaswag
    polarquant bench google/gemma-4-31B-it --ppl --compare gguf,awq,fp16 --chart
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perplexity computation — sliding-window on WikiText-2
# ---------------------------------------------------------------------------

def _compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Standard sliding-window perplexity on WikiText-2.

    Returns the test-set perplexity as a float.
    """
    from datasets import load_dataset

    print(f"  Loading dataset {dataset}/{dataset_config} ({split})...")
    ds = load_dataset(dataset, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    seq_len = input_ids.size(1)
    print(f"  Sequence length: {seq_len:,} tokens")
    print(f"  Sliding window: max_length={max_length}, stride={stride}")

    nlls: list[float] = []
    prev_end = 0
    n_windows = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # tokens we actually score
        input_chunk = input_ids[:, begin:end]

        target = input_chunk.clone()
        target[:, :-trg_len] = -100  # mask context overlap

        with torch.no_grad():
            outputs = model(input_chunk, labels=target)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.float().item())
        prev_end = end
        n_windows += 1

        if n_windows % 10 == 0:
            running_ppl = math.exp(sum(nlls) / len(nlls))
            print(f"    [{n_windows}] running PPL = {running_ppl:.2f}")

        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / len(nlls))
    print(f"  Final PPL: {ppl:.4f} ({n_windows} windows)")
    return ppl


# ---------------------------------------------------------------------------
# lm-eval harness integration
# ---------------------------------------------------------------------------

def _run_lm_eval(
    model: torch.nn.Module,
    tokenizer,
    tasks: List[str],
    num_fewshot: int = 5,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Run lm-eval tasks and return the results dict."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("  [!] lm-eval not installed. Install with: pip install lm-eval")
        return {}

    print(f"  Wrapping model for lm-eval (batch_size={batch_size})...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    print(f"  Running tasks: {', '.join(tasks)}...")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    return results


# ---------------------------------------------------------------------------
# Comparison loaders — FP16, GGUF, AWQ baselines
# ---------------------------------------------------------------------------

def _load_fp16(model_id: str):
    """Load the original FP16/BF16 model as a baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading FP16 baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model, tokenizer


def _load_awq(model_id: str):
    """Load an AWQ-quantized model if available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading AWQ model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model, tokenizer


def _load_gguf(model_id: str):
    """Placeholder for GGUF comparison (requires llama-cpp-python)."""
    print("  [!] GGUF comparison requires llama-cpp-python.")
    print("      Install: pip install llama-cpp-python")
    print("      For now, provide a pre-computed PPL via --compare-ppl gguf=<value>")
    return None, None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_results(results_dict: Dict[str, Any]) -> str:
    """Format benchmark results as a markdown table.

    results_dict structure:
    {
        "model": str,
        "methods": {
            "polarquant": {"ppl": float, "vram_gb": float, "eval": {...}},
            "fp16":       {"ppl": float, "vram_gb": float, "eval": {...}},
            ...
        }
    }
    """
    lines: list[str] = []
    model_name = results_dict.get("model", "unknown")
    methods = results_dict.get("methods", {})

    lines.append(f"# PolarQuant Benchmark: {model_name}")
    lines.append("")

    # -- Perplexity + VRAM table --
    has_ppl = any(m.get("ppl") is not None for m in methods.values())
    if has_ppl:
        lines.append("## Perplexity & VRAM")
        lines.append("")
        lines.append("| Method | PPL | VRAM (GB) | vs FP16 |")
        lines.append("|--------|----:|----------:|--------:|")

        fp16_ppl = methods.get("fp16", {}).get("ppl")
        for name, data in methods.items():
            ppl = data.get("ppl")
            vram = data.get("vram_gb")
            ppl_str = f"{ppl:.2f}" if ppl is not None else "—"
            vram_str = f"{vram:.1f}" if vram is not None else "—"

            if fp16_ppl and ppl and name != "fp16":
                delta = ((ppl - fp16_ppl) / fp16_ppl) * 100
                delta_str = f"{delta:+.1f}%"
            elif name == "fp16":
                delta_str = "baseline"
            else:
                delta_str = "—"

            lines.append(f"| {name} | {ppl_str} | {vram_str} | {delta_str} |")
        lines.append("")

    # -- lm-eval table --
    has_eval = any(m.get("eval") for m in methods.values())
    if has_eval:
        # Collect all task names
        all_tasks: set[str] = set()
        for data in methods.values():
            if data.get("eval"):
                all_tasks.update(data["eval"].keys())
        task_list = sorted(all_tasks)

        lines.append("## Benchmark Scores")
        lines.append("")
        header = "| Task |" + "|".join(f" {n} " for n in methods) + "|"
        sep = "|------|" + "|".join("------:" for _ in methods) + "|"
        lines.append(header)
        lines.append(sep)

        for task in task_list:
            row = f"| {task} |"
            for name, data in methods.items():
                score = data.get("eval", {}).get(task)
                if score is not None:
                    row += f" {score*100:.1f}% |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def _generate_chart(results_dict: Dict[str, Any], output_path: str):
    """Generate a comparison bar chart with matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [!] matplotlib not installed. Install with: pip install matplotlib")
        return

    methods = results_dict.get("methods", {})
    model_name = results_dict.get("model", "unknown")

    # Collect metrics for charting
    method_names = list(methods.keys())
    ppls = [methods[m].get("ppl") for m in method_names]
    vrams = [methods[m].get("vram_gb") for m in method_names]

    has_ppl = any(p is not None for p in ppls)
    has_vram = any(v is not None for v in vrams)

    # Gather eval tasks
    all_tasks: set[str] = set()
    for data in methods.values():
        if data.get("eval"):
            all_tasks.update(data["eval"].keys())
    task_list = sorted(all_tasks)
    has_eval = len(task_list) > 0

    n_panels = sum([has_ppl, has_vram, has_eval])
    if n_panels == 0:
        print("  [!] No data to chart.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    ax_idx = 0
    x = np.arange(len(method_names))

    if has_ppl:
        ax = axes[ax_idx]
        vals = [p if p is not None else 0 for p in ppls]
        bars = ax.bar(x, vals, color=colors[:len(method_names)], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=30, ha="right")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("WikiText-2 PPL")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax_idx += 1

    if has_vram:
        ax = axes[ax_idx]
        vals = [v if v is not None else 0 for v in vrams]
        bars = ax.bar(x, vals, color=colors[:len(method_names)], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=30, ha="right")
        ax.set_ylabel("VRAM (GB)")
        ax.set_title("Memory Usage")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax_idx += 1

    if has_eval:
        ax = axes[ax_idx]
        n_tasks = len(task_list)
        width = 0.8 / len(method_names)
        for i, name in enumerate(method_names):
            scores = []
            for task in task_list:
                s = methods[name].get("eval", {}).get(task)
                scores.append(s * 100 if s is not None else 0)
            offsets = np.arange(n_tasks) + i * width - 0.4 + width / 2
            ax.bar(offsets, scores, width=width, label=name,
                   color=colors[i % len(colors)])
        ax.set_xticks(np.arange(n_tasks))
        ax.set_xticklabels(task_list, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("lm-eval Benchmarks")
        ax.legend(fontsize=8)
        ax_idx += 1

    fig.suptitle(f"PolarQuant Benchmark — {model_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# VRAM measurement helper
# ---------------------------------------------------------------------------

def _get_vram_gb() -> Optional[float]:
    """Return current GPU memory allocated in GB, or None if not on CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_bench(args):
    """Main benchmark runner.

    Orchestrates perplexity, lm-eval, and comparison workflows.
    """
    model_id: str = args.model
    do_ppl: bool = getattr(args, "ppl", False)
    eval_tasks_raw: Optional[str] = getattr(args, "eval_tasks", None)
    compare_raw: Optional[str] = getattr(args, "compare", None)
    compare_ppl_raw: Optional[str] = getattr(args, "compare_ppl", None)
    output_file: Optional[str] = getattr(args, "output", None)
    do_chart: bool = getattr(args, "chart", False)
    max_length: int = getattr(args, "max_length", 2048)
    stride: int = getattr(args, "stride", 512)
    num_fewshot: int = getattr(args, "fewshot", 5)
    batch_size: int = getattr(args, "batch_size", 4)

    eval_tasks: List[str] = []
    if eval_tasks_raw:
        eval_tasks = [t.strip() for t in eval_tasks_raw.split(",") if t.strip()]

    compare_methods: List[str] = []
    if compare_raw:
        compare_methods = [m.strip().lower() for m in compare_raw.split(",") if m.strip()]

    # Pre-supplied PPL values for comparison methods (e.g. --compare-ppl gguf=8.5,awq=7.2)
    manual_ppl: Dict[str, float] = {}
    if compare_ppl_raw:
        for pair in compare_ppl_raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    manual_ppl[k.strip().lower()] = float(v.strip())
                except ValueError:
                    pass

    # If nothing was requested, default to PPL
    if not do_ppl and not eval_tasks and not compare_methods:
        do_ppl = True

    print(f"{'=' * 60}")
    print(f" PolarQuant Benchmark")
    print(f"{'=' * 60}")
    print(f"  Model:      {model_id}")
    print(f"  PPL:        {'yes' if do_ppl else 'no'}")
    print(f"  Eval tasks: {', '.join(eval_tasks) if eval_tasks else 'none'}")
    print(f"  Compare:    {', '.join(compare_methods) if compare_methods else 'none'}")
    print(f"{'=' * 60}")
    print()

    results: Dict[str, Any] = {
        "model": model_id,
        "methods": {},
    }

    # ---- Load PolarQuant model ----
    print("[1/4] Loading PolarQuant model...")
    t0 = time.time()

    from .cmd_chat import _load_model_streaming

    model, tokenizer, processor, num_layers, num_kv_heads, head_dim = \
        _load_model_streaming(model_id, vision=False)

    load_time = time.time() - t0
    vram = _get_vram_gb()
    print(f"  Loaded in {load_time:.1f}s | VRAM: {vram:.1f} GB" if vram else
          f"  Loaded in {load_time:.1f}s | VRAM: N/A (CPU)")

    pq_data: Dict[str, Any] = {"vram_gb": vram, "load_time_s": round(load_time, 1)}

    # ---- PPL on PolarQuant model ----
    if do_ppl:
        print()
        print("[2/4] Computing WikiText-2 perplexity (PolarQuant)...")
        pq_ppl = _compute_perplexity(
            model, tokenizer,
            max_length=max_length, stride=stride,
        )
        pq_data["ppl"] = round(pq_ppl, 4)
    else:
        pq_data["ppl"] = None

    # ---- lm-eval on PolarQuant model ----
    if eval_tasks:
        print()
        print("[3/4] Running lm-eval benchmarks (PolarQuant)...")
        eval_results = _run_lm_eval(
            model, tokenizer, eval_tasks,
            num_fewshot=num_fewshot, batch_size=batch_size,
        )
        task_scores: Dict[str, float] = {}
        for task, res in eval_results.get("results", {}).items():
            metric = "acc,none" if "acc,none" in res else list(res.keys())[0]
            score = res.get(metric, 0)
            if isinstance(score, (int, float)):
                task_scores[task] = score
        pq_data["eval"] = task_scores
    else:
        pq_data["eval"] = {}

    results["methods"]["polarquant"] = pq_data

    # ---- Free PolarQuant model before loading comparisons ----
    if compare_methods:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Comparison methods ----
    if compare_methods:
        print()
        print("[4/4] Running comparison methods...")

        for method in compare_methods:
            print(f"\n  --- {method.upper()} ---")
            comp_data: Dict[str, Any] = {"ppl": None, "vram_gb": None, "eval": {}}

            # Use manual PPL if provided
            if method in manual_ppl:
                comp_data["ppl"] = manual_ppl[method]
                print(f"  Using provided PPL: {manual_ppl[method]:.2f}")
                results["methods"][method] = comp_data
                continue

            # Try to load the comparison model
            comp_model = None
            comp_tokenizer = None

            try:
                if method == "fp16":
                    comp_model, comp_tokenizer = _load_fp16(model_id)
                elif method == "awq":
                    comp_model, comp_tokenizer = _load_awq(model_id)
                elif method == "gguf":
                    comp_model, comp_tokenizer = _load_gguf(model_id)
                else:
                    print(f"  [!] Unknown method '{method}'. Skipping.")
                    results["methods"][method] = comp_data
                    continue
            except Exception as e:
                print(f"  [!] Failed to load {method}: {e}")
                results["methods"][method] = comp_data
                continue

            if comp_model is None:
                results["methods"][method] = comp_data
                continue

            comp_data["vram_gb"] = _get_vram_gb()

            if do_ppl:
                print(f"  Computing PPL ({method})...")
                try:
                    comp_ppl = _compute_perplexity(
                        comp_model, comp_tokenizer,
                        max_length=max_length, stride=stride,
                    )
                    comp_data["ppl"] = round(comp_ppl, 4)
                except Exception as e:
                    print(f"  [!] PPL failed for {method}: {e}")

            if eval_tasks:
                print(f"  Running lm-eval ({method})...")
                try:
                    comp_eval = _run_lm_eval(
                        comp_model, comp_tokenizer, eval_tasks,
                        num_fewshot=num_fewshot, batch_size=batch_size,
                    )
                    for task, res in comp_eval.get("results", {}).items():
                        metric = "acc,none" if "acc,none" in res else list(res.keys())[0]
                        score = res.get(metric, 0)
                        if isinstance(score, (int, float)):
                            comp_data["eval"][task] = score
                except Exception as e:
                    print(f"  [!] lm-eval failed for {method}: {e}")

            results["methods"][method] = comp_data

            # Free comparison model
            del comp_model, comp_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print()
        print("[4/4] No comparison methods requested. Skipping.")

    # ---- Format and display results ----
    print()
    print("=" * 60)
    md_table = _format_results(results)
    print(md_table)
    print("=" * 60)

    # ---- Save output ----
    if output_file:
        ext = os.path.splitext(output_file)[1].lower()
        if ext == ".json":
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file} (JSON)")
        else:
            with open(output_file, "w") as f:
                f.write(md_table)
            print(f"\nResults saved to {output_file} (Markdown)")

    # ---- Chart ----
    if do_chart:
        chart_path = output_file.replace(os.path.splitext(output_file)[1], ".png") \
            if output_file else "polarquant_bench.png"
        _generate_chart(results, chart_path)

    return results
