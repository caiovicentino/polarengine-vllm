"""polarquant demo — interactive Gradio demo with chat, KV cache charts, and model info.

Usage:
    polarquant demo google/gemma-4-31B-it
    polarquant demo caiovicentino1/model --share --port 7860
    polarquant demo caiovicentino1/model --kv-nbits 3
"""

from __future__ import annotations

import gc
import io
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Model Loader (reuses cmd_chat pattern)
# ═══════════════════════════════════════════════════════════════════

def _load_model(model_id: str):
    """Load PolarQuant model with streaming PQ5+INT4."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    import torchao.quantization.utils as _tao_utils
    from scipy.stats import norm as sp_norm

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    BS = 128
    _C = {}

    def get_centroids(bits):
        if bits in _C:
            return _C[bits]
        n = 1 << bits
        bd = torch.linspace(-4.0, 4.0, n + 1)
        ct = torch.zeros(n)
        for _ in range(100):
            for i in range(n):
                a, b = bd[i].item(), bd[i + 1].item()
                pa, pb = sp_norm.cdf(a), sp_norm.cdf(b)
                ct[i] = (sp_norm.pdf(a) - sp_norm.pdf(b)) / (pb - pa) if pb - pa > 1e-12 else (a + b) / 2
            for i in range(1, n):
                bd[i] = (ct[i - 1] + ct[i]) / 2
        _C[bits] = ct
        return ct

    def _build_H(n):
        if n == 1:
            return torch.tensor([[1.0]])
        h = _build_H(n // 2)
        return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)

    def should_quantize(name, param):
        if param.ndim < 2 or param.numel() < 256:
            return False
        if any(k in name for k in ["norm", "layernorm", "rmsnorm"]):
            return False
        if "bias" in name and param.ndim == 1:
            return False
        if name.endswith(".gate.weight") or "router" in name:
            return False
        return True

    # Guard patch
    _orig = _tao_utils.guard_dtype_size
    def _patched(t, n, dtype=None, size=None):
        if dtype is not None and t.dtype != dtype:
            t.data = t.data.to(dtype)
        if size is not None and t.size() != size:
            raise ValueError(f"{size} vs {t.size()}")
    _tao_utils.guard_dtype_size = _patched

    print(f"Loading {model_id} on CPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa",
    )
    print(f"  Loaded in {time.time() - t0:.0f}s")

    config = model.config
    text_config = config.text_config if hasattr(config, "text_config") else config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", 128)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9

    # Streaming PQ5+INT4
    print("PQ5 dequant + INT4 (streaming)...")
    H_dev = _build_H(BS).to("cuda")
    ct5 = get_centroids(5).to("cuda")
    int4_config = Int4WeightOnlyConfig(group_size=128)
    n_q = 0
    t0 = time.time()

    for name, child in list(model.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if child.weight.device.type == "meta":
            continue
        if not should_quantize(name, child.weight):
            continue

        w = child.weight.data.float().to("cuda")
        out_f, in_f = w.shape
        pad = (BS - in_f % BS) % BS
        if pad > 0:
            w = F.pad(w, (0, pad))
        nb = w.shape[1] // BS
        w = w.reshape(out_f, nb, BS)

        for i in range(0, out_f, 64):
            e = min(i + 64, out_f)
            w[i:e] = (w[i:e].reshape(-1, BS) @ H_dev).reshape(e - i, nb, BS)

        norms = w.norm(dim=2, keepdim=True).clamp(min=1e-10)
        w.div_(norms).mul_(math.sqrt(BS))

        QC = 256
        codes = torch.empty(out_f, nb, BS, dtype=torch.int8, device="cuda")
        for ci in range(0, out_f, QC):
            ce = min(ci + QC, out_f)
            codes[ci:ce] = (w[ci:ce].unsqueeze(-1) - ct5.view(1, 1, 1, -1)).abs().argmin(-1).to(torch.int8)

        del w
        vals = torch.empty(out_f, nb, BS, dtype=torch.float32, device="cuda")
        for ci in range(0, out_f, QC):
            ce = min(ci + QC, out_f)
            vals[ci:ce] = ct5[codes[ci:ce].long()] / math.sqrt(BS)
        del codes
        torch.cuda.empty_cache()

        for i in range(0, out_f, 64):
            e = min(i + 64, out_f)
            vals[i:e] = (vals[i:e].reshape(-1, BS) @ H_dev).reshape(e - i, nb, BS)
        vals *= norms
        del norms
        bf16_w = vals.reshape(out_f, -1)[:, :in_f].to(torch.bfloat16)
        del vals
        torch.cuda.empty_cache()

        try:
            with torch.device("meta"):
                dummy = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
            dummy[0].weight = nn.Parameter(bf16_w)
            quantize_(dummy, int4_config)
            child.weight = dummy[0].weight
            del dummy
        except Exception:
            child.weight.data = bf16_w.cpu()
        del bf16_w
        torch.cuda.empty_cache()

        n_q += 1
        if n_q % 100 == 0:
            print(f"  {n_q} layers ({torch.cuda.memory_allocated() / 1e9:.1f} GB)...")

    _tao_utils.guard_dtype_size = _orig
    print(f"  {n_q} layers quantized in {time.time() - t0:.0f}s")

    for _, p in model.named_parameters():
        if p.device.type == "cpu":
            p.data = p.data.to("cuda")
    for _, b in model.named_buffers():
        if b.device.type == "cpu":
            b.data = b.data.to("cuda")

    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Ready! VRAM: {vram:.1f} GB")

    polar_config = {
        "model_id": model_id,
        "num_params_b": num_params,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "n_quantized_layers": n_q,
        "vram_gb": vram,
        "quant_method": "PQ5 + INT4 (Walsh-Hadamard + Lloyd-Max)",
    }

    return model, tokenizer, polar_config


# ═══════════════════════════════════════════════════════════════════
# KV Cache Memory Calculations
# ═══════════════════════════════════════════════════════════════════

def _kv_memory_gb(num_layers: int, num_kv_heads: int, head_dim: int,
                  seq_len: int, nbits: int = 16) -> float:
    """Calculate KV cache memory in GB for a given configuration."""
    if nbits == 16:
        # FP16: 2 bytes per element, K + V
        bytes_total = num_layers * 2 * num_kv_heads * head_dim * seq_len * 2
    else:
        # PolarQuant: nbits per value + 2-byte norm per vector
        bits_per_vec = head_dim * nbits
        norm_bytes = 2
        bytes_per_vec = bits_per_vec / 8 + norm_bytes
        bytes_total = num_layers * 2 * num_kv_heads * bytes_per_vec * seq_len
    return bytes_total / (1024 ** 3)


def _build_kv_chart(num_layers: int, num_kv_heads: int, head_dim: int):
    """Build matplotlib chart comparing KV cache memory at different context lengths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ctx_lengths = [4096, 16384, 65536, 131072]
    labels = ["4K", "16K", "64K", "128K"]

    fp16_mem = [_kv_memory_gb(num_layers, num_kv_heads, head_dim, s, 16) for s in ctx_lengths]
    q4_mem = [_kv_memory_gb(num_layers, num_kv_heads, head_dim, s, 4) for s in ctx_lengths]
    q3_mem = [_kv_memory_gb(num_layers, num_kv_heads, head_dim, s, 3) for s in ctx_lengths]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars_fp16 = ax.bar(x - width, fp16_mem, width, label="FP16 (baseline)", color="#ff6b6b", edgecolor="#ff6b6b", alpha=0.85)
    bars_q4 = ax.bar(x, q4_mem, width, label="PolarQuant Q4 (4x)", color="#ffd93d", edgecolor="#ffd93d", alpha=0.85)
    bars_q3 = ax.bar(x + width, q3_mem, width, label="PolarQuant Q3 (5.3x)", color="#6bcb77", edgecolor="#6bcb77", alpha=0.85)

    # Value labels on bars
    for bars in [bars_fp16, bars_q4, bars_q3]:
        for bar in bars:
            h = bar.get_height()
            if h >= 1.0:
                label = f"{h:.1f}"
            else:
                label = f"{h:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * max(fp16_mem),
                    label, ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    ax.set_xlabel("Context Length", fontsize=12, color="white")
    ax.set_ylabel("KV Cache Memory (GB)", fontsize=12, color="white")
    ax.set_title("KV Cache Memory: FP16 vs PolarQuant", fontsize=14, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=11, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, color="#8b949e")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════
# Streaming Generator
# ═══════════════════════════════════════════════════════════════════

def _generate(message, history, model, tokenizer, max_tokens=512, temperature=0.7):
    """Streaming text generation for Gradio ChatInterface."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    messages = list(history)
    messages.append({"role": "user", "content": str(message)})

    chat_out = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    input_ids = chat_out["input_ids"] if hasattr(chat_out, "input_ids") else chat_out
    inputs = {"input_ids": input_ids.to("cuda")}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-4),
        top_p=0.9,
        repetition_penalty=1.3,
        streamer=streamer,
    )

    t_start = time.time()
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    n_tokens = 0
    for text in streamer:
        partial += text
        n_tokens += 1
        yield partial
    thread.join()

    elapsed = time.time() - t_start
    tps = n_tokens / elapsed if elapsed > 0 else 0
    vram = torch.cuda.memory_allocated() / 1e9
    partial += f"\n\n---\n*{n_tokens} tokens | {tps:.1f} tok/s | {vram:.1f} GB VRAM*"
    yield partial


# ═══════════════════════════════════════════════════════════════════
# Build Gradio Interface
# ═══════════════════════════════════════════════════════════════════

def _build_interface(model, tokenizer, polar_config, kv_nbits: Optional[int] = None):
    """Build Gradio Blocks with three tabs: Chat, KV Cache, Model Info."""
    import gradio as gr
    from PIL import Image

    model_name = polar_config["model_id"].split("/")[-1]

    # Pre-build KV chart
    kv_chart_buf = _build_kv_chart(
        polar_config["num_layers"],
        polar_config["num_kv_heads"],
        polar_config["head_dim"],
    )
    kv_chart_img = Image.open(kv_chart_buf)

    # Model info markdown
    info_md = f"""
## Model Information

| Property | Value |
|----------|-------|
| **Model** | `{polar_config['model_id']}` |
| **Parameters** | {polar_config['num_params_b']:.1f}B |
| **Layers** | {polar_config['num_layers']} |
| **KV Heads** | {polar_config['num_kv_heads']} |
| **Head Dim** | {polar_config['head_dim']} |
| **Quantized Layers** | {polar_config['n_quantized_layers']} |
| **VRAM Usage** | {polar_config['vram_gb']:.1f} GB |
| **Method** | {polar_config['quant_method']} |

## How PolarQuant Works

1. **Walsh-Hadamard Rotation** -- Distributes weight energy uniformly across dimensions,
   making all values similarly important (no outliers).

2. **Lloyd-Max Quantization** -- Optimal centroids for the resulting near-Gaussian distribution.
   5-bit = 32 levels, MSE-optimal for N(0,1).

3. **INT4 Repacking** -- After PQ5 dequant, weights are repacked into torchao INT4
   for fast GPU inference with CUDA-native kernels.

## KV Cache Compression

PolarQuant also compresses the KV cache using the same Walsh-Hadamard + Lloyd-Max approach:
- **Q3 KV**: 5.3x compression (3 bits per value)
- **Q4 KV**: 4x compression (4 bits per value)
- Enables 128K+ context on consumer GPUs

## Links

- Paper: [arxiv.org/abs/2603.29078](https://arxiv.org/abs/2603.29078)
- GitHub: [github.com/caiovicentino/polarengine-vllm](https://github.com/caiovicentino/polarengine-vllm)
- PyPI: `pip install polarquant`
"""

    # KV comparison table
    ctx_targets = [4096, 16384, 65536, 131072]
    ctx_labels = ["4K", "16K", "64K", "128K"]
    kv_rows = []
    for s, label in zip(ctx_targets, ctx_labels):
        fp16 = _kv_memory_gb(polar_config["num_layers"], polar_config["num_kv_heads"],
                             polar_config["head_dim"], s, 16)
        q4 = _kv_memory_gb(polar_config["num_layers"], polar_config["num_kv_heads"],
                           polar_config["head_dim"], s, 4)
        q3 = _kv_memory_gb(polar_config["num_layers"], polar_config["num_kv_heads"],
                           polar_config["head_dim"], s, 3)
        kv_rows.append(f"| {label} | {fp16:.2f} GB | {q4:.2f} GB ({fp16/q4:.1f}x) | {q3:.2f} GB ({fp16/q3:.1f}x) |")

    kv_table = "| Context | FP16 | PolarQuant Q4 | PolarQuant Q3 |\n"
    kv_table += "|---------|------|---------------|---------------|\n"
    kv_table += "\n".join(kv_rows)

    # Chat function with closure
    @torch.no_grad()
    def chat_fn(message, history):
        yield from _generate(message, history, model, tokenizer)

    # Build Blocks
    with gr.Blocks(
        title=f"PolarQuant Demo - {model_name}",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as demo:

        gr.Markdown(
            f"# PolarQuant Demo -- {model_name}\n"
            f"**{polar_config['num_params_b']:.1f}B params** | "
            f"**{polar_config['vram_gb']:.1f} GB VRAM** | "
            f"**{polar_config['quant_method']}**"
        )

        with gr.Tabs():
            # ── Tab 1: Chat ──────────────────────────────────────
            with gr.Tab("Chat"):
                gr.ChatInterface(
                    chat_fn,
                    type="messages",
                    examples=[
                        "Explain quantum computing in simple terms.",
                        "Write a Python function for binary search with type hints.",
                        "What are the northern lights and why do they happen?",
                        "Compare TCP and UDP in a concise table.",
                    ],
                    description=(
                        f"Chat with **{model_name}** running PolarQuant Q5+INT4. "
                        f"Using {polar_config['vram_gb']:.1f} GB VRAM."
                    ),
                )

            # ── Tab 2: KV Cache ──────────────────────────────────
            with gr.Tab("KV Cache Comparison"):
                gr.Markdown("## KV Cache Memory: FP16 vs PolarQuant\n\n"
                            "PolarQuant compresses the KV cache using Walsh-Hadamard rotation + "
                            "Lloyd-Max optimal centroids -- the same math used for weight quantization.")
                gr.Image(value=kv_chart_img, label="KV Cache Memory Comparison",
                         show_label=True, show_download_button=False)
                gr.Markdown(kv_table)
                gr.Markdown(
                    f"\n*Model: {polar_config['num_layers']} layers, "
                    f"{polar_config['num_kv_heads']} KV heads, "
                    f"head_dim={polar_config['head_dim']}*"
                )

            # ── Tab 3: Model Info ────────────────────────────────
            with gr.Tab("Model Info"):
                gr.Markdown(info_md)

    return demo


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Points
# ═══════════════════════════════════════════════════════════════════

def run_demo(args):
    """Launch Gradio demo from CLI (polarquant demo)."""
    model, tokenizer, polar_config = _load_model(args.model)

    kv_nbits = getattr(args, "kv_nbits", None)
    demo = _build_interface(model, tokenizer, polar_config, kv_nbits=kv_nbits)
    demo.launch(
        server_port=args.port,
        share=args.share,
        quiet=False,
    )


def launch_space(model_id: str, port: int = 7860, share: bool = False):
    """Launch demo for HuggingFace Spaces (app.py entry point)."""
    model, tokenizer, polar_config = _load_model(model_id)
    demo = _build_interface(model, tokenizer, polar_config)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
    )
