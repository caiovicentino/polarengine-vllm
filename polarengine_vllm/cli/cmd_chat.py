"""polarquant chat — start Gradio chat with streaming PolarQuant loader.

Usage:
    polarquant chat google/gemma-4-31B-it
    polarquant chat google/gemma-4-31B-it --vision
    polarquant chat caiovicentino1/model --nbits-kv 2
"""

from __future__ import annotations

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_model_streaming(model_id: str, vision: bool = False):
    """Load model with streaming PQ5+INT4 (per-module, fits 24GB)."""
    from transformers import AutoTokenizer, AutoProcessor
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    import torchao.quantization.utils as _tao_utils
    from scipy.stats import norm as sp_norm

    # Choose model class
    if vision:
        from transformers import AutoModelForMultimodalLM as ModelClass
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoModelForCausalLM as ModelClass
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = None

    # PolarQuant math
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
        if vision and ("vision_tower" in name or "vision_model" in name or "multi_modal_projector" in name):
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

    # Load on CPU
    print(f"Loading {model_id} on CPU...")
    t0 = time.time()
    model = ModelClass.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="sdpa",
    )
    print(f"  Loaded in {time.time() - t0:.0f}s")

    # Config
    config = model.config
    text_config = config.text_config if hasattr(config, "text_config") else config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", 128)

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

    # Move remaining to GPU
    for _, p in model.named_parameters():
        if p.device.type == "cpu":
            p.data = p.data.to("cuda")
    for _, b in model.named_buffers():
        if b.device.type == "cpu":
            b.data = b.data.to("cuda")

    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"✅ Ready! VRAM: {vram:.1f} GB")

    return model, tokenizer, processor, num_layers, num_kv_heads, head_dim


def run_chat(args):
    """Launch Gradio chat UI with PolarQuant model."""
    import gradio as gr
    from transformers import TextIteratorStreamer
    from threading import Thread

    model, tokenizer, processor, num_layers, num_kv_heads, head_dim = \
        _load_model_streaming(args.model, vision=args.vision)

    @torch.no_grad()
    def chat_fn(message, history):
        messages = list(history)

        # Handle multimodal input
        if isinstance(message, dict):
            content = []
            if message.get("files"):
                for f in message["files"]:
                    content.append({"type": "image", "url": f})
            if message.get("text"):
                content.append({"type": "text", "text": message["text"]})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": str(message)})

        # Tokenize
        if args.vision and processor is not None:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            ).to("cuda")
            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        else:
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
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.3,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        partial = ""
        for text in streamer:
            partial += text
            yield partial
        thread.join()

    vram = torch.cuda.memory_allocated() / 1e9
    title = f"🧊 {args.model.split('/')[-1]} — PolarQuant Q5+INT4"
    if args.vision:
        title += " (Vision)"

    demo = gr.ChatInterface(
        chat_fn,
        title=title,
        description=f"VRAM: {vram:.0f} GB | temp={args.temperature} top_p={args.top_p}",
        examples=[
            "Explain quantum computing in simple terms.",
            "Write a Python function for binary search.",
            "What causes the northern lights?",
        ],
        multimodal=args.vision,
        type="messages",
    )
    demo.launch(share=not args.no_share, server_port=args.port, quiet=True)
    print(f"\n🔗 Chat running on port {args.port}")
