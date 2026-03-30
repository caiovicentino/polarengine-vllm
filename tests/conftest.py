"""Shared fixtures for PolarEngine tests."""

import math

import pytest
import torch


@pytest.fixture
def device():
    """Return the best available device (CUDA preferred, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def block_size():
    """Default block size used across PolarEngine."""
    return 128


@pytest.fixture
def sample_codes(device, block_size):
    """Random Q4 codes for a 4096x4096 layer.

    Returns (codes, out_f, in_f_padded, n_blocks) tuple.
    """
    bits = 4
    n_levels = 1 << bits
    out_f = 4096
    in_f = 4096
    in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
    n_blocks = in_f_padded // block_size

    codes = torch.randint(
        0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=device
    )
    return codes, out_f, in_f_padded, n_blocks


@pytest.fixture
def sample_norms(sample_codes, device):
    """Random per-block norms matching sample_codes dimensions."""
    _, out_f, _, n_blocks = sample_codes
    return (
        torch.randn(out_f, n_blocks, dtype=torch.float32, device=device).abs()
        + 0.1
    )


@pytest.fixture
def sample_centroids(device, block_size):
    """Pre-scaled centroids (4-bit) for testing."""
    bits = 4
    n_levels = 1 << bits
    ct = torch.linspace(-1.5, 1.5, n_levels, dtype=torch.float32, device=device)
    ct_scaled = ct / math.sqrt(block_size)
    return ct_scaled


@pytest.fixture
def sample_polar_config():
    """Sample polar_config.json as dict."""
    return {
        "format": "polar_engine_v4",
        "quantization": "polarengine",
        "block_size": 128,
        "bit_assignment": {
            "q_proj": 5,
            "k_proj": 5,
            "v_proj": 5,
            "o_proj": 6,
            "gate_up_proj": 3,
            "gate_proj": 3,
            "up_proj": 3,
            "down_proj": 4,
            "embed": 5,
            "lm_head": 6,
        },
        "layers": {
            "model.layers.0.self_attn.q_proj": {
                "in_features": 4096,
                "out_features": 4096,
                "in_features_padded": 4096,
                "n_blocks": 32,
                "bits": 5,
                "block_size": 128,
                "packed": False,
                "scale_dtype": "float16",
            },
            "model.layers.0.self_attn.k_proj": {
                "in_features": 4096,
                "out_features": 1024,
                "in_features_padded": 4096,
                "n_blocks": 32,
                "bits": 5,
                "block_size": 128,
                "packed": False,
                "scale_dtype": "float16",
            },
            "model.layers.0.self_attn.o_proj": {
                "in_features": 4096,
                "out_features": 4096,
                "in_features_padded": 4096,
                "n_blocks": 32,
                "bits": 6,
                "block_size": 128,
                "packed": False,
                "scale_dtype": "float16",
            },
            "model.layers.0.mlp.gate_proj": {
                "in_features": 4096,
                "out_features": 12288,
                "in_features_padded": 4096,
                "n_blocks": 32,
                "bits": 3,
                "block_size": 128,
                "packed": True,
                "scale_dtype": "float16",
            },
            "model.layers.0.mlp.down_proj": {
                "in_features": 12288,
                "out_features": 4096,
                "in_features_padded": 12288,
                "n_blocks": 96,
                "bits": 4,
                "block_size": 128,
                "packed": True,
                "scale_dtype": "float16",
            },
        },
    }


@pytest.fixture
def hadamard_matrix(block_size, device):
    """Pre-built Hadamard matrix for reference computations."""
    from polarengine_vllm.kernels.fwht import build_hadamard

    return build_hadamard(block_size, device=torch.device(device))
