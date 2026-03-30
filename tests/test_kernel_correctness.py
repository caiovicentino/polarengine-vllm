"""Test Triton GEMV kernels against torch reference."""

import math

import pytest
import torch
import torch.nn.functional as F

from polarengine_vllm.kernels.fwht import build_hadamard
from polarengine_vllm.kernels.polar_gemv import (
    HAS_TRITON,
    pack_codes_int4,
    polar_gemv,
    polar_gemv_packed,
)

# Skip the entire module if CUDA or Triton is unavailable.
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    ),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton required"),
]

# Test shapes matching real model layers.
SHAPES = [
    (4096, 4096),    # q_proj, o_proj
    (1024, 4096),    # k_proj, v_proj (GQA)
    (12288, 4096),   # down_proj
    (4096, 12288),   # gate_up_proj (one half)
    (24576, 4096),   # gate_up_proj (fused)
    (2048, 8192),    # edge case: small M, large K
    (512, 4096),     # small layer
]

DEVICE = "cuda"
BLOCK_SIZE = 128
BITS = 4
N_LEVELS = 1 << BITS


def _build_test_data(out_f, in_f, block_size=BLOCK_SIZE, bits=BITS):
    """Build random quantized data for a given layer shape.

    Returns a dict with codes, norms, x, x_transformed, ct_scaled,
    in_f_padded, n_blocks, and the Hadamard matrix.
    """
    n_levels = 1 << bits
    in_f_padded = ((in_f + block_size - 1) // block_size) * block_size
    n_blocks = in_f_padded // block_size

    ct = torch.linspace(-1.5, 1.5, n_levels, dtype=torch.float32, device=DEVICE)
    ct_scaled = ct / math.sqrt(block_size)

    codes = torch.randint(
        0, n_levels, (out_f, in_f_padded), dtype=torch.int8, device=DEVICE
    )
    norms = (
        torch.randn(out_f * n_blocks, dtype=torch.float32, device=DEVICE).abs()
        + 0.1
    )
    x = torch.randn(in_f, dtype=torch.float32, device=DEVICE)

    H = build_hadamard(block_size, device=torch.device(DEVICE))

    x_padded = F.pad(x, (0, in_f_padded - in_f)) if in_f_padded > in_f else x
    x_transformed = torch.matmul(
        x_padded.view(-1, block_size), H
    ).reshape(-1)

    return {
        "codes": codes,
        "norms": norms,
        "x": x,
        "x_padded": x_padded,
        "x_transformed": x_transformed,
        "ct_scaled": ct_scaled,
        "H": H,
        "in_f_padded": in_f_padded,
        "n_blocks": n_blocks,
        "out_f": out_f,
        "in_f": in_f,
    }


def _reference_output(data):
    """Compute the torch reference output (centroid lookup + norm + dot).

    This reproduces the exact math that the Triton kernel implements:
    ct_scaled[codes] * norms, then dot with x_transformed. No inverse
    Hadamard is needed because both sides are in the Hadamard domain.
    """
    out_f = data["out_f"]
    n_blocks = data["n_blocks"]
    block_size = BLOCK_SIZE
    ct_scaled = data["ct_scaled"]
    codes = data["codes"]
    norms = data["norms"]
    x_tf = data["x_transformed"]

    codes_3d = codes.view(out_f, n_blocks, block_size)
    norms_3d = norms.view(out_f, n_blocks, 1)
    values = ct_scaled[codes_3d.long()] * norms_3d  # (out_f, n_blocks, block_size)

    x_tf_3d = x_tf.view(n_blocks, block_size)
    # Dot product: sum over (n_blocks, block_size)
    ref = torch.einsum("mnk,nk->m", values, x_tf_3d)
    return ref


class TestUnpackedKernel:
    """Tests for the unpacked (int8 codes) Triton GEMV kernel."""

    @pytest.mark.parametrize("M,K", SHAPES)
    def test_correctness(self, M, K):
        """Kernel output matches torch reference (cos > 0.9999, rel_err < 0.01)."""
        data = _build_test_data(M, K)
        ref = _reference_output(data)

        kernel_out = polar_gemv(
            data["codes"],
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            data["in_f_padded"],
            data["n_blocks"],
            BLOCK_SIZE,
        )

        cos = F.cosine_similarity(
            ref.unsqueeze(0), kernel_out.unsqueeze(0)
        ).item()
        rel_err = (
            (torch.abs(ref - kernel_out) / (torch.abs(ref) + 1e-8)).mean().item()
        )

        assert cos > 0.9999, f"Cosine similarity too low: {cos:.6f}"
        assert rel_err < 0.01, f"Relative error too high: {rel_err:.6f}"

    @pytest.mark.parametrize("M,K", SHAPES)
    def test_no_nan(self, M, K):
        """Output contains no NaN or Inf values."""
        data = _build_test_data(M, K)

        kernel_out = polar_gemv(
            data["codes"],
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            data["in_f_padded"],
            data["n_blocks"],
            BLOCK_SIZE,
        )

        assert not torch.isnan(kernel_out).any(), "Output contains NaN"
        assert not torch.isinf(kernel_out).any(), "Output contains Inf"

    @pytest.mark.parametrize("M,K", [(4096, 4096), (512, 4096)])
    def test_deterministic(self, M, K):
        """Two runs with the same data produce the same output."""
        data = _build_test_data(M, K)

        out1 = polar_gemv(
            data["codes"],
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            data["in_f_padded"],
            data["n_blocks"],
            BLOCK_SIZE,
        )
        out2 = polar_gemv(
            data["codes"],
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            data["in_f_padded"],
            data["n_blocks"],
            BLOCK_SIZE,
        )

        assert torch.allclose(out1, out2, atol=1e-6), "Kernel is non-deterministic"

    def test_output_shape(self):
        """Output tensor has shape (out_f,)."""
        data = _build_test_data(512, 4096)
        kernel_out = polar_gemv(
            data["codes"],
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            data["in_f_padded"],
            data["n_blocks"],
            BLOCK_SIZE,
        )
        assert kernel_out.shape == (512,)
        assert kernel_out.dtype == torch.float32


class TestPackedKernel:
    """Tests for the packed INT4 nibble Triton GEMV kernel."""

    @pytest.mark.parametrize("M,K", SHAPES)
    def test_correctness(self, M, K):
        """Packed kernel output matches torch reference (cos > 0.9999, rel_err < 0.01)."""
        data = _build_test_data(M, K)
        ref = _reference_output(data)

        packed_codes = pack_codes_int4(data["codes"], BLOCK_SIZE)
        n_blocks = data["n_blocks"]
        half_bk = BLOCK_SIZE // 2
        in_f_half = n_blocks * half_bk

        kernel_out = polar_gemv_packed(
            packed_codes,
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            in_f_half,
            n_blocks,
            BLOCK_SIZE,
        )

        cos = F.cosine_similarity(
            ref.unsqueeze(0), kernel_out.unsqueeze(0)
        ).item()
        rel_err = (
            (torch.abs(ref - kernel_out) / (torch.abs(ref) + 1e-8)).mean().item()
        )

        assert cos > 0.9999, f"Cosine similarity too low: {cos:.6f}"
        assert rel_err < 0.01, f"Relative error too high: {rel_err:.6f}"

    @pytest.mark.parametrize("M,K", SHAPES)
    def test_no_nan(self, M, K):
        """Packed kernel output contains no NaN or Inf values."""
        data = _build_test_data(M, K)
        packed_codes = pack_codes_int4(data["codes"], BLOCK_SIZE)
        n_blocks = data["n_blocks"]
        in_f_half = n_blocks * (BLOCK_SIZE // 2)

        kernel_out = polar_gemv_packed(
            packed_codes,
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            in_f_half,
            n_blocks,
            BLOCK_SIZE,
        )

        assert not torch.isnan(kernel_out).any(), "Output contains NaN"
        assert not torch.isinf(kernel_out).any(), "Output contains Inf"

    def test_packed_equals_unpacked(self):
        """Packed and unpacked kernels give the same result on the same data."""
        data = _build_test_data(4096, 4096)

        unpacked_out = polar_gemv(
            data["codes"],
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            data["in_f_padded"],
            data["n_blocks"],
            BLOCK_SIZE,
        )

        packed_codes = pack_codes_int4(data["codes"], BLOCK_SIZE)
        n_blocks = data["n_blocks"]
        in_f_half = n_blocks * (BLOCK_SIZE // 2)

        packed_out = polar_gemv_packed(
            packed_codes,
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            in_f_half,
            n_blocks,
            BLOCK_SIZE,
        )

        cos = F.cosine_similarity(
            unpacked_out.unsqueeze(0), packed_out.unsqueeze(0)
        ).item()
        rel_err = (
            (torch.abs(unpacked_out - packed_out) / (torch.abs(unpacked_out) + 1e-8))
            .mean()
            .item()
        )

        assert cos > 0.9999, f"Packed/unpacked cosine too low: {cos:.6f}"
        assert rel_err < 0.01, f"Packed/unpacked rel_err too high: {rel_err:.6f}"

    def test_output_shape(self):
        """Packed kernel output tensor has shape (out_f,)."""
        data = _build_test_data(512, 4096)
        packed_codes = pack_codes_int4(data["codes"], BLOCK_SIZE)
        n_blocks = data["n_blocks"]
        in_f_half = n_blocks * (BLOCK_SIZE // 2)

        kernel_out = polar_gemv_packed(
            packed_codes,
            data["x_transformed"],
            data["norms"],
            data["ct_scaled"],
            data["out_f"],
            in_f_half,
            n_blocks,
            BLOCK_SIZE,
        )
        assert kernel_out.shape == (512,)
        assert kernel_out.dtype == torch.float32
