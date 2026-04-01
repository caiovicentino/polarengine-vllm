"""PolarEngine Triton kernels for quantized GEMV/GEMM inference."""

from polarengine_vllm.kernels.polar_gemv import (
    polar_gemv,
    polar_gemv_packed,
    pack_codes_int4,
    HAS_TRITON,
)
from polarengine_vllm.kernels.polar_gemm import (
    polar_gemm,
    polar_gemm_packed,
    polar_matmul,
)
from polarengine_vllm.kernels.polar_gemv_splitk import (
    polar_gemv_splitk,
    polar_gemv_packed_splitk,
)

__all__ = [
    # GEMV (single vector, decode)
    "polar_gemv",
    "polar_gemv_packed",
    # GEMV SplitK (single vector, large-K decode)
    "polar_gemv_splitk",
    "polar_gemv_packed_splitk",
    # GEMM (batched, prefill)
    "polar_gemm",
    "polar_gemm_packed",
    # Adaptive dispatch (auto-selects GEMV, SplitK, or GEMM)
    "polar_matmul",
    # Utilities
    "pack_codes_int4",
    "HAS_TRITON",
]
