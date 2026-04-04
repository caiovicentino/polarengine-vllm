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
from polarengine_vllm.kernels.polar_quantize import (
    polar_quantize,
)
from polarengine_vllm.kernels.fwht_train import (
    fwht_train,
    fwht_triton,
    fwht_matmul_train,
    FWHTLayer,
)
from polarengine_vllm.kernels.gla_retention import (
    gla_retention,
    gla_retention_reference,
    benchmark_gla_retention,
    run_full_benchmark as gla_benchmark,
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
    # Fused quantization (training)
    "polar_quantize",
    # FWHT (training-compatible Walsh-Hadamard transform)
    "fwht_train",
    "fwht_triton",
    "fwht_matmul_train",
    "FWHTLayer",
    # GLA Retention (fused gated linear attention with exponential decay)
    "gla_retention",
    "gla_retention_reference",
    "benchmark_gla_retention",
    "gla_benchmark",
    # Utilities
    "pack_codes_int4",
    "HAS_TRITON",
]
