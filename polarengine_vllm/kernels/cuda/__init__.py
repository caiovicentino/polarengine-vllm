"""PolarEngine CUDA kernels -- Python wrappers with Triton fallback.

If the CUDA extension ``polarengine_cuda`` has been compiled and installed,
the functions in this module dispatch directly to the fused CUDA kernels.
Otherwise they fall back to the Triton implementations so that the engine
still works (at the cost of Triton dispatch overhead).
"""

import torch

try:
    import polarengine_cuda
    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False


# --------------------------------------------------------------------------
# GEMV: unpacked int8 codes
# --------------------------------------------------------------------------

def polar_gemv_cuda(
    codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_padded: int,
    n_blocks: int,
    block_size: int = 128,
) -> torch.Tensor:
    """CUDA GEMV kernel wrapper. Falls back to Triton if CUDA extension not built."""
    if not HAS_CUDA_KERNELS:
        from ..polar_gemv import polar_gemv
        return polar_gemv(codes, x_transformed, norms, ct_scaled, out_f, in_f_padded, n_blocks, block_size)
    return polarengine_cuda.polar_gemv(
        codes, x_transformed, norms, ct_scaled,
        out_f, in_f_padded, n_blocks, block_size,
    )


# --------------------------------------------------------------------------
# GEMV: nibble-packed uint8 codes (Q4)
# --------------------------------------------------------------------------

def polar_gemv_packed_cuda(
    packed_codes: torch.Tensor,
    x_transformed: torch.Tensor,
    norms: torch.Tensor,
    ct_scaled: torch.Tensor,
    out_f: int,
    in_f_half: int,
    n_blocks: int,
    block_size: int = 128,
) -> torch.Tensor:
    """CUDA packed GEMV kernel wrapper. Falls back to Triton if not built."""
    if not HAS_CUDA_KERNELS:
        from ..polar_gemv import polar_gemv_packed
        return polar_gemv_packed(
            packed_codes, x_transformed, norms, ct_scaled,
            out_f, in_f_half, n_blocks, block_size,
        )
    return polarengine_cuda.polar_gemv_packed(
        packed_codes, x_transformed, norms, ct_scaled,
        out_f, in_f_half, n_blocks, block_size,
    )


# --------------------------------------------------------------------------
# FWHT: Fast Walsh-Hadamard Transform (butterfly, O(n log n))
# --------------------------------------------------------------------------

def fwht_cuda(data: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """CUDA FWHT kernel wrapper. Returns transformed copy (not in-place).

    Args:
        data: float32 tensor whose last dimension is divisible by block_size.
              Reshaped internally to (n_blocks, block_size).
        block_size: must be 128.

    Returns:
        Transformed float32 tensor with the same shape as input.
    """
    if not HAS_CUDA_KERNELS:
        # Fallback: matmul-based FWHT via Hadamard matrix
        import math
        shape = data.shape
        flat = data.reshape(-1, block_size).float()
        # Build H on the fly (cached by caller in practice)
        def _build_H(n):
            if n == 1:
                return torch.tensor([[1.0]], device=data.device)
            h = _build_H(n // 2)
            return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)
        H = _build_H(block_size).to(data.device)
        result = torch.matmul(flat, H)
        return result.reshape(shape)
    return polarengine_cuda.fwht(data.contiguous().float(), block_size).reshape(data.shape)
