"""PolarQuant KV Cache — core quantizer and cache manager.

Provides:
- PolarKVQuantizer: stateless quantize/dequantize for KV tensors
- PolarKVLayer: per-layer cache with residual buffer
- PolarKVCache: full cache manager for all layers
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from scipy.stats import norm as sp_norm

from .config import PolarKVConfig


# ═══════════════════════════════════════════════════════════════════
# Lloyd-Max Centroids
# ═══════════════════════════════════════════════════════════════════

_CENTROID_CACHE: dict[int, torch.Tensor] = {}


def get_centroids(nbits: int) -> torch.Tensor:
    """Compute Lloyd-Max optimal centroids for N(0,1).

    Uses iterative refinement (100 steps) to find MSE-optimal
    quantization levels for a standard normal distribution.
    Results are cached globally.
    """
    if nbits in _CENTROID_CACHE:
        return _CENTROID_CACHE[nbits]

    n_levels = 1 << nbits
    lo, hi = -4.0, 4.0
    boundaries = torch.linspace(lo, hi, n_levels + 1)
    centroids = torch.zeros(n_levels)

    for _ in range(100):
        for i in range(n_levels):
            a, b = boundaries[i].item(), boundaries[i + 1].item()
            pa, pb = sp_norm.cdf(a), sp_norm.cdf(b)
            if pb - pa < 1e-12:
                centroids[i] = (a + b) / 2
            else:
                centroids[i] = (sp_norm.pdf(a) - sp_norm.pdf(b)) / (pb - pa)
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    _CENTROID_CACHE[nbits] = centroids
    return centroids


# Pre-warm cache
for _b in (2, 3, 4):
    get_centroids(_b)


# ═══════════════════════════════════════════════════════════════════
# Walsh-Hadamard Matrix
# ═══════════════════════════════════════════════════════════════════

_H_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def build_hadamard(n: int, device: str = "cpu") -> torch.Tensor:
    """Build orthogonal Walsh-Hadamard matrix of size n×n.

    Cached per (n, device). n must be a power of 2.
    """
    key = (n, device)
    if key in _H_CACHE:
        return _H_CACHE[key]

    def _build(sz: int) -> torch.Tensor:
        if sz == 1:
            return torch.tensor([[1.0]])
        half = _build(sz // 2)
        return torch.cat([
            torch.cat([half, half], dim=1),
            torch.cat([half, -half], dim=1),
        ], dim=0) / math.sqrt(2)

    H = _build(n).to(device)
    _H_CACHE[key] = H
    return H


# ═══════════════════════════════════════════════════════════════════
# BitPacker — real bit-packing (2/3/4 bit)
# ═══════════════════════════════════════════════════════════════════

class BitPacker:
    """Pack/unpack integer codes into bit-packed uint8 tensors."""

    @staticmethod
    def pack(codes: torch.Tensor, nbits: int) -> torch.Tensor:
        """Pack (N, D) codes into bit-packed uint8.

        Args:
            codes: integer codes in [0, 2^nbits - 1]
            nbits: 2, 3, or 4

        Returns:
            uint8 packed tensor
        """
        c = codes.long()
        N = c.shape[0]

        if nbits == 2:
            c = c.reshape(N, -1, 4)
            return ((c[:, :, 0] << 6) | (c[:, :, 1] << 4) |
                    (c[:, :, 2] << 2) | c[:, :, 3]).to(torch.uint8)

        elif nbits == 3:
            c = c.reshape(N, -1, 8)
            b0 = (c[:, :, 0] << 5) | (c[:, :, 1] << 2) | (c[:, :, 2] >> 1)
            b1 = ((c[:, :, 2] & 1) << 7) | (c[:, :, 3] << 4) | (c[:, :, 4] << 1) | (c[:, :, 5] >> 2)
            b2 = ((c[:, :, 5] & 3) << 6) | (c[:, :, 6] << 3) | c[:, :, 7]
            return torch.stack([b0, b1, b2], dim=-1).reshape(N, -1).to(torch.uint8)

        elif nbits == 4:
            return ((c[:, 0::2] << 4) | c[:, 1::2]).to(torch.uint8)

        return codes.to(torch.uint8)

    @staticmethod
    def unpack(packed: torch.Tensor, nbits: int, D: int) -> torch.Tensor:
        """Unpack bit-packed uint8 back to (N, D) codes.

        Args:
            packed: uint8 packed tensor
            nbits: 2, 3, or 4
            D: original dimension

        Returns:
            long tensor of codes
        """
        p = packed.long()
        N = p.shape[0]

        if nbits == 2:
            return torch.stack(
                [(p >> 6) & 3, (p >> 4) & 3, (p >> 2) & 3, p & 3], dim=-1
            ).reshape(N, D)

        elif nbits == 3:
            p3 = p.reshape(N, -1, 3)
            b0, b1, b2 = p3[:, :, 0], p3[:, :, 1], p3[:, :, 2]
            return torch.stack([
                (b0 >> 5) & 7, (b0 >> 2) & 7, ((b0 & 3) << 1) | ((b1 >> 7) & 1),
                (b1 >> 4) & 7, (b1 >> 1) & 7, ((b1 & 1) << 2) | ((b2 >> 6) & 3),
                (b2 >> 3) & 7, b2 & 7,
            ], dim=-1).reshape(N, D)

        elif nbits == 4:
            return torch.stack(
                [(p >> 4) & 0xF, p & 0xF], dim=-1
            ).reshape(N, D)

        return p

    @staticmethod
    def packed_size(D: int, nbits: int) -> int:
        """Number of packed bytes per vector of dimension D."""
        if nbits == 2:
            return D // 4
        elif nbits == 3:
            return (D // 8) * 3
        elif nbits == 4:
            return D // 2
        return D


# ═══════════════════════════════════════════════════════════════════
# PolarKVQuantizer — stateless quantize/dequantize
# ═══════════════════════════════════════════════════════════════════

class PolarKVQuantizer:
    """Quantizes and dequantizes KV cache vectors.

    Uses Walsh-Hadamard rotation + Lloyd-Max optimal centroids.
    Stateless — call quantize() and dequantize() independently.
    """

    def __init__(self, head_dim: int, nbits: int = 3, device: str = "cuda"):
        self.head_dim = head_dim
        self.nbits = nbits
        self.device = device
        self.scale = math.sqrt(head_dim)

        self.centroids = get_centroids(nbits).to(device)
        self.H = build_hadamard(head_dim, device)

    def quantize(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize KV tensor.

        Args:
            tensor: (B, H, S, D) or (N, D) BF16/FP32 tensor

        Returns:
            packed: uint8 bit-packed codes, shape (N, packed_bytes)
            norms: BF16 per-vector L2 norms, shape (N,)
        """
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, self.head_dim).float()

        # L2 normalize + Hadamard rotate
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-10)
        rotated = (flat / norms) @ self.H * self.scale

        # Lloyd-Max quantize (chunked for large tensors)
        N = rotated.shape[0]
        QC = 4096
        codes = torch.empty(N, self.head_dim, dtype=torch.int8, device=self.device)
        ct = self.centroids.view(1, 1, -1)
        for i in range(0, N, QC):
            j = min(i + QC, N)
            codes[i:j] = (rotated[i:j].unsqueeze(-1) - ct).abs().argmin(-1).to(torch.int8)

        # Bit-pack
        packed = BitPacker.pack(codes, self.nbits)
        return packed, norms.bfloat16().squeeze(1)

    def dequantize(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Dequantize packed codes back to BF16 tensor.

        Args:
            packed: uint8 bit-packed codes
            norms: BF16 per-vector norms
            shape: target output shape (B, H, S, D)

        Returns:
            BF16 tensor of requested shape
        """
        codes = BitPacker.unpack(packed, self.nbits, self.head_dim)
        values = self.centroids[codes] / self.scale
        values = (values @ self.H) * norms.float().unsqueeze(1)
        return values.bfloat16().reshape(shape)


# ═══════════════════════════════════════════════════════════════════
# PolarKVLayer — per-layer cache with residual buffer
# ═══════════════════════════════════════════════════════════════════

class PolarKVLayer:
    """Manages quantized KV cache for a single transformer layer.

    Maintains a residual buffer of recent tokens in BF16 and
    a compressed buffer of older tokens in PolarQuant format.
    """

    def __init__(self, quantizer: PolarKVQuantizer, residual_length: int = 128):
        self.quantizer = quantizer
        self.residual_length = residual_length

        # Compressed storage
        self._packed: Optional[torch.Tensor] = None
        self._norms: Optional[torch.Tensor] = None
        self._q_count = 0

        # BF16 residual (recent tokens)
        self.residual: Optional[torch.Tensor] = None

        # Shape tracking
        self._B: Optional[int] = None
        self._H: Optional[int] = None
        self._D: Optional[int] = None
        self._can_quantize = True

    def update(self, new_tensor: torch.Tensor) -> torch.Tensor:
        """Append new KV entries and return full (decompressed + residual) cache.

        Args:
            new_tensor: (B, num_heads, new_seq_len, head_dim) BF16

        Returns:
            Full cache tensor (B, num_heads, total_seq_len, head_dim) BF16
        """
        if self._B is None:
            self._B, self._H = new_tensor.shape[0], new_tensor.shape[1]
            self._D = new_tensor.shape[3]
            self._can_quantize = (
                self._D == self.quantizer.head_dim
                and (self._D & (self._D - 1)) == 0
            )

        # Append to residual
        if self.residual is None:
            self.residual = new_tensor
        else:
            self.residual = torch.cat([self.residual, new_tensor], dim=2)

        # Quantize overflow beyond residual_length
        if self._can_quantize and self.residual.shape[2] > self.residual_length:
            n_q = self.residual.shape[2] - self.residual_length
            to_quantize = self.residual[:, :, :n_q, :]
            self.residual = self.residual[:, :, n_q:, :].contiguous()

            packed, norms = self.quantizer.quantize(to_quantize)
            if self._packed is None:
                self._packed, self._norms = packed, norms
            else:
                self._packed = torch.cat([self._packed, packed], dim=0)
                self._norms = torch.cat([self._norms, norms], dim=0)
            self._q_count += n_q

        # Return full cache (decompressed + residual)
        if self._packed is not None:
            B, H, D = self._B, self._H, self._D
            S_q = self._packed.shape[0] // (B * H)
            decompressed = self.quantizer.dequantize(
                self._packed, self._norms, (B, H, S_q, D)
            )
            return torch.cat([decompressed, self.residual], dim=2)
        return self.residual

    def get_seq_length(self) -> int:
        """Total cached sequence length (quantized + residual)."""
        q = 0
        if self._packed is not None and self._B and self._H:
            q = self._packed.shape[0] // (self._B * self._H)
        return q + (self.residual.shape[2] if self.residual is not None else 0)

    def memory_bytes(self) -> int:
        """Total memory used by this layer's cache."""
        total = 0
        if self._packed is not None:
            total += self._packed.numel()  # uint8
            total += self._norms.numel() * 2  # bf16
        if self.residual is not None:
            total += self.residual.numel() * 2  # bf16
        return total

    def reset(self):
        """Clear all cached data."""
        self._packed = None
        self._norms = None
        self._q_count = 0
        self.residual = None


# ═══════════════════════════════════════════════════════════════════
# PolarKVCache — full cache manager
# ═══════════════════════════════════════════════════════════════════

class PolarKVCache:
    """Full PolarQuant KV cache for all transformer layers.

    Drop-in replacement for vLLM's cache when used with
    PolarKVAttentionWrapper.
    """

    def __init__(self, config: PolarKVConfig):
        self.config = config
        self.quantizer = PolarKVQuantizer(
            head_dim=config.head_dim,
            nbits=config.nbits,
            device="cuda",
        )

        self.k_layers = []
        self.v_layers = []
        for i in range(config.num_layers):
            if i in config.skip_layers:
                # Skipped layers use None (fall back to FP16 in wrapper)
                self.k_layers.append(None)
                self.v_layers.append(None)
            else:
                self.k_layers.append(
                    PolarKVLayer(self.quantizer, config.residual_length)
                )
                self.v_layers.append(
                    PolarKVLayer(self.quantizer, config.residual_length)
                )

        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a layer and return full K/V.

        Args:
            key_states: (B, num_kv_heads, new_seq, head_dim)
            value_states: same shape
            layer_idx: transformer layer index

        Returns:
            (full_keys, full_values) including history
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        k_layer = self.k_layers[layer_idx]
        v_layer = self.v_layers[layer_idx]

        if k_layer is None:
            # Skipped layer — no compression, just accumulate BF16
            # (handled by caller or fallback)
            return key_states, value_states

        return k_layer.update(key_states), v_layer.update(value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        k = self.k_layers[layer_idx]
        if k is None:
            return 0
        return k.get_seq_length()

    def memory_bytes(self) -> int:
        """Total memory across all layers."""
        total = 0
        for k, v in zip(self.k_layers, self.v_layers):
            if k is not None:
                total += k.memory_bytes() + v.memory_bytes()
        return total

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)

    def reset(self):
        """Clear all cached data across all layers."""
        for k, v in zip(self.k_layers, self.v_layers):
            if k is not None:
                k.reset()
                v.reset()
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def stats(self) -> dict:
        """Return cache statistics."""
        fp16_bytes = (
            self.config.num_layers * 2 * self.config.num_kv_heads
            * self.config.head_dim * self.get_seq_length() * 2
        )
        actual_bytes = self.memory_bytes()
        return {
            "seq_length": self.get_seq_length(),
            "memory_mb": self.memory_mb(),
            "fp16_memory_mb": fp16_bytes / (1024 * 1024),
            "compression_ratio": fp16_bytes / max(actual_bytes, 1),
            "seen_tokens": self._seen_tokens,
        }
