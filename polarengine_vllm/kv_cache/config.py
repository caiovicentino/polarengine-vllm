"""Configuration for PolarQuant KV cache compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PolarKVConfig:
    """Configuration for PolarQuant KV cache.

    Args:
        nbits: Quantization bits (2, 3, or 4). Default 3 = 5.3x compression.
        residual_length: Keep last N tokens in BF16 (not quantized).
            Recent tokens matter most for attention quality.
        head_dim: Model's KV head dimension. Must be power of 2 for Hadamard.
        num_kv_heads: Number of key-value heads (GQA/MQA).
        num_layers: Number of transformer layers.
        enabled: Enable/disable compression globally.
        skip_layers: Layer indices to skip (keep FP16 KV). Useful for
            hybrid models where some layers have incompatible head_dim.
    """

    nbits: int = 3
    residual_length: int = 128
    head_dim: int = 128
    num_kv_heads: int = 8
    num_layers: int = 32
    enabled: bool = True
    skip_layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert self.nbits in (2, 3, 4), f"nbits must be 2, 3, or 4, got {self.nbits}"
        assert self.head_dim > 0, f"head_dim must be positive, got {self.head_dim}"
        # Check power of 2
        if self.head_dim & (self.head_dim - 1) != 0:
            raise ValueError(
                f"head_dim={self.head_dim} is not a power of 2. "
                f"PolarQuant requires power-of-2 head_dim for Walsh-Hadamard transform."
            )

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio vs FP16."""
        return 16.0 / self.nbits

    @property
    def n_levels(self) -> int:
        return 1 << self.nbits

    def bytes_per_token(self, fp16: bool = False) -> float:
        """Bytes per token per layer (both K and V)."""
        if fp16:
            return self.num_kv_heads * self.head_dim * 2 * 2  # K + V, 2 bytes each
        # Quantized: nbits per value + norm overhead
        bits_per_vec = self.head_dim * self.nbits
        norm_bytes = 2  # BF16 norm per vector
        bytes_per_vec = bits_per_vec / 8 + norm_bytes
        return self.num_kv_heads * bytes_per_vec * 2  # K + V

    def max_context(self, budget_gb: float) -> int:
        """Max context tokens for a given KV cache VRAM budget."""
        budget_bytes = budget_gb * 1024 ** 3
        bytes_per_tok = self.bytes_per_token() * self.num_layers
        return int(budget_bytes / bytes_per_tok)

    @classmethod
    def for_gemma4_31b(cls, nbits: int = 3) -> "PolarKVConfig":
        """Pre-configured for Gemma 4 31B-it."""
        return cls(
            nbits=nbits,
            head_dim=256,
            num_kv_heads=16,
            num_layers=60,
            residual_length=128,
        )

    @classmethod
    def for_llama3(cls, nbits: int = 3, size: str = "8b") -> "PolarKVConfig":
        """Pre-configured for Llama 3 models."""
        configs = {
            "8b": dict(head_dim=128, num_kv_heads=8, num_layers=32),
            "70b": dict(head_dim=128, num_kv_heads=8, num_layers=80),
        }
        return cls(nbits=nbits, residual_length=128, **configs[size])

    @classmethod
    def for_qwen35(cls, nbits: int = 3, size: str = "9b") -> "PolarKVConfig":
        """Pre-configured for Qwen3.5 models."""
        configs = {
            "9b": dict(head_dim=128, num_kv_heads=8, num_layers=48),
            "27b": dict(head_dim=128, num_kv_heads=4, num_layers=48),
        }
        return cls(nbits=nbits, residual_length=128, **configs[size])
