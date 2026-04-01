"""
CPU-pinned expert store for PolarQuant MoE expert offloading.

Loads all routed expert weights from safetensors into CPU pinned memory,
enabling fast DMA transfers to GPU via PCIe without going through the
CPU page cache.  Each expert's PolarQuant components (codes, norms,
ct_scaled) are stored contiguously for efficient bulk transfer.

Designed for Nemotron-Cascade-2-30B-A3B:
  - 23 MoE layers, 128 routed experts per layer
  - Expert naming: backbone.layers.X.mixer.experts.Y.{gate_proj,up_proj,down_proj}
  - Per-expert: ~15M params (up_proj: 1856x2688, down_proj: 2688x1856)
  - Router: backbone.layers.X.mixer.gate.weight (stays FP16, never offloaded)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# Regex to extract (layer_idx, expert_id, proj_name) from Nemotron expert keys.
# Matches both "backbone.layers.X.mixer.experts.Y.up_proj" and
# "model.layers.X.mlp.experts.Y.up_proj" style naming.
_EXPERT_PATTERN = re.compile(
    r"(?:backbone|model)\.layers\.(\d+)\."
    r"(?:mixer|mlp)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)"
)


class ExpertOffloadStore:
    """CPU pinned memory store for all PolarQuant MoE expert weights.

    At initialisation, loads ALL expert weights from a
    :class:`~polarengine_vllm.loader.PolarWeightLoader` into CPU pinned
    memory.  Pinned memory enables DMA transfers to GPU without going
    through the CPU page cache, achieving near-peak PCIe bandwidth.

    Each expert is stored as a dict with keys per projection
    (``gate_proj``, ``up_proj``, ``down_proj``), each containing:
      - ``codes``     (int8 / uint8)
      - ``norms``     (fp16)
      - ``ct_scaled`` (fp32)

    Args:
        polar_config: Parsed ``polar_config.json`` dict.
        weight_loader: An open :class:`PolarWeightLoader` instance.
        device: CPU device string (default ``'cpu'``).  Tensors are
                stored on CPU in pinned memory.
    """

    def __init__(
        self,
        polar_config: dict,
        weight_loader: Any,  # PolarWeightLoader, but avoid circular import
        device: str = "cpu",
    ) -> None:
        self.polar_config = polar_config
        self.weight_loader = weight_loader
        self.device = device

        # (layer_idx, expert_id) -> {
        #   'gate_proj': {'codes': ..., 'norms': ..., 'ct_scaled': ...},
        #   'up_proj':   {'codes': ..., 'norms': ..., 'ct_scaled': ...},
        #   'down_proj': {'codes': ..., 'norms': ..., 'ct_scaled': ...},
        # }
        self._experts: Dict[Tuple[int, int], Dict[str, Dict[str, torch.Tensor]]] = {}

        # Set of MoE layer indices discovered during loading
        self._moe_layer_indices: Set[int] = set()

        # Total expert count
        self._total_experts: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all_experts(self) -> None:
        """Load all expert weights into CPU pinned memory.

        Scans the weight loader's weight map for expert tensor keys,
        groups them by (layer_idx, expert_id), and loads each expert's
        PolarQuant components into pinned memory.

        This can be a long operation (loading ~23 GB for Nemotron).
        Call once at model initialisation time.
        """
        weight_map = self.weight_loader.weight_map
        layers_meta = self.polar_config.get("layers", {})

        # Discover all expert tensor keys and group by (layer, expert, proj)
        expert_keys: Dict[Tuple[int, int, str], List[str]] = {}

        for key in weight_map:
            match = _EXPERT_PATTERN.search(key)
            if match:
                layer_idx = int(match.group(1))
                expert_id = int(match.group(2))
                proj_name = match.group(3)
                group_key = (layer_idx, expert_id, proj_name)
                expert_keys.setdefault(group_key, []).append(key)
                self._moe_layer_indices.add(layer_idx)

        if not expert_keys:
            logger.warning(
                "No expert tensors found in weight map. "
                "Expert offloading will have no effect."
            )
            return

        # Group by (layer_idx, expert_id)
        experts_to_load: Dict[Tuple[int, int], Dict[str, List[str]]] = {}
        for (layer_idx, expert_id, proj_name), keys in expert_keys.items():
            loc = (layer_idx, expert_id)
            experts_to_load.setdefault(loc, {})[proj_name] = keys

        logger.info(
            "Loading %d experts from %d MoE layers into CPU pinned memory...",
            len(experts_to_load),
            len(self._moe_layer_indices),
        )

        for (layer_idx, expert_id), proj_dict in experts_to_load.items():
            expert_data: Dict[str, Dict[str, torch.Tensor]] = {}

            for proj_name, keys in proj_dict.items():
                proj_tensors: Dict[str, torch.Tensor] = {}

                for key in keys:
                    # Determine component from key suffix
                    if key.endswith(".codes"):
                        component = "codes"
                    elif key.endswith(".norms"):
                        component = "norms"
                    elif key.endswith(".ct_scaled"):
                        component = "ct_scaled"
                    elif key.endswith(".weight"):
                        component = "weight"
                    elif key.endswith(".bias"):
                        component = "bias"
                    else:
                        # Bare key or unknown suffix -- treat as weight
                        component = "weight"

                    tensor = self.weight_loader.load_tensor(key, device="cpu")
                    proj_tensors[component] = self._pin_tensor(tensor)

                expert_data[proj_name] = proj_tensors

            self._experts[(layer_idx, expert_id)] = expert_data

        self._total_experts = len(self._experts)
        logger.info(
            "Loaded %d experts into CPU pinned memory. "
            "MoE layers: %s",
            self._total_experts,
            sorted(self._moe_layer_indices),
        )

    def get_expert(
        self, layer_idx: int, expert_id: int
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return CPU-pinned expert tensors.

        Args:
            layer_idx: MoE layer index (0-based).
            expert_id: Expert index within the layer (0-127 for Nemotron).

        Returns:
            Dict mapping projection name (``'gate_proj'``, ``'up_proj'``,
            ``'down_proj'``) to a dict of PolarQuant component tensors
            (``'codes'``, ``'norms'``, ``'ct_scaled'``) in CPU pinned memory.

        Raises:
            KeyError: If the requested expert has not been loaded.
        """
        key = (layer_idx, expert_id)
        if key not in self._experts:
            raise KeyError(
                f"Expert (layer={layer_idx}, expert={expert_id}) "
                f"not found in offload store. "
                f"Loaded experts: {self._total_experts}"
            )
        return self._experts[key]

    def transfer_to_gpu(
        self,
        layer_idx: int,
        expert_id: int,
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Copy an expert's tensors from CPU pinned memory to GPU.

        Performs a synchronous (blocking) copy.  For overlapped transfers,
        use :meth:`async_transfer` with a dedicated CUDA stream.

        Args:
            layer_idx: MoE layer index.
            expert_id: Expert index within the layer.
            device:    Target GPU device (default ``'cuda'``).

        Returns:
            Dict with the same structure as :meth:`get_expert`, but with
            all tensors on the GPU device.
        """
        cpu_expert = self.get_expert(layer_idx, expert_id)
        gpu_expert: Dict[str, Dict[str, torch.Tensor]] = {}

        for proj_name, proj_tensors in cpu_expert.items():
            gpu_proj: Dict[str, torch.Tensor] = {}
            for component, tensor in proj_tensors.items():
                gpu_proj[component] = tensor.to(device, non_blocking=False)
            gpu_expert[proj_name] = gpu_proj

        return gpu_expert

    def async_transfer(
        self,
        layer_idx: int,
        expert_id: int,
        stream: torch.cuda.Stream,
        device: str = "cuda",
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Asynchronously copy an expert's tensors to GPU on a CUDA stream.

        The caller must synchronise the stream before using the returned
        tensors.  Using pinned CPU memory enables DMA transfers that
        overlap with GPU compute on the default stream.

        Args:
            layer_idx: MoE layer index.
            expert_id: Expert index within the layer.
            stream:    CUDA stream for the async copy.
            device:    Target GPU device (default ``'cuda'``).

        Returns:
            Dict with GPU tensors.  The tensors are valid only after
            ``stream.synchronize()`` or an appropriate event wait.
        """
        cpu_expert = self.get_expert(layer_idx, expert_id)
        gpu_expert: Dict[str, Dict[str, torch.Tensor]] = {}

        with torch.cuda.stream(stream):
            for proj_name, proj_tensors in cpu_expert.items():
                gpu_proj: Dict[str, torch.Tensor] = {}
                for component, tensor in proj_tensors.items():
                    gpu_proj[component] = tensor.to(device, non_blocking=True)
                gpu_expert[proj_name] = gpu_proj

        return gpu_expert

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    @property
    def num_experts_loaded(self) -> int:
        """Total number of experts currently in the store."""
        return self._total_experts

    @property
    def moe_layer_indices(self) -> list[int]:
        """Sorted list of MoE layer indices found during loading."""
        return sorted(self._moe_layer_indices)

    def has_expert(self, layer_idx: int, expert_id: int) -> bool:
        """Check if an expert is loaded in the store."""
        return (layer_idx, expert_id) in self._experts

    def experts_for_layer(self, layer_idx: int) -> list[int]:
        """Return sorted list of expert IDs loaded for a given layer."""
        return sorted(
            eid for (lidx, eid) in self._experts if lidx == layer_idx
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Move a CPU tensor to pinned memory for fast DMA transfers.

        If CUDA is not available, returns the tensor as-is (pinning
        requires CUDA initialisation).
        """
        if not torch.cuda.is_available():
            return tensor.contiguous()

        if tensor.is_pinned():
            return tensor

        return tensor.contiguous().pin_memory()

    def __repr__(self) -> str:
        return (
            f"ExpertOffloadStore(experts={self._total_experts}, "
            f"moe_layers={len(self._moe_layer_indices)}, "
            f"device='{self.device}')"
        )
