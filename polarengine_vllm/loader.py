"""Model loader for PolarQuant quantized models.

Loads sharded safetensors files with PolarQuant format:
- codes (int8/uint8 packed) per quantized layer
- norms (fp16) per quantized layer
- ct_scaled (fp32) per quantized layer
- Standard FP16 weights for non-quantized layers

Compatible with both standalone usage and vLLM's weight loading pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from safetensors import safe_open

from polarengine_vllm.packing import unpack_codes_q5

logger = logging.getLogger(__name__)

# Suffixes that identify PolarQuant tensor components.
_POLAR_SUFFIXES = (".codes", ".norms", ".ct_scaled")


# ===================================================================
# Config loading
# ===================================================================

def load_polar_config(model_dir: str) -> dict:
    """Load polar_config.json from model directory.

    Args:
        model_dir: Path to the directory containing polar_config.json.

    Returns:
        Parsed JSON dictionary with keys such as ``format``, ``base_model``,
        ``block_size``, ``bit_assignment``, and ``layers``.

    Raises:
        FileNotFoundError: If polar_config.json does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    config_path = os.path.join(model_dir, "polar_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"polar_config.json not found in {model_dir}"
        )
    with open(config_path, "r") as f:
        config = json.load(f)

    fmt = config.get("format", "unknown")
    if fmt not in ("polar_engine_v4", "polar_engine_v5"):
        logger.warning(
            "Unrecognised PolarEngine format '%s'. "
            "Proceeding, but weight loading may fail.",
            fmt,
        )

    return config


# ===================================================================
# Weight map (sharded safetensors)
# ===================================================================

def get_weight_map(model_dir: str) -> dict:
    """Read model.safetensors.index.json to get weight -> shard mapping.

    For single-shard models (no index file), scans for .safetensors files
    and builds a mapping by inspecting each file's metadata.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Dictionary mapping tensor name to the safetensors filename
        (relative to model_dir) that contains it.  For example::

            {"model.layers.0.self_attn.q_proj.codes": "model-00001-of-00002.safetensors"}
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        logger.info(
            "Loaded weight map from index.json: %d entries across shards",
            len(weight_map),
        )
        return weight_map

    # No index file -- build the map by scanning safetensors files directly.
    weight_map: Dict[str, str] = {}
    model_dir_path = Path(model_dir)
    st_files = sorted(model_dir_path.glob("*.safetensors"))

    if not st_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}"
        )

    for st_file in st_files:
        filename = st_file.name
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = filename

    logger.info(
        "Built weight map from %d safetensors file(s): %d entries",
        len(st_files),
        len(weight_map),
    )
    return weight_map


# ===================================================================
# Standalone model loading
# ===================================================================

def load_polar_model(
    model_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, torch.Tensor], dict, dict]:
    """Load a PolarQuant model for standalone inference (without vLLM).

    Loads every tensor from the safetensors shards into a flat state dict.
    Non-quantized tensors are cast to ``dtype``; quantized components
    (codes, norms, ct_scaled) retain their native dtypes.

    Args:
        model_dir: Path to the quantized model directory.
        device:    Target device (e.g. ``"cuda"`` or ``"cpu"``).
        dtype:     dtype for non-quantized weights (default ``torch.float16``).

    Returns:
        Tuple of ``(state_dict, hf_config, polar_config)`` where:
        - ``state_dict`` maps tensor names to tensors on ``device``.
        - ``hf_config`` is the parsed ``config.json`` (HuggingFace model config),
          or an empty dict if not present.
        - ``polar_config`` is the parsed ``polar_config.json``.
    """
    polar_config = load_polar_config(model_dir)

    # Optionally load HuggingFace config.json
    hf_config: dict = {}
    hf_config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(hf_config_path):
        with open(hf_config_path, "r") as f:
            hf_config = json.load(f)

    weight_map = get_weight_map(model_dir)
    quantized_layers: Set[str] = set(polar_config.get("layers", {}).keys())

    # Group keys by shard file to minimize file open/close overhead.
    shard_to_keys: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    state_dict: Dict[str, torch.Tensor] = {}

    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(model_dir, shard_file)
        logger.info("Loading shard: %s (%d tensors)", shard_file, len(keys))

        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys:
                tensor = f.get_tensor(key)

                # Determine if this is a PolarQuant component that should
                # keep its native dtype, or a regular weight to cast.
                is_polar_component = any(
                    key.endswith(suffix) for suffix in _POLAR_SUFFIXES
                )

                if not is_polar_component:
                    # Standard weight or bias -- cast to target dtype.
                    # Exception: codes-like int tensors should not be cast.
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype)

                state_dict[key] = tensor

    logger.info(
        "Loaded %d tensors from %d shard(s) onto %s",
        len(state_dict),
        len(shard_to_keys),
        device,
    )

    return state_dict, hf_config, polar_config


# ===================================================================
# PolarWeightLoader (vLLM integration)
# ===================================================================

class PolarWeightLoader:
    """Loads PolarQuant weights into a vLLM model.

    Handles:
    - Sharded safetensors (multiple files)
    - Mapping safetensors keys to vLLM model structure
    - Tensor parallelism (shard codes along output dim)
    - Both packed (INT4) and unpacked (int8) codes

    Usage::

        config = load_polar_config(model_dir)
        with PolarWeightLoader(model_dir, config) as loader:
            weights = loader.load_layer_weights("model.layers.0.self_attn.q_proj")
            # weights['codes'], weights['norms'], weights['ct_scaled']
    """

    def __init__(self, model_dir: str, polar_config: dict) -> None:
        self.model_dir = model_dir
        self.polar_config = polar_config
        self.weight_map = get_weight_map(model_dir)
        self._open_files: Dict[str, Any] = {}  # shard filename -> safe_open handle
        self._quantized_layers: Set[str] = set(
            polar_config.get("layers", {}).keys()
        )
        # Build a reverse lookup: for each base layer name, which .codes /
        # .norms / .ct_scaled keys exist in the weight map.
        self._layer_keys: Dict[str, List[str]] = {}
        for key in self.weight_map:
            for suffix in _POLAR_SUFFIXES:
                if key.endswith(suffix):
                    base = key[: -len(suffix)]
                    self._layer_keys.setdefault(base, []).append(key)
                    break

    # -- file handle management -------------------------------------------

    def _get_file(self, shard_filename: str) -> Any:
        """Return an open safetensors handle for the given shard, caching it."""
        if shard_filename not in self._open_files:
            path = os.path.join(self.model_dir, shard_filename)
            self._open_files[shard_filename] = safe_open(
                path, framework="pt", device="cpu"
            )
        return self._open_files[shard_filename]

    def close(self) -> None:
        """Close all open safetensors file handles."""
        for handle in self._open_files.values():
            # safe_open context managers support __exit__; calling it
            # directly is safe and idempotent.
            if hasattr(handle, "__exit__"):
                try:
                    handle.__exit__(None, None, None)
                except Exception:
                    pass
        self._open_files.clear()

    def __enter__(self) -> "PolarWeightLoader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- tensor access ----------------------------------------------------

    def load_tensor(self, key: str, device: str = "cpu") -> torch.Tensor:
        """Load a single tensor from the appropriate shard.

        Args:
            key:    Full tensor name as stored in the safetensors file
                    (e.g. ``"model.layers.0.self_attn.q_proj.codes"``).
            device: Target device for the returned tensor.

        Returns:
            The loaded tensor on the requested device.

        Raises:
            KeyError: If the key is not found in any shard.
        """
        if key not in self.weight_map:
            raise KeyError(
                f"Tensor '{key}' not found in weight map. "
                f"Available keys (first 10): {list(self.weight_map.keys())[:10]}"
            )

        shard_filename = self.weight_map[key]
        handle = self._get_file(shard_filename)
        tensor = handle.get_tensor(key)

        if device != "cpu":
            tensor = tensor.to(device)

        return tensor

    def load_layer_weights(self, layer_name: str) -> Dict[str, torch.Tensor]:
        """Load all tensors for a PolarQuant layer.

        For a quantized layer, returns a dict with keys:
        - ``"codes"``:     int8 (unpacked) or uint8 (packed INT4) codes
        - ``"norms"``:     fp16 per-block norms
        - ``"ct_scaled"``: fp32 pre-scaled centroids
        - ``"bias"``:      fp16 bias (only if present)

        For a non-quantized layer (not in polar_config["layers"]), returns:
        - ``"weight"``:    fp16 weight tensor
        - ``"bias"``:      fp16 bias (only if present)

        Args:
            layer_name: Base layer name without suffix, e.g.
                        ``"model.layers.0.self_attn.q_proj"``.

        Returns:
            Dict mapping component names to tensors (on CPU).

        Raises:
            KeyError: If the layer's tensors cannot be found in any shard.
        """
        result: Dict[str, torch.Tensor] = {}

        if self.is_quantized_layer(layer_name):
            # Load mandatory PolarQuant components.
            for suffix in _POLAR_SUFFIXES:
                key = layer_name + suffix
                component_name = suffix.lstrip(".")  # "codes", "norms", "ct_scaled"
                result[component_name] = self.load_tensor(key)

            # Detect Q5-packed codes and unpack at load time.
            # After unpacking, kernels receive standard int8 codes (values 0-31).
            layers_meta = self.polar_config.get("layers", {})
            layer_meta = layers_meta.get(layer_name, {})
            if layer_meta.get("packed_q5", False):
                block_size = layer_meta.get("block_size", self.polar_config.get("block_size", 128))
                result["codes"] = unpack_codes_q5(result["codes"], block_size)

            # Bias is optional.
            bias_key = layer_name + ".bias"
            if bias_key in self.weight_map:
                result["bias"] = self.load_tensor(bias_key)

        else:
            # Non-quantized layer -- load the plain weight.
            weight_key = layer_name + ".weight"
            if weight_key in self.weight_map:
                result["weight"] = self.load_tensor(weight_key)
            elif layer_name in self.weight_map:
                # Fallback: bare parameter key without .weight suffix
                # (some models, e.g. Nemotron, store params under the
                # bare name rather than name + ".weight").
                result["weight"] = self.load_tensor(layer_name)
            else:
                raise KeyError(
                    f"Cannot find weight tensor for non-quantized layer "
                    f"'{layer_name}'. Tried keys '{weight_key}' and "
                    f"'{layer_name}'."
                )

            bias_key = layer_name + ".bias"
            if bias_key in self.weight_map:
                result["bias"] = self.load_tensor(bias_key)

        return result

    def is_quantized_layer(self, layer_name: str) -> bool:
        """Check if a layer is quantized (present in polar_config layers).

        A layer is considered quantized if it appears in the ``layers``
        section of ``polar_config.json``, or if a ``.codes`` tensor exists
        for it in the safetensors weight map.

        Args:
            layer_name: Base layer name (e.g. ``"model.layers.0.self_attn.q_proj"``).

        Returns:
            True if the layer has PolarQuant quantized weights.
        """
        if layer_name in self._quantized_layers:
            return True
        # Fallback: check if .codes exists in the weight map.
        codes_key = layer_name + ".codes"
        return codes_key in self.weight_map

    def get_layer_bits(self, layer_name: str) -> int:
        """Get quantization bits for a layer from polar_config.

        Lookup order:
        1. Per-layer ``bits`` in ``polar_config["layers"]``.
        2. Pattern match against ``polar_config["bit_assignment"]``.
        3. Fallback to 5 bits if no rule matches.

        Args:
            layer_name: Base layer name (e.g. ``"model.layers.0.self_attn.q_proj"``).

        Returns:
            Bit width as an integer (e.g. 3, 4, 5, 6).
        """
        layers_meta = self.polar_config.get("layers", {})
        meta = layers_meta.get(layer_name)
        if meta and "bits" in meta:
            return int(meta["bits"])

        # Pattern match against bit_assignment (longest match wins).
        bit_assignment = self.polar_config.get("bit_assignment", {})
        best_match: Optional[str] = None
        for pattern in bit_assignment:
            if pattern in layer_name:
                if best_match is None or len(pattern) > len(best_match):
                    best_match = pattern

        if best_match is not None:
            return int(bit_assignment[best_match])

        return 5  # default fallback

    def is_packed(self, layer_name: str) -> bool:
        """Check if a quantized layer uses packed codes (INT4 nibble or Q5 5-bit).

        Packed layers have:
        - bits <= 4: nibble-packed (in_f_padded // 2 columns)
        - bits == 5 with packed_q5: 5-bit packed (in_f_padded * 5 // 8 columns)

        Args:
            layer_name: Base layer name.

        Returns:
            True if the layer's codes are packed (either INT4 nibble or Q5 5-bit).
        """
        layers_meta = self.polar_config.get("layers", {})
        meta = layers_meta.get(layer_name)

        # Check for Q5 packing via explicit metadata flag
        if meta and meta.get("packed_q5", False):
            return True

        bits = self.get_layer_bits(layer_name)
        if bits > 4:
            return False

        # Verify by checking the actual tensor shape against metadata.
        if meta:
            in_f_padded = meta.get("in_f_padded", 0)
            codes_key = layer_name + ".codes"
            if codes_key in self.weight_map:
                # Peek at the shape without loading the full tensor.
                shard_filename = self.weight_map[codes_key]
                handle = self._get_file(shard_filename)
                # safetensors doesn't expose shape without loading on all
                # backends, so we load and check.
                codes = handle.get_tensor(codes_key)
                if codes.shape[-1] == in_f_padded // 2:
                    return True
                return False

        # If bits <= 4, assume packed.
        return True

    def get_all_layer_names(self) -> List[str]:
        """Return all quantized layer names from polar_config.

        Returns:
            Sorted list of layer names that have PolarQuant quantization.
        """
        return sorted(self.polar_config.get("layers", {}).keys())


# ===================================================================
# Tensor parallelism helpers
# ===================================================================

def shard_codes_for_tp(
    codes: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Shard codes tensor for tensor parallelism.

    For output dimension (dim=0): split rows across ranks.
    For input dimension (dim=1): each rank gets the full tensor (no split).

    Args:
        codes:   The codes tensor, shape ``(out_f, in_f_padded)`` or
                 ``(out_f, in_f_padded // 2)`` for packed.
        tp_rank: This worker's rank in the tensor-parallel group.
        tp_size: Total number of tensor-parallel workers.
        dim:     Dimension to shard along.  0 for column-parallel
                 (output dim), 1 for row-parallel (input dim).

    Returns:
        The shard of codes belonging to ``tp_rank``.
    """
    if tp_size <= 1:
        return codes

    if dim == 1:
        # Row-parallel: each rank needs the full codes to produce its
        # partial sum.  No sharding needed.
        return codes

    # Column-parallel (dim=0): split output rows evenly.
    total_rows = codes.shape[0]
    if total_rows % tp_size != 0:
        raise ValueError(
            f"Cannot evenly shard {total_rows} output rows across "
            f"{tp_size} TP ranks. Rows must be divisible by tp_size."
        )
    shard_size = total_rows // tp_size
    start = tp_rank * shard_size
    end = start + shard_size
    return codes[start:end].contiguous()


def shard_norms_for_tp(
    norms: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    n_blocks: int,
    dim: int = 0,
) -> torch.Tensor:
    """Shard norms tensor for tensor parallelism.

    Norms have shape ``(out_f, n_blocks)``.  Sharding follows the same
    logic as :func:`shard_codes_for_tp`.

    Args:
        norms:    Per-block norms, shape ``(out_f, n_blocks)``.
        tp_rank:  This worker's rank.
        tp_size:  Total number of TP workers.
        n_blocks: Number of blocks per row (used for reshaping if needed).
        dim:      Dimension to shard (0 = output, 1 = input).

    Returns:
        The norms shard for ``tp_rank``.
    """
    if tp_size <= 1:
        return norms

    if dim == 1:
        return norms

    total_rows = norms.shape[0]
    if total_rows % tp_size != 0:
        raise ValueError(
            f"Cannot evenly shard {total_rows} norm rows across "
            f"{tp_size} TP ranks."
        )
    shard_size = total_rows // tp_size
    start = tp_rank * shard_size
    end = start + shard_size
    return norms[start:end].contiguous()


# ===================================================================
# Standalone test
# ===================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        config = load_polar_config(model_dir)
        print(f"Model: {config.get('base_model')}")
        print(f"Format: {config.get('format')}")
        print(f"Block size: {config.get('block_size')}")
        print(f"Bit assignment: {config.get('bit_assignment', {})}")
        print(f"Layers: {len(config.get('layers', {}))}")

        with PolarWeightLoader(model_dir, config) as loader:
            layer_names = loader.get_all_layer_names()
            print(f"\nQuantized layers: {len(layer_names)}")

            # Load first quantized layer as test
            for name in layer_names:
                bits = loader.get_layer_bits(name)
                weights = loader.load_layer_weights(name)
                codes = weights["codes"]
                norms = weights["norms"]
                ct = weights["ct_scaled"]
                packed_str = " (packed)" if codes.dtype == torch.uint8 else ""
                print(
                    f"  {name}:\n"
                    f"    bits={bits}{packed_str}\n"
                    f"    codes={codes.shape} {codes.dtype}\n"
                    f"    norms={norms.shape} {norms.dtype}\n"
                    f"    ct_scaled={ct.shape} {ct.dtype}"
                )
                if "bias" in weights:
                    print(f"    bias={weights['bias'].shape} {weights['bias'].dtype}")
                break

            # Summary statistics
            total_params = 0
            bits_dist: Dict[int, int] = {}
            for name in layer_names:
                meta = config.get("layers", {}).get(name, {})
                b = loader.get_layer_bits(name)
                out_f = meta.get("out_features", 0)
                in_f = meta.get("in_features", 0)
                n_params = out_f * in_f
                total_params += n_params
                bits_dist[b] = bits_dist.get(b, 0) + n_params

            print(f"\nTotal quantized parameters: {total_params:,}")
            print("Bit distribution:")
            for b in sorted(bits_dist.keys()):
                count = bits_dist[b]
                pct = 100.0 * count / total_params if total_params > 0 else 0
                print(f"  {b}-bit: {count:>12,} params ({pct:5.1f}%)")
    else:
        print("Usage: python loader.py <model_dir>")
        print()
        print("Loads a PolarQuant model directory and prints summary info.")
        print("The directory should contain polar_config.json and .safetensors files.")
