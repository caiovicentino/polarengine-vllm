"""Test PolarQuantConfig parsing."""

import pytest
import torch

from polarengine_vllm.config import PolarQuantConfig


class TestConfig:
    """Test PolarQuantConfig construction and interface."""

    def test_parse_polar_config(self, sample_polar_config):
        """Parse a sample polar_config.json dict."""
        config = PolarQuantConfig.from_config(sample_polar_config)

        assert config.block_size == 128
        assert isinstance(config.bit_assignment, dict)
        assert len(config.bit_assignment) > 0
        assert isinstance(config.layers_meta, dict)
        assert len(config.layers_meta) == len(sample_polar_config["layers"])

    def test_from_config_defaults(self):
        """from_config uses defaults for missing keys."""
        config = PolarQuantConfig.from_config({})
        assert config.block_size == 128
        assert config.bit_assignment == {}
        assert config.layers_meta == {}

    def test_resolve_bits_exact_match(self, sample_polar_config):
        """Bit resolution: exact match in layers_meta takes priority."""
        config = PolarQuantConfig.from_config(sample_polar_config)

        # This layer has bits=5 in layers_meta
        bits = config._resolve_bits("model.layers.0.self_attn.q_proj")
        assert bits == 5

        # This layer has bits=6 in layers_meta
        bits = config._resolve_bits("model.layers.0.self_attn.o_proj")
        assert bits == 6

    def test_resolve_bits_pattern_match(self, sample_polar_config):
        """Bit resolution: pattern match from bit_assignment when no exact match."""
        config = PolarQuantConfig.from_config(sample_polar_config)

        # Layer not in layers_meta but matches "q_proj" pattern
        bits = config._resolve_bits("model.layers.5.self_attn.q_proj")
        assert bits == 5

        # Matches "down_proj" pattern
        bits = config._resolve_bits("model.layers.5.mlp.down_proj")
        assert bits == 4

    def test_resolve_bits_no_match(self, sample_polar_config):
        """Bit resolution: None when no rule matches."""
        config = PolarQuantConfig.from_config(sample_polar_config)

        bits = config._resolve_bits("model.layers.0.some_unknown_layer")
        assert bits is None

    def test_resolve_bits_longest_pattern_wins(self):
        """When multiple patterns match, the longest one wins."""
        config_dict = {
            "format": "polar_engine_v4",
            "bit_assignment": {
                "proj": 3,
                "gate_proj": 4,
                "mlp.gate_proj": 5,
            },
            "layers": {},
        }
        config = PolarQuantConfig.from_config(config_dict)

        bits = config._resolve_bits("model.layers.0.mlp.gate_proj")
        assert bits == 5, (
            "Expected longest pattern 'mlp.gate_proj' (5 bits) to win"
        )

    def test_config_filenames(self):
        """get_config_filenames returns ['polar_config.json']."""
        filenames = PolarQuantConfig.get_config_filenames()
        assert filenames == ["polar_config.json"]

    def test_get_name(self):
        """get_name returns 'polarengine'."""
        assert PolarQuantConfig.get_name() == "polarengine"

    def test_get_supported_act_dtypes(self):
        """Supported activation dtypes include float16 and bfloat16."""
        dtypes = PolarQuantConfig.get_supported_act_dtypes()
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes

    def test_get_min_capability(self):
        """Minimum CUDA capability is Volta (SM 7.0 = 70)."""
        assert PolarQuantConfig.get_min_capability() == 70

    def test_repr(self, sample_polar_config):
        """__repr__ includes block_size and layer count."""
        config = PolarQuantConfig.from_config(sample_polar_config)
        r = repr(config)
        assert "PolarQuantConfig" in r
        assert "block_size=128" in r

    def test_get_quant_method_returns_none_for_unknown(self, sample_polar_config):
        """get_quant_method returns None for layers with no bit rule."""
        config = PolarQuantConfig.from_config({
            "format": "polar_engine_v4",
            "bit_assignment": {},
            "layers": {},
        })

        # A dummy module -- the method only checks the prefix
        module = torch.nn.Linear(128, 128)
        result = config.get_quant_method(module, "model.layers.99.unknown_layer")
        assert result is None

    def test_get_quant_method_returns_linear_method(self, sample_polar_config):
        """get_quant_method returns a PolarQuantLinearMethod for known layers."""
        config = PolarQuantConfig.from_config(sample_polar_config)
        module = torch.nn.Linear(4096, 4096)

        method = config.get_quant_method(
            module, "model.layers.0.self_attn.q_proj"
        )
        assert method is not None

        from polarengine_vllm.linear_method import PolarQuantLinearMethod
        assert isinstance(method, PolarQuantLinearMethod)
        assert method.bits == 5
        assert method.block_size == 128

    def test_constructor_defaults(self):
        """Direct construction with defaults works."""
        config = PolarQuantConfig()
        assert config.block_size == 128
        assert config.bit_assignment == {}
        assert config.layers_meta == {}

    def test_constructor_custom(self):
        """Direct construction with custom values."""
        config = PolarQuantConfig(
            block_size=64,
            bit_assignment={"q_proj": 3},
            layers_meta={"layer.0": {"bits": 3}},
        )
        assert config.block_size == 64
        assert config.bit_assignment == {"q_proj": 3}
        assert "layer.0" in config.layers_meta
