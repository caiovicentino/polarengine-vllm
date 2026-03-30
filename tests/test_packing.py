"""Test INT4 nibble packing roundtrip and edge cases."""

import pytest
import torch

from polarengine_vllm.packing import (
    pack_codes_half_block,
    pack_model_codes,
    unpack_codes_half_block,
    verify_packing_roundtrip,
)
from polarengine_vllm.kernels.polar_gemv import pack_codes_int4


BLOCK_SIZE = 128
HALF_BK = BLOCK_SIZE // 2
N_LEVELS = 16  # INT4: 0-15


class TestPackingRoundtrip:
    """Test that pack -> unpack recovers the original codes."""

    @pytest.mark.parametrize(
        "shape",
        [(4096, 4096), (12288, 4096), (24576, 4096)],
    )
    def test_roundtrip(self, shape):
        """pack -> unpack gives back original codes."""
        out_f, in_f = shape
        in_f_padded = ((in_f + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        codes = torch.randint(
            0, N_LEVELS, (out_f, in_f_padded), dtype=torch.int8
        )

        packed = pack_codes_half_block(codes, BLOCK_SIZE)
        unpacked = unpack_codes_half_block(packed, BLOCK_SIZE)

        assert torch.equal(
            codes.to(torch.uint8), unpacked.to(torch.uint8)
        ), f"Roundtrip failed for shape ({out_f}, {in_f_padded})"

    @pytest.mark.parametrize(
        "shape",
        [(4096, 4096), (12288, 4096), (24576, 4096)],
    )
    def test_verify_roundtrip_helper(self, shape):
        """verify_packing_roundtrip returns True for valid codes."""
        out_f, in_f = shape
        in_f_padded = ((in_f + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        codes = torch.randint(
            0, N_LEVELS, (out_f, in_f_padded), dtype=torch.int8
        )
        assert verify_packing_roundtrip(codes, BLOCK_SIZE)

    def test_edge_values(self):
        """All zeros, all 15s, alternating patterns."""
        shape = (256, 1024)

        # All zeros
        codes_zeros = torch.zeros(*shape, dtype=torch.int8)
        assert verify_packing_roundtrip(codes_zeros, BLOCK_SIZE)

        # All fifteens
        codes_max = torch.full(shape, 15, dtype=torch.int8)
        assert verify_packing_roundtrip(codes_max, BLOCK_SIZE)

        # Alternating 0 and 15
        codes_alt = torch.zeros(*shape, dtype=torch.int8)
        codes_alt[:, ::2] = 15
        assert verify_packing_roundtrip(codes_alt, BLOCK_SIZE)

        # Sequential pattern 0..15 repeating
        codes_seq = torch.arange(16, dtype=torch.int8).repeat(
            shape[0], shape[1] // 16
        )
        assert verify_packing_roundtrip(codes_seq, BLOCK_SIZE)

    def test_half_block_order(self):
        """Verify first half -> low nibble, second half -> high nibble."""
        out_f, in_f_padded = 128, 512

        # Create codes: first half of each block = 3, second half = 12
        codes = torch.zeros(out_f, in_f_padded, dtype=torch.int8)
        codes_blocked = codes.view(out_f, -1, BLOCK_SIZE)
        codes_blocked[:, :, :HALF_BK] = 3
        codes_blocked[:, :, HALF_BK:] = 12

        packed = pack_codes_half_block(codes, BLOCK_SIZE)
        packed_view = packed.view(out_f, -1, HALF_BK)

        # Expected byte: (12 << 4) | 3 = 0xC3 = 195
        expected = (12 << 4) | 3
        assert (packed_view == expected).all(), (
            f"Expected all bytes to be 0x{expected:02X}, "
            f"got unique values: {packed_view.unique().tolist()}"
        )

    def test_nibble_placement_reverse(self):
        """First half = 15, second half = 0 -> byte = 0x0F."""
        out_f, in_f_padded = 128, 512
        codes = torch.zeros(out_f, in_f_padded, dtype=torch.int8)
        codes_blocked = codes.view(out_f, -1, BLOCK_SIZE)
        codes_blocked[:, :, :HALF_BK] = 15
        codes_blocked[:, :, HALF_BK:] = 0

        packed = pack_codes_half_block(codes, BLOCK_SIZE)
        expected = (0 << 4) | 15  # 0x0F = 15
        assert (packed.view(out_f, -1, HALF_BK) == expected).all()

    def test_packed_shape(self):
        """Packed tensor has exactly half the columns."""
        out_f, in_f_padded = 4096, 4096
        codes = torch.randint(
            0, N_LEVELS, (out_f, in_f_padded), dtype=torch.int8
        )
        packed = pack_codes_half_block(codes, BLOCK_SIZE)

        assert packed.shape == (out_f, in_f_padded // 2)
        assert packed.dtype == torch.uint8

    def test_packed_dtype(self):
        """Packed result is always uint8."""
        codes = torch.randint(0, N_LEVELS, (64, 512), dtype=torch.int8)
        packed = pack_codes_half_block(codes, BLOCK_SIZE)
        assert packed.dtype == torch.uint8

    @pytest.mark.parametrize("bs", [32, 64, 128, 256])
    def test_various_block_sizes(self, bs):
        """Roundtrip works for non-default block sizes."""
        in_f_padded = bs * 4  # 4 blocks
        codes = torch.randint(0, N_LEVELS, (64, in_f_padded), dtype=torch.int8)
        packed = pack_codes_half_block(codes, bs)
        unpacked = unpack_codes_half_block(packed, bs)
        assert torch.equal(codes.to(torch.uint8), unpacked.to(torch.uint8))


class TestConsistencyWithPolarGemv:
    """Verify packing.py and polar_gemv.py produce identical packed tensors."""

    @pytest.mark.parametrize(
        "shape",
        [(512, 1024), (2048, 4096), (4096, 4096)],
    )
    def test_pack_functions_match(self, shape):
        """pack_codes_half_block matches pack_codes_int4 from polar_gemv."""
        out_f, in_f = shape
        in_f_padded = ((in_f + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        codes = torch.randint(
            0, N_LEVELS, (out_f, in_f_padded), dtype=torch.int8
        )

        packed_packing = pack_codes_half_block(codes, BLOCK_SIZE)
        packed_gemv = pack_codes_int4(codes, BLOCK_SIZE)

        assert torch.equal(packed_packing, packed_gemv), (
            "pack_codes_half_block and pack_codes_int4 produce different results"
        )


class TestPackModelCodes:
    """Test the in-place model packing utility."""

    def test_pack_model(self):
        """pack_model_codes finds and packs Q3/Q4 layers."""

        class FakeLayer(torch.nn.Module):
            def __init__(self, out_f, in_f_padded, bits):
                super().__init__()
                self.bits = bits
                self.register_buffer(
                    "codes",
                    torch.randint(0, N_LEVELS, (out_f, in_f_padded), dtype=torch.int8),
                )

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Q3 layer -- should be packed
                self.layer_q3 = FakeLayer(1024, 4096, bits=3)
                # Q4 layer -- should be packed
                self.layer_q4 = FakeLayer(4096, 4096, bits=4)
                # Q5 layer -- should NOT be packed (bits > 4)
                self.layer_q5 = FakeLayer(4096, 4096, bits=5)

        model = FakeModel()
        stats = pack_model_codes(model, BLOCK_SIZE)

        assert stats["layers_packed"] == 2, (
            f"Expected 2 layers packed, got {stats['layers_packed']}"
        )
        assert stats["saved_gb"] > 0, "Should save some VRAM"

        # Q3 and Q4 layers should have _codes_packed flag
        assert getattr(model.layer_q3, "_codes_packed", False)
        assert getattr(model.layer_q4, "_codes_packed", False)

        # Q5 layer should not be packed
        assert not getattr(model.layer_q5, "_codes_packed", False)

    def test_pack_model_no_packable_layers(self):
        """pack_model_codes returns 0 layers_packed when nothing to pack."""

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 128)

        model = FakeModel()
        stats = pack_model_codes(model, BLOCK_SIZE)
        assert stats["layers_packed"] == 0
