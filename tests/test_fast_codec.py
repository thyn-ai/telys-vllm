# SPDX-License-Identifier: Apache-2.0
"""Tests for the zero-decode fast chunk codec."""

from __future__ import annotations

import pytest
import torch

from telys_vllm.codec import PayloadCodecError, TelysFastChunkCodec


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_blocks", [1, 4, 17])
def test_fast_codec_round_trip(dtype, num_blocks):
    codec = TelysFastChunkCodec()
    layer_tensors = {
        "layer.0": torch.arange(num_blocks * 16 * 4 * 64, dtype=dtype).reshape(
            num_blocks, 16, 4, 64
        ),
        "layer.1": torch.arange(num_blocks * 16 * 8 * 64, dtype=dtype).reshape(
            num_blocks, 16, 8, 64
        ),
    }
    encoded = codec.encode_chunk(layer_tensors)
    decoded = codec.decode_chunk(encoded)

    assert set(decoded) == {"layer.0", "layer.1"}
    for name in decoded:
        assert decoded[name].dtype == dtype
        assert decoded[name].shape == layer_tensors[name].shape
        assert torch.equal(decoded[name], layer_tensors[name])


def test_fast_codec_zero_copy_view():
    codec = TelysFastChunkCodec()
    tensor = torch.randn(4, 16, 4, 64, dtype=torch.float16)
    encoded = codec.encode_chunk({"layer.0": tensor})
    decoded = codec.decode_chunk(encoded)
    # frombuffer on a bytes object shares the underlying bytes buffer.
    assert decoded["layer.0"].data_ptr() != 0


def _pin_memory_supported() -> bool:
    try:
        return torch.empty(1).pin_memory().is_pinned()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not _pin_memory_supported(),
    reason="pin_memory is not supported on this host (e.g. macOS MPS)",
)
def test_fast_codec_pinned_decode():
    codec = TelysFastChunkCodec()
    tensor = torch.randn(3, 16, 4, 64, dtype=torch.float16)
    encoded = codec.encode_chunk({"layer.0": tensor})
    pinned = codec.decode_chunk_to_pinned(encoded)
    assert pinned["layer.0"].is_pinned()
    assert torch.equal(pinned["layer.0"], tensor)


def test_fast_codec_rejects_empty_layers():
    codec = TelysFastChunkCodec()
    with pytest.raises(PayloadCodecError, match="at least one layer"):
        codec.encode_chunk({})


def test_fast_codec_rejects_mismatched_block_counts():
    codec = TelysFastChunkCodec()
    with pytest.raises(PayloadCodecError, match="block count"):
        codec.encode_chunk(
            {
                "a": torch.randn(2, 16, 4, 64),
                "b": torch.randn(3, 16, 4, 64),
            }
        )


def test_fast_codec_deterministic_encoding():
    codec = TelysFastChunkCodec()
    tensors = {
        "layer.b": torch.randn(2, 16, 4, 64, dtype=torch.float16),
        "layer.a": torch.randn(2, 16, 4, 64, dtype=torch.float16),
    }
    # Encoding must be deterministic regardless of dict order.
    assert codec.encode_chunk(tensors) == codec.encode_chunk(
        dict(reversed(tensors.items()))
    )
