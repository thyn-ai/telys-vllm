# SPDX-License-Identifier: Apache-2.0
"""GPU KV extraction/injection layout tests.

vLLM v1 always hands the connector a *logical* per-layer KV view of shape
``[B, H, N, C]`` (blocks, heads, token-in-block, head-dim), regardless of the
physical layout of the backing allocation. The physical layout string reported
in logs (e.g. ``LBNHC``) describes the 5-D stride permutation; the per-layer
view is produced by ``vllm.v1.kv_cache_interface.create_kv_cache_views`` and
``allocate_kv_cache`` documents this explicitly as "View a flat int8 buffer as
one 4D ``[B, H, N, C]`` view per layer".

The reference vLLM connector (``example_connector.py``) uses the exact same
indexing pattern we do::

    block_idxs = slot_mapping // block_size
    offsets = slot_mapping % block_size
    dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache

These tests construct real physical LBNHC/LBHNC/BLNHC buffers, derive the
logical ``[B, H, N, C]`` view the same way vLLM does, and verify that
``telys_vllm.gpu_ops`` reads and writes the slots that correspond to the
requested logical coordinates.
"""

from __future__ import annotations

from math import prod

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only dev environments
    torch = None  # type: ignore[assignment]

from telys_vllm.gpu_ops import extract_kv_blocks, inject_kv_blocks

pytestmark = pytest.mark.skipif(torch is None, reason="torch required")

# vLLM KVCacheLayout stride permutations over logical axes [L, B, H, N, C].
# LBNHC is the layout reported by the headline bench run.
LAYOUTS = {
    "LBHNC": (0, 1, 2, 3, 4),  # identity: physical [L, B, H, N, C]
    "LBNHC": (0, 1, 3, 2, 4),  # physical [L, B, N, H, C]
    "BLNHC": (1, 0, 3, 2, 4),  # physical [B, L, N, H, C]
}

LAYER = 0
NUM_LAYERS = 2
NUM_BLOCKS = 4
NUM_HEADS = 3
BLOCK_SIZE = 8
HEAD_DIM = 5


def _make_logical_view(layout: tuple[int, ...]):
    """Return a (logical_5d, physical_base) pair for the given layout.

    The logical tensor has shape ``[L, B, H, N, C]``; ``logical_5d[l]`` is the
    ``[B, H, N, C]`` view that vLLM would pass to the connector for layer ``l``.
    The physical base tensor is contiguous in the requested layout order.
    """
    logical_shape = (NUM_LAYERS, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM)
    # A value that encodes its logical flat index makes it easy to verify that
    # the right logical slot was touched.
    flat = torch.arange(prod(logical_shape), dtype=torch.int32)
    logical = flat.reshape(logical_shape)
    # Permute to physical order and make it contiguous in that order.
    base = logical.permute(layout).contiguous()
    # Invert the permutation to recover the logical view with the physical
    # layout's strides.
    inv = tuple(layout.index(i) for i in range(5))
    logical_view = base.permute(inv)
    return logical_view, base


def _expected_block(kv_layer: torch.Tensor, block_id: int) -> torch.Tensor:
    """Expected content for one extracted block: ``[BLOCK_SIZE, H, C]``."""
    # kv_layer is [B, H, N, C]; slice out one block and move N to the front.
    return kv_layer[block_id, :, :, :].permute(1, 0, 2)


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_extract_maps_logical_slots(name: str):
    """Extraction must return the logical [B,H,N,C] values for each block."""
    layout = LAYOUTS[name]
    logical_5d, _ = _make_logical_view(layout)
    kv_layer = logical_5d[LAYER]
    block_ids = [1, 3]

    blocks = extract_kv_blocks(kv_layer, block_ids, BLOCK_SIZE, None)

    assert len(blocks) == len(block_ids)
    for block_tensor, block_id in zip(blocks, block_ids):
        assert block_tensor.shape == (BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        expected = _expected_block(kv_layer, block_id)
        assert torch.equal(block_tensor, expected)


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_injection_writes_logical_slots(name: str):
    """Injection must modify the physical slots for the requested blocks."""
    layout = LAYOUTS[name]
    logical_5d, _ = _make_logical_view(layout)
    kv_layer = logical_5d[LAYER]
    block_ids = [0, 2]

    injected = [
        torch.full((BLOCK_SIZE, NUM_HEADS, HEAD_DIM), 1000 + i, dtype=torch.int32)
        for i in range(len(block_ids))
    ]
    inject_kv_blocks(kv_layer, block_ids, BLOCK_SIZE, None, injected)

    for block_tensor, block_id in zip(injected, block_ids):
        actual = _expected_block(kv_layer, block_id)
        assert torch.equal(actual, block_tensor)

    # Other blocks must stay untouched.
    untouched = [b for b in range(NUM_BLOCKS) if b not in block_ids]
    logical_5d_before, _ = _make_logical_view(layout)
    for block_id in untouched:
        assert torch.equal(
            _expected_block(kv_layer, block_id),
            _expected_block(logical_5d_before[LAYER], block_id),
        )


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_roundtrip_through_extract_inject(name: str):
    """Extract -> inject into a fresh view -> re-extract must be identity."""
    layout = LAYOUTS[name]
    logical_5d_src, _ = _make_logical_view(layout)
    logical_5d_dst, _ = _make_logical_view(layout)
    kv_src = logical_5d_src[LAYER]
    kv_dst = logical_5d_dst[LAYER]
    block_ids = [1]

    blocks = extract_kv_blocks(kv_src, block_ids, BLOCK_SIZE, None)
    inject_kv_blocks(kv_dst, block_ids, BLOCK_SIZE, None, blocks)

    for block_id in block_ids:
        assert torch.equal(
            _expected_block(kv_dst, block_id),
            _expected_block(kv_src, block_id),
        )
