# SPDX-License-Identifier: Apache-2.0
"""Measure the warm-load data-path cost: legacy chunk vs fast zero-decode path.

Calibrates to a ~1.5B-class model: 28 layers, block_size=16, 12 heads, 128 head
 dimension, fp16 KV tensors.  The benchmark isolates the per-chunk work that
happens on every warm cache hit:

    blob read (memcpy from store) + decode + stage to pinned CPU + H2D transfer

The "transfer" is mocked with a CPU-to-CPU copy because this benchmark is meant
to run on a laptop; the pinned staging and async-copy structure are still
exercised.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from telys_vllm.codec import TelysChunkCodec, TelysFastChunkCodec
from telys_vllm.gpu_ops import serialize_kv_tensor

BLOCKS = 106
LAYERS = 28
BLOCK_SIZE = 16
NUM_HEADS = 12
HEAD_DIM = 128
DTYPE = torch.float16
RUNS = 5


def make_layer_tensors() -> dict[str, torch.Tensor]:
    """Return stacked CPU tensors for every layer: [BLOCKS, BLOCK_SIZE, H, D]."""
    return {
        f"layer.{i}": torch.randn(BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
        for i in range(LAYERS)
    }


def _payload_size(payload: bytes) -> str:
    mb = len(payload) / (1024 * 1024)
    return f"{mb:.1f} MB"


def build_old_chunk(layer_tensors: dict[str, torch.Tensor]) -> bytes:
    """Legacy chunk: per-block serialized bytes for each layer."""
    codec = TelysChunkCodec()
    layer_blobs: dict[str, list[bytes]] = {}
    for name, stacked in layer_tensors.items():
        blobs: list[bytes] = []
        for block_idx in range(BLOCKS):
            blobs.append(serialize_kv_tensor(stacked[block_idx]))
        layer_blobs[name] = blobs
    return codec.encode_chunk(layer_blobs)


def build_new_chunk(layer_tensors: dict[str, torch.Tensor]) -> bytes:
    """Fast chunk: one raw-byte tensor per layer."""
    codec = TelysFastChunkCodec()
    return codec.encode_chunk(layer_tensors)


def old_load_transfer(payload: bytes) -> dict[str, torch.Tensor]:
    """Legacy load: decode, deserialize each block, stack, stage, transfer."""
    from telys_vllm.gpu_ops import deserialize_kv_tensor

    chunk_codec = TelysChunkCodec()

    # 1) blob read is implicit: `payload` is already a fresh bytes object.
    per_block_blobs = chunk_codec.decode_chunk(payload)

    # 2) decode + stage per layer.
    staged: dict[str, torch.Tensor] = {}
    layer_names = sorted(per_block_blobs[0].keys())
    for name in layer_names:
        block_tensors = [
            deserialize_kv_tensor(block_blobs[name]) for block_blobs in per_block_blobs
        ]
        stacked = torch.stack(block_tensors)
        staged[name] = torch.empty(stacked.shape, dtype=stacked.dtype, pin_memory=True)
        staged[name].copy_(stacked)

    # 3) mock H2D transfer.
    gpu: dict[str, torch.Tensor] = {}
    for name, pinned in staged.items():
        gpu[name] = torch.empty_like(pinned)
        gpu[name].copy_(pinned)
    return gpu


def new_load_transfer(payload: bytes) -> dict[str, torch.Tensor]:
    """Fast load: zero-decode to pinned tensor, then mock H2D transfer."""
    codec = TelysFastChunkCodec()

    # 1) blob read implicit, 2) zero-decode into pinned CPU staging.
    pinned = codec.decode_chunk_to_pinned(payload)

    # 3) mock H2D transfer.
    gpu: dict[str, torch.Tensor] = {}
    for name, staged in pinned.items():
        gpu[name] = torch.empty_like(staged)
        gpu[name].copy_(staged)
    return gpu


def timed(fn: callable[[], Any], runs: int = RUNS) -> tuple[float, Any]:
    """Return average elapsed seconds and the result of the last run."""
    result: Any = None
    total = 0.0
    for _ in range(runs):
        # Best-effort cache/allocator isolation.
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        total += t1 - t0
    return total / runs, result


def main() -> None:
    layer_tensors = make_layer_tensors()
    payload_per_block_bytes = sum(
        tensor.element_size() * tensor.numel() // BLOCKS
        for tensor in layer_tensors.values()
    )
    payload_total_bytes = payload_per_block_bytes * BLOCKS

    print(
        f"config: blocks={BLOCKS}, layers={LAYERS}, block_size={BLOCK_SIZE}, "
        f"heads={NUM_HEADS}, head_dim={HEAD_DIM}, dtype={DTYPE}"
    )
    print(f"KV payload per block: {payload_per_block_bytes / 1024 / 1024:.2f} MB")
    print(f"KV payload per chunk: {payload_total_bytes / 1024 / 1024:.2f} MB")

    old_payload = build_old_chunk(layer_tensors)
    new_payload = build_new_chunk(layer_tensors)
    print(f"old encoded size: {_payload_size(old_payload)}")
    print(f"new encoded size: {_payload_size(new_payload)}")

    # Warm up.
    old_load_transfer(bytes(old_payload))
    new_load_transfer(bytes(new_payload))

    old_dt, old_result = timed(lambda: old_load_transfer(bytes(old_payload)))
    new_dt, new_result = timed(lambda: new_load_transfer(bytes(new_payload)))

    # Sanity: results have the same bytes (tensors equal).
    for name in sorted(layer_tensors):
        assert torch.equal(old_result[name], new_result[name]), name

    print(f"\nload+stage+transfer per chunk (avg of {RUNS} runs):")
    print(f"  old: {old_dt * 1000:.2f} ms")
    print(f"  new: {new_dt * 1000:.2f} ms")
    print(f"  speedup: {old_dt / new_dt:.2f}x")
    print(f"  time saved: {(old_dt - new_dt) * 1000:.2f} ms")
    print(
        f"  effective throughput: "
        f"{payload_total_bytes / (1024 * 1024) / new_dt:.1f} MB/s (new)"
    )


if __name__ == "__main__":
    main()
