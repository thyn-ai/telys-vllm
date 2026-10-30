# SPDX-License-Identifier: Apache-2.0
"""Measure the connector's data-plane encode/decode overhead: v2 vs v3 vs v4.

v2: one base64-JSON metadata blob per block.
v3: one raw-bytes chunk record per contiguous run of blocks.
v4: same raw-bytes chunk as v3, but payload bytes live in an append-only slab
    file; Telys stores only the slab pointer + provenance metadata.

Synthetic payload sizes are calibrated to a ~1.5B-class model with
block_size=16 and ~28 attention layers.
"""

import base64
import json
import random
import tempfile
import time
from pathlib import Path

from telys_vllm.codec import LengthPrefixedCodec, TelysChunkCodec
from telys_vllm.slab import KVSlab

BLOCKS = 106
LAYERS = 28
BLOCK_BLOB_BYTES = 200_000  # ~torch.save'd KV block tensor for one layer
RUNS = 3


def make_block_blobs() -> dict[str, bytes]:
    rng = random.Random(42)
    return {
        f"layer.{i}": bytes(rng.randint(0, 255) for _ in range(BLOCK_BLOB_BYTES))
        for i in range(LAYERS)
    }


def old_per_block_encode_decode(
    block_blobs: dict[str, bytes],
) -> tuple[float, float, int]:
    """v2 path: JSON+base64 wrapper for every block, repeated BLOCKS times."""
    codec = LengthPrefixedCodec()
    block_payload = codec.encode_block(block_blobs)

    encode_times = []
    decode_times = []
    encoded_records = []

    for _ in range(BLOCKS):
        t0 = time.perf_counter()
        blob = {
            "payload": base64.b64encode(block_payload).decode("ascii"),
            "metadata": {"vllm.model": "fake"},
        }
        encoded = json.dumps(blob, separators=(",", ":"), sort_keys=True)
        t1 = time.perf_counter()
        encode_times.append(t1 - t0)
        encoded_records.append(encoded)

    for encoded in encoded_records:
        t0 = time.perf_counter()
        raw = json.loads(encoded)
        decoded_bytes = base64.b64decode(raw["payload"], validate=True)
        codec.decode_block(decoded_bytes)
        t1 = time.perf_counter()
        decode_times.append(t1 - t0)

    return sum(encode_times), sum(decode_times), len(encoded_records)


def new_chunk_encode_decode(block_blobs: dict[str, bytes]) -> tuple[float, float, int]:
    """v3 path: one raw-bytes chunk record for the whole 106-block run."""
    codec = TelysChunkCodec()
    layer_blobs = {name: [blob] * BLOCKS for name, blob in block_blobs.items()}

    t0 = time.perf_counter()
    payload = codec.encode_chunk(layer_blobs)
    t1 = time.perf_counter()
    encode_time = t1 - t0

    t0 = time.perf_counter()
    codec.decode_chunk(payload)
    t1 = time.perf_counter()
    decode_time = t1 - t0

    return encode_time, decode_time, 1


def slab_chunk_encode_decode(
    block_blobs: dict[str, bytes], slab: KVSlab
) -> tuple[float, float, float, int]:
    """v4 path: encode a chunk, append it to the slab, then read it back and decode."""
    codec = TelysChunkCodec()
    layer_blobs = {name: [blob] * BLOCKS for name, blob in block_blobs.items()}

    t0 = time.perf_counter()
    payload = codec.encode_chunk(layer_blobs)
    t1 = time.perf_counter()
    encode_time = t1 - t0

    t0 = time.perf_counter()
    offset, length = slab.append(payload)
    t1 = time.perf_counter()
    append_time = t1 - t0

    t0 = time.perf_counter()
    read_payload = slab.read(offset, length)
    decoded = codec.decode_chunk(read_payload)
    t1 = time.perf_counter()
    decode_time = t1 - t0

    # Touch the decoded data so the interpreter doesn't optimize away the work.
    assert len(decoded) == BLOCKS
    return encode_time, append_time, decode_time, 1


def main():
    block_blobs = make_block_blobs()
    print(f"block_blob_bytes={BLOCK_BLOB_BYTES}, blocks={BLOCKS}, layers={LAYERS}")

    old_encode_total = old_decode_total = 0.0
    new_encode_total = new_decode_total = 0.0
    v4_encode_total = v4_append_total = v4_decode_total = 0.0

    with tempfile.TemporaryDirectory(prefix="telys-vllm-bench-") as td:
        slab = KVSlab(Path(td) / "kv.slab")
        try:
            for run in range(RUNS):
                old_enc, old_dec, old_records = old_per_block_encode_decode(block_blobs)
                new_enc, new_dec, new_records = new_chunk_encode_decode(block_blobs)
                v4_enc, v4_app, v4_dec, v4_records = slab_chunk_encode_decode(
                    block_blobs, slab
                )
                print(
                    f"run {run + 1}: "
                    f"old={old_enc * 1000:.2f}ms encode / {old_dec * 1000:.2f}ms decode "
                    f"({old_records} records); "
                    f"v3={new_enc * 1000:.2f}ms encode / {new_dec * 1000:.2f}ms decode "
                    f"({new_records} record); "
                    f"v4={v4_enc * 1000:.2f}ms encode / {v4_app * 1000:.2f}ms append / "
                    f"{v4_dec * 1000:.2f}ms decode ({v4_records} record)"
                )
                old_encode_total += old_enc
                old_decode_total += old_dec
                new_encode_total += new_enc
                new_decode_total += new_dec
                v4_encode_total += v4_enc
                v4_append_total += v4_app
                v4_decode_total += v4_dec
        finally:
            slab.close()

    print("\nAverages:")
    print(
        f"  v2: encode={(old_encode_total / RUNS) * 1000:.2f}ms, "
        f"decode={(old_decode_total / RUNS) * 1000:.2f}ms, records={BLOCKS}"
    )
    print(
        f"  v3: encode={(new_encode_total / RUNS) * 1000:.2f}ms, "
        f"decode={(new_decode_total / RUNS) * 1000:.2f}ms, records=1"
    )
    print(
        f"  v4: encode={(v4_encode_total / RUNS) * 1000:.2f}ms, "
        f"append={(v4_append_total / RUNS) * 1000:.2f}ms, "
        f"decode={(v4_decode_total / RUNS) * 1000:.2f}ms, records=1"
    )
    print(
        f"  v2->v3 encode speedup: {old_encode_total / new_encode_total:.1f}x, "
        f"decode speedup: {old_decode_total / new_decode_total:.1f}x"
    )
    print(
        f"  v3->v4 append overhead: {v4_append_total / new_encode_total * 100:.1f}% "
        f"of v3 encode time"
    )


if __name__ == "__main__":
    main()
